use crate::model::{Dimension, Model, Vehicle};
use crate::types::{Cost, NodeId, VehicleId};

/// Routes indexed by vehicle. Each holds the visits *between* the vehicle's
/// start and end nodes, terminals excluded.
pub type Routes = Vec<Vec<NodeId>>;

/// Cost of running `route` on vehicle `v`, or `None` if it is infeasible.
///
/// The central primitive: construction and every operator route through here.
/// Always true cost — guided local search layers its penalties on top of this
/// in the search module, never inside this loop.
///
/// Feasibility is a full forward pass, O(route length), recomputed on every
/// call. That is the deliberate ceiling; to lift it, cache cumul prefixes
/// per route.
#[inline]
pub fn eval_route(m: &Model, route: &[NodeId], v: VehicleId) -> Option<Cost> {
    let (arcs, soft) = eval_route_split(m, route, v)?;
    Some(arcs + soft)
}

/// `eval_route` with the arc cost and the soft-bound penalty kept apart.
pub fn eval_route_split(m: &Model, route: &[NodeId], v: VehicleId) -> Option<(Cost, Cost)> {
    // An unused vehicle is free, it never leaves the depot.
    if route.is_empty() {
        return Some((0, 0));
    }
    let veh = m.vehicle(v);

    // The early-out keeps an unrestricted vehicle at one branch per call;
    // the scan itself is one bit test per node.
    if !veh.forbidden.is_empty() && route.iter().any(|&n| veh.forbids(n)) {
        return None;
    }

    let mut soft = 0;
    // A dropped node's window or ordering must not block dropping it.
    if m.unserved_vehicle() != Some(v) {
        if m.has_precedence() && !precedence_holds(m, route) {
            return None;
        }
        for d in m.dimensions() {
            // Per-dimension weighting of the prices belongs here.
            if !walk(m, d, route, veh, v, |_, _, cost| soft += cost, |_, _, _| {}) {
                return None;
            }
        }
    }

    let mut cost = 0;
    let mut prev = veh.start;
    for &node in route.iter().chain(std::iter::once(&veh.end)) {
        cost += m.eval(veh.cost_class, prev, node);
        prev = node;
    }
    Some((cost, soft))
}

/// Forward pass of `route` on `d`, `false` once a hard bound breaks. Every
/// priced unit goes to `excess` as `(node, units, cost)`: soft excess at a
/// node, a priced wait at a node, `None` for the vehicle's peak. Every stop,
/// start and end included, goes to `visit` as `(node, arrive, cumul)`; the
/// gap between the two is the wait. The start has no arrival, so both are
/// its departure.
fn walk(
    m: &Model,
    d: &Dimension,
    route: &[NodeId],
    veh: &Vehicle,
    v: VehicleId,
    mut excess: impl FnMut(Option<NodeId>, i64, Cost),
    mut visit: impl FnMut(NodeId, i64, i64),
) -> bool {
    let cap = d.max_cumul[v.index()];
    let wait_cost = d.wait_cost[v.index()];
    let mut cumul = d.lower_bound[veh.start.index()];
    if cumul > cap {
        return false;
    }
    visit(veh.start, cumul, cumul);
    let mut peak = cumul;
    let mut prev = veh.start;
    for &node in route.iter().chain(std::iter::once(&veh.end)) {
        // Late is infeasible before the clamp; early waits via the clamp.
        let arrive = cumul + m.eval(d.transit, prev, node);
        if arrive > d.upper_bound[node.index()] {
            return false;
        }
        if arrive > d.soft_upper_bound[node.index()] {
            let units = arrive - d.soft_upper_bound[node.index()];
            excess(
                Some(node),
                units,
                units * d.soft_upper_bound_cost[node.index()],
            );
        }
        cumul = arrive.max(d.lower_bound[node.index()]);
        if cumul > cap {
            return false;
        }
        if wait_cost > 0 && cumul > arrive {
            excess(Some(node), cumul - arrive, (cumul - arrive) * wait_cost);
        }
        visit(node, arrive, cumul);
        peak = peak.max(cumul);
        prev = node;
    }
    if peak > d.soft_max_cumul[v.index()] {
        let units = peak - d.soft_max_cumul[v.index()];
        excess(None, units, units * d.soft_max_cumul_cost[v.index()]);
    }
    true
}

/// One priced thing: a soft bound exceeded at `node`, a priced wait at
/// `node`, or `None` for the vehicle's peak. `penalty` is what it added to
/// the cost. `walk_route` tells a wait from a late arrival.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Violation {
    pub dimension: usize,
    pub vehicle: VehicleId,
    pub node: Option<NodeId>,
    pub excess: i64,
    pub penalty: Cost,
}

/// Every soft bound `sol` exceeds, and what each one cost.
pub fn violations(m: &Model, sol: &Routes) -> Vec<Violation> {
    let mut out = Vec::new();
    for (v, route) in sol.iter().enumerate() {
        let vehicle = VehicleId(v as u32);
        if m.unserved_vehicle() == Some(vehicle) {
            continue;
        }
        for (dimension, d) in m.dimensions().iter().enumerate() {
            walk(
                m,
                d,
                route,
                m.vehicle(vehicle),
                vehicle,
                |node, excess, penalty| {
                    out.push(Violation {
                        dimension,
                        vehicle,
                        node,
                        excess,
                        penalty,
                    })
                },
                |_, _, _| {},
            );
        }
    }
    out
}

/// Forward pass of `route` on vehicle `v` over dimension `dim`, one call per
/// stop as `(node, arrive, cumul)`, start and end included. `cumul - arrive`
/// is the wait; at the start both are the departure. `false` once a hard
/// bound breaks, the stops before it already visited. The drop sink has no
/// timetable.
pub fn walk_route(
    m: &Model,
    route: &[NodeId],
    v: VehicleId,
    dim: usize,
    visit: impl FnMut(NodeId, i64, i64),
) -> bool {
    let d = &m.dimensions()[dim];
    walk(m, d, route, m.vehicle(v), v, |_, _, _| {}, visit)
}

// ponytail: linear `route[..i]` scan. Swap for a timestamped position array
// if dense pickup-and-delivery ever makes this the hot spot.
fn precedence_holds(m: &Model, route: &[NodeId]) -> bool {
    route
        .iter()
        .enumerate()
        .all(|(i, &n)| m.successors(n).iter().all(|s| !route[..i].contains(s)))
}

/// True cost of every route, or `None` if any is infeasible.
pub fn eval_routes(m: &Model, sol: &Routes) -> Option<Cost> {
    (0..sol.len()).try_fold(0, |acc, v| {
        Some(acc + eval_route(m, &sol[v], VehicleId(v as u32))?)
    })
}

/// Every non-terminal node visited exactly once.
pub fn visits_all_nodes(m: &Model, sol: &Routes) -> bool {
    let mut seen = vec![0u32; m.node_count()];
    for route in sol {
        for &n in route {
            seen[n.index()] += 1;
        }
    }
    (0..m.node_count()).all(|i| {
        let expected = if m.is_terminal(NodeId(i as u32)) {
            0
        } else {
            1
        };
        seen[i] == expected
    })
}
