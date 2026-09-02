use crate::model::Model;
use crate::types::{Cost, NodeId, VehicleId};

/// Routes indexed by vehicle. Each holds the visits *between* the vehicle's
/// start and end nodes, terminals excluded.
pub type Routes = Vec<Vec<NodeId>>;

/// Cost of running `route` on vehicle `v`, or `None` if it is infeasible.
///
/// Always true cost, and stateless. The solver evaluates through
/// [`Search`](crate::Search) instead, which caches what this recomputes and
/// can carry a penalized objective; this one stays the reference the cached
/// path is checked against. Reach for it when you want the cost of a route
/// and nothing else.
///
/// Feasibility is a full forward pass, recomputed on every call.
pub fn eval_route(m: &Model, route: &[NodeId], v: VehicleId) -> Option<Cost> {
    // An unused vehicle is free: it never leaves the depot.
    if route.is_empty() {
        return Some(0);
    }
    if !vehicle_allows(m, route, v) {
        return None;
    }
    // A dropped node's window or ordering must not block dropping it.
    if m.unserved_vehicle() != Some(v) {
        if m.has_precedence() && !precedence_holds(m, route) {
            return None;
        }
        if !dimensions_hold(m, route, v) {
            return None;
        }
    }

    let veh = m.vehicle(v);
    let mut cost = 0;
    let mut prev = veh.start;
    for &node in route.iter().chain(std::iter::once(&veh.end)) {
        cost += m.eval(veh.cost_class, prev, node);
        prev = node;
    }
    Some(cost)
}

/// No node on `route` is one that `v` refuses to carry.
///
/// The early-out keeps an unrestricted vehicle at one branch per call.
pub(crate) fn vehicle_allows(m: &Model, route: &[NodeId], v: VehicleId) -> bool {
    let veh = m.vehicle(v);
    veh.forbidden.is_empty() || !route.iter().any(|&n| veh.forbids(n))
}

/// Every dimension stays inside its per-vehicle limit and every node's window.
///
/// Shared with [`Search::eval`](crate::Search::eval): precedence is the only
/// part of feasibility the two check differently, so it is the only part
/// written twice.
pub(crate) fn dimensions_hold(m: &Model, route: &[NodeId], v: VehicleId) -> bool {
    let veh = m.vehicle(v);
    for d in m.dimensions() {
        let cap = d.max_cumul[v.index()];
        let mut cumul = d.start_cumul.max(d.lower_bound[veh.start.index()]);
        if cumul > cap {
            return false;
        }
        let mut prev = veh.start;
        for &node in route.iter().chain(std::iter::once(&veh.end)) {
            // Late is infeasible before the clamp; early waits via the clamp.
            let arrive = cumul + m.eval(d.transit, prev, node);
            if arrive > d.upper_bound[node.index()] {
                return false;
            }
            cumul = arrive.max(d.lower_bound[node.index()]);
            if cumul > cap {
                return false;
            }
            prev = node;
        }
    }
    true
}

// The linear scan stays: a reference that shared `Search`'s stamped index
// would not catch that index going wrong.
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
