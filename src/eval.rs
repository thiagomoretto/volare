use crate::model::Model;
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
pub fn eval_route(m: &Model, route: &[NodeId], v: VehicleId) -> Option<Cost> {
    // ponytail: an unused vehicle is free, it never leaves the depot.
    if route.is_empty() {
        return Some(0);
    }
    let veh = m.vehicle(v);

    // The early-out keeps an unrestricted vehicle at one branch per call;
    // the scan itself is one bit test per node.
    if !veh.forbidden.is_empty() && route.iter().any(|&n| veh.forbids(n)) {
        return None;
    }

    // The unserved sink skips both checks: a dropped node's window or its
    // ordering must not make dropping it infeasible.
    if m.unserved_vehicle() != Some(v) {
        if m.has_precedence() && !precedence_holds(m, route) {
            return None;
        }
        for d in m.dimensions() {
            let cap = d.max_cumul[v.index()];
            let mut cumul = d.start_cumul.max(d.lower_bound[veh.start.index()]);
            if cumul > cap {
                return None;
            }
            let mut prev = veh.start;
            for &node in route.iter().chain(std::iter::once(&veh.end)) {
                // Late is infeasible before the clamp; early waits via the clamp.
                let arrive = cumul + m.eval(d.transit, prev, node);
                if arrive > d.upper_bound[node.index()] {
                    return None;
                }
                cumul = arrive.max(d.lower_bound[node.index()]);
                if cumul > cap {
                    return None;
                }
                prev = node;
            }
        }
    }

    let mut cost = 0;
    let mut prev = veh.start;
    for &node in route.iter().chain(std::iter::once(&veh.end)) {
        cost += m.eval(veh.cost_class, prev, node);
        prev = node;
    }
    Some(cost)
}

/// Every ordered pair the route holds both ends of runs `before` first.
///
/// One backward scan per declared successor. A node with no successors costs
/// nothing, which is every node in a model that never called `precede`, and
/// nearly every node in one that did.
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
