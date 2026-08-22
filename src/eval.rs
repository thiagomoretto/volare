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

    for d in m.dimensions() {
        let cap = d.max_cumul[v.index()];
        let mut cumul = d.start_cumul.max(d.lower_bound[veh.start.index()]);
        if cumul > cap {
            return None;
        }
        let mut prev = veh.start;
        for &node in route.iter().chain(std::iter::once(&veh.end)) {
            cumul = (cumul + m.eval(d.transit, prev, node)).max(d.lower_bound[node.index()]);
            if cumul > cap {
                return None;
            }
            prev = node;
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
