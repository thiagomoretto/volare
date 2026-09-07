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
            let mut wait = 0;
            // Per-dimension weighting of the prices belongs here.
            let ok = walk(
                m,
                d,
                route,
                veh,
                v,
                |_, _, cost| soft += cost,
                |_, arrive, cumul| wait += cumul - arrive,
            );
            if !ok {
                return None;
            }
            soft += wait * d.wait_cost[v.index()];
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

/// Forward pass of `route` on `d`, `false` once a hard bound breaks. Soft
/// excess goes to `excess` as `(node, units, cost)`, `None` for the
/// vehicle's peak. Every stop, start and end included, goes to `visit` as
/// `(node, arrive, cumul)`; the gap between the two is the wait. The start
/// has no arrival, so both are its departure. Shared by the hot loop, the
/// violation report and the schedule.
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
    let mut cumul = d.start_cumul.max(d.lower_bound[veh.start.index()]);
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

/// One soft bound exceeded: at `node`, or `None` for the vehicle's peak.
/// `penalty` is what it added to the cost.
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

/// One visit on a finished route, with every dimension's value at it.
/// Both vectors are indexed by dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stop {
    pub node: NodeId,
    /// Before any wait. At a start node, the departure.
    pub arrive: Vec<i64>,
    /// After the wait for the node's lower bound: when service starts.
    pub cumul: Vec<i64>,
}

impl Stop {
    pub fn wait(&self, dimension: usize) -> i64 {
        self.cumul[dimension] - self.arrive[dimension]
    }
}

/// Every dimension at every stop of a solution, for reading finished routes:
/// arrival, service start, wait, load. Built once, then looked up by vehicle
/// or by node. Nodes on the drop sink have no stop.
pub struct Schedule {
    routes: Vec<Vec<Stop>>,
}

impl Schedule {
    /// Panics on an infeasible route; a solver result never is.
    pub fn of(m: &Model, sol: &Routes) -> Schedule {
        let dims = m.dimensions().len();
        let mut routes = Vec::with_capacity(sol.len());
        for (v, route) in sol.iter().enumerate() {
            let vehicle = VehicleId(v as u32);
            if m.unserved_vehicle() == Some(vehicle) {
                routes.push(Vec::new());
                continue;
            }
            let veh = m.vehicle(vehicle);
            let mut stops: Vec<Stop> = std::iter::once(&veh.start)
                .chain(route)
                .chain(std::iter::once(&veh.end))
                .map(|&node| Stop {
                    node,
                    arrive: vec![0; dims],
                    cumul: vec![0; dims],
                })
                .collect();
            for (k, d) in m.dimensions().iter().enumerate() {
                let mut i = 0;
                let ok = walk(
                    m,
                    d,
                    route,
                    veh,
                    vehicle,
                    |_, _, _| {},
                    |_, arrive, cumul| {
                        stops[i].arrive[k] = arrive;
                        stops[i].cumul[k] = cumul;
                        i += 1;
                    },
                );
                assert!(ok, "vehicle {v} runs an infeasible route");
            }
            routes.push(stops);
        }
        Schedule { routes }
    }

    /// Start through end, in visiting order. Empty for the drop sink.
    pub fn route(&self, v: VehicleId) -> &[Stop] {
        &self.routes[v.index()]
    }

    /// The stop serving `n`, `None` if `n` is unserved or a terminal.
    /// Terminals are shared between vehicles, so they are reachable through
    /// `route` only.
    // ponytail: linear scan per lookup. Index node -> (vehicle, position) if
    // a caller ever asks for every node of a big solution.
    pub fn stop(&self, n: NodeId) -> Option<&Stop> {
        self.routes
            .iter()
            .flat_map(|r| r.get(1..r.len().saturating_sub(1)).unwrap_or(&[]))
            .find(|s| s.node == n)
    }

    /// Index of dimension `name` into a stop's vectors.
    pub fn dimension(m: &Model, name: &str) -> usize {
        m.dimensions()
            .iter()
            .position(|d| d.name == name)
            .expect("unknown dimension")
    }
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
