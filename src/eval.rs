use std::cell::Cell;

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
/// call. `RouteCache` prices a single insertion without it.
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
    if forbids_any(veh, route) {
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

    Some((route_arcs(m, veh, route), soft))
}

/// The early-out keeps an unrestricted vehicle at one branch per call; the
/// scan itself is one bit test per node.
#[inline]
fn forbids_any(veh: &Vehicle, route: &[NodeId]) -> bool {
    !veh.forbidden.is_empty() && route.iter().any(|&n| veh.forbids(n))
}

/// Arc cost of `route` from the vehicle's start to its end.
#[inline]
fn route_arcs(m: &Model, veh: &Vehicle, route: &[NodeId]) -> Cost {
    let mut cost = 0;
    let mut prev = veh.start;
    for &node in route.iter().chain(std::iter::once(&veh.end)) {
        cost += m.eval(veh.cost_class, prev, node);
        prev = node;
    }
    cost
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
    let lim = Limits::of(d, v);
    let mut cumul = d.lower_bound[veh.start.index()];
    if cumul > lim.cap {
        return false;
    }
    visit(veh.start, cumul, cumul);
    let mut peak = cumul;
    let mut prev = veh.start;
    for &node in route.iter().chain(std::iter::once(&veh.end)) {
        let arrive = cumul + m.eval(d.transit, prev, node);
        let Some(next) = settle(d, &lim, node, arrive) else {
            return false;
        };
        cumul = next;
        if arrive > d.soft_upper_bound[node.index()] {
            let units = arrive - d.soft_upper_bound[node.index()];
            excess(
                Some(node),
                units,
                units * d.soft_upper_bound_cost[node.index()],
            );
        }
        let wait = cumul - arrive;
        if wait > 0 && lim.wait_cost > 0 {
            excess(Some(node), wait, wait * lim.wait_cost);
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

/// One dimension's per-vehicle limits.
struct Limits {
    cap: i64,
    wait_cost: Cost,
    wait_cap: i64,
}

impl Limits {
    #[inline]
    fn of(d: &Dimension, v: VehicleId) -> Self {
        Limits {
            cap: d.max_cumul[v.index()],
            wait_cost: d.wait_cost[v.index()],
            wait_cap: d.max_wait[v.index()],
        }
    }
}

/// The cumul at `node` after arriving at `arrive`, or `None` if a hard bound
/// breaks. Late is infeasible before the clamp; early waits via the clamp.
#[inline]
fn settle(d: &Dimension, lim: &Limits, node: NodeId, arrive: i64) -> Option<i64> {
    let n = node.index();
    if arrive > d.upper_bound[n] {
        return None;
    }
    let cumul = arrive.max(d.lower_bound[n]);
    if cumul > lim.cap {
        return None;
    }
    let wait = cumul - arrive;
    if wait > 0 && (wait > lim.wait_cap || wait > d.max_wait_at[n]) {
        return None;
    }
    Some(cumul)
}

/// What one stop adds to the price: lateness past its soft bound, and the
/// wait.
#[inline]
fn stop_price(d: &Dimension, lim: &Limits, node: NodeId, arrive: i64, cumul: i64) -> Cost {
    let soft = d.soft_upper_bound[node.index()];
    let late = if arrive > soft {
        (arrive - soft) * d.soft_upper_bound_cost[node.index()]
    } else {
        0
    };
    late + (cumul - arrive) * lim.wait_cost
}

#[inline]
fn peak_price(d: &Dimension, v: VehicleId, peak: i64) -> Cost {
    let soft = d.soft_max_cumul[v.index()];
    if peak > soft {
        (peak - soft) * d.soft_max_cumul_cost[v.index()]
    } else {
        0
    }
}

/// One feasible route's forward pass, kept so that inserting a node is priced
/// from the stop before it rather than from the start. `insert_cost` returns
/// what `eval_route` would on the route with the node inserted.
#[derive(Default)]
pub(crate) struct RouteCache {
    cost: Cost,
    arcs: Cost,
    dims: Vec<Trace>,
}

/// One dimension's pass over a route, per stop: start, visits, end.
#[derive(Default)]
struct Trace {
    arrive: Vec<i64>,
    cumul: Vec<i64>,
    /// Stop prices summed through this stop. The peak price is not in it.
    price: Vec<Cost>,
    /// Highest cumul through this stop, and from this stop to the end.
    peak: Vec<i64>,
    tail_peak: Vec<i64>,
    /// The most this stop may be reached late with every stop from here on
    /// still feasible. A late arrival shrinks by each wait it meets, so the
    /// slack grows by each wait going backward.
    slack: Vec<i64>,
    /// Some stop or the vehicle carries a price. A delay can move a price, so
    /// the slack alone cannot answer and the suffix must be walked.
    priced: bool,
}

/// A route's stops by index: 0 is the start, then the visits, then the end.
#[derive(Clone, Copy)]
struct Stops<'a> {
    veh: &'a Vehicle,
    route: &'a [NodeId],
}

impl Stops<'_> {
    #[inline]
    fn at(self, k: usize) -> NodeId {
        match k {
            0 => self.veh.start,
            k if k <= self.route.len() => self.route[k - 1],
            _ => self.veh.end,
        }
    }

    /// Index of the end stop.
    #[inline]
    fn end(self) -> usize {
        self.route.len() + 1
    }
}

impl RouteCache {
    /// The route's true cost, as `eval_route` prices it.
    #[inline]
    pub(crate) fn cost(&self) -> Cost {
        self.cost
    }

    /// The route's arc cost alone, without soft prices.
    #[inline]
    pub(crate) fn arcs(&self) -> Cost {
        self.arcs
    }

    /// Record `route` on `v`; `false` if it is infeasible, and the cache is
    /// then unusable until the next rebuild. Rebuilding in place keeps the
    /// buffers.
    pub(crate) fn rebuild(&mut self, m: &Model, route: &[NodeId], v: VehicleId) -> bool {
        let veh = m.vehicle(v);
        self.dims.resize_with(m.dimensions().len(), Trace::default);
        self.arcs = 0;
        self.cost = 0;
        if route.is_empty() {
            return true;
        }
        if forbids_any(veh, route) {
            return false;
        }
        self.arcs = route_arcs(m, veh, route);
        self.cost = self.arcs;
        if m.unserved_vehicle() == Some(v) {
            return true;
        }
        if m.has_precedence() && !precedence_holds(m, route) {
            return false;
        }
        let stops = Stops { veh, route };
        for (d, t) in m.dimensions().iter().zip(&mut self.dims) {
            let Some(price) = t.fill(m, d, stops, v) else {
                return false;
            };
            self.cost += price;
        }
        true
    }

    /// Cost of `route` with `u` inserted before `route[pos]`, or `None` if
    /// that is infeasible. `route` and `v` must be what the cache was built
    /// from.
    pub(crate) fn insert_cost(
        &self,
        m: &Model,
        route: &[NodeId],
        v: VehicleId,
        pos: usize,
        u: NodeId,
    ) -> Option<Cost> {
        let veh = m.vehicle(v);
        if veh.forbids(u) {
            return None;
        }
        if route.is_empty() {
            return eval_route(m, &[u], v);
        }
        let stops = Stops { veh, route };
        let (prev, next) = (stops.at(pos), stops.at(pos + 1));
        let cc = veh.cost_class;
        let arcs = self.arcs + m.eval(cc, prev, u) + m.eval(cc, u, next) - m.eval(cc, prev, next);
        if m.unserved_vehicle() == Some(v) {
            return Some(arcs);
        }
        if m.has_precedence()
            && (m.successors(u).iter().any(|s| route[..pos].contains(s))
                || route[pos..].iter().any(|&n| m.successors(n).contains(&u)))
        {
            return None;
        }
        let mut cost = arcs;
        for (d, t) in m.dimensions().iter().zip(&self.dims) {
            cost += t.insert_price(m, d, v, stops, pos, u)?;
        }
        Some(cost)
    }
}

impl Trace {
    /// Record `walk` over `stops`, and return what it priced the route at.
    fn fill(&mut self, m: &Model, d: &Dimension, stops: Stops, v: VehicleId) -> Option<Cost> {
        self.arrive.clear();
        self.cumul.clear();
        self.price.clear();
        self.peak.clear();
        // `walk` reports a stop's prices before the stop itself, and the
        // peak's after the last stop; both closures need the running sums.
        let (paid, total) = (Cell::new(0), Cell::new(0));
        let mut peak = i64::MIN;
        let feasible = walk(
            m,
            d,
            stops.route,
            stops.veh,
            v,
            |node, _, cost| {
                total.set(total.get() + cost);
                if node.is_some() {
                    paid.set(paid.get() + cost);
                }
            },
            |_, arrive, cumul| {
                peak = peak.max(cumul);
                self.arrive.push(arrive);
                self.cumul.push(cumul);
                self.price.push(paid.get());
                self.peak.push(peak);
            },
        );
        if !feasible {
            return None;
        }

        let lim = Limits::of(d, v);
        let n = self.cumul.len();
        self.tail_peak.clear();
        self.tail_peak.resize(n, i64::MIN);
        self.slack.clear();
        self.slack.resize(n, i64::MAX);
        let (mut tail, mut later) = (i64::MIN, i64::MAX);
        for k in (0..n).rev() {
            tail = tail.max(self.cumul[k]);
            self.tail_peak[k] = tail;
            let wait = self.cumul[k] - self.arrive[k];
            let room = lim.cap.saturating_sub(self.cumul[k]).min(later);
            later = d.upper_bound[stops.at(k).index()]
                .saturating_sub(self.arrive[k])
                .min(wait.saturating_add(room));
            self.slack[k] = later;
        }
        self.priced = lim.wait_cost > 0
            || d.soft_max_cumul[v.index()] != i64::MAX
            || (1..n).any(|k| d.soft_upper_bound[stops.at(k).index()] != i64::MAX);
        Some(total.get())
    }

    /// What this dimension prices the route at with `u` inserted after stop
    /// `pos`, `None` if infeasible. The walk stops as soon as the new cumul
    /// meets the cached one: from that stop on, nothing differs.
    fn insert_price(
        &self,
        m: &Model,
        d: &Dimension,
        v: VehicleId,
        stops: Stops,
        pos: usize,
        u: NodeId,
    ) -> Option<Cost> {
        let lim = Limits::of(d, v);
        let last = stops.end();
        let arrive = self.cumul[pos] + m.eval(d.transit, stops.at(pos), u);
        let mut cumul = settle(d, &lim, u, arrive)?;
        let mut price = self.price[pos] + stop_price(d, &lim, u, arrive, cumul);
        let mut peak = self.peak[pos].max(cumul);
        let mut prev = u;
        for k in pos + 1..=last {
            let node = stops.at(k);
            let arrive = cumul + m.eval(d.transit, prev, node);
            if !self.priced && arrive >= self.arrive[k] {
                return (arrive - self.arrive[k] <= self.slack[k]).then_some(price);
            }
            cumul = settle(d, &lim, node, arrive)?;
            price += stop_price(d, &lim, node, arrive, cumul);
            peak = peak.max(cumul);
            if cumul == self.cumul[k] {
                price += self.price[last] - self.price[k];
                peak = peak.max(self.tail_peak[k]);
                break;
            }
            prev = node;
        }
        Some(price + peak_price(d, v, peak))
    }
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
