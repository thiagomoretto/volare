use std::ops::ControlFlow;

use super::{Rng, SearchEvent, candidate_vehicles};
use crate::eval::{RouteCache, Routes};
use crate::model::Model;
use crate::types::{Cost, NodeId, VehicleId};

/// One in this many split strings stops growing its kept run at each step.
const SPLIT_STOP: usize = 100;
/// One in this many insertion positions is skipped without being priced.
const BLINK: usize = 100;
/// Neighbors kept per customer. A ruin walks out from its seed until it has
/// cut enough routes, which happens long before this list runs out.
const NEIGHBORS: usize = 100;

/// How ruin and recreate searches. Start from `SisrParams::new` and set the
/// fields you want to change: new fields can then arrive without breaking
/// your code.
///
/// ```
/// # use volare::SisrParams;
/// let mut p = SisrParams::new(50_000);
/// p.avg_removed = 15;
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct SisrParams {
    /// Rounds to run.
    pub iters: usize,
    /// Same seed, same solution.
    pub seed: u64,
    /// Customers one ruin removes, on average. More reaches further from the
    /// current solution per round, and makes each round cost more.
    pub avg_removed: usize,
    /// Longest string one ruin cuts out of a route.
    pub max_string: usize,
    /// Temperature at the first round, as a fraction of the mean arc cost of
    /// the starting solution. A round that is worse by `d` is accepted with
    /// chance `exp(-d / temperature)`, so this sets how far uphill the search
    /// is willing to walk. Zero accepts only strict improvements.
    pub start_temperature: f64,
    /// Temperature at the last round, on the same scale. The schedule falls
    /// geometrically from the start to here.
    pub end_temperature: f64,
}

impl SisrParams {
    /// Defaults from the paper, with the temperature rescaled to the model's
    /// own arc costs.
    pub fn new(iters: usize) -> Self {
        SisrParams {
            iters,
            seed: 0,
            avg_removed: 10,
            max_string: 10,
            start_temperature: 0.5,
            end_temperature: 0.005,
        }
    }
}

/// Ruin and recreate with slack induction by string removals (Christiaens and
/// Vanden Berghe). Each round cuts a few short strings out of routes that lie
/// near one random customer, puts the customers back by cheapest insertion
/// with a small chance to skip each position, and accepts the result under
/// simulated annealing. The best true-cost solution seen is what comes back.
///
/// Constraints stay a black box: every insertion is priced as `eval_route`
/// would price it, through a per-route cache of the forward pass, and a
/// dropped node is an insertion into the unserved sink. A round that leaves a
/// cut route infeasible, or cannot place every customer, is thrown away.
pub fn ruin_recreate(m: &Model, sol: &mut Routes, p: SisrParams) {
    ruin_recreate_with(m, sol, p, |_| ControlFlow::Continue(()))
}

/// `ruin_recreate` reporting a `SisrBest` per new best cost, a `SisrRound` per
/// round, and a final `Done`. Stopped early by the callback, it still returns
/// the best solution so far.
pub fn ruin_recreate_with(
    m: &Model,
    sol: &mut Routes,
    p: SisrParams,
    mut log: impl FnMut(SearchEvent) -> ControlFlow<()>,
) {
    p.check();
    let mut draft = Draft::new(m, sol);
    let search = Search::new(m, p, &draft);
    let mut rng = Rng(p.seed);

    let mut cost: Vec<Cost> = draft.caches.iter().map(RouteCache::cost).collect();
    let mut cur_cost: Cost = cost.iter().sum();
    let mut best = sol.clone();
    let mut best_cost = cur_cost;
    let mut stopped = log(SearchEvent::SisrBest {
        iter: 0,
        cost: best_cost,
    })
    .is_break();

    let mut at: Vec<(usize, usize)> = vec![(usize::MAX, 0); m.node_count()];
    let mut removed: Vec<NodeId> = Vec::new();
    for iter in 1..=p.iters {
        if stopped {
            break;
        }
        locate(&draft.routes, &mut at);
        search.ruin(&mut rng, &mut draft, &at, &mut removed);
        let cut_ok = draft.rebuild_touched(m);
        search.order(&mut rng, &mut removed);

        let mut accepted = false;
        if cut_ok && search.recreate(&mut rng, &mut draft, &removed) {
            let new_cost = cur_cost + draft.delta(&cost);
            accepted = (new_cost as f64) < cur_cost as f64 + search.temperature(iter) * rng.exp1();
            if accepted {
                cur_cost = new_cost;
            }
        }
        if accepted {
            draft.commit(sol, &mut cost);
        } else {
            draft.rollback(m, sol);
        }

        if accepted && cur_cost < best_cost {
            best_cost = cur_cost;
            best.clone_from(sol);
            stopped = log(SearchEvent::SisrBest {
                iter,
                cost: best_cost,
            })
            .is_break();
        }
        stopped = stopped
            || log(SearchEvent::SisrRound {
                iter,
                cost: cur_cost,
            })
            .is_break();
    }

    *sol = best;
    let _ = log(SearchEvent::Done { cost: best_cost });
}

impl SisrParams {
    fn check(&self) {
        assert!(self.avg_removed >= 1, "avg_removed must be at least 1");
        assert!(self.max_string >= 1, "max_string must be at least 1");
        assert!(
            self.end_temperature >= 0.0 && self.end_temperature <= self.start_temperature,
            "temperatures must satisfy 0 <= end <= start"
        );
        assert!(
            self.start_temperature.is_finite(),
            "start_temperature must be finite"
        );
    }
}

/// What a search knows that never changes while it runs.
struct Search<'a> {
    m: &'a Model,
    p: SisrParams,
    sink: Option<usize>,
    customers: Vec<NodeId>,
    /// Each customer's nearest customers, itself first.
    adj: Vec<Vec<NodeId>>,
    depot: NodeId,
    cost_class: usize,
    /// Mean served arc cost of the starting solution: the temperature's unit.
    /// Dropped nodes and soft prices are not distances, so they stay out.
    mean_arc: f64,
}

impl<'a> Search<'a> {
    fn new(m: &'a Model, p: SisrParams, start: &Draft) -> Self {
        let sink = m.unserved_vehicle().map(|v| v.index());
        let customers: Vec<NodeId> = (0..m.node_count() as u32)
            .map(NodeId)
            .filter(|&n| !m.is_terminal(n))
            .collect();
        let veh = m.vehicle(VehicleId(0));
        let (mut served, mut arcs) = (0usize, 0);
        for v in (0..start.routes.len()).filter(|&v| Some(v) != sink) {
            served += start.routes[v].len();
            arcs += start.caches[v].arcs();
        }
        Search {
            m,
            p,
            sink,
            adj: neighbors(m, &customers, veh.cost_class),
            customers,
            depot: veh.start,
            cost_class: veh.cost_class,
            mean_arc: arcs as f64 / served.max(1) as f64,
        }
    }

    /// Geometric from the start temperature to the end one over the run.
    fn temperature(&self, iter: usize) -> f64 {
        let p = &self.p;
        if p.start_temperature == 0.0 {
            return 0.0;
        }
        let fall = p.end_temperature / p.start_temperature;
        self.mean_arc * p.start_temperature * fall.powf(iter as f64 / p.iters as f64)
    }

    /// Cut strings out of routes near a random seed customer, into `removed`.
    /// The routes cut are the draft's touched set.
    fn ruin(
        &self,
        rng: &mut Rng,
        draft: &mut Draft,
        at: &[(usize, usize)],
        removed: &mut Vec<NodeId>,
    ) {
        removed.clear();
        let (mut used, mut visits) = (0, 0);
        for (v, route) in draft.routes.iter().enumerate() {
            if Some(v) != self.sink && !route.is_empty() {
                used += 1;
                visits += route.len();
            }
        }
        let max_len = (visits / used.max(1)).clamp(1, self.p.max_string);
        let strings = 1 + rng.below((4 * self.p.avg_removed / (1 + max_len)).max(1));

        let seed = self.customers[rng.below(self.customers.len())];
        for &c in &self.adj[seed.index()] {
            if draft.touched.len() >= strings {
                break;
            }
            let (v, pos) = at[c.index()];
            if draft.marked[v] {
                continue;
            }
            let route = &mut draft.routes[v];
            let len = 1 + rng.below(route.len().min(max_len));
            let mut keep = 0;
            if rng.below(2) != 0 && len < route.len() {
                // Grow a run of kept customers inside the window; the run is
                // what makes the removal leave slack in the middle of the
                // route.
                keep = 1;
                while len + keep < route.len() && rng.below(SPLIT_STOP) != 0 {
                    keep += 1;
                }
            }
            cut_string(rng, route, pos, len, keep, removed);
            draft.touch(v);
        }
    }

    /// Reinsertion order: mostly random, sometimes farthest from the depot
    /// first, now and then nearest first.
    ///
    /// ponytail: the paper also sorts by demand. There is no single demand in
    /// a model of named dimensions; add it when one dimension can be marked
    /// as load.
    fn order(&self, rng: &mut Rng, removed: &mut [NodeId]) {
        let from_depot = |n: NodeId| self.m.eval(self.cost_class, self.depot, n);
        match rng.below(7) {
            0..=3 => {
                for i in (1..removed.len()).rev() {
                    removed.swap(i, rng.below(i + 1));
                }
            }
            4..=5 => removed.sort_by_key(|&n| std::cmp::Reverse(from_depot(n))),
            _ => removed.sort_by_key(|&n| from_depot(n)),
        }
    }

    /// Put every node of `removed` back at its cheapest unskipped feasible
    /// position, `false` as soon as one fits nowhere.
    fn recreate(&self, rng: &mut Rng, draft: &mut Draft, removed: &[NodeId]) -> bool {
        let nv = draft.routes.len();
        for &u in removed {
            candidate_vehicles(self.m, &draft.routes, &mut draft.vehicles);
            let mut best = self.best_insertion(rng, draft, &draft.vehicles, u);
            // Per-vehicle forbids make empty vehicles differ, so the one empty
            // candidate is not enough when it refuses `u`.
            if best.is_none() && draft.vehicles.len() < nv {
                let rest: Vec<usize> = (0..nv).filter(|v| !draft.vehicles.contains(v)).collect();
                best = self.best_insertion(rng, draft, &rest, u);
            }
            let Some((_, v, pos)) = best else {
                return false;
            };
            if !draft.insert(self.m, v, pos, u) {
                return false;
            }
        }
        true
    }

    /// Cheapest `(delta, vehicle, position)` for `u` over `vs`. The sink's
    /// order is meaningless, so it is priced at one position only, and never
    /// skipped.
    fn best_insertion(
        &self,
        rng: &mut Rng,
        draft: &Draft,
        vs: &[usize],
        u: NodeId,
    ) -> Option<(Cost, usize, usize)> {
        let mut best: Option<(Cost, usize, usize)> = None;
        for &v in vs {
            let route = &draft.routes[v];
            let cache = &draft.caches[v];
            let is_sink = Some(v) == self.sink;
            let first = if is_sink { route.len() } else { 0 };
            for pos in first..=route.len() {
                if !is_sink && rng.below(BLINK) == 0 {
                    continue;
                }
                let Some(c) = cache.insert_cost(self.m, route, VehicleId(v as u32), pos, u) else {
                    continue;
                };
                let delta = c - cache.cost();
                if best.is_none_or(|(bd, ..)| delta < bd) {
                    best = Some((delta, v, pos));
                }
            }
        }
        best
    }
}

/// The solution a round edits, with every route's cache kept in step. A
/// round ends in `commit` or `rollback`, so between rounds the draft equals
/// the accepted solution and nothing is touched.
struct Draft {
    routes: Routes,
    caches: Vec<RouteCache>,
    /// Routes this round changed, once each, and a mark per vehicle.
    touched: Vec<usize>,
    marked: Vec<bool>,
    /// Reused by `candidate_vehicles`.
    vehicles: Vec<usize>,
}

impl Draft {
    fn new(m: &Model, sol: &Routes) -> Self {
        let mut caches: Vec<RouteCache> = (0..sol.len()).map(|_| RouteCache::default()).collect();
        for (v, (c, route)) in caches.iter_mut().zip(sol).enumerate() {
            assert!(c.rebuild(m, route, VehicleId(v as u32)), "infeasible start");
        }
        Draft {
            routes: sol.clone(),
            caches,
            touched: Vec::new(),
            marked: vec![false; sol.len()],
            vehicles: Vec::new(),
        }
    }

    fn touch(&mut self, v: usize) {
        if !self.marked[v] {
            self.marked[v] = true;
            self.touched.push(v);
        }
    }

    /// Refresh the caches of the routes cut so far. Removing a stop can break
    /// a route: an earlier arrival can wait past a cap, and a transit need
    /// not obey the triangle inequality.
    fn rebuild_touched(&mut self, m: &Model) -> bool {
        let Draft {
            routes,
            caches,
            touched,
            ..
        } = self;
        touched
            .iter()
            .all(|&v| caches[v].rebuild(m, &routes[v], VehicleId(v as u32)))
    }

    fn insert(&mut self, m: &Model, v: usize, pos: usize, u: NodeId) -> bool {
        self.routes[v].insert(pos, u);
        self.touch(v);
        let fits = self.caches[v].rebuild(m, &self.routes[v], VehicleId(v as u32));
        debug_assert!(fits, "insert_cost priced an infeasible insertion");
        fits
    }

    /// How much the draft costs over `cost`, the accepted routes' costs.
    fn delta(&self, cost: &[Cost]) -> Cost {
        self.touched
            .iter()
            .map(|&v| self.caches[v].cost() - cost[v])
            .sum()
    }

    fn commit(&mut self, sol: &mut Routes, cost: &mut [Cost]) {
        for &v in &self.touched {
            sol[v].clone_from(&self.routes[v]);
            cost[v] = self.caches[v].cost();
            self.marked[v] = false;
        }
        self.touched.clear();
    }

    fn rollback(&mut self, m: &Model, sol: &Routes) {
        for &v in &self.touched {
            self.routes[v].clone_from(&sol[v]);
            self.caches[v].rebuild(m, &self.routes[v], VehicleId(v as u32));
            self.marked[v] = false;
        }
        self.touched.clear();
    }
}

/// Each customer's nearest customers, itself first, by `cost_class`.
fn neighbors(m: &Model, customers: &[NodeId], cost_class: usize) -> Vec<Vec<NodeId>> {
    let mut adj = vec![Vec::new(); m.node_count()];
    let mut by_cost: Vec<(Cost, NodeId)> = Vec::with_capacity(customers.len());
    for &c in customers {
        by_cost.clear();
        by_cost.extend(customers.iter().map(|&o| {
            let d = if o == c {
                Cost::MIN
            } else {
                m.eval(cost_class, c, o)
            };
            (d, o)
        }));
        let k = NEIGHBORS.min(by_cost.len());
        if k < by_cost.len() {
            by_cost.select_nth_unstable(k);
        }
        by_cost.truncate(k);
        by_cost.sort_unstable();
        adj[c.index()] = by_cost.iter().map(|&(_, o)| o).collect();
    }
    adj
}

/// `at[n]` becomes `(route, position)` of every routed node.
fn locate(sol: &Routes, at: &mut [(usize, usize)]) {
    for (v, route) in sol.iter().enumerate() {
        for (i, &n) in route.iter().enumerate() {
            at[n.index()] = (v, i);
        }
    }
}

/// Remove `len` customers from a window of `len + keep` around `pos`, leaving
/// one run of `keep` in place.
fn cut_string(
    rng: &mut Rng,
    route: &mut Vec<NodeId>,
    pos: usize,
    len: usize,
    keep: usize,
    removed: &mut Vec<NodeId>,
) {
    let w = len + keep;
    let lo = (pos + 1).saturating_sub(w);
    let hi = pos.min(route.len() - w);
    let start = lo + rng.below(hi - lo + 1);
    let kept_at = start + rng.below(len + 1);
    let mut i = start;
    let mut k = 0;
    route.retain(|&n| {
        let in_window = i >= start && i < start + w;
        let in_kept = i >= kept_at && i < kept_at + keep;
        i += 1;
        if in_window && !in_kept {
            removed.push(n);
            k += 1;
            false
        } else {
            true
        }
    });
    debug_assert_eq!(k, len);
}
