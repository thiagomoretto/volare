use std::ops::ControlFlow;

use super::construct::Rng;
use super::{SearchEvent, candidate_vehicles};
use crate::eval::{RouteCache, Routes, eval_route_split};
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
    assert!(p.avg_removed >= 1, "avg_removed must be at least 1");
    assert!(p.max_string >= 1, "max_string must be at least 1");
    assert!(
        p.end_temperature >= 0.0 && p.end_temperature <= p.start_temperature,
        "temperatures must satisfy 0 <= end <= start"
    );
    assert!(
        p.start_temperature.is_finite(),
        "start_temperature must be finite"
    );
    let sink = m.unserved_vehicle().map(|v| v.index());
    let mut rng = Rng(p.seed);
    let customers: Vec<NodeId> = (0..m.node_count() as u32)
        .map(NodeId)
        .filter(|&n| !m.is_terminal(n))
        .collect();
    let adj = neighbors(m, &customers);
    let depot = m.vehicle(VehicleId(0)).start;
    let cost_class = m.vehicle(VehicleId(0)).cost_class;

    // `caches` follows `work`; `cost` holds each route's cost in `sol`.
    let mut caches: Vec<RouteCache> = (0..sol.len()).map(|_| RouteCache::default()).collect();
    for (v, (c, route)) in caches.iter_mut().zip(sol.iter()).enumerate() {
        assert!(c.rebuild(m, route, VehicleId(v as u32)), "infeasible start");
    }
    let mut cost: Vec<Cost> = caches.iter().map(RouteCache::cost).collect();
    let mut cur_cost: Cost = cost.iter().sum();
    let mut best = sol.clone();
    let mut best_cost = cur_cost;

    // Same scaling as the GLS lambda: dropped nodes and soft prices are not
    // distances, so only served arcs set the temperature.
    let (mut served, mut arc_cost) = (0usize, 0);
    for (v, route) in sol.iter().enumerate() {
        if Some(v) != sink {
            served += route.len();
            arc_cost += eval_route_split(m, route, VehicleId(v as u32))
                .expect("infeasible start")
                .0;
        }
    }
    let mean_arc = arc_cost as f64 / served.max(1) as f64;
    let temperature = |iter: usize| {
        if p.start_temperature == 0.0 {
            return 0.0;
        }
        let fall = p.end_temperature / p.start_temperature;
        mean_arc * p.start_temperature * fall.powf(iter as f64 / p.iters as f64)
    };

    let mut stopped = log(SearchEvent::SisrBest {
        iter: 0,
        cost: best_cost,
    })
    .is_break();

    // A round edits `work`, then either copies the routes it touched into
    // `sol` or restores them from it; the untouched ones never move.
    let mut work = sol.clone();
    let mut at: Vec<(usize, usize)> = vec![(usize::MAX, 0); m.node_count()];
    let mut ruined: Vec<usize> = Vec::new();
    let mut removed: Vec<NodeId> = Vec::new();
    let mut ctx = Recreate::new(sol.len());

    for iter in 1..=p.iters {
        if stopped {
            break;
        }
        locate(&work, &mut at);
        ruin(
            &mut rng,
            &adj,
            &customers,
            sink,
            &mut work,
            &at,
            p,
            &mut ruined,
            &mut removed,
        );
        for &v in &ruined {
            ctx.touch(v);
        }
        // Removing a stop can break a route: an earlier arrival can wait
        // past a cap, and a transit need not obey the triangle inequality.
        let cut_ok = ruined
            .iter()
            .all(|&v| caches[v].rebuild(m, &work[v], VehicleId(v as u32)));
        order(m, &mut rng, &mut removed, depot, cost_class);

        let mut accepted = false;
        if cut_ok
            && recreate(
                m,
                &mut rng,
                &mut work,
                &mut caches,
                &removed,
                sink,
                &mut ctx,
            )
        {
            let new_cost = cur_cost
                + ctx
                    .touched
                    .iter()
                    .map(|&v| caches[v].cost() - cost[v])
                    .sum::<Cost>();
            if (new_cost as f64) < cur_cost as f64 + temperature(iter) * exp1(&mut rng) {
                accepted = true;
                cur_cost = new_cost;
            }
        }
        for &v in &ctx.touched {
            if accepted {
                sol[v].clone_from(&work[v]);
                cost[v] = caches[v].cost();
            } else {
                work[v].clone_from(&sol[v]);
                caches[v].rebuild(m, &work[v], VehicleId(v as u32));
            }
            ctx.marked[v] = false;
        }
        ctx.touched.clear();

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

/// Each customer's nearest customers, itself first, by the first vehicle's
/// arc cost.
fn neighbors(m: &Model, customers: &[NodeId]) -> Vec<Vec<NodeId>> {
    let cost_class = m.vehicle(VehicleId(0)).cost_class;
    let mut adj = vec![Vec::new(); m.node_count()];
    let mut by_cost: Vec<(Cost, NodeId)> = Vec::with_capacity(customers.len());
    for &c in customers {
        by_cost.clear();
        by_cost.extend(customers.iter().map(|&o| {
            (
                if o == c {
                    Cost::MIN
                } else {
                    m.eval(cost_class, c, o)
                },
                o,
            )
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

/// Cut strings out of routes near a random seed customer, into `removed`.
/// `ruined` collects the routes cut.
#[allow(clippy::too_many_arguments)]
fn ruin(
    rng: &mut Rng,
    adj: &[Vec<NodeId>],
    customers: &[NodeId],
    sink: Option<usize>,
    sol: &mut Routes,
    at: &[(usize, usize)],
    p: SisrParams,
    ruined: &mut Vec<usize>,
    removed: &mut Vec<NodeId>,
) {
    ruined.clear();
    removed.clear();

    let (mut used, mut visits) = (0, 0);
    for (v, route) in sol.iter().enumerate() {
        if Some(v) != sink && !route.is_empty() {
            used += 1;
            visits += route.len();
        }
    }
    let max_len = (visits / used.max(1)).clamp(1, p.max_string);
    let strings = 1 + rng.below((4 * p.avg_removed / (1 + max_len)).max(1));

    let seed = customers[rng.below(customers.len())];
    for &c in &adj[seed.index()] {
        if ruined.len() >= strings {
            break;
        }
        let (v, pos) = at[c.index()];
        if ruined.contains(&v) {
            continue;
        }
        let route = &mut sol[v];
        let len = 1 + rng.below(route.len().min(max_len));
        if rng.below(2) == 0 || len == route.len() {
            cut_string(rng, route, pos, len, 0, removed);
        } else {
            // Grow a run of kept customers inside the window; the run is what
            // makes the removal leave slack in the middle of the route.
            let mut keep = 1;
            while len + keep < route.len() && rng.below(SPLIT_STOP) != 0 {
                keep += 1;
            }
            cut_string(rng, route, pos, len, keep, removed);
        }
        ruined.push(v);
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

/// Reinsertion order: mostly random, sometimes farthest from the depot first,
/// now and then nearest first.
///
/// ponytail: the paper also sorts by demand. There is no single demand in a
/// model of named dimensions; add it when one dimension can be marked as load.
fn order(m: &Model, rng: &mut Rng, removed: &mut [NodeId], depot: NodeId, cost_class: usize) {
    match rng.below(7) {
        0..=3 => {
            for i in (1..removed.len()).rev() {
                removed.swap(i, rng.below(i + 1));
            }
        }
        4..=5 => removed.sort_by_key(|&n| std::cmp::Reverse(m.eval(cost_class, depot, n))),
        _ => removed.sort_by_key(|&n| m.eval(cost_class, depot, n)),
    }
}

/// Buffers `recreate` reuses across rounds, and the routes a round touched.
struct Recreate {
    vehicles: Vec<usize>,
    touched: Vec<usize>,
    marked: Vec<bool>,
}

impl Recreate {
    fn new(vehicles: usize) -> Self {
        Recreate {
            vehicles: Vec::new(),
            touched: Vec::new(),
            marked: vec![false; vehicles],
        }
    }

    fn touch(&mut self, v: usize) {
        if !self.marked[v] {
            self.marked[v] = true;
            self.touched.push(v);
        }
    }
}

/// Put every node of `removed` back at its cheapest unskipped feasible
/// position, `false` as soon as one fits nowhere.
fn recreate(
    m: &Model,
    rng: &mut Rng,
    sol: &mut Routes,
    caches: &mut [RouteCache],
    removed: &[NodeId],
    sink: Option<usize>,
    ctx: &mut Recreate,
) -> bool {
    let nv = sol.len();
    for &u in removed {
        candidate_vehicles(m, sol, &mut ctx.vehicles);
        let mut best = best_insertion(m, rng, sol, caches, &ctx.vehicles, u, sink);
        // Per-vehicle forbids make empty vehicles differ, so the one empty
        // candidate is not enough when it refuses `u`.
        if best.is_none() && ctx.vehicles.len() < nv {
            let rest: Vec<usize> = (0..nv).filter(|v| !ctx.vehicles.contains(v)).collect();
            best = best_insertion(m, rng, sol, caches, &rest, u, sink);
        }
        let Some((_, v, pos)) = best else {
            return false;
        };
        sol[v].insert(pos, u);
        ctx.touch(v);
        let fits = caches[v].rebuild(m, &sol[v], VehicleId(v as u32));
        debug_assert!(fits, "insert_cost priced an infeasible insertion");
        if !fits {
            return false;
        }
    }
    true
}

/// Cheapest `(delta, vehicle, position)` for `u` over `vs`. The sink's order
/// is meaningless, so it is priced at one position only, and never skipped.
fn best_insertion(
    m: &Model,
    rng: &mut Rng,
    sol: &Routes,
    caches: &[RouteCache],
    vs: &[usize],
    u: NodeId,
    sink: Option<usize>,
) -> Option<(Cost, usize, usize)> {
    let mut best: Option<(Cost, usize, usize)> = None;
    for &v in vs {
        let route = &sol[v];
        let cache = &caches[v];
        let first = if Some(v) == sink { route.len() } else { 0 };
        for pos in first..=route.len() {
            if Some(v) != sink && rng.below(BLINK) == 0 {
                continue;
            }
            let Some(c) = cache.insert_cost(m, route, VehicleId(v as u32), pos, u) else {
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

/// A draw from the unit exponential, `-ln U` for `U` in `(0, 1]`.
fn exp1(rng: &mut Rng) -> f64 {
    let u = ((rng.next() >> 11) + 1) as f64 / (1u64 << 53) as f64;
    -u.ln()
}
