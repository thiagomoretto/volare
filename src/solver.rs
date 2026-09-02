use std::collections::VecDeque;
use std::time::Instant;

use crate::eval::{Routes, eval_routes};
use crate::model::Model;
use crate::operators::{try_relocate, try_swap, try_two_opt, try_two_opt_star};
use crate::search::Search;
use crate::types::{Cost, NodeId, VehicleId};

/// The neighborhood operator that accepted an improving move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operator {
    Relocate,
    Swap,
    TwoOpt,
    TwoOptStar,
    /// One of the operators handed to [`solve_with_operators`], by its
    /// position in that slice.
    Custom(usize),
}

impl std::fmt::Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operator::Relocate => write!(f, "relocate"),
            Operator::Swap => write!(f, "swap"),
            Operator::TwoOpt => write!(f, "2-opt"),
            Operator::TwoOptStar => write!(f, "2-opt*"),
            Operator::Custom(i) => write!(f, "custom operator {i}"),
        }
    }
}

/// A progress point during a solve. Hand a callback to `solve_with`,
/// `first_solution_with` or `local_search_with` to observe them; `search_log`
/// builds one that prints progress lines. Costs are whole-solution
/// totals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchEvent {
    /// Construction placed every node; the first complete solution exists.
    FirstSolution { cost: Cost },
    /// `operator` accepted a move and the total cost dropped to `cost`.
    Improvement { operator: Operator, cost: Cost },
    /// Guided local search finished round `iter` holding a solution cheaper
    /// than anything before it. `cost` is the true cost, never the penalized
    /// one the descent was reading.
    GuidedBest { iter: usize, cost: Cost },
    /// Local search converged; the solution is final.
    Done { cost: Cost },
}

/// An event callback that prints progress lines to stderr,
/// prefixed with elapsed time since the closure was created:
///
/// ```text
/// #search    0.012s  relocate improved, cost 5900
/// ```
///
/// ```no_run
/// # use volare::solver::{Construct, Improve, solve_with, search_log};
/// # let model: volare::Model = todo!();
/// let routes = solve_with(
///     &model,
///     Construct::CheapestInsertion,
///     Improve::HillClimb,
///     search_log(),
/// );
/// ```
pub fn search_log() -> impl FnMut(SearchEvent) {
    let started = Instant::now();
    move |event| {
        let t = started.elapsed().as_secs_f64();
        match event {
            SearchEvent::FirstSolution { cost } => {
                eprintln!("#search {t:7.3}s  first solution, cost {cost}")
            }
            SearchEvent::Improvement { operator, cost } => {
                eprintln!("#search {t:7.3}s  {operator} improved, cost {cost}")
            }
            SearchEvent::GuidedBest { iter, cost } => {
                eprintln!("#search {t:7.3}s  gls round {iter}, new best cost {cost}")
            }
            SearchEvent::Done { cost } => eprintln!("#search {t:7.3}s  done, cost {cost}"),
        }
    }
}

/// How the first solution is built.
pub enum Construct {
    CheapestInsertion,
    /// Each step draws from the `k` cheapest, not the first. Same seed, same
    /// solution. `k = 1` is `CheapestInsertion`.
    GreedyRandomized {
        seed: u64,
        k: usize,
    },
}

/// How that solution is then made cheaper.
pub enum Improve {
    /// Descend to the first local optimum and stop.
    HillClimb,
    /// Guided local search: keep descending, penalizing the arcs that keep
    /// coming back, for `iters` rounds.
    Gls { iters: usize },
}

/// `cost` is the true cost — never the penalized number a GLS descent was
/// reading — and the solver only returns feasible routes, so it is a plain
/// `Cost`, not an `Option`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Solution {
    pub routes: Routes,
    pub cost: Cost,
}

impl Solution {
    /// Nodes left unserved (their penalties are already inside `cost`).
    pub fn unserved<'a>(&'a self, m: &'a Model) -> &'a [NodeId] {
        m.unserved_vehicle()
            .map_or(&[], |v| &self.routes[v.index()])
    }
}

/// The model is borrowed shared: every piece of mutable search state, GLS
/// penalties included, is owned by the call. One model can therefore back any
/// number of solves running at once.
pub fn solve(m: &Model, construct: Construct, improve: Improve) -> Solution {
    solve_with(m, construct, improve, |_| {})
}

/// `solve` with an observer for search progress. The callback runs on the
/// solver thread; keep it cheap or it becomes part of the measured time.
pub fn solve_with(
    m: &Model,
    construct: Construct,
    improve: Improve,
    log: impl FnMut(SearchEvent),
) -> Solution {
    solve_with_operators(m, construct, improve, &mut [], log)
}

/// `solve_with`, plus operators of your own in every descent it runs.
///
/// Yours ride the node queue and the don't-look bits with the shipped four
/// rather than running in a pass around them: each is offered the node the
/// descent just woke, and a move any of them accepts re-wakes the routes it
/// touched. They are tried after relocate and swap and before 2-opt. An
/// accepted move reports as [`Operator::Custom`] carrying its index.
///
/// Both [`Improve`] modes take them. Under guided local search the objective
/// carries arc penalties and [`Search`] applies them to everything it prices,
/// so an operator comparing `cx.eval` against the `cost` it was handed is
/// already minimizing what the round is minimizing. Those rounds descend
/// silently, though, so no [`Operator::Custom`] event escapes them; count
/// accepted moves inside your own operator instead.
pub fn solve_with_operators(
    m: &Model,
    construct: Construct,
    improve: Improve,
    operators: &mut [CustomOperator],
    mut log: impl FnMut(SearchEvent),
) -> Solution {
    // One context for the whole solve, so construction's buffers are the
    // descent's.
    let mut cx = Search::new(m);
    let mut sol = first_solution_in(&mut cx, construct, &mut log);
    match improve {
        Improve::HillClimb => descend(&mut cx, &mut sol, operators, &mut log),
        Improve::Gls { iters } => guided_in(&mut cx, &mut sol, iters, operators, &mut log),
    }
    let cost = eval_routes(m, &sol).expect("solver produced an infeasible solution");
    Solution { routes: sol, cost }
}

/// Vehicles worth trying an insertion on: every route in use, plus one empty
/// one so the fleet can still grow.
///
/// One empty is enough only while empty vehicles are interchangeable. Once
/// per-vehicle `forbid` sets make them differ, the caller retries with all of
/// them. The unserved sink is always a candidate: dropping must be on offer
/// even when it is empty.
pub(crate) fn candidate_vehicles(m: &Model, sol: &Routes, out: &mut Vec<usize>) {
    out.clear();
    out.extend((0..sol.len()).filter(|&i| !sol[i].is_empty()));
    if let Some(empty) = (0..sol.len()).find(|&i| sol[i].is_empty()) {
        out.push(empty);
    }
    if let Some(uv) = m.unserved_vehicle()
        && !out.contains(&uv.index())
    {
        out.push(uv.index());
    }
}

pub fn first_solution_with(
    m: &Model,
    construct: Construct,
    log: impl FnMut(SearchEvent),
) -> Routes {
    first_solution_in(&mut Search::new(m), construct, log)
}

fn first_solution_in(
    cx: &mut Search,
    construct: Construct,
    log: impl FnMut(SearchEvent),
) -> Routes {
    match construct {
        Construct::CheapestInsertion => insertion(cx, 1, 0, log),
        Construct::GreedyRandomized { seed, k } => insertion(cx, k, seed, log),
    }
}

/// A candidate insertion: its delta cost, the vehicle, and the position.
type Insertion = (Cost, usize, usize);

/// Cheapest feasible position for `u` in route `v`, or `None` if it has none.
fn best_in_route(
    cx: &mut Search,
    sol: &Routes,
    cost: &[Cost],
    v: usize,
    u: NodeId,
) -> Option<Insertion> {
    let mut best: Option<Insertion> = None;
    for pos in 0..=sol[v].len() {
        let Some(c) = cx.eval_splice(&sol[v], pos..pos, &[u], VehicleId(v as u32)) else {
            continue;
        };
        let delta = c - cost[v];
        if best.is_none_or(|(bd, ..)| delta < bd) {
            best = Some((delta, v, pos));
        }
    }
    best
}

/// Cheapest feasible position for `u` across `vs`; ties go to the first `vs`.
fn best_over(
    cx: &mut Search,
    sol: &Routes,
    cost: &[Cost],
    vs: &[usize],
    u: NodeId,
) -> Option<Insertion> {
    let mut best: Option<Insertion> = None;
    for &v in vs {
        if let Some(c) = best_in_route(cx, sol, cost, v, u)
            && best.is_none_or(|(bd, ..)| c.0 < bd)
        {
            best = Some(c);
        }
    }
    best
}

/// SplitMix64. Not cryptographic. Seeds reproduce solutions, no dependency.
struct Rng(u64);

impl Rng {
    #[inline]
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform over `0..n`. The modulo bias is negligible at these `n`.
    #[inline]
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Always the cheapest insertion on offer.
pub fn cheapest_insertion(m: &Model, log: impl FnMut(SearchEvent)) -> Routes {
    insertion(&mut Search::new(m), 1, 0, log)
}

/// Draws from the `k` cheapest. Not a better first solution, a different one
/// per seed, so callers can race several and keep the cheapest.
pub fn greedy_randomized(m: &Model, seed: u64, k: usize, log: impl FnMut(SearchEvent)) -> Routes {
    insertion(&mut Search::new(m), k, seed, log)
}

/// Insertion with a candidate list of width `k`. The cache below is
/// indifferent to the draw: one route still grows per step.
fn insertion(cx: &mut Search, k: usize, seed: u64, mut log: impl FnMut(SearchEvent)) -> Routes {
    let m = cx.model();
    let nv = m.vehicle_count();
    let mut rng = Rng(seed);
    let mut sol: Routes = vec![Vec::new(); nv];
    let mut cost = vec![0 as Cost; nv];
    let mut unrouted: Vec<NodeId> = (0..m.node_count() as u32)
        .map(NodeId)
        .filter(|&n| !m.is_terminal(n))
        .collect();
    let mut cands = Vec::new();

    // Each node's cheapest insertion among the used routes, parallel to
    // `unrouted`. Empty candidates stay out: they cost one position to price,
    // and one becomes a used route on nearly every insertion.
    let mut best: Vec<Option<Insertion>> = vec![None; unrouted.len()];
    let mut dirty = vec![true; unrouted.len()];
    // The route the last insertion grew, the only one that can be stale.
    let mut changed: Option<usize> = None;
    // Each node's best this round, the draw picks from here. Reused.
    let mut ranked: Vec<(Insertion, usize)> = Vec::new();

    while !unrouted.is_empty() {
        // `candidate_vehicles` keeps used routes in a fixed order, so a cached
        // entry stays comparable with a fresh one.
        candidate_vehicles(m, &sol, &mut cands);
        let (used, empty): (Vec<usize>, Vec<usize>) =
            cands.iter().partition(|&&v| !sol[v].is_empty());

        ranked.clear();
        for i in 0..unrouted.len() {
            let u = unrouted[i];
            if dirty[i] {
                best[i] = best_over(cx, &sol, &cost, &used, u);
                dirty[i] = false;
            } else if let Some(t) = changed {
                let held = best[i];
                let fresh_t = best_in_route(cx, &sol, &cost, t, u);
                // Every route but `t` was already worse and none of them
                // moved, so only a `t` that got worse needs a full rescan.
                best[i] = match (held, fresh_t) {
                    (Some((d, v, _)), Some(c)) if v == t && c.0 <= d => Some(c),
                    (Some((_, v, _)), _) if v == t => best_over(cx, &sol, &cost, &used, u),
                    (_, Some(c)) if held.is_none_or(|b| c < b) => Some(c),
                    _ => held,
                };
            }
            let fresh = best_over(cx, &sol, &cost, &empty, u);
            let node_best = match (best[i], fresh) {
                (Some(b), Some(f)) if f.0 < b.0 => Some(f),
                (Some(b), _) => Some(b),
                (None, f) => f,
            };
            if let Some(c) = node_best {
                ranked.push((c, i));
            }
        }

        // The one empty candidate may be forbidden for every remaining node
        // while another empty vehicle is not. That is the only way a narrow
        // scan can miss a feasible insertion, so only then widen it.
        let widened = ranked.is_empty() && cands.len() < nv;
        if widened {
            let all: Vec<usize> = (0..nv).collect();
            for (i, &u) in unrouted.iter().enumerate() {
                if let Some(c) = best_over(cx, &sol, &cost, &all, u) {
                    ranked.push((c, i));
                }
            }
        }

        if ranked.is_empty() {
            if let Some(n) = unrouted
                .iter()
                .find(|&&n| (0..nv).all(|v| m.vehicle(VehicleId(v as u32)).forbids(n)))
            {
                panic!("node {} is unroutable: forbidden on every vehicle", n.0);
            }
            panic!("no feasible insertion left — fleet too small?");
        }

        // Unique `i` per entry makes `(delta, i)` total, so the j-th smallest
        // is unique: selecting beats sorting and `k = 1` still draws the old
        // scan's pick. Draw before selecting, the rank is what we select on.
        let j = rng.below(k.clamp(1, ranked.len()));
        ranked.select_nth_unstable_by_key(j, |&((delta, ..), i)| (delta, i));
        let ((delta, v, pos), ui) = ranked[j];
        let node = unrouted.swap_remove(ui);
        best.swap_remove(ui);
        dirty.swap_remove(ui);
        sol[v].insert(pos, node);
        cost[v] += delta;
        changed = Some(v);

        // Widening ranked entries against a vehicle set the narrow scan omits.
        if widened {
            dirty.iter_mut().for_each(|d| *d = true);
        }
    }
    log(SearchEvent::FirstSolution {
        cost: cost.iter().sum(),
    });
    sol
}

/// First-improvement hill climb over relocate + swap + 2-opt, with don't-look
/// bits: a node is only re-examined after a move touched its route. 2-opt*
/// stays out of the per-node cascade — it fires once the fine operators reach
/// a fixpoint, as a partition-level kick, then the sweep resumes.
pub fn local_search(m: &Model, sol: &mut Routes) {
    local_search_with(m, sol, |_| {})
}

/// `local_search` reporting an `Improvement` per accepted move and a final
/// `Done`.
pub fn local_search_with(m: &Model, sol: &mut Routes, log: impl FnMut(SearchEvent)) {
    descend(&mut Search::new(m), sol, &mut [], log)
}

/// An operator of your own, called with the node the descent just popped and
/// the route holding it.
///
/// Apply one improving change and return the other vehicle you touched, the
/// same route again if the move stayed inside it, or `None` if you found
/// nothing. Two rules:
///
/// * Leave `cost[v]` correct for every route you rewrote; the descent reports
///   totals from it and never recomputes.
/// * Return `Some` only for a move you applied that made the solution
///   cheaper. Anything else will not terminate.
///
/// Price routes through the [`Search`], and keep any buffers you need in the
/// closure.
pub type CustomOperator<'a> =
    &'a mut dyn FnMut(&mut Search, &mut Routes, &mut [Cost], NodeId, usize) -> Option<usize>;

/// The descent itself, on whatever objective `cx` currently carries. Public
/// callers get true cost; only guided local search leaves penalties on it.
fn descend(
    cx: &mut Search,
    sol: &mut Routes,
    operators: &mut [CustomOperator],
    mut log: impl FnMut(SearchEvent),
) {
    let m = cx.model();
    let mut cost: Vec<Cost> = (0..sol.len())
        .map(|v| {
            cx.eval(&sol[v], VehicleId(v as u32))
                .expect("infeasible start solution")
        })
        .collect();

    // Held across evaluations, so they belong to the descent, not the context.
    let mut without = Vec::new();
    let mut cands = Vec::new();

    let mut queued = vec![false; m.node_count()];
    let mut index = vec![u32::MAX; m.node_count()];
    // Route-level don't-look bit: 2-opt takes a route and ignores the popped
    // node, so without this it rescans one route once per node in it.
    let mut two_opt_dirty = vec![true; sol.len()];

    // Draining the queue is not a fixpoint: a move only re-wakes the two
    // routes it touched, and a node elsewhere may now have an improving move
    // into them. Re-sweep everything until a whole sweep finds nothing.
    loop {
        let mut queue: VecDeque<NodeId> = sol.iter().flatten().copied().collect();
        // node -> route, rebuilt per sweep; a move re-stamps only the routes
        // it touched. Replaces an O(n) route scan per queue pop.
        for (r, route) in sol.iter().enumerate() {
            for &n in route {
                index[n.index()] = r as u32;
            }
        }
        queued.iter_mut().for_each(|q| *q = false);
        for &n in &queue {
            queued[n.index()] = true;
        }
        let mut improved = false;

        while let Some(u) = queue.pop_front() {
            queued[u.index()] = false;
            // Every queued node is in exactly one route, so this never sees
            // the u32::MAX sentinel.
            let r = index[u.index()] as usize;

            // Cheapest operator first: relocate and swap each cost O(n) route
            // evaluations, 2-opt costs O(n^2).
            let relocated = try_relocate(cx, sol, &mut cost, u, r, &mut without, &mut cands);
            let (other, operator) = match relocated {
                Some(v) => (Some(v), Operator::Relocate),
                None => match try_swap(cx, sol, &mut cost, u, r) {
                    Some(v) => (Some(v), Operator::Swap),
                    None => {
                        let mut custom = None;
                        for (i, op) in operators.iter_mut().enumerate() {
                            if let Some(v) = op(cx, sol, &mut cost, u, r) {
                                custom = Some((v, i));
                                break;
                            }
                        }
                        match custom {
                            Some((v, i)) => (Some(v), Operator::Custom(i)),
                            None if !two_opt_dirty[r] => continue,
                            None if try_two_opt(cx, sol, &mut cost, r) => (None, Operator::TwoOpt),
                            None => {
                                two_opt_dirty[r] = false;
                                continue;
                            }
                        }
                    }
                },
            };
            improved = true;
            log(SearchEvent::Improvement {
                operator,
                cost: cost.iter().sum(),
            });
            for t in [Some(r), other.filter(|&v| v != r)].into_iter().flatten() {
                two_opt_dirty[t] = true;
                for &n in &sol[t] {
                    index[n.index()] = t as u32;
                    if !queued[n.index()] {
                        queued[n.index()] = true;
                        queue.push_back(n);
                    }
                }
            }
        }

        if !improved {
            // The fine operators are at a fixpoint: one 2-opt* pass over all
            // routes. Firing it here rather than per node keeps the big tail
            // swaps from disrupting routes relocate would have fixed for less.
            for r in 0..sol.len() {
                if sol[r].is_empty() {
                    continue;
                }
                if let Some(v) = try_two_opt_star(cx, sol, &mut cost, r) {
                    two_opt_dirty[r] = true;
                    two_opt_dirty[v] = true;
                    improved = true;
                    log(SearchEvent::Improvement {
                        operator: Operator::TwoOptStar,
                        cost: cost.iter().sum(),
                    });
                }
            }
        }

        if !improved {
            log(SearchEvent::Done {
                cost: cost.iter().sum(),
            });
            return;
        }
    }
}

/// Fraction of the average arc cost that one penalty is worth. The usual
/// starting point for CVRP; higher diversifies harder, lower descends harder.
const LAMBDA_NUMERATOR: Cost = 1;
const LAMBDA_DENOMINATOR: Cost = 10;

/// Guided local search. Descend, then repeatedly punish the arcs that look
/// worst — expensive and not yet punished much — and descend again on the
/// penalized cost. The penalties push the search out of a local optimum
/// without ever lying about which solution is actually cheapest: the best
/// true-cost solution seen is kept aside and restored at the end.
///
/// No RNG, so this stays as deterministic as the hill climb it wraps.
///
/// Every iteration restarts a full local search. Waking only the nodes touched
/// by the arcs just penalized is the win, but seeding the queue is capped by
/// `descend`'s outer re-sweep, which starts a full sweep anyway.
pub fn guided_local_search(m: &Model, sol: &mut Routes, iters: usize) {
    guided_local_search_with(m, sol, iters, |_| {})
}

/// `guided_local_search` reporting a `GuidedBest` per new best true cost and a
/// final `Done`.
///
/// The descents themselves run silent on purpose. Every round after the first
/// is minimizing penalized cost, so its `Improvement` costs would be numbers no
/// solution ever actually costs — a log that reads as progress while telling
/// you nothing true.
pub fn guided_local_search_with(
    m: &Model,
    sol: &mut Routes,
    iters: usize,
    log: impl FnMut(SearchEvent),
) {
    guided_in(&mut Search::new(m), sol, iters, &mut [], log)
}

fn guided_in(
    cx: &mut Search,
    sol: &mut Routes,
    iters: usize,
    operators: &mut [CustomOperator],
    mut log: impl FnMut(SearchEvent),
) {
    let m = cx.model();
    descend(cx, sol, operators, |_| {});

    let mut best = sol.clone();
    let mut best_cost = eval_routes(m, sol).expect("infeasible local optimum");
    log(SearchEvent::GuidedBest {
        iter: 0,
        cost: best_cost,
    });

    // Scale one penalty into arc-cost units, comparable to the distances the
    // operators trade against it. Dropped nodes are not customers, so they
    // must not dilute it.
    let customers: usize = sol
        .iter()
        .enumerate()
        .filter(|(v, _)| m.unserved_vehicle() != Some(VehicleId(*v as u32)))
        .map(|(_, r)| r.len())
        .sum();
    cx.set_lambda(
        (LAMBDA_NUMERATOR * best_cost / (LAMBDA_DENOMINATOR * customers.max(1) as Cost)).max(1),
    );

    for iter in 1..=iters {
        penalize_worst_arcs(cx, sol);
        descend(cx, sol, operators, |_| {});

        // The descent just optimized penalized cost, which is not the cost we
        // rank solutions by. `eval_routes` is always the true one.
        let cost = eval_routes(m, sol).expect("infeasible after descent");
        if cost < best_cost {
            best_cost = cost;
            best.clone_from(sol);
            log(SearchEvent::GuidedBest { iter, cost });
        }
    }

    // The penalties die with `cx`; no caller can ever see a penalized cost.
    *sol = best;
    log(SearchEvent::Done { cost: best_cost });
}

/// Penalize every arc of `sol` with maximal utility `cost / (1 + penalty)` —
/// long arcs first, but an arc already hit often loses priority to a fresh one.
///
/// The comparison is cross-multiplied rather than divided: this repo has no
/// floats, and ties must break the same way on every run.
fn penalize_worst_arcs(cx: &mut Search, sol: &Routes) {
    let m = cx.model();
    let mut worst: Vec<(NodeId, NodeId)> = Vec::new();
    // (arc cost, 1 + times penalized) of the best candidate so far.
    let mut top: Option<(Cost, Cost)> = None;

    for (v, route) in sol.iter().enumerate() {
        // Sink arcs are penalties, not travel; never punish them.
        if m.unserved_vehicle() == Some(VehicleId(v as u32)) {
            continue;
        }
        if route.is_empty() {
            continue;
        }
        let veh = m.vehicle(VehicleId(v as u32));
        let mut prev = veh.start;
        for &node in route.iter().chain(std::iter::once(&veh.end)) {
            // True arc cost, not `cx.arc`: utility ranks by what the arc
            // really costs.
            let cost = m.eval(veh.cost_class, prev, node);
            let seen = 1 + cx.penalty(prev, node);
            let arc = if prev <= node {
                (prev, node)
            } else {
                (node, prev)
            };
            match top {
                None => {
                    top = Some((cost, seen));
                    worst.push(arc);
                }
                Some((top_cost, top_seen)) => {
                    let (lhs, rhs) = (cost * top_seen, top_cost * seen);
                    if lhs > rhs {
                        top = Some((cost, seen));
                        worst.clear();
                        worst.push(arc);
                    } else if lhs == rhs {
                        worst.push(arc);
                    }
                }
            }
            prev = node;
        }
    }

    // A one-customer route visits the depot twice, so {depot, x} can show up as
    // both of its arcs. Penalize it once.
    worst.sort_unstable();
    worst.dedup();
    for (a, b) in worst {
        cx.penalize(a, b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{eval_route, visits_all_nodes};
    use crate::model::ModelBuilder;

    /// Depot at 0, customers 1..=5 on a line; arc cost is line distance.
    fn line_model() -> Model {
        let dist = |a: NodeId, b: NodeId| (a.0 as i64 - b.0 as i64).abs() * 10;
        let mut b = ModelBuilder::new(6);
        let cost = b.cost_class(dist);
        b.vehicle(NodeId(0), NodeId(0), cost);
        b.vehicle(NodeId(0), NodeId(0), cost);
        b.dimension(
            "load",
            |_from, to| if to == NodeId(0) { 0 } else { 1 },
            vec![3, 3],
        );
        b.build()
    }

    /// `k = 1` is greedy, seeds reproduce, some seed diverges. Without the
    /// last, the draw is a no-op and multi-start has nothing to search.
    #[test]
    fn randomized_insertion_is_seeded_not_arbitrary() {
        let m = line_model();
        let greedy = cheapest_insertion(&m, |_| {});

        assert_eq!(
            greedy_randomized(&m, 7, 1, |_| {}),
            greedy,
            "k = 1 is greedy"
        );
        assert_eq!(
            greedy_randomized(&m, 7, 3, |_| {}),
            greedy_randomized(&m, 7, 3, |_| {}),
            "same seed, same solution"
        );

        let mut diverged = 0;
        for seed in 0..32 {
            let sol = greedy_randomized(&m, seed, 3, |_| {});
            assert!(visits_all_nodes(&m, &sol), "seed {seed} lost a node");
            assert!(eval_routes(&m, &sol).is_some(), "seed {seed} is infeasible");
            if sol != greedy {
                diverged += 1;
            }
        }
        assert!(diverged > 0, "32 seeds all returned the greedy solution");
    }

    /// Forbids and the drop sink are ordinary candidates. Order is random,
    /// feasibility is not.
    #[test]
    fn randomized_insertion_respects_drops_and_forbids() {
        let dist = |a: NodeId, b: NodeId| (a.0 as i64 - b.0 as i64).abs() * 10;
        let mut b = ModelBuilder::new(6);
        let cost = b.cost_class(dist);
        let v0 = b.vehicle(NodeId(0), NodeId(0), cost);
        b.vehicle(NodeId(0), NodeId(0), cost);
        b.forbid(v0, NodeId(1));
        b.dimension(
            "load",
            |_from, to| if to == NodeId(0) { 0 } else { 1 },
            vec![3, 3],
        );
        b.allow_drop(NodeId(5), 15);
        let m = b.build();

        for seed in 0..16 {
            let sol = greedy_randomized(&m, seed, 4, |_| {});
            assert!(visits_all_nodes(&m, &sol), "seed {seed} lost a node");
            assert!(eval_routes(&m, &sol).is_some(), "seed {seed} is infeasible");
            assert!(
                !sol[v0.index()].contains(&NodeId(1)),
                "seed {seed} put node 1 on the vehicle that forbids it"
            );
        }
    }

    /// A node cheaper to drop than to serve pays its penalty instead.
    #[test]
    fn dropped_nodes_pay_their_penalty() {
        let build = |penalty: Cost| {
            let dist = |a: NodeId, b: NodeId| (a.0 as i64 - b.0 as i64).abs() * 10;
            let mut b = ModelBuilder::new(4);
            let cost = b.cost_class(dist);
            b.vehicle(NodeId(0), NodeId(0), cost);
            b.dimension(
                "load",
                |_from, to| if to == NodeId(0) { 0 } else { 1 },
                vec![3],
            );
            b.allow_drop(NodeId(3), penalty);
            b.build()
        };

        // Serving node 3 extends the route by 20; penalty 15 wins, 30 loses.
        let m = build(15);
        let sol = solve(&m, Construct::CheapestInsertion, Improve::HillClimb);
        assert_eq!(sol.unserved(&m), &[NodeId(3)]);
        assert_eq!(sol.cost, 40 + 15);
        assert!(visits_all_nodes(&m, &sol.routes));

        let m = build(30);
        let sol = solve(&m, Construct::CheapestInsertion, Improve::HillClimb);
        assert!(sol.unserved(&m).is_empty());
        assert_eq!(sol.cost, 60);
        assert!(visits_all_nodes(&m, &sol.routes));
    }

    /// Both routes are full, so no node can move anywhere: only a trade helps.
    #[test]
    fn swap_finds_what_relocate_cannot() {
        let dist = |a: NodeId, b: NodeId| (a.0 as i64 - b.0 as i64).abs() * 10;
        let mut b = ModelBuilder::new(5);
        let cost = b.cost_class(dist);
        b.vehicle(NodeId(0), NodeId(0), cost);
        b.vehicle(NodeId(0), NodeId(0), cost);
        b.dimension(
            "load",
            |_from, to| if to == NodeId(0) { 0 } else { 1 },
            vec![2, 2],
        );
        let m = b.build();

        let mut sol = vec![vec![NodeId(1), NodeId(3)], vec![NodeId(2), NodeId(4)]];
        assert_eq!(eval_routes(&m, &sol), Some(140));

        let mut ops = Vec::new();
        local_search_with(&m, &mut sol, |e| {
            if let SearchEvent::Improvement { operator, .. } = e {
                ops.push(operator);
            }
        });
        assert!(
            ops.contains(&Operator::Swap),
            "expected a swap, got {ops:?}"
        );
        assert_eq!(eval_routes(&m, &sol), Some(120));
    }

    /// A node forbidden on one vehicle must land on the other, and stay
    /// there through local search: every operator re-validates via
    /// `eval_route`, so no move can drag it back.
    #[test]
    fn forbidden_node_rides_the_allowed_vehicle() {
        let dist = |a: NodeId, b: NodeId| (a.0 as i64 - b.0 as i64).abs() * 10;
        let mut b = ModelBuilder::new(6);
        let cost = b.cost_class(dist);
        let v0 = b.vehicle(NodeId(0), NodeId(0), cost);
        b.vehicle(NodeId(0), NodeId(0), cost);
        // Node 1 is closest to the depot; v0 would take it if it could.
        b.forbid(v0, NodeId(1));
        b.dimension(
            "load",
            |_from, to| if to == NodeId(0) { 0 } else { 1 },
            vec![3, 3],
        );
        let m = b.build();

        assert_eq!(eval_route(&m, &[NodeId(1)], VehicleId(0)), None);
        assert!(eval_route(&m, &[NodeId(1)], VehicleId(1)).is_some());

        let sol = solve(&m, Construct::CheapestInsertion, Improve::Gls { iters: 5 });
        assert!(visits_all_nodes(&m, &sol.routes));
        assert!(!sol.routes[0].contains(&NodeId(1)));
        assert!(sol.routes[1].contains(&NodeId(1)));
    }

    /// Forbidden everywhere is not a fleet problem; the panic must say so.
    #[test]
    #[should_panic(expected = "unroutable")]
    fn node_forbidden_everywhere_panics_clearly() {
        let dist = |a: NodeId, b: NodeId| (a.0 as i64 - b.0 as i64).abs() * 10;
        let mut b = ModelBuilder::new(6);
        let cost = b.cost_class(dist);
        let v0 = b.vehicle(NodeId(0), NodeId(0), cost);
        let v1 = b.vehicle(NodeId(0), NodeId(0), cost);
        b.forbid(v0, NodeId(3));
        b.forbid(v1, NodeId(3));
        let m = b.build();
        solve(&m, Construct::CheapestInsertion, Improve::HillClimb);
    }

    #[test]
    fn search_events_trace_the_solve() {
        let m = line_model();
        let mut events = Vec::new();
        let sol = solve_with(&m, Construct::CheapestInsertion, Improve::HillClimb, |e| {
            events.push(e)
        });

        let SearchEvent::FirstSolution { cost: first } = events[0] else {
            panic!("first event must be FirstSolution, got {:?}", events[0]);
        };
        let &SearchEvent::Done { cost: done } = events.last().unwrap() else {
            panic!("last event must be Done, got {:?}", events.last());
        };

        let mut prev = i64::MAX;
        for e in events.iter() {
            if let SearchEvent::Improvement { cost, .. } = e {
                assert!(*cost < prev, "improvements must strictly decrease");
                prev = *cost;
            }
        }

        assert!(done <= first, "local search never worsens the solution");
        assert_eq!(Some(done), eval_routes(&m, &sol.routes));
        // The silent entry point solves to the same cost.
        let silent = solve(&m, Construct::CheapestInsertion, Improve::HillClimb);
        assert_eq!(silent.cost, done);
    }
}
