use std::collections::{HashMap, VecDeque};
use std::hash::{BuildHasherDefault, Hasher};
use std::time::Instant;

use crate::eval::{Routes, eval_route, eval_routes};
use crate::model::Model;
use crate::types::{Cost, NodeId, VehicleId};

/// The neighborhood operator that accepted an improving move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operator {
    Relocate,
    Swap,
    TwoOpt,
    TwoOptStar,
}

impl std::fmt::Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operator::Relocate => write!(f, "relocate"),
            Operator::Swap => write!(f, "swap"),
            Operator::TwoOpt => write!(f, "2-opt"),
            Operator::TwoOptStar => write!(f, "2-opt*"),
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
}

/// How that solution is then made cheaper.
pub enum Improve {
    /// Descend to the first local optimum and stop.
    HillClimb,
    /// Guided local search: keep descending, penalizing the arcs that keep
    /// coming back, for `iters` rounds.
    Gls { iters: usize },
}

/// What a solve leaves behind: the routes and their true total cost.
///
/// `cost` is the true cost — never the penalized number a GLS descent was
/// reading — and the solver only returns feasible routes, so it is a plain
/// `Cost`, not an `Option`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Solution {
    pub routes: Routes,
    pub cost: Cost,
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
    mut log: impl FnMut(SearchEvent),
) -> Solution {
    let mut sol = first_solution_with(m, construct, &mut log);
    match improve {
        Improve::HillClimb => local_search_with(m, &mut sol, &mut log),
        Improve::Gls { iters } => guided_local_search_with(m, &mut sol, iters, &mut log),
    }
    let cost = eval_routes(m, &sol).expect("solver produced an infeasible solution");
    Solution { routes: sol, cost }
}

/// Vehicles worth trying an insertion on: every route in use, plus one empty
/// one so the fleet can still grow.
///
/// ponytail: "one empty" is right for a homogeneous fleet. With a real
/// heterogeneous fleet this should be one empty vehicle per cost class,
/// otherwise the cheaper vehicle types are never opened.
fn candidate_vehicles(sol: &Routes) -> Vec<usize> {
    let mut v: Vec<usize> = (0..sol.len()).filter(|&i| !sol[i].is_empty()).collect();
    if let Some(empty) = (0..sol.len()).find(|&i| sol[i].is_empty()) {
        v.push(empty);
    }
    v
}

fn with_insert(route: &[NodeId], pos: usize, node: NodeId, out: &mut Vec<NodeId>) {
    out.clear();
    out.extend_from_slice(&route[..pos]);
    out.push(node);
    out.extend_from_slice(&route[pos..]);
}

pub fn first_solution_with(
    m: &Model,
    construct: Construct,
    log: impl FnMut(SearchEvent),
) -> Routes {
    match construct {
        Construct::CheapestInsertion => cheapest_insertion(m, log),
    }
}

pub fn cheapest_insertion(m: &Model, mut log: impl FnMut(SearchEvent)) -> Routes {
    let nv = m.vehicle_count();
    let mut sol: Routes = vec![Vec::new(); nv];
    let mut cost = vec![0 as Cost; nv];
    let mut unrouted: Vec<NodeId> = (0..m.node_count() as u32)
        .map(NodeId)
        .filter(|&n| !m.is_terminal(n))
        .collect();
    let mut scratch = Vec::new();

    while !unrouted.is_empty() {
        let candidates = candidate_vehicles(&sol);
        // (delta, unrouted index, vehicle, position)
        let mut best: Option<(Cost, usize, usize, usize)> = None;

        for (ui, &node) in unrouted.iter().enumerate() {
            for &v in &candidates {
                for pos in 0..=sol[v].len() {
                    with_insert(&sol[v], pos, node, &mut scratch);
                    let Some(c) = eval_route(m, &scratch, VehicleId(v as u32)) else {
                        continue;
                    };
                    let delta = c - cost[v];
                    if best.is_none_or(|(bd, ..)| delta < bd) {
                        best = Some((delta, ui, v, pos));
                    }
                }
            }
        }

        let (delta, ui, v, pos) = best.expect("no feasible insertion left — fleet too small?");
        let node = unrouted.swap_remove(ui);
        sol[v].insert(pos, node);
        cost[v] += delta;
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
    descend(m, sol, eval_route, log)
}

/// The descent itself, on whatever cost `p` defines. Public callers get true
/// cost; only guided local search passes a non-empty table.
fn descend<F>(m: &Model, sol: &mut Routes, eval: F, mut log: impl FnMut(SearchEvent))
where
    F: Fn(&Model, &[NodeId], VehicleId) -> Option<Cost>,
{
    let mut cost: Vec<Cost> = (0..sol.len())
        .map(|v| eval(m, &sol[v], VehicleId(v as u32)).expect("infeasible start solution"))
        .collect();

    let mut queued = vec![false; m.node_count()];

    // Draining the queue is not a fixpoint: a move only re-wakes the two
    // routes it touched, and a node elsewhere may now have an improving move
    // into them. Re-sweep everything until a whole sweep finds nothing.
    loop {
        let mut queue: VecDeque<NodeId> = sol.iter().flatten().copied().collect();
        queued.iter_mut().for_each(|q| *q = false);
        for &n in &queue {
            queued[n.index()] = true;
        }
        let mut improved = false;

        while let Some(u) = queue.pop_front() {
            queued[u.index()] = false;
            let Some(r) = sol.iter().position(|route| route.contains(&u)) else {
                continue;
            };

            // Cheapest operator first: relocate and swap each cost O(n) route
            // evaluations, 2-opt costs O(n^2).
            let (touched, operator) = match try_relocate(m, sol, &eval, &mut cost, u, r) {
                Some(v) => (vec![r, v], Operator::Relocate),
                None => match try_swap(m, sol, &eval, &mut cost, u, r) {
                    Some(v) => (vec![r, v], Operator::Swap),
                    None if try_two_opt(m, sol, &eval, &mut cost, r) => (vec![r], Operator::TwoOpt),
                    None => continue,
                },
            };
            improved = true;
            log(SearchEvent::Improvement {
                operator,
                cost: cost.iter().sum(),
            });
            for t in touched {
                for &n in &sol[t] {
                    if !queued[n.index()] {
                        queued[n.index()] = true;
                        queue.push_back(n);
                    }
                }
            }
        }

        if !improved {
            // The fine operators are at a fixpoint: one 2-opt* pass over all
            // routes. Firing it here instead of per node keeps the big tail
            // swaps from disrupting routes that relocate would have fixed for
            // less (X-n143-k7 regressed 14.5% -> 21.1% the other way).
            for r in 0..sol.len() {
                if sol[r].is_empty() {
                    continue;
                }
                if let Some(_v) = try_two_opt_star(m, sol, &eval, &mut cost, r) {
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

/// Arc penalties for guided local search, and the weight one penalty carries.
///
/// Owned by the search that is running, never by the `Model`: the model stays a
/// pure description of the problem, shareable by `&` across concurrent solves.
/// Created and dropped inside `gls`, so a penalty cannot outlive the search
/// that made it.
struct Penalties {
    lambda: Cost,
    map: HashMap<u64, Cost, BuildHasherDefault<ArcHasher>>,
}

impl Penalties {
    /// An empty table whose penalties each cost `lambda`.
    fn new(lambda: Cost) -> Self {
        Penalties {
            lambda,
            map: HashMap::default(),
        }
    }

    /// Weight applied to each penalty.
    #[inline]
    fn lambda(&self) -> Cost {
        self.lambda
    }

    /// How many times arc `{a, b}` has been penalized.
    #[inline]
    fn arc(&self, a: NodeId, b: NodeId) -> Cost {
        self.map.get(&arc_key(a, b)).copied().unwrap_or(0)
    }

    fn penalize(&mut self, a: NodeId, b: NodeId) {
        *self.map.entry(arc_key(a, b)).or_insert(0) += 1;
    }
}

/// Arcs are penalized as unordered pairs, packed into one `u64`. 2-opt reverses
/// whole segments, so a directed key would make a reversal silently change the
/// penalty it carries.
#[inline]
fn arc_key(a: NodeId, b: NodeId) -> u64 {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    (lo.0 as u64) << 32 | hi.0 as u64
}

/// Hasher for `Penalties`, and for nothing else.
///
/// GLS reads that map once per arc — it is the hottest line in the project.
/// The default `RandomState` runs SipHash there, which a profile showed costing
/// more than evaluating the arc it was pricing. The keys are packed node pairs,
/// not adversarial input, so one multiply is enough.
///
/// The map is only ever read by key, never iterated, so nothing downstream can
/// observe the change in hash order.
#[derive(Default)]
struct ArcHasher(u64);

impl Hasher for ArcHasher {
    #[inline]
    fn write_u64(&mut self, n: u64) {
        // Fibonacci hashing puts the entropy in the high bits; hashbrown picks
        // its bucket from the low ones, so fold the high half back down.
        let h = n.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        self.0 = h ^ (h >> 32);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, _: &[u8]) {
        unreachable!("ArcHasher only ever hashes a packed arc key");
    }
}

/// Penalized cost for the GLS descent: true cost and feasibility from
/// `eval_route`, plus one `lambda` per penalty on the route's arcs. The second
/// walk sums penalties only, so GLS pays for its own lookup.
fn eval_route_penalized(m: &Model, p: &Penalties, route: &[NodeId], v: VehicleId) -> Option<Cost> {
    let base = eval_route(m, route, v)?;
    let veh = m.vehicle(v);
    let mut prev = veh.start;
    let mut extra = 0;
    for &node in route.iter().chain(std::iter::once(&veh.end)) {
        extra += p.arc(prev, node);
        prev = node;
    }
    Some(base + p.lambda() * extra)
}

/// Guided local search. Descend, then repeatedly punish the arcs that look
/// worst — expensive and not yet punished much — and descend again on the
/// penalized cost. The penalties push the search out of a local optimum
/// without ever lying about which solution is actually cheapest: the best
/// true-cost solution seen is kept aside and restored at the end.
///
/// No RNG, so this stays as deterministic as the hill climb it wraps.
///
/// ponytail: every iteration restarts a full local search from scratch. Waking
/// only the nodes touched by the arcs just penalized is the 3-5x speedup —
/// add it when the iteration count needs to go past a few hundred.
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
    mut log: impl FnMut(SearchEvent),
) {
    local_search(m, sol);

    let mut best = sol.clone();
    let mut best_cost = eval_routes(m, sol).expect("infeasible local optimum");
    log(SearchEvent::GuidedBest {
        iter: 0,
        cost: best_cost,
    });

    // Scale one penalty into arc-cost units, so lambda * penalty is comparable
    // to the distances the operators are trading against it.
    let customers: usize = sol.iter().map(Vec::len).sum();
    let lambda =
        (LAMBDA_NUMERATOR * best_cost / (LAMBDA_DENOMINATOR * customers.max(1) as Cost)).max(1);
    let mut penalties = Penalties::new(lambda);

    for iter in 1..=iters {
        penalize_worst_arcs(m, &mut penalties, sol);
        descend(
            m,
            sol,
            |m, route, v| eval_route_penalized(m, &penalties, route, v),
            |_| {},
        );

        // The descent just optimized penalized cost, which is not the cost we
        // rank solutions by. `eval_routes` is always the true one.
        let cost = eval_routes(m, sol).expect("infeasible after descent");
        if cost < best_cost {
            best_cost = cost;
            best.clone_from(sol);
            log(SearchEvent::GuidedBest { iter, cost });
        }
    }

    // `penalties` dies here; no caller can ever see a penalized cost.
    *sol = best;
    log(SearchEvent::Done { cost: best_cost });
}

/// Penalize every arc of `sol` with maximal utility `cost / (1 + penalty)` —
/// long arcs first, but an arc already hit often loses priority to a fresh one.
///
/// The comparison is cross-multiplied rather than divided: this repo has no
/// floats, and ties must break the same way on every run.
fn penalize_worst_arcs(m: &Model, p: &mut Penalties, sol: &Routes) {
    let mut worst: Vec<(NodeId, NodeId)> = Vec::new();
    // (arc cost, 1 + times penalized) of the best candidate so far.
    let mut top: Option<(Cost, Cost)> = None;

    for (v, route) in sol.iter().enumerate() {
        if route.is_empty() {
            continue;
        }
        let veh = m.vehicle(VehicleId(v as u32));
        let mut prev = veh.start;
        for &node in route.iter().chain(std::iter::once(&veh.end)) {
            let cost = m.eval(veh.cost_class, prev, node);
            let seen = 1 + p.arc(prev, node);
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
        p.penalize(a, b);
    }
}

/// Move `u` out of route `r` to its first improving position anywhere,
/// including back into `r`. Returns the receiving vehicle.
fn try_relocate<F>(
    m: &Model,
    sol: &mut Routes,
    eval: &F,
    cost: &mut [Cost],
    u: NodeId,
    r: usize,
) -> Option<usize>
where
    F: Fn(&Model, &[NodeId], VehicleId) -> Option<Cost>,
{
    let at = sol[r].iter().position(|&x| x == u)?;
    let mut without = sol[r].clone();
    without.remove(at);
    let without_cost = eval(m, &without, VehicleId(r as u32))?;

    let mut scratch = Vec::new();
    for v in candidate_vehicles(sol) {
        let base = if v == r { &without } else { &sol[v] };
        for pos in 0..=base.len() {
            with_insert(base, pos, u, &mut scratch);
            let Some(c) = eval(m, &scratch, VehicleId(v as u32)) else {
                continue;
            };
            let delta = if v == r {
                c - cost[r]
            } else {
                (without_cost - cost[r]) + (c - cost[v])
            };
            if delta < 0 {
                if v == r {
                    sol[r].clone_from(&scratch);
                    cost[r] = c;
                } else {
                    sol[r].clone_from(&without);
                    cost[r] = without_cost;
                    sol[v].clone_from(&scratch);
                    cost[v] = c;
                }
                return Some(v);
            }
        }
    }
    None
}

/// Trade `u` for a customer on another route, first improving pair wins.
/// Returns the other vehicle.
///
/// Relocate cannot reach these moves when both routes are near capacity: moving
/// a node either way overflows, and only an even trade fits. On the X set that
/// is most of the search space, since the reference fleet is sized tight.
fn try_swap<F>(
    m: &Model,
    sol: &mut Routes,
    eval: F,
    cost: &mut [Cost],
    u: NodeId,
    r: usize,
) -> Option<usize>
where
    F: Fn(&Model, &[NodeId], VehicleId) -> Option<Cost>,
{
    let at = sol[r].iter().position(|&x| x == u)?;
    for v in 0..sol.len() {
        if v == r || sol[v].is_empty() {
            continue;
        }
        for q in 0..sol[v].len() {
            let w = sol[v][q];
            sol[r][at] = w;
            sol[v][q] = u;
            let new =
                eval(m, &sol[r], VehicleId(r as u32)).zip(eval(m, &sol[v], VehicleId(v as u32)));
            match new {
                Some((a, b)) if a + b < cost[r] + cost[v] => {
                    cost[r] = a;
                    cost[v] = b;
                    return Some(v);
                }
                _ => {
                    sol[r][at] = u;
                    sol[v][q] = w;
                }
            }
        }
    }
    None
}

/// Intra-route 2-opt: reverse one segment, keep it if it is cheaper.
fn try_two_opt<F>(m: &Model, sol: &mut Routes, eval: F, cost: &mut [Cost], r: usize) -> bool
where
    F: Fn(&Model, &[NodeId], VehicleId) -> Option<Cost>,
{
    let len = sol[r].len();
    for i in 0..len {
        for j in i + 1..len {
            sol[r][i..=j].reverse();
            match eval(m, &sol[r], VehicleId(r as u32)) {
                Some(c) if c < cost[r] => {
                    cost[r] = c;
                    return true;
                }
                _ => sol[r][i..=j].reverse(),
            }
        }
    }
    false
}

/// A candidate 2-opt* move: penalised gain, the other vehicle, the two
/// rebuilt routes, and their true costs.
type TwoOptStarMove = (Cost, usize, Vec<NodeId>, Vec<NodeId>, Cost, Cost);

/// Inter-route 2-opt*: cut one arc in route `r` and one in another route,
/// trade the tails. Returns the other vehicle.
///
/// This is the operator that changes the customer-to-route partition in
/// chunks: relocate and swap shift one customer at a time, which cannot undo
/// a bad layout once routes fill up. The delta is O(1) arc arithmetic — only
/// the two cut arcs and the two reconnecting arcs change — and `eval_route`
/// runs only on improving candidates, to confirm capacity.
///
/// Best-improvement, unlike the cheaper operators: a tail swap commits many
/// customers at once, so taking the first improving cut drags the descent
/// into noticeably worse local optima (X-n143-k7 went 14.5% -> 21.1% gap).
fn try_two_opt_star<F>(
    m: &Model,
    sol: &mut Routes,
    eval: &F,
    cost: &mut [Cost],
    r: usize,
) -> Option<usize>
where
    F: Fn(&Model, &[NodeId], VehicleId) -> Option<Cost>,
{
    let veh_r = m.vehicle(VehicleId(r as u32));
    // let lambda = p.lambda();
    let mut best: Option<TwoOptStarMove> = None;
    for v in 0..sol.len() {
        if v == r || sol[v].is_empty() {
            continue;
        }
        let veh_v = m.vehicle(VehicleId(v as u32));
        for i in 0..sol[r].len() {
            for j in 0..sol[v].len() {
                let tail_r = sol[r].get(i + 1).copied();
                let tail_v = sol[v].get(j + 1).copied();
                if tail_r.is_none() && tail_v.is_none() {
                    continue; // both tails empty: no arc changes
                }
                let arc = |class: usize, a: NodeId, b: NodeId| {
                    // FIXME: delta ranks by true arc cost; penalties only steer acceptance.
                    // let mut c = m.eval(class, a, b);
                    // if lambda != 0 {
                    //     c += lambda * p.arc(a, b);
                    // }
                    // c
                    m.eval(class, a, b)
                };
                let out = arc(veh_r.cost_class, sol[r][i], tail_r.unwrap_or(veh_r.end))
                    + arc(veh_v.cost_class, sol[v][j], tail_v.unwrap_or(veh_v.end));
                let into = arc(veh_r.cost_class, sol[r][i], tail_v.unwrap_or(veh_r.end))
                    + arc(veh_v.cost_class, sol[v][j], tail_r.unwrap_or(veh_v.end));
                if into - out >= best.as_ref().map_or(0, |b| b.0) {
                    continue;
                }
                let new_r: Vec<NodeId> = sol[r][..=i]
                    .iter()
                    .chain(&sol[v][j + 1..])
                    .copied()
                    .collect();
                let new_v: Vec<NodeId> = sol[v][..=j]
                    .iter()
                    .chain(&sol[r][i + 1..])
                    .copied()
                    .collect();
                let new =
                    eval(m, &new_r, VehicleId(r as u32)).zip(eval(m, &new_v, VehicleId(v as u32)));
                if let Some((a, b)) = new
                    && a + b < cost[r] + cost[v]
                {
                    best = Some((into - out, v, new_r, new_v, a, b));
                }
            }
        }
    }
    let (_, v, new_r, new_v, a, b) = best?;
    sol[r] = new_r;
    sol[v] = new_v;
    cost[r] = a;
    cost[v] = b;
    Some(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::eval_routes;
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

    /// Crossing routes on tight capacity: no single-customer move helps, only
    /// trading tails between routes does.
    #[test]
    fn two_opt_star_repartitions_routes() {
        // Depot at the origin, customers on the compass points.
        let xy = [(0, 0), (10, 0), (0, 10), (-10, 0), (0, -10)];
        let dist = move |a: NodeId, b: NodeId| {
            let (ax, ay) = xy[a.index()];
            let (bx, by) = xy[b.index()];
            let (dx, dy) = (ax - bx, ay - by);
            ((dx * dx + dy * dy) as f64).sqrt().round() as i64
        };
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

        // East rides with west, north with south: both routes cross the map.
        let mut sol = vec![vec![NodeId(1), NodeId(3)], vec![NodeId(2), NodeId(4)]];
        let mut cost: Vec<Cost> = (0..2)
            .map(|v| eval_route(&m, &sol[v], VehicleId(v as u32)).unwrap())
            .collect();
        assert_eq!(cost.iter().sum::<Cost>(), 80);

        let v = try_two_opt_star(&m, &mut sol, &eval_route, &mut cost, 0);
        assert_eq!(v, Some(1));
        assert_eq!(
            sol,
            vec![vec![NodeId(1), NodeId(4)], vec![NodeId(2), NodeId(3)]]
        );
        assert_eq!(cost.iter().sum::<Cost>(), 68);
        // A local optimum of this neighborhood: a second call finds nothing.
        assert_eq!(
            try_two_opt_star(&m, &mut sol, &eval_route, &mut cost, 0),
            None
        );
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
