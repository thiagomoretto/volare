use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use super::SearchEvent;
use super::descent::{descend, local_search};
use crate::eval::{Routes, eval_route, eval_routes};
use crate::model::Model;
use crate::types::{Cost, NodeId, VehicleId};

/// Fraction of the average arc cost that one penalty is worth. The usual
/// starting point for CVRP; higher diversifies harder, lower descends harder.
const LAMBDA_NUMERATOR: Cost = 1;
const LAMBDA_DENOMINATOR: Cost = 10;

/// Arc penalties for guided local search, and the weight one penalty carries.
///
/// Owned by the search that is running, never by the `Model`: the model stays a
/// pure description of the problem, shareable by `&` across concurrent solves.
/// Created and dropped inside `guided_local_search_with`, so a penalty cannot
/// outlive the search that made it.
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
/// only the nodes touched by the arcs just penalized is the win, but seeding
/// the queue alone is capped by `descend`'s outer re-sweep, which starts a
/// full sweep anyway — tighten that first, then seed. Both are worth it when
/// the iteration count needs to go past a few hundred.
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
    // Dropped nodes are not customers: they must not dilute lambda.
    let customers: usize = sol
        .iter()
        .enumerate()
        .filter(|(v, _)| m.unserved_vehicle() != Some(VehicleId(*v as u32)))
        .map(|(_, r)| r.len())
        .sum();
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
