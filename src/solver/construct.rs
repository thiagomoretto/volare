use super::{Construct, SearchEvent, candidate_vehicles, with_front};
use crate::eval::{Routes, eval_route};
use crate::model::Model;
use crate::types::{Cost, NodeId, VehicleId};

pub fn first_solution_with(
    m: &Model,
    construct: Construct,
    log: impl FnMut(SearchEvent),
) -> Routes {
    match construct {
        Construct::CheapestInsertion => cheapest_insertion(m, log),
        Construct::GreedyRandomized { seed, k } => greedy_randomized(m, seed, k, log),
    }
}

/// A candidate insertion: its delta cost, the vehicle, and the position.
type Insertion = (Cost, usize, usize);

/// Cheapest feasible position for `u` in route `v`, or `None` if it has none.
fn best_in_route(
    m: &Model,
    sol: &Routes,
    cost: &[Cost],
    v: usize,
    u: NodeId,
    scratch: &mut Vec<NodeId>,
) -> Option<Insertion> {
    let mut best: Option<Insertion> = None;
    with_front(&sol[v], u, scratch);
    for pos in 0..=sol[v].len() {
        if pos > 0 {
            scratch.swap(pos - 1, pos);
        }
        let Some(c) = eval_route(m, scratch, VehicleId(v as u32)) else {
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
    m: &Model,
    sol: &Routes,
    cost: &[Cost],
    vs: &[usize],
    u: NodeId,
    scratch: &mut Vec<NodeId>,
) -> Option<Insertion> {
    let mut best: Option<Insertion> = None;
    for &v in vs {
        if let Some(c) = best_in_route(m, sol, cost, v, u, scratch)
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

    /// Uniform over `0..n`. Modulo bias under 2^-55 at these `n`, ignore it.
    #[inline]
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Always the cheapest insertion on offer.
pub fn cheapest_insertion(m: &Model, log: impl FnMut(SearchEvent)) -> Routes {
    insertion(m, 1, 0, log)
}

/// Draws from the `k` cheapest. Not a better first solution, a different one
/// per seed, so callers can race several and keep the cheapest.
pub fn greedy_randomized(m: &Model, seed: u64, k: usize, log: impl FnMut(SearchEvent)) -> Routes {
    insertion(m, k, seed, log)
}

/// Insertion with a candidate list of width `k`. The cache below is
/// indifferent to the draw: one route still grows per step.
fn insertion(m: &Model, k: usize, seed: u64, mut log: impl FnMut(SearchEvent)) -> Routes {
    let nv = m.vehicle_count();
    let mut rng = Rng(seed);
    let mut sol: Routes = vec![Vec::new(); nv];
    let mut cost = vec![0 as Cost; nv];
    let mut unrouted: Vec<NodeId> = (0..m.node_count() as u32)
        .map(NodeId)
        .filter(|&n| !m.is_terminal(n))
        .collect();
    let mut scratch = Vec::new();
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
        let split = cands
            .iter()
            .position(|&v| sol[v].is_empty())
            .unwrap_or(cands.len());
        let (used, empty) = cands.split_at(split);

        ranked.clear();
        for i in 0..unrouted.len() {
            let u = unrouted[i];
            if dirty[i] {
                best[i] = best_over(m, &sol, &cost, used, u, &mut scratch);
                dirty[i] = false;
            } else if let Some(t) = changed {
                let held = best[i];
                let fresh_t = best_in_route(m, &sol, &cost, t, u, &mut scratch);
                // Every route but `t` was already worse and none of them
                // moved, so only a `t` that got worse needs a full rescan.
                best[i] = match (held, fresh_t) {
                    (Some((d, v, _)), Some(c)) if v == t && c.0 <= d => Some(c),
                    (Some((_, v, _)), _) if v == t => {
                        best_over(m, &sol, &cost, used, u, &mut scratch)
                    }
                    (_, Some(c)) if held.is_none_or(|b| c < b) => Some(c),
                    _ => held,
                };
            }
            let fresh = best_over(m, &sol, &cost, empty, u, &mut scratch);
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
                if let Some(c) = best_over(m, &sol, &cost, &all, u, &mut scratch) {
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
