use std::collections::VecDeque;

use super::operators::{try_or_opt, try_relocate, try_swap, try_two_opt, try_two_opt_star};
use super::{Operator, RouteEval, Scratch, SearchEvent};
use crate::eval::{Routes, eval_route};
use crate::model::Model;
use crate::types::{Cost, NodeId, VehicleId};

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

/// The descent itself, on whatever cost `eval` defines. Public callers get true
/// cost; only guided local search passes a penalized one.
pub(super) fn descend(
    m: &Model,
    sol: &mut Routes,
    eval: impl RouteEval,
    mut log: impl FnMut(SearchEvent),
) {
    let mut cost: Vec<Cost> = (0..sol.len())
        .map(|v| eval(m, &sol[v], VehicleId(v as u32)).expect("infeasible start solution"))
        .collect();

    let mut queued = vec![false; m.node_count()];
    let mut index = vec![u32::MAX; m.node_count()];
    // Route-level don't-look bit: 2-opt takes a route and ignores the popped
    // node, so without this it rescans one route once per node in it.
    let mut two_opt_dirty = vec![true; sol.len()];
    let mut sx = Scratch::default();

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

            // Cheapest operator first: relocate and swap cost O(n) route
            // evaluations, or-opt O(n) across all vehicles, 2-opt O(n^2).
            // Or-opt moves pairs; a single-node chain is relocate.
            let (other, operator) =
                if let Some(v) = try_relocate(m, sol, &eval, &mut cost, u, r, &mut sx) {
                    (Some(v), Operator::Relocate)
                } else if let Some(v) = try_swap(m, sol, &eval, &mut cost, u, r) {
                    (Some(v), Operator::Swap)
                } else if two_opt_dirty[r] && try_two_opt(m, sol, &eval, &mut cost, r) {
                    (None, Operator::TwoOpt)
                } else if let Some(v) = try_or_opt(m, sol, &eval, &mut cost, u, r, &mut sx) {
                    (Some(v), Operator::OrOpt)
                } else {
                    two_opt_dirty[r] = false;
                    continue;
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
            // routes. Firing it here instead of per node keeps the big tail
            // swaps from disrupting routes that relocate would have fixed for
            // less (X-n143-k7 regressed sharply the other way).
            for r in 0..sol.len() {
                if sol[r].is_empty() {
                    continue;
                }
                if let Some(v) = try_two_opt_star(m, sol, &eval, &mut cost, r) {
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
