use std::collections::VecDeque;

use super::operators::{try_or_opt, try_relocate, try_swap, try_two_opt, try_two_opt_star};
use super::{Operator, RouteEval, Scratch, SearchEvent};
use crate::eval::{Routes, eval_route};
use crate::model::Model;
use crate::types::{Cost, NodeId, VehicleId};

/// An accepted improving move: which operator fired, and the second route it
/// schanged.
pub(super) struct Move {
    pub operator: Operator,
    pub other_route: Option<usize>,
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

struct OperatorCtx<'a, E: RouteEval> {
    m: &'a Model,
    sol: &'a mut Routes,
    eval: &'a E,
    cost: &'a mut [Cost],
    sx: &'a mut Scratch,
}

/// Fires a improving move: Cheapest operator first.
#[inline]
fn improving_move<E: RouteEval>(
    c: &mut OperatorCtx<'_, E>,
    u: NodeId,
    r: usize,
    route_stale: bool,
) -> Option<Move> {
    if let Some(v) = try_relocate(c.m, c.sol, c.eval, c.cost, u, r, c.sx) {
        Some(Move {
            operator: Operator::Relocate,
            other_route: Some(v),
        })
    } else if let Some(v) = try_swap(c.m, c.sol, c.eval, c.cost, u, r) {
        Some(Move {
            operator: Operator::Swap,
            other_route: Some(v),
        })
    } else if route_stale && try_two_opt(c.m, c.sol, c.eval, c.cost, r) {
        Some(Move {
            operator: Operator::TwoOpt,
            other_route: None,
        })
    } else {
        try_or_opt(c.m, c.sol, c.eval, c.cost, u, r, c.sx).map(|v| Move {
            operator: Operator::OrOpt,
            other_route: Some(v),
        })
    }
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
    let mut route_stale = vec![true; sol.len()];
    let mut sx = Scratch::default();

    // Draining the queue is not a fixpoint: a move only re-wakes the two
    // routes it touched, and a node elsewhere may now have an improving move
    // into them. Re-sweep everything until a whole sweep finds nothing.
    loop {
        // Future: Switch picking strategy.
        // Now is going from 0...n. Deterministic order.
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
            let Some(Move {
                operator,
                other_route,
            }) = improving_move(
                &mut OperatorCtx {
                    m,
                    sol,
                    eval: &eval,
                    cost: &mut cost,
                    sx: &mut sx,
                },
                u,
                r,
                route_stale[r],
            )
            else {
                // No improvements, this reset route-level operator staleless and move on.
                route_stale[r] = false;
                continue;
            };

            improved = true;
            log(SearchEvent::Improvement {
                operator,
                cost: cost.iter().sum(),
            });
            for t in [Some(r), other_route.filter(|&v| v != r)]
                .into_iter()
                .flatten()
            {
                // The route has changed, any route-pass operator is now allowed.
                route_stale[t] = true;
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
            // routes. It fires here instead of per node because a tail swap
            // is partition-level: run mid-cascade it disrupts routes the
            // fine operators would have fixed for less.
            for r in 0..sol.len() {
                if sol[r].is_empty() {
                    continue;
                }
                if let Some(v) = try_two_opt_star(m, sol, &eval, &mut cost, r) {
                    route_stale[r] = true;
                    route_stale[v] = true;
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
