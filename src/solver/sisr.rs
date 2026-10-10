use std::ops::ControlFlow;

use super::construct::Rng;
use super::{SearchEvent, candidate_vehicles};
use crate::eval::{Routes, eval_route, eval_route_split};
use crate::model::Model;
use crate::types::{Cost, NodeId, VehicleId};

/// Customers one ruin removes, on average.
const AVG_REMOVED: usize = 10;
/// Longest string one ruin cuts out of a route.
const MAX_STRING: usize = 10;
/// One in this many split strings stops growing its kept run at each step.
const SPLIT_STOP: usize = 100;
/// One in this many insertion positions is skipped without being priced.
const BLINK: usize = 100;
/// Neighbors kept per customer. A ruin walks out from its seed until it has
/// cut enough routes, which happens long before this list runs out.
const NEIGHBORS: usize = 100;
/// The start temperature divides the mean arc cost by this, and the final
/// temperature divides the start by `COOLING`, so the schedule follows the
/// model's cost scale instead of fixed units.
const HEAT: f64 = 2.0;
const COOLING: f64 = 100.0;

/// Ruin and recreate with slack induction by string removals (Christiaens and
/// Vanden Berghe). Each round cuts a few short strings out of routes that lie
/// near one random customer, puts the customers back by cheapest insertion
/// with a small chance to skip each position, and accepts the result under
/// simulated annealing. The best true-cost solution seen is what comes back.
///
/// Constraints stay a black box: every insertion is priced by `eval_route`,
/// and a dropped node is an insertion into the unserved sink. A round that
/// cannot place every customer is thrown away.
///
/// ponytail: each insertion position costs a full `eval_route` pass, so a
/// round is quadratic in route length. Cached cumul prefixes per route turn
/// that check into a constant-time slack test.
pub fn ruin_recreate(m: &Model, sol: &mut Routes, iters: usize, seed: u64) {
    ruin_recreate_with(m, sol, iters, seed, |_| ControlFlow::Continue(()))
}

/// `ruin_recreate` reporting a `SisrBest` per new best cost, a `SisrRound` per
/// round, and a final `Done`. Stopped early by the callback, it still returns
/// the best solution so far.
pub fn ruin_recreate_with(
    m: &Model,
    sol: &mut Routes,
    iters: usize,
    seed: u64,
    mut log: impl FnMut(SearchEvent) -> ControlFlow<()>,
) {
    let sink = m.unserved_vehicle().map(|v| v.index());
    let mut rng = Rng(seed);
    let customers: Vec<NodeId> = (0..m.node_count() as u32)
        .map(NodeId)
        .filter(|&n| !m.is_terminal(n))
        .collect();
    let adj = neighbors(m, &customers);
    let depot = m.vehicle(VehicleId(0)).start;
    let cost_class = m.vehicle(VehicleId(0)).cost_class;

    let mut cost: Vec<Cost> = route_costs(m, sol);
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
    let t0 = (arc_cost as f64 / served.max(1) as f64 / HEAT).max(1.0);

    let mut stopped = log(SearchEvent::SisrBest {
        iter: 0,
        cost: best_cost,
    })
    .is_break();

    let mut work = sol.clone();
    let mut work_cost = cost.clone();
    let mut at: Vec<(usize, usize)> = vec![(usize::MAX, 0); m.node_count()];
    let mut ruined: Vec<usize> = Vec::new();
    let mut removed: Vec<NodeId> = Vec::new();
    let mut ctx = Recreate::default();

    for iter in 1..=iters {
        if stopped {
            break;
        }
        work.clone_from(sol);
        work_cost.clone_from(&cost);

        locate(&work, &mut at);
        ruin(
            &mut rng,
            &adj,
            &customers,
            sink,
            &mut work,
            &at,
            &mut ruined,
            &mut removed,
        );
        for &v in &ruined {
            work_cost[v] = eval_route(m, &work[v], VehicleId(v as u32)).expect("a cut route");
        }
        order(m, &mut rng, &mut removed, depot, cost_class);

        if recreate(
            m,
            &mut rng,
            &mut work,
            &mut work_cost,
            &removed,
            sink,
            &mut ctx,
        ) {
            let new_cost: Cost = work_cost.iter().sum();
            let t = t0 * COOLING.powf(-(iter as f64) / iters as f64);
            if (new_cost as f64) < cur_cost as f64 + t * exp1(&mut rng) {
                std::mem::swap(sol, &mut work);
                std::mem::swap(&mut cost, &mut work_cost);
                cur_cost = new_cost;
                if cur_cost < best_cost {
                    best_cost = cur_cost;
                    best.clone_from(sol);
                    stopped = log(SearchEvent::SisrBest {
                        iter,
                        cost: best_cost,
                    })
                    .is_break();
                }
            }
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

fn route_costs(m: &Model, sol: &Routes) -> Vec<Cost> {
    sol.iter()
        .enumerate()
        .map(|(v, r)| eval_route(m, r, VehicleId(v as u32)).expect("infeasible start"))
        .collect()
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
    let max_len = (visits / used.max(1)).clamp(1, MAX_STRING);
    let strings = 1 + rng.below((4 * AVG_REMOVED / (1 + max_len)).max(1));

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

/// Buffers `recreate` reuses across rounds.
#[derive(Default)]
struct Recreate {
    vehicles: Vec<usize>,
    candidate: Vec<NodeId>,
}

/// Put every node of `removed` back at its cheapest unskipped feasible
/// position, `false` as soon as one fits nowhere.
fn recreate(
    m: &Model,
    rng: &mut Rng,
    sol: &mut Routes,
    cost: &mut [Cost],
    removed: &[NodeId],
    sink: Option<usize>,
    ctx: &mut Recreate,
) -> bool {
    let nv = sol.len();
    for &u in removed {
        candidate_vehicles(m, sol, &mut ctx.vehicles);
        let mut best = best_insertion(
            m,
            rng,
            sol,
            cost,
            &ctx.vehicles,
            u,
            sink,
            &mut ctx.candidate,
        );
        // Per-vehicle forbids make empty vehicles differ, so the one empty
        // candidate is not enough when it refuses `u`.
        if best.is_none() && ctx.vehicles.len() < nv {
            let rest: Vec<usize> = (0..nv).filter(|v| !ctx.vehicles.contains(v)).collect();
            best = best_insertion(m, rng, sol, cost, &rest, u, sink, &mut ctx.candidate);
        }
        let Some((delta, v, pos)) = best else {
            return false;
        };
        sol[v].insert(pos, u);
        cost[v] += delta;
    }
    true
}

/// Cheapest `(delta, vehicle, position)` for `u` over `vs`. The sink's order
/// is meaningless, so it is priced at one position only, and never skipped.
#[allow(clippy::too_many_arguments)]
fn best_insertion(
    m: &Model,
    rng: &mut Rng,
    sol: &Routes,
    cost: &[Cost],
    vs: &[usize],
    u: NodeId,
    sink: Option<usize>,
    candidate: &mut Vec<NodeId>,
) -> Option<(Cost, usize, usize)> {
    let mut best: Option<(Cost, usize, usize)> = None;
    for &v in vs {
        let route = &sol[v];
        let first = if Some(v) == sink { route.len() } else { 0 };
        candidate.clear();
        candidate.extend_from_slice(&route[..first]);
        candidate.push(u);
        candidate.extend_from_slice(&route[first..]);
        for pos in first..=route.len() {
            if pos > first {
                candidate.swap(pos - 1, pos);
            }
            if Some(v) != sink && rng.below(BLINK) == 0 {
                continue;
            }
            let Some(c) = eval_route(m, candidate, VehicleId(v as u32)) else {
                continue;
            };
            let delta = c - cost[v];
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
