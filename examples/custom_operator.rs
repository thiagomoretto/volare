//! Bring your own operator: or-opt, written entirely outside the library.
//!
//! volare ships relocate, swap, 2-opt and 2-opt\*. Or-opt is none of them — it
//! lifts a run of two or three consecutive stops and drops the whole run
//! somewhere else, either way round. This file implements it against nothing
//! but the public API, which is the point: a `Search` is the whole contract an
//! algorithm needs.
//!
//! Three things make it work, and they are the three an operator always wants:
//!
//! * `eval_splice` prices a route with one stretch replaced. Lifting a run is
//!   a stretch replaced by nothing; dropping it is nothing replaced by a run.
//! * `spliced` hands back the route that was just priced, so an accepted move
//!   commits without being rebuilt.
//! * `arc` prices one arc on the objective in force, so an operator that ranks
//!   by arc arithmetic ranks on what acceptance will actually use.
//!
//! ```sh
//! cargo run --release --example custom_operator
//! ```

use volare::eval::eval_routes;
use volare::solver::{Construct, first_solution_with};
use volare::{Cost, ModelBuilder, NodeId, Routes, Search, VehicleId};

/// Runs no longer than this get moved. Beyond three the neighborhood grows
/// faster than it pays.
const MAX_RUN: usize = 3;

fn main() {
    let coords = ring_of_clusters();
    let mut b = ModelBuilder::new(coords.len());
    let cost = b.cost_class(move |from, to| {
        let (p, q) = (coords[from.index()], coords[to.index()]);
        (p.0 - q.0).hypot(p.1 - q.1).round() as i64
    });
    for _ in 0..6 {
        b.vehicle(NodeId(0), NodeId(0), cost);
    }
    b.dimension(
        "load",
        |_from, to| if to == NodeId(0) { 0 } else { 1 },
        vec![8; 6],
    );
    let m = b.build();

    let mut sol = first_solution_with(&m, Construct::CheapestInsertion, |_| {});
    let before = eval_routes(&m, &sol).expect("construction is feasible");

    // One context for the run: its buffers outlive every individual move.
    let mut cx = Search::new(&m);
    let mut cost: Vec<Cost> = (0..sol.len())
        .map(|v| cx.eval(&sol[v], VehicleId(v as u32)).expect("feasible"))
        .collect();

    let mut moves = 0;
    let mut run = Vec::new();
    let mut without = Vec::new();
    while or_opt(&mut cx, &mut sol, &mut cost, &mut run, &mut without) {
        moves += 1;
    }

    let after = eval_routes(&m, &sol).expect("or-opt kept the solution feasible");
    assert_eq!(after, cost.iter().sum::<Cost>(), "tracked cost drifted");
    println!("construction {before}, after {moves} or-opt moves {after}");
    for (v, route) in sol.iter().enumerate() {
        if !route.is_empty() {
            println!(
                "  vehicle {v}: {:?}",
                route.iter().map(|n| n.0).collect::<Vec<_>>()
            );
        }
    }
}

/// Applies the first improving or-opt move, or reports a fixpoint.
///
/// `run` and `without` are the caller's: they are held *across* an evaluation,
/// and `Search` only owns the buffers evaluation itself consumes.
fn or_opt(
    cx: &mut Search,
    sol: &mut Routes,
    cost: &mut [Cost],
    run: &mut Vec<NodeId>,
    without: &mut Vec<NodeId>,
) -> bool {
    for r in 0..sol.len() {
        for len in 1..=MAX_RUN {
            for i in 0..sol[r].len().saturating_sub(len - 1) {
                if i + len > sol[r].len() {
                    break;
                }
                let vr = VehicleId(r as u32);
                // Lift the run: the stretch replaced by nothing.
                let Some(lifted_cost) = cx.eval_splice(&sol[r], i..i + len, &[], vr) else {
                    continue;
                };
                run.clear();
                run.extend_from_slice(&sol[r][i..i + len]);
                without.clear();
                without.extend_from_slice(cx.spliced());

                let Some((v, c)) = best_drop(cx, sol, cost, r, lifted_cost, run, without) else {
                    continue;
                };
                // `best_drop` left the winning route in the context, and no
                // evaluation has run since, so it is still there.
                if v != r {
                    sol[r].clear();
                    sol[r].extend_from_slice(without);
                    cost[r] = lifted_cost;
                }
                sol[v].clear();
                sol[v].extend_from_slice(cx.spliced());
                cost[v] = c;
                return true;
            }
        }
    }
    false
}

/// First position, on any vehicle and either way round, that drops `run` for
/// less than leaving it where it was. Probe only — nothing is committed, and
/// `sol` stays borrowed shared throughout.
fn best_drop(
    cx: &mut Search,
    sol: &Routes,
    cost: &[Cost],
    r: usize,
    lifted_cost: Cost,
    run: &mut [NodeId],
    without: &[NodeId],
) -> Option<(usize, Cost)> {
    for v in 0..sol.len() {
        let base: &[NodeId] = if v == r { without } else { &sol[v] };
        // A run of one reads the same both ways.
        for reversed in [false, run.len() > 1] {
            if reversed {
                run.reverse();
            }
            for pos in 0..=base.len() {
                let Some(c) = cx.eval_splice(base, pos..pos, run, VehicleId(v as u32)) else {
                    continue;
                };
                let delta = if v == r {
                    c - cost[r]
                } else {
                    (lifted_cost - cost[r]) + (c - cost[v])
                };
                if delta < 0 {
                    return Some((v, c));
                }
            }
        }
    }
    None
}

/// Four tight clusters on a ring. Cheapest insertion threads routes between
/// them, which is exactly the layout a run-length move repairs.
fn ring_of_clusters() -> Vec<(f64, f64)> {
    let mut coords = vec![(0.0, 0.0)];
    for (cx, cy) in [(40.0, 0.0), (0.0, 40.0), (-40.0, 0.0), (0.0, -40.0)] {
        for k in 0..8 {
            let a = k as f64 * std::f64::consts::TAU / 8.0;
            coords.push((cx + 6.0 * a.cos(), cy + 6.0 * a.sin()));
        }
    }
    coords
}
