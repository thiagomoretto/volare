//! Bring your own operator and hand it to the solver.
//!
//! An operator is a function that finds one improving change and applies it.
//! This is the smallest one worth writing: take the stop the descent just
//! woke together with the one after it, and try trading that pair for a pair
//! on another route. volare ships a swap that trades one stop for one; two for
//! two is a move it does not have.
//!
//! `local_search_with_operators` puts it in the rotation with volare's own
//! four. It is not a pass wrapped around the solver — the operator rides the
//! same node queue and the same don't-look bits as relocate, swap and 2-opt,
//! and a move it accepts re-wakes the routes it touched, exactly as theirs do.
//!
//! The contract is two functions wide. [`Search`] prices routes:
//!
//! * `cx.eval(route, v)` prices a route. Reach for it when a move *rearranges*
//!   a route, because the stops are the same ones in a different order.
//! * `cx.eval_splice(route, range, repl, v)` prices a route with one stretch
//!   replaced. Reach for it when a move *changes what is on* the route —
//!   inserting, removing, moving a run to another vehicle — because then
//!   there is a new route to build and no reason to allocate one.
//!
//! Your side of the bargain: leave `cost[v]` correct for any route you
//! rewrote, and return `Some` only for a move you actually applied that made
//! the solution cheaper.
//!
//! ```sh
//! cargo run --release --example custom_operator
//! ```

use volare::eval::eval_routes;
use volare::solver::{
    Construct, Operator, SearchEvent, first_solution_with, local_search_with,
    local_search_with_operators,
};
use volare::{Cost, ModelBuilder, NodeId, Routes, Search, VehicleId};

fn main() {
    let m = windowed_model();
    let start = first_solution_with(&m, Construct::CheapestInsertion, |_| {});

    // What the shipped four reach on their own, to measure against.
    let mut alone = start.clone();
    local_search_with(&m, &mut alone, |_| {});
    let alone_cost = eval_routes(&m, &alone).expect("local search is feasible");

    // The same descent with yours in the rotation.
    let mut sol = start;
    let mut trades = 0;
    let mut mine = |cx: &mut Search, sol: &mut Routes, cost: &mut [Cost], u, r| {
        trade_pairs(cx, sol, cost, u, r)
    };
    local_search_with_operators(&m, &mut sol, &mut [&mut mine], |e| {
        if let SearchEvent::Improvement {
            operator: Operator::Custom(_),
            ..
        } = e
        {
            trades += 1;
        }
    });

    let together = eval_routes(&m, &sol).expect("the operator kept the solution feasible");
    println!("shipped operators alone   {alone_cost}");
    println!("with yours in the mix     {together}, after {trades} trades of its own");
}

/// Trade the pair of stops starting at `u` for a pair on another route, first
/// improvement wins. Returns the other vehicle.
///
/// The whole operator. volare ships `swap`, which trades one stop for one
/// stop; trading two for two reaches arrangements that no sequence of single
/// swaps can, because each single step through them costs more than both ends.
///
/// Both routes keep their length, so the four stops can be exchanged in place
/// and put back if the trade does not pay — no buffers, and nothing stored
/// between calls. `cx.eval` answers `None` for a route that breaks a
/// constraint, so the fall-through arm covers infeasible and unprofitable
/// alike.
fn trade_pairs(
    cx: &mut Search,
    sol: &mut Routes,
    cost: &mut [Cost],
    u: NodeId,
    r: usize,
) -> Option<usize> {
    let at = sol[r].iter().position(|&x| x == u)?;
    if at + 1 >= sol[r].len() {
        return None; // `u` ends its route, so there is no pair to trade
    }
    for v in 0..sol.len() {
        if v == r {
            continue;
        }
        for q in 0..sol[v].len().saturating_sub(1) {
            let mine = (sol[r][at], sol[r][at + 1]);
            let theirs = (sol[v][q], sol[v][q + 1]);
            sol[r][at] = theirs.0;
            sol[r][at + 1] = theirs.1;
            sol[v][q] = mine.0;
            sol[v][q + 1] = mine.1;

            let traded = match cx.eval(&sol[r], VehicleId(r as u32)) {
                Some(a) => cx.eval(&sol[v], VehicleId(v as u32)).map(|b| (a, b)),
                None => None,
            };
            match traded {
                Some((a, b)) if a + b < cost[r] + cost[v] => {
                    cost[r] = a;
                    cost[v] = b;
                    return Some(v);
                }
                _ => {
                    sol[r][at] = mine.0;
                    sol[r][at + 1] = mine.1;
                    sol[v][q] = theirs.0;
                    sol[v][q + 1] = theirs.1;
                }
            }
        }
    }
    None
}

/// Sixty stops scattered over a square, eight vans, and a delivery window on
/// every other stop.
///
/// The windows are what make a trade worth trying. Without them a route's
/// order is unconstrained and 2-opt flattens it; with them most reversals turn
/// infeasible, and exchanging two stops is one of the few rearrangements left.
fn windowed_model() -> volare::Model {
    let coords = scattered_stops(60);
    let fleet = 8;
    let drive = |xy: Vec<(f64, f64)>| {
        move |from: NodeId, to: NodeId| {
            let (p, q) = (xy[from.index()], xy[to.index()]);
            (p.0 - q.0).hypot(p.1 - q.1).round() as i64
        }
    };

    let mut b = ModelBuilder::new(coords.len());
    let cost_class = b.cost_class(drive(coords.clone()));
    for _ in 0..fleet {
        b.vehicle(NodeId(0), NodeId(0), cost_class);
    }
    b.dimension(
        "load",
        |_from, to| if to == NodeId(0) { 0 } else { 1 },
        vec![10; fleet],
    );
    b.dimension("time", drive(coords.clone()), vec![i64::MAX; fleet]);

    let depot = coords[0];
    for i in (1..coords.len()).step_by(2) {
        let (p, q) = (coords[i], depot);
        let direct = (p.0 - q.0).hypot(p.1 - q.1).round() as i64;
        b.cumul_bounds("time", NodeId(i as u32), direct, direct + 250);
    }
    b.build()
}

/// Seeded by hand, so the example prints the same numbers on every machine.
fn scattered_stops(n: usize) -> Vec<(f64, f64)> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 33) as f64 / (1u64 << 31) as f64
    };
    let mut coords = vec![(0.0, 0.0)];
    for _ in 0..n {
        coords.push((next() * 200.0 - 100.0, next() * 200.0 - 100.0));
    }
    coords
}
