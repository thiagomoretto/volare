//! Bring your own operator: trade two stops on the same route.
//!
//! An operator is a function that finds one improving change and applies it.
//! This is the smallest one worth writing: pick two stops on a route, swap
//! them, keep the swap if the route got cheaper, put them back if it did not.
//!
//! volare does not ship this move. Its `swap` trades a stop with one on a
//! *different* route, and its `2-opt` reverses a stretch of one route — a
//! stretch of two is the only exchange those two already cover. So trading
//! two stops that are further apart on one route is a real gap, and a
//! four-line change away from the operators the library does ship.
//!
//! The whole contract is [`Search`]: it prices routes, and it is the only
//! thing an operator needs.
//!
//! * `cx.eval(route, v)` prices a route. Reach for it when a move *rearranges*
//!   a route, because the stops are the same ones in a different order.
//! * `cx.eval_splice(route, range, repl, v)` prices a route with one stretch
//!   replaced. Reach for it when a move *changes what is on* the route —
//!   inserting, removing, or moving a run to another vehicle — because then
//!   there is a new route to build and no reason to allocate one.
//!
//! ```sh
//! cargo run --release --example custom_operator
//! ```

use volare::eval::eval_routes;
use volare::{Cost, ModelBuilder, NodeId, Routes, Search, VehicleId};

fn main() {
    // Ten stops evenly spaced on a circle, depot in the middle. The cheapest
    // tour walks the rim in order, so any out-of-order stop costs a detour.
    let coords = circle_of_stops(10);
    let mut b = ModelBuilder::new(coords.len());
    let cost_class = b.cost_class(move |from, to| {
        let (p, q) = (coords[from.index()], coords[to.index()]);
        (p.0 - q.0).hypot(p.1 - q.1).round() as i64
    });
    b.vehicle(NodeId(0), NodeId(0), cost_class);
    let m = b.build();

    // A deliberately jumbled route, so there is something to fix.
    let mut sol: Routes = vec![
        vec![3, 7, 1, 9, 5, 2, 10, 4, 8, 6]
            .into_iter()
            .map(NodeId)
            .collect(),
    ];

    // One context for the whole run; its buffers outlive every single move.
    let mut cx = Search::new(&m);
    let mut cost: Vec<Cost> = (0..sol.len())
        .map(|v| cx.eval(&sol[v], VehicleId(v as u32)).expect("feasible"))
        .collect();
    let before = cost[0];

    let mut moves = 0;
    while trade_two_stops(&mut cx, &mut sol, &mut cost) {
        moves += 1;
    }

    let after = eval_routes(&m, &sol).expect("the operator kept the route feasible");
    assert_eq!(after, cost.iter().sum::<Cost>(), "tracked cost drifted");
    println!("start {before}, after {moves} trades {after}");
    println!("route {:?}", sol[0].iter().map(|n| n.0).collect::<Vec<_>>());
}

/// Apply the first exchange of two stops that makes a route cheaper, or report
/// that there is none left.
///
/// The whole operator. `cx.eval` reports `None` for a route that breaks a
/// constraint, and the fall-through arm puts the stops back, so an exchange is
/// kept only when it is both feasible and an improvement.
fn trade_two_stops(cx: &mut Search, sol: &mut Routes, cost: &mut [Cost]) -> bool {
    for r in 0..sol.len() {
        let v = VehicleId(r as u32);
        for i in 0..sol[r].len() {
            for j in i + 1..sol[r].len() {
                sol[r].swap(i, j);
                match cx.eval(&sol[r], v) {
                    Some(c) if c < cost[r] => {
                        cost[r] = c;
                        return true;
                    }
                    _ => sol[r].swap(i, j),
                }
            }
        }
    }
    false
}

/// `n` stops on a circle of radius 100, depot at the centre as node 0.
fn circle_of_stops(n: usize) -> Vec<(f64, f64)> {
    let mut coords = vec![(0.0, 0.0)];
    for k in 0..n {
        let a = k as f64 * std::f64::consts::TAU / n as f64;
        coords.push((100.0 * a.cos(), 100.0 * a.sin()));
    }
    coords
}
