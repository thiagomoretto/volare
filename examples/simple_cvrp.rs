//! A tiny CVRP: one depot, a few stops, two vans. Run with:
//!
//! ```sh
//! cargo run --example simple_cvrp
//! ```

use volare::{Construct, Improve, ModelBuilder, NodeId, solve};

fn main() {
    // Node 0 is the depot; the rest are stops with a demand.
    let coords: [(f64, f64); 7] = [
        (0.0, 0.0),   // depot
        (10.0, 5.0),  //
        (12.0, -8.0), //
        (-6.0, 9.0),  //
        (-14.0, 2.0), //
        (3.0, 15.0),  //
        (8.0, -14.0), //
    ];
    let demands = [0, 3, 4, 2, 5, 3, 4];

    let mut b = ModelBuilder::new(coords.len());

    let cost = b.cost_class(move |from, to| {
        let (p, q) = (coords[from.index()], coords[to.index()]);
        (p.0 - q.0).hypot(p.1 - q.1).round() as i64
    });

    // Two vehicles, both starting and ending at the depot. Vehicles come
    // before dimensions: cumul limits are indexed by vehicle.
    b.vehicle(NodeId(0), NodeId(0), cost);
    b.vehicle(NodeId(0), NodeId(0), cost);
    b.dimension("demand", move |_from, to| demands[to.index()], vec![11, 11]);

    let model = b.build();
    let sol = solve(
        &model,
        Construct::CheapestInsertion,
        Improve::Gls { iters: 200 },
    );

    for (i, route) in sol.routes.iter().enumerate() {
        let stops: Vec<usize> = route.iter().map(|n| n.index()).collect();
        println!("vehicle {i}: {stops:?}");
    }
    println!("total cost: {}", sol.cost);
}
