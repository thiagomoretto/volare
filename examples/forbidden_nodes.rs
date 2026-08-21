//! Vehicle-node exclusion: a vehicle that may not visit a given customer.
//!
//! `ModelBuilder::forbid(vehicle, node)` makes the pair infeasible; the
//! solver then routes that customer on some other vehicle, in construction
//! and in every improvement move. Run with:
//!
//! ```sh
//! cargo run --example forbidden_nodes
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

    let van = b.vehicle(NodeId(0), NodeId(0), cost);
    let truck = b.vehicle(NodeId(0), NodeId(0), cost);
    b.dimension("demand", move |_from, to| demands[to.index()], vec![11, 11]);

    // Stop 4's delivery doesn't fit in the van, so the truck has to take it.
    b.forbid(van, NodeId(4));

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

    // The constraint is hard: whatever the search does, stop 4 is never on
    // the van.
    assert!(!sol.routes[van.index()].contains(&NodeId(4)));
    assert!(sol.routes[truck.index()].contains(&NodeId(4)));
}
