//! `precede(before, after)` holds whenever a single vehicle serves both
//! stops. A pair split across two vehicles is unordered.
//!
//! ```sh
//! cargo run --example precedence
//! ```

use volare::{Construct, Improve, ModelBuilder, NodeId, solve};

fn main() {
    // A rear-door van: two pallets block each other only on the same van.
    let coords: [(f64, f64); 7] = [
        (0.0, 0.0),    // 0 depot
        (12.0, 3.0),   // 1
        (15.0, -4.0),  // 2
        (4.0, 14.0),   // 3
        (-13.0, 6.0),  // 4
        (-9.0, -12.0), // 5
        (7.0, -13.0),  // 6
    ];

    let blocked_by = [
        (NodeId(1), NodeId(2)),
        (NodeId(3), NodeId(4)),
        (NodeId(5), NodeId(6)),
    ];

    let mut b = ModelBuilder::new(coords.len());

    let cost = b.cost_class(move |from, to| {
        let (p, q) = (coords[from.index()], coords[to.index()]);
        (p.0 - q.0).hypot(p.1 - q.1).round() as i64
    });

    b.vehicle(NodeId(0), NodeId(0), cost);
    b.vehicle(NodeId(0), NodeId(0), cost);
    b.dimension(
        "pallets",
        |_from, to| if to == NodeId(0) { 0 } else { 1 },
        vec![4, 4],
    );

    for (front, behind) in blocked_by {
        b.precede(front, behind);
    }

    let model = b.build();
    let sol = solve(
        &model,
        Construct::CheapestInsertion,
        Improve::Gls { iters: 200 },
    );

    for (v, route) in sol.routes.iter().enumerate() {
        let stops: Vec<usize> = route.iter().map(|n| n.index()).collect();
        println!("van {v}: {stops:?}");
    }
    println!("total cost: {}", sol.cost);

    for (front, behind) in blocked_by {
        let (f, t) = (front.index(), behind.index());
        let Some(route) = sol
            .routes
            .iter()
            .find(|r| r.contains(&front) && r.contains(&behind))
        else {
            println!("{f} and {t} went to different vans, unordered");
            continue;
        };
        let at = |n: NodeId| route.iter().position(|&x| x == n).unwrap();
        assert!(at(front) < at(behind), "{f} is behind {t}");
        println!("{f} before {t} on the same van");
    }
}
