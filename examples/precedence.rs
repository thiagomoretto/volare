//! Ordering within a route: one stop served ahead of another.
//!
//! `ModelBuilder::precede(before, after)` holds whenever a single vehicle
//! serves both stops, in construction and in every improvement move. It says
//! nothing about a pair split across two vehicles, and it never pulls the two
//! onto one vehicle — pinning a pair to one van is a separate constraint that
//! volare does not have yet. Run with:
//!
//! ```sh
//! cargo run --example precedence
//! ```

use volare::{Construct, Improve, ModelBuilder, NodeId, solve};

fn main() {
    // A rear-door van: loaded once at the depot, unloaded from the back. The
    // warehouse packs to a fixed order, so a pallet packed early sits deeper
    // in the van than one packed late. Two pallets only block each other when
    // they ride the same van, which is exactly what `precede` says.
    let coords: [(f64, f64); 7] = [
        (0.0, 0.0),    // 0 depot
        (12.0, 3.0),   // 1
        (15.0, -4.0),  // 2
        (4.0, 14.0),   // 3
        (-13.0, 6.0),  // 4
        (-9.0, -12.0), // 5
        (7.0, -13.0),  // 6
    ];

    // Packed deep first. Stop 2's pallet went in before stop 1's, so on a van
    // carrying both, stop 1 comes out first.
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

    // The constraint is hard where it applies, and absent where it does not.
    for (front, behind) in blocked_by {
        let on = |n: NodeId| {
            sol.routes
                .iter()
                .enumerate()
                .find_map(|(v, r)| r.iter().position(|&x| x == n).map(|i| (v, i)))
                .expect("every stop is served")
        };
        let (van_front, at_front) = on(front);
        let (van_behind, at_behind) = on(behind);

        if van_front == van_behind {
            assert!(
                at_front < at_behind,
                "{} is behind {} in van {van_front}",
                front.index(),
                behind.index()
            );
            println!(
                "{} before {} on van {van_front}",
                front.index(),
                behind.index()
            );
        } else {
            // Two vans, two separate loads: neither pallet is in the other's
            // way, so the solver is free to order them however it likes.
            println!(
                "{} and {} went to different vans, unordered",
                front.index(),
                behind.index()
            );
        }
    }
}
