use volare::{Construct, Improve, ModelBuilder, NodeId, solve};

// x: 0 10 20 30 40, plus a free sink at index 5.
fn dist(from: NodeId, to: NodeId) -> i64 {
    let x = [0.0f64, 10.0, 20.0, 30.0, 40.0, 0.0];
    if from.index() == 5 || to.index() == 5 {
        return 0;
    }
    (x[from.index()] - x[to.index()]).abs() as i64
}

#[test]
fn ends_at_a_different_node() {
    let mut b = ModelBuilder::new(5);
    let cost = b.cost_class(dist);
    b.vehicle(NodeId(0), NodeId(4), cost);
    b.dimension(
        "demand",
        |_, to| if to == NodeId(0) { 0 } else { 1 },
        vec![10],
    );
    let m = b.build();
    let sol = solve(&m, Construct::CheapestInsertion, Improve::Gls { iters: 200 });
    println!("end-at-4: {:?} cost {}", sol.routes, sol.cost);
    assert_eq!(sol.cost, 40);
}

#[test]
fn open_route_via_free_sink() {
    let mut b = ModelBuilder::new(6);
    let cost = b.cost_class(dist);
    b.vehicle(NodeId(0), NodeId(5), cost);
    b.dimension(
        "demand",
        |_, to| {
            if to.index() == 0 || to.index() == 5 {
                0
            } else {
                1
            }
        },
        vec![10],
    );
    let m = b.build();
    let sol = solve(&m, Construct::CheapestInsertion, Improve::Gls { iters: 200 });
    println!("open: {:?} cost {}", sol.routes, sol.cost);
    assert_eq!(sol.cost, 40);
}
