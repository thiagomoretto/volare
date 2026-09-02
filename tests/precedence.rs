//! Ordering within a route. See `ModelBuilder::precede` for the semantics.

use volare::{Construct, Improve, ModelBuilder, NodeId, VehicleId, eval_route, eval_routes, solve};

/// Depot 0, customers 1..=5 spaced along a line, arc cost the line distance.
fn builder(vehicles: usize) -> ModelBuilder {
    let mut b = ModelBuilder::new(6);
    let cost = b.cost_class(|a: NodeId, b: NodeId| (a.0 as i64 - b.0 as i64).abs() * 10);
    for _ in 0..vehicles {
        b.vehicle(NodeId(0), NodeId(0), cost);
    }
    b.dimension(
        "load",
        |_from, to| if to == NodeId(0) { 0 } else { 1 },
        vec![5; vehicles],
    );
    b
}

#[test]
fn out_of_order_is_infeasible() {
    let mut b = builder(1);
    b.precede(NodeId(3), NodeId(1));
    let m = b.build();

    assert_eq!(
        eval_route(&m, &[NodeId(1), NodeId(2), NodeId(3)], VehicleId(0)),
        None,
        "1 is served before its predecessor 3"
    );
    assert!(eval_route(&m, &[NodeId(3), NodeId(2), NodeId(1)], VehicleId(0)).is_some());
}

#[test]
fn a_pair_split_across_vehicles_is_unordered() {
    let mut b = builder(2);
    b.precede(NodeId(3), NodeId(1));
    let m = b.build();

    let split = vec![vec![NodeId(1)], vec![NodeId(3)]];
    assert!(eval_routes(&m, &split).is_some());
}

#[test]
fn unrelated_nodes_sit_anywhere() {
    let mut b = builder(1);
    b.precede(NodeId(4), NodeId(2));
    let m = b.build();

    for route in [
        vec![NodeId(4), NodeId(1), NodeId(2)],
        vec![NodeId(1), NodeId(4), NodeId(2)],
        vec![NodeId(4), NodeId(2), NodeId(1)],
    ] {
        assert!(
            eval_route(&m, &route, VehicleId(0)).is_some(),
            "{route:?} keeps 4 ahead of 2"
        );
    }
}

#[test]
fn chains_compose() {
    let mut b = builder(1);
    b.precede(NodeId(5), NodeId(3));
    b.precede(NodeId(3), NodeId(1));
    let m = b.build();

    assert!(eval_route(&m, &[NodeId(5), NodeId(3), NodeId(1)], VehicleId(0)).is_some());
    assert_eq!(
        eval_route(&m, &[NodeId(5), NodeId(1), NodeId(3)], VehicleId(0)),
        None
    );
    assert_eq!(
        eval_route(&m, &[NodeId(3), NodeId(5), NodeId(1)], VehicleId(0)),
        None
    );
}

#[test]
fn the_solver_never_returns_an_out_of_order_route() {
    // Cheapest order on a line runs 1..5; requiring 2 ahead of 1 and 4 ahead
    // of 5 rules out both monotone orders.
    let mut b = builder(2);
    b.precede(NodeId(2), NodeId(1));
    b.precede(NodeId(4), NodeId(5));
    let m = b.build();

    let sol = solve(&m, Construct::CheapestInsertion, Improve::Gls { iters: 50 });

    assert!(volare::eval::visits_all_nodes(&m, &sol.routes));
    for route in &sol.routes {
        let at = |n: NodeId| route.iter().position(|&x| x == n);
        for (before, after) in [(NodeId(2), NodeId(1)), (NodeId(4), NodeId(5))] {
            if let (Some(i), Some(j)) = (at(before), at(after)) {
                assert!(i < j, "{route:?} serves {after:?} before {before:?}");
            }
        }
    }
}

#[test]
fn the_unserved_sink_ignores_ordering() {
    let mut b = builder(1);
    b.precede(NodeId(2), NodeId(1));
    b.allow_drop(NodeId(1), 1);
    b.allow_drop(NodeId(2), 1);
    let m = b.build();

    let sink = m.unserved_vehicle().expect("drops declared");
    assert!(eval_route(&m, &[NodeId(1), NodeId(2)], sink).is_some());
}

#[test]
#[should_panic(expected = "vacuous")]
fn ordering_against_a_terminal_is_rejected() {
    let mut b = builder(1);
    b.precede(NodeId(1), NodeId(0));
    b.build();
}
