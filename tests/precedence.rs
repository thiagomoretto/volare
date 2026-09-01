//! Ordering within a route: a declared pair runs in order when one vehicle
//! holds both ends, and is unconstrained when two vehicles split it.

use volare::{Construct, Improve, ModelBuilder, NodeId, VehicleId, eval_route, eval_routes, solve};

/// Depot 0, customers 1..=5 spaced along a line, arc cost the line distance.
fn builder(capacity: i64, vehicles: usize) -> ModelBuilder {
    let mut b = ModelBuilder::new(6);
    let cost = b.cost_class(|a: NodeId, b: NodeId| (a.0 as i64 - b.0 as i64).abs() * 10);
    for _ in 0..vehicles {
        b.vehicle(NodeId(0), NodeId(0), cost);
    }
    b.dimension(
        "load",
        |_from, to| if to == NodeId(0) { 0 } else { 1 },
        vec![capacity; vehicles],
    );
    b
}

#[test]
fn out_of_order_is_infeasible() {
    let mut b = builder(5, 1);
    b.precede(NodeId(3), NodeId(1));
    let m = b.build();

    assert_eq!(
        eval_route(&m, &[NodeId(1), NodeId(2), NodeId(3)], VehicleId(0)),
        None,
        "1 is served before its predecessor 3"
    );
    assert!(eval_route(&m, &[NodeId(3), NodeId(2), NodeId(1)], VehicleId(0)).is_some());
}

/// The line that separates this from a pickup-and-delivery pair: nothing
/// forces the two onto one vehicle, and apart they are unordered.
#[test]
fn a_pair_split_across_vehicles_is_unordered() {
    let mut b = builder(5, 2);
    b.precede(NodeId(3), NodeId(1));
    let m = b.build();

    let split = vec![vec![NodeId(1)], vec![NodeId(3)]];
    assert!(eval_routes(&m, &split).is_some());
}

/// Only the pair's own ends matter; an unconstrained node sits anywhere.
#[test]
fn unrelated_nodes_are_untouched() {
    let mut b = builder(5, 1);
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
    assert_eq!(
        eval_route(&m, &[NodeId(2), NodeId(1), NodeId(4)], VehicleId(0)),
        None
    );
}

/// A chain constrains transitively without being declared transitively:
/// 5 before 3 and 3 before 1 leaves one order for the three of them.
#[test]
fn chains_compose() {
    let mut b = builder(5, 1);
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

/// Construction and every improvement move share one feasibility gate, so a
/// solved model cannot come back out of order.
#[test]
fn the_solver_never_returns_an_out_of_order_route() {
    // Cheapest order on a line runs 1..5; requiring 2 ahead of 1 and 4 ahead
    // of 5 rules out both monotone orders.
    let mut b = builder(5, 2);
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

/// A dropped node is out of the routing, so its ordering must not be what
/// keeps it in.
#[test]
fn the_unserved_sink_ignores_ordering() {
    // Only nodes 1 and 2 exist, and serving them costs 40 against a penalty
    // of 1 each, so the sink is where they belong.
    let mut b = ModelBuilder::new(3);
    let cost = b.cost_class(|a: NodeId, b: NodeId| (a.0 as i64 - b.0 as i64).abs() * 10);
    b.vehicle(NodeId(0), NodeId(0), cost);
    b.dimension(
        "load",
        |_from, to| if to == NodeId(0) { 0 } else { 1 },
        vec![5],
    );
    b.precede(NodeId(2), NodeId(1));
    b.allow_drop(NodeId(1), 1);
    b.allow_drop(NodeId(2), 1);
    let m = b.build();

    let sink = m.unserved_vehicle().expect("drops declared");
    assert!(
        eval_route(&m, &[NodeId(1), NodeId(2)], sink).is_some(),
        "the sink holds dropped nodes in any order"
    );

    let sol = solve(&m, Construct::CheapestInsertion, Improve::HillClimb);
    assert_eq!(sol.unserved(&m).len(), 2, "both are cheaper dropped");
}

#[test]
#[should_panic(expected = "vacuous")]
fn ordering_against_a_terminal_is_rejected() {
    let mut b = builder(5, 1);
    b.precede(NodeId(1), NodeId(0));
    b.build();
}
