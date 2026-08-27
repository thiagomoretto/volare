//! Cumul windows: late arrival is infeasible, early arrival waits, and a
//! window never blocks dropping the node.

use volare::{ModelBuilder, NodeId, VehicleId, eval_route};

/// Depot 0, customers 1 and 2, every arc 10 time units.
fn builder() -> ModelBuilder {
    let mut b = ModelBuilder::new(3);
    let cost = b.cost_class(|_, _| 10);
    b.vehicle(NodeId(0), NodeId(0), cost);
    b.dimension("time", |_, _| 10, vec![i64::MAX]);
    b
}

#[test]
fn late_arrival_is_infeasible() {
    let mut b = builder();
    b.cumul_bounds("time", NodeId(1), 0, 5);
    let m = b.build();
    assert_eq!(
        eval_route(&m, &[NodeId(1)], VehicleId(0)),
        None,
        "arrive 10 > close 5"
    );
}

#[test]
fn early_arrival_waits() {
    let mut b = builder();
    // Arrive at 1 at t=10, wait until 15; arrive at 2 at t=25.
    b.cumul_bounds("time", NodeId(1), 15, 100);
    b.cumul_bounds("time", NodeId(2), 0, 25);
    let m = b.build();
    assert!(eval_route(&m, &[NodeId(1), NodeId(2)], VehicleId(0)).is_some());

    let mut b = builder();
    b.cumul_bounds("time", NodeId(1), 15, 100);
    b.cumul_bounds("time", NodeId(2), 0, 24);
    let m = b.build();
    assert_eq!(
        eval_route(&m, &[NodeId(1), NodeId(2)], VehicleId(0)),
        None,
        "the wait at 1 pushes arrival at 2 past its close"
    );
}

#[test]
fn window_never_blocks_a_drop() {
    let mut b = builder();
    b.cumul_bounds("time", NodeId(1), 0, 5);
    b.allow_drop(NodeId(1), 999);
    let m = b.build();
    let sink = m.unserved_vehicle().unwrap();
    assert_eq!(eval_route(&m, &[NodeId(1)], sink), Some(999));
}
