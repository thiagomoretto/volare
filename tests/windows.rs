//! Cumul windows: late arrival is infeasible, early arrival waits, and a
//! window never blocks dropping the node. A soft close prices lateness up to
//! the hard one.

use volare::{
    ModelBuilder, NodeId, VehicleId, Violation, eval_route, eval_route_split, eval_routes,
    violations,
};

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

#[test]
fn late_past_the_soft_close_pays_per_unit() {
    let mut b = builder();
    b.soft_upper_bound("time", NodeId(1), 4, 3);
    let m = b.build();
    // Arrive at 10, six late, three each: 18 on top of the 20 of arcs.
    assert_eq!(
        eval_route_split(&m, &[NodeId(1)], VehicleId(0)),
        Some((20, 18))
    );
    assert_eq!(eval_route(&m, &[NodeId(1)], VehicleId(0)), Some(38));
}

#[test]
fn hard_close_still_binds_above_the_soft_one() {
    let mut b = builder();
    b.cumul_bounds("time", NodeId(1), 0, 8);
    b.soft_upper_bound("time", NodeId(1), 4, 3);
    let m = b.build();
    assert_eq!(eval_route(&m, &[NodeId(1)], VehicleId(0)), None);
}

#[test]
fn lateness_propagates_down_the_route() {
    let mut b = builder();
    b.soft_upper_bound("time", NodeId(1), 5, 1);
    b.soft_upper_bound("time", NodeId(2), 15, 1);
    let m = b.build();
    // Arrive 10 at node 1 (5 late), leave late, arrive 20 at node 2 (5 late).
    let sol = vec![vec![NodeId(1), NodeId(2)]];
    assert_eq!(eval_routes(&m, &sol), Some(30 + 10));
    let late = |node| Violation {
        dimension: 0,
        vehicle: VehicleId(0),
        node: Some(node),
        excess: 5,
        penalty: 5,
    };
    assert_eq!(violations(&m, &sol), vec![late(NodeId(1)), late(NodeId(2))]);
}

#[test]
fn soft_close_never_charges_a_drop() {
    let mut b = builder();
    b.soft_upper_bound("time", NodeId(1), 0, 1000);
    b.allow_drop(NodeId(1), 999);
    let m = b.build();
    let sink = m.unserved_vehicle().unwrap();
    assert_eq!(eval_route(&m, &[NodeId(1)], sink), Some(999));
    assert!(violations(&m, &vec![vec![], vec![NodeId(1)]]).is_empty());
}

#[test]
fn wait_is_priced_per_unit() {
    let mut b = builder();
    // Arrive at 1 at t=10, wait until 15: five units at 3 each.
    b.cumul_bounds("time", NodeId(1), 15, 100);
    b.wait_cost("time", VehicleId(0), 3);
    let m = b.build();
    assert_eq!(
        eval_route_split(&m, &[NodeId(1)], VehicleId(0)),
        Some((20, 15))
    );
    assert!(
        violations(&m, &vec![vec![NodeId(1)]]).is_empty(),
        "a wait is not a violation"
    );
}

#[test]
fn no_wait_no_charge() {
    let mut b = builder();
    b.wait_cost("time", VehicleId(0), 3);
    let m = b.build();
    assert_eq!(eval_route(&m, &[NodeId(1)], VehicleId(0)), Some(20));
}
