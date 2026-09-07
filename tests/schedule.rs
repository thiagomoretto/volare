//! A schedule reads every dimension off a finished route.

use volare::{ModelBuilder, NodeId, Schedule, VehicleId};

#[test]
fn schedule_reports_arrival_wait_and_load() {
    // Depot 0, customers 1 and 2, ten time units per arc, demand 4 each.
    let mut b = ModelBuilder::new(3);
    let cost = b.cost_class(|_, _| 10);
    b.vehicle(NodeId(0), NodeId(0), cost);
    b.dimension("time", |_, _| 10, vec![i64::MAX]);
    b.dimension(
        "load",
        |_, to| if to.index() == 0 { 0 } else { 4 },
        vec![10],
    );
    b.cumul_bounds("time", NodeId(1), 15, 100);
    b.allow_drop(NodeId(2), 999);
    let m = b.build();
    let sink = m.unserved_vehicle().unwrap();
    let mut routes = vec![Vec::new(); sink.index() + 1];
    routes[0] = vec![NodeId(1)];
    routes[sink.index()] = vec![NodeId(2)];

    let s = Schedule::of(&m, &routes);
    let t = Schedule::dimension(&m, "time");
    let load = Schedule::dimension(&m, "load");

    let r = s.route(VehicleId(0));
    let nodes: Vec<_> = r.iter().map(|s| s.node).collect();
    assert_eq!(nodes, [NodeId(0), NodeId(1), NodeId(0)]);
    assert_eq!(r[0].arrive[t], 0);
    assert_eq!(r[1].arrive[t], 10);
    assert_eq!(r[1].cumul[t], 15);
    assert_eq!(r[1].wait(t), 5);
    assert_eq!(r[2].arrive[t], 25, "the wait pushes the return");
    assert_eq!(r[1].cumul[load], 4);

    assert_eq!(s.stop(NodeId(1)).map(|s| s.cumul[t]), Some(15));
    assert_eq!(s.stop(NodeId(2)), None, "dropped nodes have no stop");
    assert_eq!(s.stop(NodeId(0)), None, "terminals are per vehicle");
    assert!(s.route(sink).is_empty());
}
