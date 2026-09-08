//! Time windows with priced idle time, and reading the timetable back.
//!
//! `cumul_bounds` on a `time` dimension is the window: arriving after the
//! close is infeasible, arriving before the open waits. `wait_cost` prices
//! every unit a vehicle stands idle. `walk_route` reads arrival, wait and
//! service start off the finished routes.
//!
//! ```sh
//! cargo run --example time_windows
//! ```

use volare::{Construct, Improve, ModelBuilder, NodeId, solve, walk_route};

fn main() {
    // Minutes from midnight. One unit of distance is one minute of driving.
    let coords: [(f64, f64); 7] = [
        (0.0, 0.0),    // 0 depot
        (30.0, 10.0),  // 1
        (35.0, -20.0), // 2
        (-15.0, 25.0), // 3
        (-40.0, 5.0),  // 4
        (10.0, 45.0),  // 5
        (20.0, -40.0), // 6
    ];
    // Service window per stop, `(open, close)`. The depot's open is when
    // the vans leave.
    let windows = [
        (8 * 60, 24 * 60),
        (10 * 60, 12 * 60), // 1: 10:00 to 12:00
        (8 * 60, 9 * 60),   // 2: 08:00 to 09:00
        (9 * 60, 11 * 60),  // 3
        (14 * 60, 15 * 60), // 4: afternoon only
        (8 * 60, 10 * 60),  // 5
        (11 * 60, 13 * 60), // 6
    ];
    let service = 15;

    let mut b = ModelBuilder::new(coords.len());

    let drive = move |from: NodeId, to: NodeId| {
        let (p, q) = (coords[from.index()], coords[to.index()]);
        (p.0 - q.0).hypot(p.1 - q.1).round() as i64
    };
    let cost = b.cost_class(drive);

    let vans = [
        b.vehicle(NodeId(0), NodeId(0), cost),
        b.vehicle(NodeId(0), NodeId(0), cost),
    ];

    // Transit is the drive plus the service at the stop just left.
    b.dimension(
        "time",
        move |from, to| {
            let served = if from == NodeId(0) { 0 } else { service };
            drive(from, to) + served
        },
        vec![i64::MAX; vans.len()],
    );
    for (n, &(open, close)) in windows.iter().enumerate() {
        b.cumul_bounds("time", NodeId(n as u32), open, close);
    }
    // An idle van costs as much per minute as a driving one.
    for &v in &vans {
        b.wait_cost("time", v, 1);
    }

    let model = b.build();
    let sol = solve(
        &model,
        Construct::CheapestInsertion,
        Improve::Gls { iters: 200 },
    );

    // Read the timetable back. Every stop is served inside its window,
    // never before the open.
    let t = model.dimension_index("time");
    let mut idle = 0;
    for &v in &vans {
        println!("van {}", v.index());
        walk_route(
            &model,
            &sol.routes[v.index()],
            v,
            t,
            |node, arrive, serve| {
                println!(
                    "  stop {:<2} arrive {}  wait {:>3}  serve {}",
                    node.index(),
                    clock(arrive),
                    serve - arrive,
                    clock(serve),
                );
                idle += serve - arrive;
                let (open, close) = windows[node.index()];
                assert!(open <= serve && arrive <= close);
            },
        );
    }
    println!("total cost: {} (of which {idle} idle minutes)", sol.cost);
}

fn clock(minutes: i64) -> String {
    format!("{:02}:{:02}", minutes / 60, minutes % 60)
}
