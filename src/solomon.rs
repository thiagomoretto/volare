//! Solomon/Gehring–Homberger VRPTW reader: VRPLIB layout, DIMACS metric
//! (euclidean truncated to one decimal). Times and costs are scaled by
//! [`SCALE`] into integers, so published costs reproduce exactly.
//!
//! Service time is folded into the time transit: `time(i, j)` is service at
//! `i` plus travel. A window's close is then the latest start of service,
//! the Solomon convention.

use std::sync::Arc;

use crate::cvrplib::SolFile;
use crate::model::{Model, ModelBuilder};
use crate::types::NodeId;

pub const SCALE: i64 = 10;

/// DIMACS VRPTW metric: euclidean, truncated after one decimal.
pub fn euc_2d_dime(a: (f64, f64), b: (f64, f64)) -> i64 {
    let (dx, dy) = (a.0 - b.0, a.1 - b.1);
    ((dx * dx + dy * dy).sqrt() * SCALE as f64).floor() as i64
}

pub struct TwInstance {
    pub name: String,
    pub capacity: i64,
    /// Scaled; applies to every customer, zero at the depot.
    pub service_time: i64,
    pub coords: Vec<(f64, f64)>,
    pub demands: Vec<i64>,
    /// Scaled `(ready, due)` per node; the depot's due is the route deadline.
    pub windows: Vec<(i64, i64)>,
    pub depot: NodeId,
}

impl TwInstance {
    pub fn parse(text: &str) -> TwInstance {
        let mut name = String::new();
        let mut dimension = 0usize;
        let mut capacity = 0i64;
        let mut service_time = 0i64;
        let mut coords = Vec::new();
        let mut demands = Vec::new();
        let mut windows = Vec::new();
        let mut depot = None;
        let mut section = "";

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line == "EOF" {
                continue;
            }
            if let Some((key, value)) = line.split_once(':') {
                let (key, value) = (key.trim(), value.trim());
                match key {
                    "NAME" => name = value.to_string(),
                    "DIMENSION" => dimension = value.parse().unwrap(),
                    "CAPACITY" => capacity = value.parse().unwrap(),
                    "SERVICE_TIME" => service_time = value.parse::<i64>().unwrap() * SCALE,
                    "EDGE_WEIGHT_TYPE" => assert_eq!(value, "EUC_2D", "unsupported metric"),
                    _ => {}
                }
                continue;
            }
            if line.ends_with("_SECTION") {
                section = match line {
                    "NODE_COORD_SECTION" => "coord",
                    "DEMAND_SECTION" => "demand",
                    "TIME_WINDOW_SECTION" => "window",
                    "DEPOT_SECTION" => "depot",
                    _ => "",
                };
                continue;
            }
            let f: Vec<&str> = line.split_whitespace().collect();
            match section {
                "coord" => coords.push((f[1].parse().unwrap(), f[2].parse().unwrap())),
                "demand" => demands.push(f[1].parse().unwrap()),
                "window" => windows.push((
                    f[1].parse::<i64>().unwrap() * SCALE,
                    f[2].parse::<i64>().unwrap() * SCALE,
                )),
                "depot" => {
                    let id: i64 = f[0].parse().unwrap();
                    if id >= 0 && depot.is_none() {
                        depot = Some(NodeId(id as u32 - 1));
                    }
                }
                _ => {}
            }
        }

        assert_eq!(coords.len(), dimension, "{name}: coord count != DIMENSION");
        assert_eq!(
            windows.len(),
            dimension,
            "{name}: window count != DIMENSION"
        );
        let depot = depot.expect("no depot");
        assert_eq!(demands[depot.index()], 0, "{name}: depot has demand");
        TwInstance {
            name,
            capacity,
            service_time,
            coords,
            demands,
            windows,
            depot,
        }
    }
}

/// Cost is distance. Time is distance plus service at the from-node, with
/// the windows as cumul bounds. The depot's due date bounds the return,
/// since the end terminal is a node like any other in the cumul pass.
pub fn vrptw_model(inst: &TwInstance, fleet: usize) -> Model {
    let n = inst.coords.len();
    let mut matrix = vec![0i64; n * n];
    for i in 0..n {
        for j in 0..n {
            matrix[i * n + j] = euc_2d_dime(inst.coords[i], inst.coords[j]);
        }
    }
    let matrix = Arc::new(matrix);
    let demands = Arc::new(inst.demands.clone());
    let mut service = vec![inst.service_time; n];
    service[inst.depot.index()] = 0;
    let service = Arc::new(service);

    let mut b = ModelBuilder::new(n);
    let dist = Arc::clone(&matrix);
    let cost_class = b.cost_class(move |from, to| dist[from.index() * n + to.index()]);
    for _ in 0..fleet {
        b.vehicle(inst.depot, inst.depot, cost_class);
    }
    b.dimension(
        "demand",
        move |_from, to| demands[to.index()],
        vec![inst.capacity; fleet],
    );
    b.dimension(
        "time",
        move |from, to| service[from.index()] + matrix[from.index() * n + to.index()],
        vec![i64::MAX; fleet],
    );
    for i in 0..n {
        let (ready, due) = inst.windows[i];
        b.cumul_bounds("time", NodeId(i as u32), ready, due);
    }
    b.build()
}

/// Same route layout as the CVRP `.sol` files, but the cost line is a real
/// number; it comes back scaled by [`SCALE`].
pub fn parse_sol(text: &str, inst: &TwInstance) -> SolFile {
    let n = inst.coords.len();
    let mut routes = Vec::new();
    let mut cost = None;

    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("Route") {
            let stops = rest.split_once(':').expect("malformed Route line").1;
            routes.push(
                stops
                    .split_whitespace()
                    .map(|t| NodeId(t.parse::<u32>().unwrap()))
                    .collect::<Vec<_>>(),
            );
        } else if let Some(c) = line.strip_prefix("Cost") {
            let c: f64 = c.trim().parse().expect("malformed Cost line");
            cost = Some((c * SCALE as f64).round() as i64);
        }
    }

    let routes_ok = routes
        .iter()
        .flatten()
        .all(|&nd| nd != inst.depot && nd.index() < n);
    assert!(routes_ok, "{}: route entry out of range", inst.name);
    SolFile {
        routes,
        cost: cost.expect("no Cost line"),
    }
}
