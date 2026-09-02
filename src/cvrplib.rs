//! CVRPLIB (TSPLIB-flavoured) reader for the plain CVRP sets.
//!
//! Coordinates live here and never enter the `Model`. The model internalizes
//! no metric; geometry only enters as build-time closures.

use crate::model::{Model, ModelBuilder};
use crate::types::NodeId;

pub struct Instance {
    pub name: String,
    pub capacity: i64,
    /// Indexed by `NodeId`; file node `i` is index `i - 1`.
    pub coords: Vec<(f64, f64)>,
    pub demands: Vec<i64>,
    pub depot: NodeId,
}

/// TSPLIB `EUC_2D`: round to nearest integer *per arc*, before summing.
/// Getting this wrong makes every gap measurement meaningless.
pub fn euc_2d(a: (f64, f64), b: (f64, f64)) -> i64 {
    let (dx, dy) = (a.0 - b.0, a.1 - b.1);
    (dx * dx + dy * dy).sqrt().round() as i64
}

impl Instance {
    pub fn parse(text: &str) -> Instance {
        let mut name = String::new();
        let mut dimension = 0usize;
        let mut capacity = 0i64;
        let mut coords = Vec::new();
        let mut demands = Vec::new();
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
                    "EDGE_WEIGHT_TYPE" => assert_eq!(value, "EUC_2D", "unsupported metric"),
                    _ => {}
                }
                continue;
            }
            if line.ends_with("_SECTION") {
                section = match line {
                    "NODE_COORD_SECTION" => "coord",
                    "DEMAND_SECTION" => "demand",
                    "DEPOT_SECTION" => "depot",
                    _ => "",
                };
                continue;
            }
            let f: Vec<&str> = line.split_whitespace().collect();
            match section {
                "coord" => coords.push((f[1].parse().unwrap(), f[2].parse().unwrap())),
                "demand" => demands.push(f[1].parse().unwrap()),
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
            demands.len(),
            dimension,
            "{name}: demand count != DIMENSION"
        );
        let depot = depot.expect("no depot");
        assert_eq!(demands[depot.index()], 0, "{name}: depot has demand");
        Instance {
            name,
            capacity,
            coords,
            demands,
            depot,
        }
    }

    pub fn dist(&self, a: NodeId, b: NodeId) -> i64 {
        euc_2d(self.coords[a.index()], self.coords[b.index()])
    }
}

/// The distances are baked into a matrix here and enter the model only as a
/// closure. The model holds no coordinates and knows no metric, so swapping
/// in an OSRM `/table` response is the same shape.
pub fn cvrp_model(inst: &Instance, fleet: usize) -> Model {
    cvrp_model_with(inst, fleet, |_| {})
}

/// `cvrp_model`, with a hook that runs after the vehicles exist and before
/// the model is built — where per-vehicle tweaks like `forbid` belong.
pub fn cvrp_model_with(
    inst: &Instance,
    fleet: usize,
    configure: impl FnOnce(&mut ModelBuilder),
) -> Model {
    let n = inst.coords.len();
    let mut matrix = vec![0i64; n * n];
    for i in 0..n {
        for j in 0..n {
            matrix[i * n + j] = euc_2d(inst.coords[i], inst.coords[j]);
        }
    }
    let demands = inst.demands.clone();

    let mut b = ModelBuilder::new(n);
    let cost_class = b.cost_class(move |from, to| matrix[from.index() * n + to.index()]);
    for _ in 0..fleet {
        b.vehicle(inst.depot, inst.depot, cost_class);
    }
    configure(&mut b);
    b.dimension(
        "demand",
        move |_from, to| demands[to.index()],
        vec![inst.capacity; fleet],
    );
    b.build()
}

pub struct SolFile {
    pub routes: Vec<Vec<NodeId>>,
    /// The published cost, our reference value for gaps.
    pub cost: i64,
}

/// Parse a `.sol`. Route entries are customer numbers, which are node ids in
/// our 0-based space (file node `i` maps to `NodeId(i - 1)`, and the depot is
/// never listed). The assertions below turn an indexing mistake into a loud
/// failure here, rather than a mysterious cost mismatch further downstream.
pub fn parse_sol(text: &str, inst: &Instance) -> SolFile {
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
        } else if let Some(rest) = line.strip_prefix("Cost") {
            // One shipped .sol writes its cost as "20215.0".
            cost = Some(rest.trim().parse::<f64>().unwrap() as i64);
        }
    }

    if routes.is_empty() {
        // BKS-only reference (e.g. XL instances publish costs, not routes).
        return SolFile {
            routes,
            cost: cost.expect(".sol has neither routes nor a Cost line"),
        };
    }

    let mut seen = vec![false; n];
    for r in &routes {
        for &node in r {
            assert!(
                node != inst.depot,
                "solution routes the depot as a customer"
            );
            assert!(node.index() < n, "solution node {node:?} out of range");
            assert!(!seen[node.index()], "solution visits {node:?} twice");
            seen[node.index()] = true;
        }
    }
    let missing = (0..n)
        .filter(|&i| !seen[i] && NodeId(i as u32) != inst.depot)
        .count();
    assert_eq!(missing, 0, "solution misses {missing} customers");

    SolFile {
        routes,
        cost: cost.expect("no Cost line"),
    }
}
