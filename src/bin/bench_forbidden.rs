//! Synthesized vehicle-node exclusion benchmark: take the X instances, mark
//! a deterministic subset of (vehicle, customer) pairs forbidden, and measure
//! what the constraint costs against the same instance unconstrained.
//!
//!   cargo run --release --bin bench_forbidden           # all X instances
//!   cargo run --release --bin bench_forbidden X-n101    # name filter
//!
//! The rule is deterministic (no RNG in this repo): customers with
//! `id % 7 == 0` (~14%) are forbidden on half the fleet. The fleet is free
//! (`n-1` vehicles, like `bench`): greedy construction cannot pack a tight
//! `k` even unconstrained, and a forbidden customer still has ~n/2 vehicles
//! to choose from, so the delta measures the pure routing cost of the
//! constraint.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use volare::cvrplib::{Instance, cvrp_model, cvrp_model_with};
use volare::eval::visits_all_nodes;
use volare::solver::{Construct, Improve, solve};
use volare::types::{NodeId, VehicleId};

fn main() {
    let filter = std::env::args().nth(1);
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));

    let mut instances: Vec<PathBuf> = fs::read_dir(root.join("instances"))
        .expect("instances/ missing")
        .filter_map(|e| {
            let p = e.unwrap().path();
            (p.extension()? == "vrp").then_some(p)
        })
        .filter(|p| match &filter {
            Some(f) => p.file_name().unwrap().to_string_lossy().contains(f),
            None => true,
        })
        .collect();
    instances.sort_by_key(|p| instance_size(p));

    println!(
        "{:<14} {:>5} {:>5} {:>9} {:>9} {:>7} {:>7}",
        "instance", "n", "pairs", "open", "forbidden", "delta%", "ms"
    );

    for vrp in &instances {
        let inst = Instance::parse(&fs::read_to_string(vrp).unwrap());
        let n = inst.coords.len();

        let started = Instant::now();
        let open = solve(
            &cvrp_model(&inst, n - 1),
            Construct::CheapestInsertion,
            Improve::HillClimb,
        )
        .cost;

        let mut pairs = 0;
        let model = cvrp_model_with(&inst, n - 1, |b| {
            for c in 0..n as u32 {
                for v in 0..(n - 1) as u32 {
                    if c % 7 == 0 && (v + c) % 2 == 0 && NodeId(c) != inst.depot {
                        b.forbid(VehicleId(v), NodeId(c));
                        pairs += 1;
                    }
                }
            }
        });
        let sol = solve(&model, Construct::CheapestInsertion, Improve::HillClimb);
        let ms = started.elapsed().as_secs_f64() * 1000.0;

        // The whole point of the run: the constraint held.
        assert!(visits_all_nodes(&model, &sol.routes));
        for (v, route) in sol.routes.iter().enumerate() {
            let veh = model.vehicle(VehicleId(v as u32));
            assert!(
                route.iter().all(|&n| !veh.forbids(n)),
                "{}: vehicle {v} visits a forbidden node",
                inst.name
            );
        }

        let delta = 100.0 * (sol.cost - open) as f64 / open as f64;
        println!(
            "{:<14} {:>5} {:>5} {:>9} {:>9} {:>7.2} {:>7.0}",
            inst.name, n, pairs, open, sol.cost, delta, ms
        );
    }
}

/// Pull <size> out of an X-n<size>-k<vehicles> filename, to sort the report.
fn instance_size(p: &Path) -> u32 {
    p.file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.split('-').nth(1))
        .and_then(|s| s.trim_start_matches('n').parse().ok())
        .unwrap_or(0)
}
