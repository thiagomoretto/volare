//! Run the solver over instances/ and report the gap against the reference
//! cost shipped in each .sol.
//!
//!   cargo run --release --bin bench                  # run and compare to baseline.csv
//!   cargo run --release --bin bench -- --write-baseline
//!   cargo run --release --bin bench -- --log         # search log to stderr
//!   cargo run --release --bin bench -- X-n101        # name filter
//!   cargo run --release --bin bench -- --gls=30      # guided local search
//!   cargo run --release --bin bench -- --scenario=forbid  # constraint vs. open delta
//!
//! A `--scenario` picks a model variant instead of the plain CVRP. Scenarios
//! are a match arm over the builder, not a binary each: a new constraint is
//! ~10 lines here, and the report is the constrained cost against the same
//! instance solved open.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use volare::Construct;
use volare::cvrplib::{Instance, cvrp_model, cvrp_model_with, parse_sol};
use volare::eval::{eval_routes, visits_all_nodes};
use volare::solver::{
    Improve, SearchEvent, first_solution_with, guided_local_search_with, local_search_with,
    search_log, solve_with,
};
use volare::types::{NodeId, VehicleId};

/// A gap that worsens by more than this against baseline.csv fails the run.
const REGRESSION_TOLERANCE: f64 = 2.0;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let write_baseline = args.iter().any(|a| a == "--write-baseline");
    let log_search = args.iter().any(|a| a == "--log");
    let filter = args.iter().find(|a| !a.starts_with("--")).cloned();
    // `--gls=30`, not `--gls 30`: a bare number would be taken as the filter.
    let gls: Option<usize> = args
        .iter()
        .find_map(|a| a.strip_prefix("--gls=")?.parse().ok());
    let scenario = args.iter().find_map(|a| a.strip_prefix("--scenario="));

    assert!(
        scenario.is_none_or(|s| s == "forbid"),
        "unknown --scenario (have: forbid)"
    );
    assert!(
        !(write_baseline && scenario.is_some()),
        "baseline.csv guards the unconstrained hill climb — a scenario run would erase it"
    );
    assert!(
        !(write_baseline && gls.is_some()),
        "baseline.csv guards the hill climb — writing it from a GLS run would erase that"
    );

    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let baseline_path = root.join("baseline.csv");
    let baseline = read_baseline(&baseline_path);

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

    match scenario {
        None => println!(
            "{:<14} {:>5} {:>9} {:>7} {:>9} {:>7} {:>7}",
            "instance", "n", "ref", "ctor%", "ours", "gap%", "ms"
        ),
        Some(_) => println!(
            "{:<14} {:>5} {:>5} {:>9} {:>9} {:>7} {:>7}",
            "instance", "n", "pairs", "open", "forbidden", "delta%", "ms"
        ),
    }

    let mut rows = Vec::new();
    let mut regressions = Vec::new();
    let mut gap_sum = 0.0;

    for vrp in &instances {
        let inst = Instance::parse(&fs::read_to_string(vrp).unwrap());
        let n = inst.coords.len();
        // One vehicle per customer is always enough; the X set has a free
        // fleet size and unused vehicles cost nothing. With a free fleet a
        // forbidden customer still has ~n/2 vehicles to choose from, so the
        // scenario delta measures the pure routing cost of the constraint.
        let fleet = n - 1;

        let started = Instant::now();
        let mut log: Box<dyn FnMut(SearchEvent)> = if log_search {
            Box::new(search_log())
        } else {
            Box::new(|_| {})
        };

        match scenario {
            None => {
                let reference = parse_sol(
                    &fs::read_to_string(vrp.with_extension("sol")).unwrap(),
                    &inst,
                )
                .cost;
                let model = cvrp_model(&inst, fleet);

                let construct = Construct::CheapestInsertion;
                let mut sol = first_solution_with(&model, construct, &mut log);
                let ctor = eval_routes(&model, &sol).expect("infeasible construction");
                match gls {
                    Some(iters) => guided_local_search_with(&model, &mut sol, iters, &mut log),
                    None => local_search_with(&model, &mut sol, &mut log),
                }
                let ms = started.elapsed().as_secs_f64() * 1000.0;
                let ctor_gap = 100.0 * (ctor - reference) as f64 / reference as f64;

                assert!(
                    visits_all_nodes(&model, &sol),
                    "{}: not all nodes routed",
                    inst.name
                );
                let cost =
                    eval_routes(&model, &sol).expect("solver returned an infeasible solution");
                let gap = 100.0 * (cost - reference) as f64 / reference as f64;
                gap_sum += gap;

                println!(
                    "{:<14} {:>5} {:>9} {:>7.2} {:>9} {:>7.2} {:>7.0}",
                    inst.name, n, reference, ctor_gap, cost, gap, ms
                );

                if let Some(&was) = baseline.get(&inst.name)
                    && gap > was + REGRESSION_TOLERANCE
                {
                    regressions.push(format!("{}: {was:.2}% -> {gap:.2}%", inst.name));
                }
                rows.push((inst.name.clone(), gap));
            }
            // Deterministic (no RNG in this repo): customers with `id % 7 == 0`
            // (~14%) are forbidden on half the fleet.
            Some(_) => {
                let fresh = || Construct::CheapestInsertion;
                let improve = || match gls {
                    Some(iters) => Improve::Gls { iters },
                    None => Improve::HillClimb,
                };
                let open = solve_with(&cvrp_model(&inst, fleet), fresh(), improve(), &mut log).cost;

                let mut pairs = 0;
                let model = cvrp_model_with(&inst, fleet, |b| {
                    for c in 0..n as u32 {
                        for v in 0..fleet as u32 {
                            if c % 7 == 0 && (v + c) % 2 == 0 && NodeId(c) != inst.depot {
                                b.forbid(VehicleId(v), NodeId(c));
                                pairs += 1;
                            }
                        }
                    }
                });
                let sol = solve_with(&model, fresh(), improve(), &mut log);
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
    }

    if scenario.is_some() {
        return;
    }

    println!(
        "\nmean gap {:.2}% over {} instances",
        gap_sum / rows.len() as f64,
        rows.len()
    );

    if write_baseline {
        let mut out = String::from("instance,gap\n");
        for (name, gap) in &rows {
            out.push_str(&format!("{name},{gap:.4}\n"));
        }
        fs::write(&baseline_path, out).unwrap();
        println!("wrote {}", baseline_path.display());
    } else if !regressions.is_empty() {
        eprintln!("\nregressed beyond {REGRESSION_TOLERANCE}%:");
        for r in &regressions {
            eprintln!("  {r}");
        }
        std::process::exit(1);
    }
}

fn read_baseline(path: &Path) -> HashMap<String, f64> {
    fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .skip(1)
        .filter_map(|l| {
            let (name, gap) = l.split_once(',')?;
            Some((name.to_string(), gap.parse().ok()?))
        })
        .collect()
}

/// Pull <size> out of an X-n<size>-k<vehicles> filename, to sort the report.
fn instance_size(p: &Path) -> u32 {
    p.file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.split('-').nth(1))
        .and_then(|s| s.trim_start_matches('n').parse().ok())
        .unwrap_or(0)
}
