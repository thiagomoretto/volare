//! Run the solver over instances/ and report the gap against the reference
//! cost shipped in each .sol.
//!
//!   cargo run --release --bin bench                  # run and compare to baseline.csv
//!   cargo run --release --bin bench -- --write-baseline
//!   cargo run --release --bin bench -- --log         # search log to stderr
//!   cargo run --release --bin bench -- X-n101        # name filter
//!   cargo run --release --bin bench -- --gls=30      # guided local search
//!   cargo run --release --bin bench -- --restarts=4 --gls=300  # multi-start
//!
//! `--restarts=N` solves N times from different randomized constructions and
//! keeps the cheapest; `--rcl=K` is how wide each greedy draw is. Give every
//! start the whole round budget. Splitting one budget across restarts loses,
//! because a restart throws away the GLS penalties that aim the next descent:
//! 8 starts of 40 rounds reach 6.01% where 1 start of 300 rounds reaches
//! 5.52%. Four starts of 300 rounds reach 4.44%. One start stays the plain
//! deterministic greedy, so baseline.csv keeps its meaning.
//!
//!   cargo run --release --bin bench -- --scenario=forbid  # constraint vs. open delta
//!
//! A `--scenario` picks a model variant instead of the plain CVRP: a new
//! constraint is a `*_model` transform plus a `check_*` arm, not a binary
//! each, and the report is the constrained cost against the same instance
//! solved open.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use volare::Construct;
use volare::cvrplib::{Instance, cvrp_model, cvrp_model_with, parse_sol};
use volare::eval::{Routes, eval_route, eval_routes, visits_all_nodes};
use volare::model::{Model, ModelBuilder};
use volare::solver::{
    Improve, SearchEvent, Solution, first_solution_with, guided_local_search_with,
    local_search_with, search_log, solve_with,
};
use volare::types::{Cost, NodeId, VehicleId};

/// A gap that worsens by more than this against baseline.csv fails the run.
const REGRESSION_TOLERANCE: f64 = 2.0;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let write_baseline = args.iter().any(|a| a == "--write-baseline");
    let log_search = args.iter().any(|a| a == "--log");
    let filter = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .map(String::as_str);
    // `--gls=30`, not `--gls 30`: a bare number would be taken as the filter.
    let gls: Option<usize> = args
        .iter()
        .find_map(|a| a.strip_prefix("--gls=")?.parse().ok());
    let scenario = args.iter().find_map(|a| a.strip_prefix("--scenario="));
    // `--restarts=8` runs 8 randomized constructions and keeps the cheapest
    // solved result; `--rcl=K` is how wide each greedy draw is.
    let restarts: usize = args
        .iter()
        .find_map(|a| a.strip_prefix("--restarts=")?.parse().ok())
        .unwrap_or(1);
    let rcl: usize = args
        .iter()
        .find_map(|a| a.strip_prefix("--rcl=")?.parse().ok())
        .unwrap_or(3);

    assert!(
        scenario.is_none_or(|s| s == "forbid" || s == "drop" || s == "tw"),
        "unknown --scenario (have: forbid, drop, tw)"
    );
    assert!(
        !(write_baseline && scenario.is_some()),
        "baseline.csv guards the unconstrained hill climb — a scenario run would erase it"
    );
    assert!(
        !(write_baseline && gls.is_some()),
        "baseline.csv guards the hill climb — writing it from a GLS run would erase that"
    );
    assert!(restarts >= 1, "--restarts must be at least 1");
    assert!(
        !(write_baseline && restarts > 1),
        "baseline.csv guards the single-start hill climb — a multi-start run would erase that"
    );

    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let instances = discover_instances(root, filter);

    match scenario {
        None => bench_reference(
            root,
            &instances,
            write_baseline,
            log_search,
            gls,
            restarts,
            rcl,
        ),
        Some(name) => bench_scenario(name, &instances, log_search, gls),
    }
}

fn bench_reference(
    root: &Path,
    instances: &[PathBuf],
    write_baseline: bool,
    log_search: bool,
    gls: Option<usize>,
    restarts: usize,
    rcl: usize,
) {
    let baseline_path = root.join("baseline.csv");
    let baseline = read_baseline(&baseline_path);

    if restarts > 1 {
        println!("{restarts} restarts, RCL {rcl}, cheapest of each; ms covers all starts");
    }
    println!(
        "{:<14} {:>5} {:>9} {:>7} {:>9} {:>7} {:>7}",
        "instance", "n", "ref", "ctor%", "ours", "gap%", "ms"
    );

    let mut rows = Vec::new();
    let mut regressions = Vec::new();
    let mut gap_sum = 0.0;

    for vrp in instances {
        let inst = Instance::parse(&fs::read_to_string(vrp).unwrap());
        let reference = parse_sol(
            &fs::read_to_string(vrp.with_extension("sol")).unwrap(),
            &inst,
        )
        .cost;
        let model = cvrp_model(&inst, free_fleet(&inst));

        let started = Instant::now();
        let mut log = logger(log_search);
        // One start stays the deterministic greedy, so a default run is the
        // same number baseline.csv holds.
        let mut winner: Option<(Cost, Cost, Routes)> = None;
        for start in 0..restarts {
            let construct = match restarts {
                1 => Construct::CheapestInsertion,
                _ => Construct::GreedyRandomized {
                    seed: start as u64,
                    k: rcl,
                },
            };
            let mut sol = first_solution_with(&model, construct, &mut log);
            let ctor = eval_routes(&model, &sol).expect("infeasible construction");
            match gls {
                Some(iters) => guided_local_search_with(&model, &mut sol, iters, &mut log),
                None => local_search_with(&model, &mut sol, &mut log),
            }
            let cost = eval_routes(&model, &sol).expect("solver returned an infeasible solution");
            if winner.as_ref().is_none_or(|w| cost < w.0) {
                winner = Some((cost, ctor, sol));
            }
        }
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        // `ctor` is the winning start's construction, not the best of them:
        // the column has to explain the solution on the same row.
        let (cost, ctor, sol) = winner.expect("restarts >= 1");

        assert!(
            visits_all_nodes(&model, &sol),
            "{}: not all nodes routed",
            inst.name
        );
        let gap = 100.0 * (cost - reference) as f64 / reference as f64;
        gap_sum += gap;

        let ctor_gap = 100.0 * (ctor - reference) as f64 / reference as f64;
        println!(
            "{:<14} {:>5} {:>9} {:>7.2} {:>9} {:>7.2} {:>7.0}",
            inst.name,
            inst.coords.len(),
            reference,
            ctor_gap,
            cost,
            gap,
            ms
        );

        if let Some(&was) = baseline.get(&inst.name)
            && gap > was + REGRESSION_TOLERANCE
        {
            regressions.push(format!("{}: {was:.2}% -> {gap:.2}%", inst.name));
        }
        rows.push((inst.name.clone(), gap));
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

fn bench_scenario(scenario: &str, instances: &[PathBuf], log_search: bool, gls: Option<usize>) {
    println!(
        "{:<14} {:>5} {:>5} {:>9} {:>9} {:>7} {:>7}",
        "instance", "n", "note", "open", "scenario", "delta%", "ms"
    );

    for vrp in instances {
        let inst = Instance::parse(&fs::read_to_string(vrp).unwrap());
        let fleet = free_fleet(&inst);
        let mut log = logger(log_search);

        let started = Instant::now();
        let open = solve(&cvrp_model(&inst, fleet), gls, &mut log).cost;
        let mut note = String::new();
        let model = cvrp_model_with(&inst, fleet, |b| match scenario {
            "forbid" => forbid_model(&inst, b, &mut note),
            "drop" => drop_model(&inst, b, &mut note),
            "tw" => tw_model(&inst, b, &mut note),
            _ => unreachable!("gated in main"),
        });
        let sol = solve(&model, gls, &mut log);
        let ms = started.elapsed().as_secs_f64() * 1000.0;

        // The whole point of the run: the constraint held.
        assert!(visits_all_nodes(&model, &sol.routes));
        match scenario {
            "forbid" => check_forbid(&model, &sol, &inst.name),
            "drop" => {
                check_drop(&model, &sol, &inst);
                note = format!("{note}->{}", sol.unserved(&model).len());
            }
            "tw" => {
                let waits = check_tw(&model, &sol, &inst.name);
                note = format!("{note}~w{waits}");
            }
            _ => unreachable!("gated in main"),
        }

        let delta = 100.0 * (sol.cost - open) as f64 / open as f64;
        println!(
            "{:<14} {:>5} {:>5} {:>9} {:>9} {:>7.2} {:>7.0}",
            inst.name,
            inst.coords.len(),
            note,
            open,
            sol.cost,
            delta,
            ms
        );
    }
}

/// Customers with `id % 7 == 0` (~14%) are forbidden on half the fleet —
/// deterministic, no RNG in this repo. With a free fleet a forbidden customer
/// still has ~n/2 vehicles to choose from, so the delta measures the pure
/// routing cost of the constraint.
fn forbid_model(inst: &Instance, b: &mut ModelBuilder, note: &mut String) {
    let n = inst.coords.len() as u32;
    let mut pairs = 0;
    for c in 0..n {
        for v in 0..free_fleet(inst) as u32 {
            if c % 7 == 0 && (v + c) % 2 == 0 && NodeId(c) != inst.depot {
                b.forbid(VehicleId(v), NodeId(c));
                pairs += 1;
            }
        }
    }
    *note = pairs.to_string();
}

/// ~9% of customers may be dropped for a detour-sized penalty; the note
/// becomes `declared->dropped` once the solution exists.
fn drop_model(inst: &Instance, b: &mut ModelBuilder, note: &mut String) {
    let n = inst.coords.len() as u32;
    let mut declared = 0;
    for c in 0..n {
        let node = NodeId(c);
        if c % 11 == 0 && node != inst.depot {
            b.allow_drop(node, drop_penalty(inst, node));
            declared += 1;
        }
    }
    *note = declared.to_string();
}

/// Time is distance, no service time. Every fifth customer gets a window
/// on its direct arrival time `t = dist(depot, c)`: open at `3t/2`, above
/// any possible arrival (triangle inequality), so early visits wait; close
/// at `3t + 10`, so a single-customer route always fits. With a free fleet
/// the scenario is feasible by construction.
fn tw_model(inst: &Instance, b: &mut ModelBuilder, note: &mut String) {
    let n = inst.coords.len();
    let coords = std::sync::Arc::new(inst.coords.clone());
    b.dimension(
        "time",
        move |from, to| volare::cvrplib::euc_2d(coords[from.index()], coords[to.index()]),
        vec![i64::MAX; free_fleet(inst)],
    );
    let mut windowed = 0;
    for c in 0..n as u32 {
        let node = NodeId(c);
        if c % 5 == 0 && node != inst.depot {
            let t = inst.dist(inst.depot, node);
            b.cumul_bounds("time", node, 3 * t / 2, 3 * t + 10);
            windowed += 1;
        }
    }
    *note = windowed.to_string();
}

/// Re-walk every route on the time dimension and assert no late arrival.
/// Returns the number of waits, proof that the clamp fires.
fn check_tw(model: &Model, sol: &Solution, name: &str) -> usize {
    let d = model
        .dimensions()
        .iter()
        .find(|d| d.name == "time")
        .expect("tw scenario has a time dimension");
    let mut waits = 0;
    for (v, route) in sol.routes.iter().enumerate() {
        let veh = model.vehicle(VehicleId(v as u32));
        let mut cumul = d.start_cumul.max(d.lower_bound[veh.start.index()]);
        let mut prev = veh.start;
        for &node in route.iter().chain(std::iter::once(&veh.end)) {
            let arrive = cumul + model.eval(d.transit, prev, node);
            assert!(
                arrive <= d.upper_bound[node.index()],
                "{name}: vehicle {v} arrives at node {} after its window closes",
                node.index()
            );
            if arrive < d.lower_bound[node.index()] {
                waits += 1;
            }
            cumul = arrive.max(d.lower_bound[node.index()]);
            prev = node;
        }
    }
    waits
}

/// Shared by `check_drop`, so the scenario and its check never drift apart.
fn drop_penalty(inst: &Instance, n: NodeId) -> i64 {
    inst.dist(inst.depot, n) / 16
}

/// The sink's route cost must equal the dropped nodes' declared penalties.
fn check_drop(model: &Model, sol: &Solution, inst: &Instance) {
    let uv = model.unserved_vehicle().expect("drop scenario has a sink");
    let sink_cost = eval_route(model, &sol.routes[uv.index()], uv).unwrap();
    let dropped_penalties: i64 = sol
        .unserved(model)
        .iter()
        .map(|&n| drop_penalty(inst, n))
        .sum();
    assert_eq!(
        sink_cost, dropped_penalties,
        "{}: sink cost != dropped nodes' penalties",
        inst.name
    );
    assert_eq!(Some(sol.cost), eval_routes(model, &sol.routes));
}

fn check_forbid(model: &Model, sol: &Solution, name: &str) {
    for (v, route) in sol.routes.iter().enumerate() {
        let veh = model.vehicle(VehicleId(v as u32));
        assert!(
            route.iter().all(|&n| !veh.forbids(n)),
            "{name}: vehicle {v} visits a forbidden node"
        );
    }
}

/// One vehicle per customer is always enough; the X set has a free fleet
/// size and unused vehicles cost nothing.
fn free_fleet(inst: &Instance) -> usize {
    inst.coords.len() - 1
}

fn solve(m: &Model, gls: Option<usize>, log: &mut dyn FnMut(SearchEvent)) -> Solution {
    let improve = match gls {
        Some(iters) => Improve::Gls { iters },
        None => Improve::HillClimb,
    };
    solve_with(m, Construct::CheapestInsertion, improve, log)
}

fn logger(on: bool) -> Box<dyn FnMut(SearchEvent)> {
    if on {
        Box::new(search_log())
    } else {
        Box::new(|_| {})
    }
}

fn discover_instances(root: &Path, filter: Option<&str>) -> Vec<PathBuf> {
    let mut instances: Vec<PathBuf> = fs::read_dir(root.join("instances"))
        .expect("instances/ missing")
        .filter_map(|e| {
            let p = e.unwrap().path();
            (p.extension()? == "vrp").then_some(p)
        })
        .filter(|p| match filter {
            Some(f) => p.file_name().unwrap().to_string_lossy().contains(f),
            None => true,
        })
        .collect();
    instances.sort_by_key(|p| instance_size(p));
    instances
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

/// Pull `size` out of an `X-n<size>-k<vehicles>` filename, to sort the report.
/// Names without the pattern (the Belgium set) are the big ones — sort them
/// last so a full run fails fast on the small instances first.
fn instance_size(p: &Path) -> u32 {
    p.file_stem()
        .and_then(|s| s.to_str())
        .and_then(|s| s.split('-').nth(1))
        .and_then(|s| s.trim_start_matches('n').parse().ok())
        .unwrap_or(u32::MAX)
}
