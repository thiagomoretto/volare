//! Cost oracle: re-evaluate the published BKS route order and check it
//! reproduces the published cost exactly.
//!
//! This is the highest-value test in the project. It catches EUC_2D rounding,
//! depot indexing and off-by-one errors, all of which otherwise show up as a
//! plausible-looking but meaningless gap percentage.

use std::fs;
use std::path::Path;

use volare::cvrplib::{Instance, cvrp_model, parse_sol};
use volare::eval::{eval_route, visits_all_nodes};
use volare::types::VehicleId;

#[test]
fn bks_routes_reproduce_published_cost() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("instances");
    let mut names: Vec<_> = fs::read_dir(&dir)
        .expect("instances/ missing")
        .filter_map(|e| {
            let p = e.unwrap().path();
            (p.extension()? == "vrp").then_some(p)
        })
        .collect();
    names.sort();
    assert!(!names.is_empty(), "no instances found");

    for vrp in &names {
        let inst = Instance::parse(&fs::read_to_string(vrp).unwrap());
        let sol = parse_sol(
            &fs::read_to_string(vrp.with_extension("sol")).unwrap(),
            &inst,
        );
        if sol.routes.is_empty() {
            continue; // cost-only BKS reference, no routes to verify
        }
        let model = cvrp_model(&inst, sol.routes.len());

        let mut total = 0;
        for (v, route) in sol.routes.iter().enumerate() {
            total += eval_route(&model, route, VehicleId(v as u32))
                .unwrap_or_else(|| panic!("{}: BKS route {v} violates capacity", inst.name));
        }
        assert_eq!(total, sol.cost, "{}: cost mismatch", inst.name);
        assert!(visits_all_nodes(&model, &sol.routes), "{}", inst.name);
    }
    println!("{} instances verified against published cost", names.len());
}

#[test]
fn empty_route_is_free() {
    let inst = Instance::parse(
        "NAME : t\nDIMENSION : 2\nCAPACITY : 10\nEDGE_WEIGHT_TYPE : EUC_2D\n\
         NODE_COORD_SECTION\n1 0 0\n2 3 4\nDEMAND_SECTION\n1 0\n2 5\nDEPOT_SECTION\n1\n-1\nEOF\n",
    );
    let model = cvrp_model(&inst, 2);
    assert_eq!(eval_route(&model, &[], VehicleId(0)), Some(0));
    // 3-4-5 triangle, out and back.
    assert_eq!(
        eval_route(&model, &[volare::NodeId(1)], VehicleId(0)),
        Some(10)
    );
}

#[test]
fn capacity_is_enforced() {
    let inst = Instance::parse(
        "NAME : t\nDIMENSION : 3\nCAPACITY : 10\nEDGE_WEIGHT_TYPE : EUC_2D\n\
         NODE_COORD_SECTION\n1 0 0\n2 1 0\n3 2 0\nDEMAND_SECTION\n1 0\n2 6\n3 6\n\
         DEPOT_SECTION\n1\n-1\nEOF\n",
    );
    let model = cvrp_model(&inst, 2);
    let (a, b) = (volare::NodeId(1), volare::NodeId(2));
    assert!(eval_route(&model, &[a], VehicleId(0)).is_some());
    assert_eq!(eval_route(&model, &[a, b], VehicleId(0)), None, "12 > 10");
}
