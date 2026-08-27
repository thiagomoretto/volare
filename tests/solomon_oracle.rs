//! VRPTW oracle: published best-known routes through `eval_route`. Every
//! route must be feasible and the total must equal the published cost
//! exactly, both in tenths. Validates the window semantics against someone
//! else's answers.

use std::fs;
use std::path::Path;

use volare::eval::{eval_route, visits_all_nodes};
use volare::solomon::{TwInstance, parse_sol, vrptw_model};
use volare::types::VehicleId;

#[test]
fn bks_routes_reproduce_published_cost() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("instances/vrptw");
    let mut checked = 0;
    for entry in fs::read_dir(&dir).expect("instances/vrptw missing") {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "vrp") {
            continue;
        }
        let inst = TwInstance::parse(&fs::read_to_string(&path).unwrap());
        let sol = parse_sol(
            &fs::read_to_string(path.with_extension("sol")).unwrap(),
            &inst,
        );

        let model = vrptw_model(&inst, sol.routes.len());
        let mut total = 0;
        for (v, route) in sol.routes.iter().enumerate() {
            total += eval_route(&model, route, VehicleId(v as u32))
                .unwrap_or_else(|| panic!("{}: BKS route {v} evaluates infeasible", inst.name));
        }
        assert!(
            visits_all_nodes(&model, &sol.routes),
            "{}: BKS does not visit every customer once",
            inst.name
        );
        assert_eq!(
            total, sol.cost,
            "{}: cost mismatch against published",
            inst.name
        );

        // Negative control: windows are directional, so reversing a route
        // should almost always break one. If most reversals stay feasible,
        // the window check is not actually biting.
        let broken = sol
            .routes
            .iter()
            .enumerate()
            .filter(|(v, route)| {
                let reversed: Vec<_> = route.iter().rev().copied().collect();
                eval_route(&model, &reversed, VehicleId(*v as u32)).is_none()
            })
            .count();
        assert!(
            broken * 2 > sol.routes.len(),
            "{}: only {broken}/{} reversed routes infeasible — windows not biting",
            inst.name,
            sol.routes.len()
        );
        checked += 1;
    }
    assert!(checked >= 4, "expected the committed vrptw instances");
}
