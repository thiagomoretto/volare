//! Guided local search must beat the hill climb it wraps.
//!
//! It used to have a second job here: proving no penalty survived the call.
//! `Penalties` is now owned by the search and dropped with it, and `Model` has
//! no penalty state to leak into, so the type system covers what that half of
//! the test covered.

use std::fs;
use std::path::Path;

use volare::cvrplib::{Instance, cvrp_model};
use volare::eval::{eval_routes, visits_all_nodes};
use volare::solver::{Construct, Improve, first_solution_with, local_search, solve};

#[test]
fn gls_improves_on_the_hill_climb() {
    let vrp = Path::new(env!("CARGO_MANIFEST_DIR")).join("instances/X-n101-k25.vrp");
    let inst = Instance::parse(&fs::read_to_string(&vrp).unwrap());
    let n = inst.coords.len();
    let m = cvrp_model(&inst, n - 1);

    let mut hill = first_solution_with(&m, Construct::CheapestInsertion, |_| {});
    local_search(&m, &mut hill);
    let hill_cost = eval_routes(&m, &hill).expect("infeasible hill climb");

    // 1000, not 20: on this instance 20 rounds improve nothing, and that count
    // would pass this test while doing no work at all — `best` starts as the
    // hill climb's own answer. The large gain lands between 300 and 1000.
    let sol = solve(
        &m,
        Construct::CheapestInsertion,
        Improve::Gls { iters: 1000 },
    );
    let gls_cost = sol.cost;

    assert!(
        visits_all_nodes(&m, &sol.routes),
        "GLS dropped or duplicated a node"
    );
    assert!(
        gls_cost < hill_cost,
        "GLS returned {gls_cost} against the hill climb's {hill_cost} — it escaped no local \
         optimum, so the penalties are not reaching the operators"
    );
}
