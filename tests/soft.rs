//! Soft caps per vehicle, and the solver on a priced objective.

use volare::{
    Construct, Improve, ModelBuilder, NodeId, VehicleId, Violation, eval_route, eval_route_split,
    eval_routes, solve, violations,
};

#[test]
fn overload_pays_the_peak_once() {
    // Loads 8, 4, -6 along the route: cumul peaks at 12 in the middle.
    let load = [0, 8, 4, -6];
    let build = |hard_cap| {
        let mut b = ModelBuilder::new(4);
        let cost = b.cost_class(|_, _| 1);
        let v = b.vehicle(NodeId(0), NodeId(0), cost);
        b.dimension("load", move |_, to| load[to.index()], vec![hard_cap]);
        b.soft_max_cumul("load", v, 10, 5);
        b.build()
    };
    let route = [NodeId(1), NodeId(2), NodeId(3)];
    let v = VehicleId(0);

    let m = build(12);
    assert_eq!(eval_route_split(&m, &route, v), Some((4, 10)));
    assert_eq!(
        violations(&m, &vec![route.to_vec()]),
        vec![Violation {
            dimension: 0,
            vehicle: v,
            node: None,
            excess: 2,
            penalty: 10,
        }]
    );
    assert_eq!(eval_route(&build(11), &route, v), None, "hard cap at 11");
}

/// Five customers on a line, two vehicles, soft bounds no solution can all
/// meet. Brute force over every assignment and order gives the true optimum.
#[test]
fn solver_finds_the_optimum_of_the_priced_objective() {
    let dist = |a: NodeId, b: NodeId| (a.0 as i64 - b.0 as i64).abs() * 10;
    let mut b = ModelBuilder::new(6);
    let cost = b.cost_class(dist);
    let v0 = b.vehicle(NodeId(0), NodeId(0), cost);
    let v1 = b.vehicle(NodeId(0), NodeId(0), cost);
    b.dimension("time", dist, vec![i64::MAX; 2]);
    b.dimension(
        "load",
        |_, to| if to == NodeId(0) { 0 } else { 1 },
        vec![3, 3],
    );
    b.soft_upper_bound("time", NodeId(5), 60, 4);
    b.soft_upper_bound("time", NodeId(1), 10, 1);
    b.soft_upper_bound("time", NodeId(2), 20, 1);
    b.soft_max_cumul("load", v0, 2, 15);
    b.soft_max_cumul("load", v1, 2, 15);
    let m = b.build();

    let mut best = i64::MAX;
    let mut perm: Vec<NodeId> = (1..=5).map(NodeId).collect();
    permutations(&mut perm, 0, &mut |p| {
        for split in 0..=p.len() {
            let sol = vec![p[..split].to_vec(), p[split..].to_vec()];
            if let Some(c) = eval_routes(&m, &sol) {
                best = best.min(c);
            }
        }
    });

    let sol = solve(&m, Construct::CheapestInsertion, Improve::Gls { iters: 20 });
    assert_eq!(eval_routes(&m, &sol.routes), Some(sol.cost));
    assert_eq!(sol.cost, best, "solver {:?}", sol.routes);
    assert!(
        !violations(&m, &sol.routes).is_empty(),
        "the optimum is over a soft bound somewhere"
    );
}

fn permutations(items: &mut [NodeId], k: usize, visit: &mut impl FnMut(&[NodeId])) {
    if k == items.len() {
        visit(items);
        return;
    }
    for i in k..items.len() {
        items.swap(k, i);
        permutations(items, k + 1, visit);
        items.swap(k, i);
    }
}

/// Tight construction pays no penalty while a penalty-free placement exists,
/// and still routes a node that has none.
#[test]
fn tight_construction_starts_penalty_free_and_falls_back() {
    use volare::solver::first_solution_with;

    let dist = |a: NodeId, b: NodeId| (a.0 as i64 - b.0 as i64).abs() * 10;
    let mut b = ModelBuilder::new(6);
    let cost = b.cost_class(dist);
    for _ in 0..3 {
        b.vehicle(NodeId(0), NodeId(0), cost);
    }
    b.dimension(
        "load",
        |_, to| if to == NodeId(0) { 0 } else { 1 },
        vec![5; 3],
    );
    // Two per vehicle is cheap to break, so the priced greedy overloads one
    // route; tight construction spreads the five parcels over three.
    for v in 0..3 {
        b.soft_max_cumul("load", VehicleId(v), 2, 1);
    }
    let m = b.build();
    let priced = first_solution_with(&m, Construct::CheapestInsertion, |_| {});
    let tight = first_solution_with(&m, Construct::TightInsertion { seed: 0, k: 1 }, |_| {});
    assert!(
        !violations(&m, &priced).is_empty(),
        "priced greedy overloads"
    );
    assert!(
        violations(&m, &tight).is_empty(),
        "tight construction does not"
    );

    // Node 5 can never meet a soft close of zero: tight construction has to
    // loosen for it rather than leave it unrouted.
    let mut b = ModelBuilder::new(6);
    let cost = b.cost_class(dist);
    b.vehicle(NodeId(0), NodeId(0), cost);
    b.dimension("time", dist, vec![i64::MAX]);
    b.soft_upper_bound("time", NodeId(5), 0, 1);
    let m = b.build();
    let tight = first_solution_with(&m, Construct::TightInsertion { seed: 0, k: 1 }, |_| {});
    assert_eq!(tight[0].len(), 5, "every node routed");
    assert_eq!(violations(&m, &tight).len(), 1);
}
