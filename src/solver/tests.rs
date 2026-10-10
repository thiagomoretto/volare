use std::ops::ControlFlow;

use super::operators::{try_or_opt, try_two_opt_star};
use super::*;
use crate::eval::{eval_route, visits_all_nodes};
use crate::model::ModelBuilder;

/// Depot at 0, customers 1..=5 on a line; arc cost is line distance.
fn line_model() -> Model {
    let dist = |a: NodeId, b: NodeId| (a.0 as i64 - b.0 as i64).abs() * 10;
    let mut b = ModelBuilder::new(6);
    let cost = b.cost_class(dist);
    b.vehicle(NodeId(0), NodeId(0), cost);
    b.vehicle(NodeId(0), NodeId(0), cost);
    b.dimension(
        "load",
        |_from, to| if to == NodeId(0) { 0 } else { 1 },
        vec![3, 3],
    );
    b.build()
}

/// `k = 1` is greedy, seeds reproduce, some seed diverges. Without the
/// last, the draw is a no-op and multi-start has nothing to search.
#[test]
fn randomized_insertion_is_seeded_not_arbitrary() {
    let m = line_model();
    let greedy = cheapest_insertion(&m, |_| {});

    assert_eq!(
        greedy_randomized(&m, 7, 1, |_| {}),
        greedy,
        "k = 1 is greedy"
    );
    assert_eq!(
        greedy_randomized(&m, 7, 3, |_| {}),
        greedy_randomized(&m, 7, 3, |_| {}),
        "same seed, same solution"
    );

    let mut diverged = 0;
    for seed in 0..32 {
        let sol = greedy_randomized(&m, seed, 3, |_| {});
        assert!(visits_all_nodes(&m, &sol), "seed {seed} lost a node");
        assert!(eval_routes(&m, &sol).is_some(), "seed {seed} is infeasible");
        if sol != greedy {
            diverged += 1;
        }
    }
    assert!(diverged > 0, "32 seeds all returned the greedy solution");
}

/// Forbids and the drop sink are ordinary candidates. Order is random,
/// feasibility is not.
#[test]
fn randomized_insertion_respects_drops_and_forbids() {
    let dist = |a: NodeId, b: NodeId| (a.0 as i64 - b.0 as i64).abs() * 10;
    let mut b = ModelBuilder::new(6);
    let cost = b.cost_class(dist);
    let v0 = b.vehicle(NodeId(0), NodeId(0), cost);
    b.vehicle(NodeId(0), NodeId(0), cost);
    b.forbid(v0, NodeId(1));
    b.dimension(
        "load",
        |_from, to| if to == NodeId(0) { 0 } else { 1 },
        vec![3, 3],
    );
    b.allow_drop(NodeId(5), 15);
    let m = b.build();

    for seed in 0..16 {
        let sol = greedy_randomized(&m, seed, 4, |_| {});
        assert!(visits_all_nodes(&m, &sol), "seed {seed} lost a node");
        assert!(eval_routes(&m, &sol).is_some(), "seed {seed} is infeasible");
        assert!(
            !sol[v0.index()].contains(&NodeId(1)),
            "seed {seed} put node 1 on the vehicle that forbids it"
        );
    }
}

/// A node cheaper to drop than to serve pays its penalty instead.
#[test]
fn dropped_nodes_pay_their_penalty() {
    let build = |penalty: Cost| {
        let dist = |a: NodeId, b: NodeId| (a.0 as i64 - b.0 as i64).abs() * 10;
        let mut b = ModelBuilder::new(4);
        let cost = b.cost_class(dist);
        b.vehicle(NodeId(0), NodeId(0), cost);
        b.dimension(
            "load",
            |_from, to| if to == NodeId(0) { 0 } else { 1 },
            vec![3],
        );
        b.allow_drop(NodeId(3), penalty);
        b.build()
    };

    // Serving node 3 extends the route by 20; penalty 15 wins, 30 loses.
    let m = build(15);
    let sol = solve(&m, Construct::CheapestInsertion, Improve::HillClimb);
    assert_eq!(sol.unserved(&m), &[NodeId(3)]);
    assert_eq!(sol.cost, 40 + 15);
    assert!(visits_all_nodes(&m, &sol.routes));

    let m = build(30);
    let sol = solve(&m, Construct::CheapestInsertion, Improve::HillClimb);
    assert!(sol.unserved(&m).is_empty());
    assert_eq!(sol.cost, 60);
    assert!(visits_all_nodes(&m, &sol.routes));
}

/// Both routes are full, so no node can move anywhere: only a trade helps.
#[test]
fn swap_finds_what_relocate_cannot() {
    let dist = |a: NodeId, b: NodeId| (a.0 as i64 - b.0 as i64).abs() * 10;
    let mut b = ModelBuilder::new(5);
    let cost = b.cost_class(dist);
    b.vehicle(NodeId(0), NodeId(0), cost);
    b.vehicle(NodeId(0), NodeId(0), cost);
    b.dimension(
        "load",
        |_from, to| if to == NodeId(0) { 0 } else { 1 },
        vec![2, 2],
    );
    let m = b.build();

    let mut sol = vec![vec![NodeId(1), NodeId(3)], vec![NodeId(2), NodeId(4)]];
    assert_eq!(eval_routes(&m, &sol), Some(140));

    let mut ops = Vec::new();
    local_search_with(&m, &mut sol, |e| {
        if let SearchEvent::Improvement { operator, .. } = e {
            ops.push(operator);
        }
        ControlFlow::Continue(())
    });
    assert!(
        ops.contains(&Operator::Swap),
        "expected a swap, got {ops:?}"
    );
    assert_eq!(eval_routes(&m, &sol), Some(120));
}

/// Crossing routes on tight capacity: no single-customer move helps, only
/// trading tails between routes does.
#[test]
fn two_opt_star_repartitions_routes() {
    // Depot at the origin, customers on the compass points.
    let xy = [(0, 0), (10, 0), (0, 10), (-10, 0), (0, -10)];
    let dist = move |a: NodeId, b: NodeId| {
        let (ax, ay) = xy[a.index()];
        let (bx, by) = xy[b.index()];
        let (dx, dy) = (ax - bx, ay - by);
        ((dx * dx + dy * dy) as f64).sqrt().round() as i64
    };
    let mut b = ModelBuilder::new(5);
    let cost = b.cost_class(dist);
    b.vehicle(NodeId(0), NodeId(0), cost);
    b.vehicle(NodeId(0), NodeId(0), cost);
    b.dimension(
        "load",
        |_from, to| if to == NodeId(0) { 0 } else { 1 },
        vec![2, 2],
    );
    let m = b.build();

    // East rides with west, north with south: both routes cross the map.
    let mut sol = vec![vec![NodeId(1), NodeId(3)], vec![NodeId(2), NodeId(4)]];
    let mut cost: Vec<Cost> = (0..2)
        .map(|v| eval_route(&m, &sol[v], VehicleId(v as u32)).unwrap())
        .collect();
    assert_eq!(cost.iter().sum::<Cost>(), 80);

    let v = try_two_opt_star(&m, &mut sol, &eval_route, &mut cost, 0);
    assert_eq!(v, Some(1));
    assert_eq!(
        sol,
        vec![vec![NodeId(1), NodeId(4)], vec![NodeId(2), NodeId(3)]]
    );
    assert_eq!(cost.iter().sum::<Cost>(), 68);
    // A local optimum of this neighborhood: a second call finds nothing.
    assert_eq!(
        try_two_opt_star(&m, &mut sol, &eval_route, &mut cost, 0),
        None
    );
}

/// A pair cheaper two stops over, while every single-node move breaks even
/// or worsens: the chain is the smallest move that helps.
#[test]
fn or_opt_moves_a_chain_relocate_cannot() {
    let dist = |a: NodeId, b: NodeId| (a.0 as i64 - b.0 as i64).abs() * 10;
    let mut b = ModelBuilder::new(5);
    let cost = b.cost_class(dist);
    b.vehicle(NodeId(0), NodeId(0), cost);
    b.vehicle(NodeId(0), NodeId(0), cost);
    b.dimension(
        "load",
        |_from, to| if to == NodeId(0) { 0 } else { 1 },
        vec![4, 4],
    );
    let m = b.build();

    // 0->4->2->3->1->0 costs 100; the pair (2,3) belongs up front.
    let mut sol = vec![vec![NodeId(4), NodeId(2), NodeId(3), NodeId(1)], vec![]];
    let mut cost: Vec<Cost> = (0..2)
        .map(|v| eval_route(&m, &sol[v], VehicleId(v as u32)).unwrap_or(0))
        .collect();
    assert_eq!(cost[0], 100);

    let mut sx = Scratch::default();
    let v = try_or_opt(&m, &mut sol, &eval_route, &mut cost, NodeId(2), 0, &mut sx);
    assert_eq!(v, Some(0));
    assert_eq!(sol[0], vec![NodeId(2), NodeId(3), NodeId(4), NodeId(1)]);
    assert_eq!(cost[0], 80);
    // A local optimum of this neighborhood: a second call finds nothing.
    assert_eq!(
        try_or_opt(&m, &mut sol, &eval_route, &mut cost, NodeId(2), 0, &mut sx),
        None
    );
}

/// A node forbidden on one vehicle must land on the other, and stay
/// there through local search: every operator re-validates via
/// `eval_route`, so no move can drag it back.
#[test]
fn forbidden_node_rides_the_allowed_vehicle() {
    let dist = |a: NodeId, b: NodeId| (a.0 as i64 - b.0 as i64).abs() * 10;
    let mut b = ModelBuilder::new(6);
    let cost = b.cost_class(dist);
    let v0 = b.vehicle(NodeId(0), NodeId(0), cost);
    b.vehicle(NodeId(0), NodeId(0), cost);
    // Node 1 is closest to the depot; v0 would take it if it could.
    b.forbid(v0, NodeId(1));
    b.dimension(
        "load",
        |_from, to| if to == NodeId(0) { 0 } else { 1 },
        vec![3, 3],
    );
    let m = b.build();

    assert_eq!(eval_route(&m, &[NodeId(1)], VehicleId(0)), None);
    assert!(eval_route(&m, &[NodeId(1)], VehicleId(1)).is_some());

    let sol = solve(&m, Construct::CheapestInsertion, Improve::Gls { iters: 5 });
    assert!(visits_all_nodes(&m, &sol.routes));
    assert!(!sol.routes[0].contains(&NodeId(1)));
    assert!(sol.routes[1].contains(&NodeId(1)));
}

/// Forbidden everywhere is not a fleet problem; the panic must say so.
#[test]
#[should_panic(expected = "unroutable")]
fn node_forbidden_everywhere_panics_clearly() {
    let dist = |a: NodeId, b: NodeId| (a.0 as i64 - b.0 as i64).abs() * 10;
    let mut b = ModelBuilder::new(6);
    let cost = b.cost_class(dist);
    let v0 = b.vehicle(NodeId(0), NodeId(0), cost);
    let v1 = b.vehicle(NodeId(0), NodeId(0), cost);
    b.forbid(v0, NodeId(3));
    b.forbid(v1, NodeId(3));
    let m = b.build();
    solve(&m, Construct::CheapestInsertion, Improve::HillClimb);
}

#[test]
fn search_events_trace_the_solve() {
    let m = line_model();
    let mut events = Vec::new();
    let sol = solve_with(&m, Construct::CheapestInsertion, Improve::HillClimb, |e| {
        events.push(e);
        ControlFlow::Continue(())
    });

    let SearchEvent::FirstSolution { cost: first } = events[0] else {
        panic!("first event must be FirstSolution, got {:?}", events[0]);
    };
    let &SearchEvent::Done { cost: done } = events.last().unwrap() else {
        panic!("last event must be Done, got {:?}", events.last());
    };

    let mut prev = i64::MAX;
    for e in events.iter() {
        if let SearchEvent::Improvement { cost, .. } = e {
            assert!(*cost < prev, "improvements must strictly decrease");
            prev = *cost;
        }
    }

    assert!(done <= first, "local search never worsens the solution");
    assert_eq!(Some(done), eval_routes(&m, &sol.routes));
    // The silent entry point solves to the same cost.
    let silent = solve(&m, Construct::CheapestInsertion, Improve::HillClimb);
    assert_eq!(silent.cost, done);
}

/// `Break` ends the search on the spot, and the answer is still the best one
/// seen, not whatever the stopped round held.
#[test]
fn monitor_stops_guided_search() {
    let m = line_model();
    let (mut rounds, mut best, mut done) = (0, i64::MAX, None);
    let sol = solve_with(
        &m,
        Construct::CheapestInsertion,
        Improve::Gls { iters: 1000 },
        |e| {
            match e {
                SearchEvent::GuidedBest { cost, .. } => best = cost,
                SearchEvent::GuidedRound { .. } => rounds += 1,
                SearchEvent::Done { cost } => done = Some(cost),
                _ => {}
            }
            if rounds == 5 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        },
    );
    assert_eq!(rounds, 5);
    assert_eq!(done, Some(best));
    assert_eq!(sol.cost, best);
}

#[test]
fn monitor_stops_before_improving() {
    let m = line_model();
    let mut events = Vec::new();
    let sol = solve_with(&m, Construct::CheapestInsertion, Improve::HillClimb, |e| {
        events.push(e);
        ControlFlow::Break(())
    });
    let first = cheapest_insertion(&m, |_| {});
    let cost = eval_routes(&m, &first).unwrap();
    assert_eq!(sol.routes, first);
    assert_eq!(
        events,
        [
            SearchEvent::FirstSolution { cost },
            SearchEvent::Done { cost }
        ]
    );
}

#[test]
fn monitor_stops_hill_climb() {
    let m = line_model();
    let start = vec![
        vec![NodeId(5), NodeId(1), NodeId(3)],
        vec![NodeId(2), NodeId(4)],
    ];
    let improvements = |stop: bool| {
        let mut sol = start.clone();
        let mut costs = Vec::new();
        local_search_with(&m, &mut sol, |e| {
            if let SearchEvent::Improvement { cost, .. } = e {
                costs.push(cost);
                if stop {
                    return ControlFlow::Break(());
                }
            }
            ControlFlow::Continue(())
        });
        (costs, eval_routes(&m, &sol))
    };
    let (all, _) = improvements(false);
    assert!(
        all.len() > 1,
        "start must take more than one move, took {all:?}"
    );
    let (first, cost) = improvements(true);
    assert_eq!(first, all[..1]);
    assert_eq!(cost, Some(first[0]));
}

/// Ruin and recreate keeps every constraint the evaluator enforces, never
/// returns worse than its start, and a seed reproduces its answer.
#[test]
fn ruin_recreate_is_seeded_feasible_and_monotone() {
    let dist = |a: NodeId, b: NodeId| (a.0 as i64 - b.0 as i64).abs() * 10;
    let mut b = ModelBuilder::new(9);
    let cost = b.cost_class(dist);
    let v0 = b.vehicle(NodeId(0), NodeId(0), cost);
    b.vehicle(NodeId(0), NodeId(0), cost);
    b.vehicle(NodeId(0), NodeId(0), cost);
    b.forbid(v0, NodeId(1));
    b.dimension(
        "load",
        |_from, to| if to == NodeId(0) { 0 } else { 1 },
        vec![3, 3, 3],
    );
    b.allow_drop(NodeId(8), 15);
    let m = b.build();

    let start = cheapest_insertion(&m, |_| {});
    let start_cost = eval_routes(&m, &start).unwrap();
    let run = |seed| {
        let mut sol = start.clone();
        let mut p = SisrParams::new(500);
        p.seed = seed;
        ruin_recreate(&m, &mut sol, p);
        sol
    };

    for seed in 0..8 {
        let sol = run(seed);
        assert_eq!(sol, run(seed), "seed {seed} did not reproduce");
        assert!(visits_all_nodes(&m, &sol), "seed {seed} lost a node");
        let c = eval_routes(&m, &sol).expect("infeasible result");
        assert!(c <= start_cost, "seed {seed} returned worse than its start");
        assert!(
            !sol[v0.index()].contains(&NodeId(1)),
            "seed {seed} put node 1 on the vehicle that forbids it"
        );
    }
}

/// Every constraint `walk` knows, on small nodes with random windows. Vehicle
/// 0 carries no price, so its routes take the slack shortcut; vehicle 1 pays
/// for waiting and for a soft peak, so its routes walk.
fn every_constraint_model(seed: u64) -> Model {
    let mut rng = Rng(seed);
    let n = 9;
    let x: Vec<i64> = (0..n).map(|_| rng.below(40) as i64).collect();
    let dist = move |a: NodeId, b: NodeId| (x[a.index()] - x[b.index()]).abs();
    let demand: Vec<i64> = (0..n)
        .map(|i| if i == 0 { 0 } else { 1 + (i % 3) as i64 })
        .collect();

    let mut b = ModelBuilder::new(n);
    let cost = b.cost_class(dist.clone());
    let v0 = b.vehicle(NodeId(0), NodeId(0), cost);
    let v1 = b.vehicle(NodeId(0), NodeId(0), cost);
    b.dimension("load", move |_, to| demand[to.index()], vec![9, 8]);
    b.soft_max_cumul("load", v1, 5, 3);
    b.dimension(
        "time",
        move |from, to| dist(from, to) + if from == NodeId(0) { 0 } else { 5 },
        vec![400, 400],
    );
    // Picks up at even nodes, delivers at odd ones: the cumul dips mid-route.
    b.dimension("pd", |_, to| if to.0 % 2 == 0 { 2 } else { -1 }, vec![5, 6]);

    for i in 1..n {
        let node = NodeId(i as u32);
        if rng.below(2) == 0 {
            let lb = rng.below(120) as i64;
            b.cumul_bounds("time", node, lb, lb + 40 + rng.below(120) as i64);
        }
        if rng.below(4) == 0 {
            b.soft_upper_bound("time", node, rng.below(100) as i64, 2);
        }
        if rng.below(5) == 0 {
            b.max_wait_at("time", node, rng.below(30) as i64);
        }
    }
    b.max_wait("time", v0, 60);
    b.wait_cost("time", v1, 1);
    b.forbid(v0, NodeId(1));
    b.precede(NodeId(2), NodeId(3));
    b.allow_drop(NodeId(8), 50);
    b.allow_drop(NodeId(7), 20);
    b.build()
}

/// `insert_cost` is a shortcut for `eval_route` on the inserted route, so it
/// must agree with it everywhere, feasible or not.
#[test]
fn route_cache_prices_insertions_like_eval_route() {
    use crate::eval::RouteCache;

    let (mut checked, mut feasible) = (0, 0);
    for seed in 0..40 {
        let m = every_constraint_model(seed);
        let mut rng = Rng(seed ^ 0xABCD);
        let customers: Vec<NodeId> = (1..m.node_count() as u32).map(NodeId).collect();
        let mut cache = RouteCache::default();

        for _ in 0..200 {
            let v = VehicleId(rng.below(m.vehicle_count()) as u32);
            let mut pool = customers.clone();
            for i in (1..pool.len()).rev() {
                pool.swap(i, rng.below(i + 1));
            }
            let (route, rest) = pool.split_at(rng.below(6));

            let truth = eval_route(&m, route, v);
            assert_eq!(cache.rebuild(&m, route, v), truth.is_some());
            let Some(c) = truth else { continue };
            assert_eq!(cache.cost(), c);

            for &u in rest {
                for pos in 0..=route.len() {
                    let mut inserted = route.to_vec();
                    inserted.insert(pos, u);
                    let want = eval_route(&m, &inserted, v);
                    assert_eq!(
                        cache.insert_cost(&m, route, v, pos, u),
                        want,
                        "seed {seed}: {u:?} at {pos} of {route:?} on {v:?}"
                    );
                    checked += 1;
                    feasible += want.is_some() as usize;
                }
            }
        }
    }
    assert!(
        feasible > 1000 && checked - feasible > 1000,
        "{feasible} of {checked} feasible"
    );
}
