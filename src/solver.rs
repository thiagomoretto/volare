use std::time::Instant;

use crate::eval::{Routes, eval_routes};
use crate::model::Model;
use crate::types::{Cost, NodeId, VehicleId};

mod construct;
mod descent;
mod gls;
mod operators;
#[cfg(test)]
mod tests;

pub use construct::{cheapest_insertion, first_solution_with, greedy_randomized};
pub use descent::{local_search, local_search_with};
pub use gls::{guided_local_search, guided_local_search_with};

/// The neighborhood operator that accepted an improving move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operator {
    Relocate,
    Swap,
    TwoOpt,
    TwoOptStar,
}

impl std::fmt::Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operator::Relocate => write!(f, "relocate"),
            Operator::Swap => write!(f, "swap"),
            Operator::TwoOpt => write!(f, "2-opt"),
            Operator::TwoOptStar => write!(f, "2-opt*"),
        }
    }
}

/// A progress point during a solve. Hand a callback to `solve_with`,
/// `first_solution_with` or `local_search_with` to observe them; `search_log`
/// builds one that prints progress lines. Costs are whole-solution
/// totals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchEvent {
    /// Construction placed every node; the first complete solution exists.
    FirstSolution { cost: Cost },
    /// `operator` accepted a move and the total cost dropped to `cost`.
    Improvement { operator: Operator, cost: Cost },
    /// Guided local search finished round `iter` holding a solution cheaper
    /// than anything before it. `cost` is the true cost, never the penalized
    /// one the descent was reading.
    GuidedBest { iter: usize, cost: Cost },
    /// Local search converged; the solution is final.
    Done { cost: Cost },
}

/// An event callback that prints progress lines to stderr,
/// prefixed with elapsed time since the closure was created:
///
/// ```text
/// #search    0.012s  relocate improved, cost 5900
/// ```
///
/// ```no_run
/// # use volare::solver::{Construct, Improve, solve_with, search_log};
/// # let model: volare::Model = todo!();
/// let routes = solve_with(
///     &model,
///     Construct::CheapestInsertion,
///     Improve::HillClimb,
///     search_log(),
/// );
/// ```
pub fn search_log() -> impl FnMut(SearchEvent) {
    let started = Instant::now();
    move |event| {
        let t = started.elapsed().as_secs_f64();
        match event {
            SearchEvent::FirstSolution { cost } => {
                eprintln!("#search {t:7.3}s  first solution, cost {cost}")
            }
            SearchEvent::Improvement { operator, cost } => {
                eprintln!("#search {t:7.3}s  {operator} improved, cost {cost}")
            }
            SearchEvent::GuidedBest { iter, cost } => {
                eprintln!("#search {t:7.3}s  gls round {iter}, new best cost {cost}")
            }
            SearchEvent::Done { cost } => eprintln!("#search {t:7.3}s  done, cost {cost}"),
        }
    }
}

/// How the first solution is built.
pub enum Construct {
    CheapestInsertion,
    /// Each step draws from the `k` cheapest, not the first. Same seed, same
    /// solution. `k = 1` is `CheapestInsertion`.
    GreedyRandomized {
        seed: u64,
        k: usize,
    },
}

/// How that solution is then made cheaper.
pub enum Improve {
    /// Descend to the first local optimum and stop.
    HillClimb,
    /// Guided local search: keep descending, penalizing the arcs that keep
    /// coming back, for `iters` rounds.
    Gls { iters: usize },
}

/// `cost` is the true cost — never the penalized number a GLS descent was
/// reading — and the solver only returns feasible routes, so it is a plain
/// `Cost`, not an `Option`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Solution {
    pub routes: Routes,
    pub cost: Cost,
}

impl Solution {
    /// Nodes left unserved (their penalties are already inside `cost`).
    pub fn unserved<'a>(&'a self, m: &'a Model) -> &'a [NodeId] {
        m.unserved_vehicle()
            .map_or(&[], |v| &self.routes[v.index()])
    }
}

/// The model is borrowed shared: every piece of mutable search state, GLS
/// penalties included, is owned by the call. One model can therefore back any
/// number of solves running at once.
pub fn solve(m: &Model, construct: Construct, improve: Improve) -> Solution {
    solve_with(m, construct, improve, |_| {})
}

/// `solve` with an observer for search progress. The callback runs on the
/// solver thread; keep it cheap or it becomes part of the measured time.
pub fn solve_with(
    m: &Model,
    construct: Construct,
    improve: Improve,
    mut log: impl FnMut(SearchEvent),
) -> Solution {
    let mut sol = first_solution_with(m, construct, &mut log);
    match improve {
        Improve::HillClimb => local_search_with(m, &mut sol, &mut log),
        Improve::Gls { iters } => guided_local_search_with(m, &mut sol, iters, &mut log),
    }
    let cost = eval_routes(m, &sol).expect("solver produced an infeasible solution");
    Solution { routes: sol, cost }
}

/// The route cost a descent minimizes: `eval_route` for the hill climb, the
/// penalized one for guided local search.
trait RouteEval: Fn(&Model, &[NodeId], VehicleId) -> Option<Cost> {}
impl<F: Fn(&Model, &[NodeId], VehicleId) -> Option<Cost>> RouteEval for F {}

/// Vehicles worth trying an insertion on: every route in use, plus one empty
/// one so the fleet can still grow.
///
/// One empty is enough only while empty vehicles are interchangeable. Once
/// they are not — per-vehicle `forbid` sets make them differ — the caller
/// retries with all of them (see `cheapest_insertion`). The unserved sink is
/// always a candidate: dropping must be on offer even when it is empty.
fn candidate_vehicles(m: &Model, sol: &Routes, out: &mut Vec<usize>) {
    out.clear();
    out.extend((0..sol.len()).filter(|&i| !sol[i].is_empty()));
    if let Some(empty) = (0..sol.len()).find(|&i| sol[i].is_empty()) {
        out.push(empty);
    }
    if let Some(uv) = m.unserved_vehicle()
        && !out.contains(&uv.index())
    {
        out.push(uv.index());
    }
}

/// Buffers the operators reuse across calls, one per descent.
#[derive(Default)]
struct Scratch {
    /// The moving node's own route with that node taken out.
    without: Vec<NodeId>,
    /// A route with the moving node inserted, the one being priced.
    candidate: Vec<NodeId>,
    /// Vehicles worth trying, from `candidate_vehicles`.
    vehicles: Vec<usize>,
}

/// `node` in front of `route`; callers slide it forward one swap per position.
fn with_front(route: &[NodeId], node: NodeId, out: &mut Vec<NodeId>) {
    out.clear();
    out.push(node);
    out.extend_from_slice(route);
}
