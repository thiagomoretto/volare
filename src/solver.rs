use std::ops::ControlFlow;
use std::time::Instant;

use crate::eval::{Routes, eval_routes};
use crate::model::Model;
use crate::types::{Cost, NodeId, VehicleId};

mod construct;
mod descent;
mod gls;
mod operators;
mod sisr;
#[cfg(test)]
mod tests;

pub use construct::{cheapest_insertion, first_solution_with, greedy_randomized};
pub use descent::{local_search, local_search_with};
pub use gls::{guided_local_search, guided_local_search_with};
pub use sisr::{SisrParams, ruin_recreate, ruin_recreate_with};

/// The neighborhood operator that accepted an improving move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operator {
    Relocate,
    Swap,
    OrOpt,
    TwoOpt,
    TwoOptStar,
}

impl std::fmt::Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operator::Relocate => write!(f, "relocate"),
            Operator::Swap => write!(f, "swap"),
            Operator::OrOpt => write!(f, "or-opt"),
            Operator::TwoOpt => write!(f, "2-opt"),
            Operator::TwoOptStar => write!(f, "2-opt*"),
        }
    }
}

/// A progress point during a solve. Hand a callback to `solve_with`,
/// `first_solution_with`, `local_search_with` or `guided_local_search_with` to
/// observe them; `search_log` builds one that prints progress lines. Costs are
/// whole-solution totals.
///
/// The callback handed to `solve_with`, `local_search_with` or
/// `guided_local_search_with` is also a search monitor: return
/// `ControlFlow::Break(())` and the search stops at once, keeps the best
/// solution found, and reports `Done`. Construction cannot stop before every
/// node is placed, so `first_solution_with` takes a plain observer; a `Break`
/// on `FirstSolution` makes `solve_with` skip the improvement phase.
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
    /// Guided local search finished round `iter`; `cost` is the true cost of
    /// the round's own solution. Fires every round, improving or not, so a
    /// monitor can stop a search that has stalled.
    GuidedRound { iter: usize, cost: Cost },
    /// Ruin and recreate finished round `iter` holding a solution cheaper
    /// than anything before it.
    SisrBest { iter: usize, cost: Cost },
    /// Ruin and recreate finished round `iter`; `cost` is the solution the
    /// search now holds, which annealing may have let get worse.
    SisrRound { iter: usize, cost: Cost },
    /// The search converged or the callback stopped it; the solution is final.
    Done { cost: Cost },
}

/// An event callback that prints progress lines to stderr, prefixed with
/// elapsed time since the closure was created. It skips `GuidedRound` and
/// `SisrRound`, which fire too often to read, and never stops the search:
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
pub fn search_log() -> impl FnMut(SearchEvent) -> ControlFlow<()> {
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
            SearchEvent::SisrBest { iter, cost } => {
                eprintln!("#search {t:7.3}s  sisr round {iter}, new best cost {cost}")
            }
            SearchEvent::GuidedRound { .. } | SearchEvent::SisrRound { .. } => {}
            SearchEvent::Done { cost } => eprintln!("#search {t:7.3}s  done, cost {cost}"),
        }
        ControlFlow::Continue(())
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Improve {
    /// Descend to the first local optimum and stop.
    HillClimb,
    /// Guided local search: keep descending, penalizing the arcs that keep
    /// coming back, for `iters` rounds.
    Gls { iters: usize },
    /// Ruin and recreate: cut strings of nearby customers out, reinsert them,
    /// accept under simulated annealing. See `SisrParams`.
    Sisr(SisrParams),
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
    solve_with(m, construct, improve, |_| ControlFlow::Continue(()))
}

/// `solve` with a monitor: it sees every `SearchEvent` and can stop the search
/// early (see `SearchEvent`). The callback runs on the solver thread; keep it
/// cheap or it becomes part of the measured time.
pub fn solve_with(
    m: &Model,
    construct: Construct,
    improve: Improve,
    mut log: impl FnMut(SearchEvent) -> ControlFlow<()>,
) -> Solution {
    let mut stopped = false;
    let mut sol = first_solution_with(m, construct, |e| stopped = log(e).is_break());
    if !stopped {
        match improve {
            Improve::HillClimb => local_search_with(m, &mut sol, &mut log),
            Improve::Gls { iters } => guided_local_search_with(m, &mut sol, iters, &mut log),
            Improve::Sisr(p) => ruin_recreate_with(m, &mut sol, p, &mut log),
        }
    }
    let cost = eval_routes(m, &sol).expect("solver produced an infeasible solution");
    if stopped {
        let _ = log(SearchEvent::Done { cost });
    }
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
    /// The moving chain, copied out of its route.
    chain: Vec<NodeId>,
    /// A route with the moving node or chain inserted, the one being priced.
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

/// SplitMix64. Not cryptographic. Seeds reproduce solutions, no dependency.
struct Rng(u64);

impl Rng {
    #[inline]
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform over `0..n`. Modulo bias under 2^-55 at these `n`, ignore it.
    #[inline]
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// A draw from the unit exponential, `-ln U` for `U` in `(0, 1]`.
    fn exp1(&mut self) -> f64 {
        let u = ((self.next() >> 11) + 1) as f64 / (1u64 << 53) as f64;
        -u.ln()
    }
}
