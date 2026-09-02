//! The neighborhood: the moves a descent tries, one function each.
//!
//! Each takes the [`Search`] it evaluates through, the solution it may rewrite,
//! and the per-route costs it has to keep in step; each returns the other
//! vehicle it touched, so the caller knows what to re-examine. Adding a move
//! means adding a function of the same shape — `examples/custom_operator`
//! writes one from outside the crate.

use crate::eval::Routes;
use crate::search::Search;
use crate::solver::candidate_vehicles;
use crate::types::{Cost, NodeId, VehicleId};

/// Move `u` out of route `r` to its first improving position anywhere,
/// including back into `r`. Returns the receiving vehicle.
///
/// `without` and `cands` belong to the caller: both are held across an
/// evaluation, which is exactly the buffer `Search` cannot own.
pub(crate) fn try_relocate(
    cx: &mut Search,
    sol: &mut Routes,
    cost: &mut [Cost],
    u: NodeId,
    r: usize,
    without: &mut Vec<NodeId>,
    cands: &mut Vec<usize>,
) -> Option<usize> {
    let at = sol[r].iter().position(|&x| x == u)?;
    without.clear();
    without.extend_from_slice(&sol[r]);
    without.remove(at);
    let without_cost = cx.eval(without, VehicleId(r as u32))?;

    // Probe first, commit after: the accepted route is still in the context,
    // and `sol` cannot be written while a candidate base borrows it.
    candidate_vehicles(cx.model(), sol, cands);
    let mut accepted = None;
    'search: for &v in cands.iter() {
        let base: &[NodeId] = if v == r { without } else { &sol[v] };
        for pos in 0..=base.len() {
            let Some(c) = cx.eval_splice(base, pos..pos, &[u], VehicleId(v as u32)) else {
                continue;
            };
            let delta = if v == r {
                c - cost[r]
            } else {
                (without_cost - cost[r]) + (c - cost[v])
            };
            if delta < 0 {
                accepted = Some((v, c));
                break 'search;
            }
        }
    }

    let (v, c) = accepted?;
    if v != r {
        sol[r].clone_from(without);
        cost[r] = without_cost;
    }
    sol[v].clear();
    sol[v].extend_from_slice(cx.spliced());
    cost[v] = c;
    Some(v)
}

/// Trade `u` for a customer on another route, first improving pair wins.
/// Returns the other vehicle.
///
/// Relocate cannot reach these moves when both routes are near capacity: moving
/// a node either way overflows, and only an even trade fits. On the X set that
/// is most of the search space, since the reference fleet is sized tight.
pub(crate) fn try_swap(
    cx: &mut Search,
    sol: &mut Routes,
    cost: &mut [Cost],
    u: NodeId,
    r: usize,
) -> Option<usize> {
    let at = sol[r].iter().position(|&x| x == u)?;
    for v in 0..sol.len() {
        if v == r || sol[v].is_empty() {
            continue;
        }
        for q in 0..sol[v].len() {
            let w = sol[v][q];
            sol[r][at] = w;
            sol[v][q] = u;
            // Route `v` goes unevaluated when `r` is already infeasible.
            let new = match cx.eval(&sol[r], VehicleId(r as u32)) {
                Some(a) => cx.eval(&sol[v], VehicleId(v as u32)).map(|b| (a, b)),
                None => None,
            };
            match new {
                Some((a, b)) if a + b < cost[r] + cost[v] => {
                    cost[r] = a;
                    cost[v] = b;
                    return Some(v);
                }
                _ => {
                    sol[r][at] = u;
                    sol[v][q] = w;
                }
            }
        }
    }
    None
}

/// Intra-route 2-opt: reverse one segment, keep it if it is cheaper.
pub(crate) fn try_two_opt(cx: &mut Search, sol: &mut Routes, cost: &mut [Cost], r: usize) -> bool {
    let len = sol[r].len();
    for i in 0..len {
        for j in i + 1..len {
            sol[r][i..=j].reverse();
            match cx.eval(&sol[r], VehicleId(r as u32)) {
                Some(c) if c < cost[r] => {
                    cost[r] = c;
                    return true;
                }
                _ => sol[r][i..=j].reverse(),
            }
        }
    }
    false
}

/// A candidate 2-opt* move: penalised gain, the other vehicle, the two
/// rebuilt routes, and their true costs.
type TwoOptStarMove = (Cost, usize, Vec<NodeId>, Vec<NodeId>, Cost, Cost);

/// Inter-route 2-opt*: cut one arc in route `r` and one in another route,
/// trade the tails. Returns the other vehicle.
///
/// This is the operator that changes the customer-to-route partition in
/// chunks: relocate and swap shift one customer at a time, which cannot undo
/// a bad layout once routes fill up. The delta is O(1) arc arithmetic — only
/// the two cut arcs and the two reconnecting arcs change — and the full
/// evaluation runs only on improving candidates, to confirm capacity.
///
/// Best-improvement, unlike the cheaper operators: a tail swap commits many
/// customers at once, so taking the first improving cut drags the descent
/// into noticeably worse local optima (measured on X-n143-k7).
pub(crate) fn try_two_opt_star(
    cx: &mut Search,
    sol: &mut Routes,
    cost: &mut [Cost],
    r: usize,
) -> Option<usize> {
    let m = cx.model();
    let veh_r = m.vehicle(VehicleId(r as u32));
    let mut best: Option<TwoOptStarMove> = None;
    for v in 0..sol.len() {
        if v == r || sol[v].is_empty() {
            continue;
        }
        let veh_v = m.vehicle(VehicleId(v as u32));
        for i in 0..sol[r].len() {
            for j in 0..sol[v].len() {
                let tail_r = sol[r].get(i + 1).copied();
                let tail_v = sol[v].get(j + 1).copied();
                if tail_r.is_none() && tail_v.is_none() {
                    continue; // both tails empty: no arc changes
                }
                // `cx.arc`, not `m.eval`: under GLS the delta must rank on the
                // same objective the acceptance below runs on, or the best
                // candidate by one measure loses to a worse one by the other.
                let out = cx.arc(veh_r.cost_class, sol[r][i], tail_r.unwrap_or(veh_r.end))
                    + cx.arc(veh_v.cost_class, sol[v][j], tail_v.unwrap_or(veh_v.end));
                let into = cx.arc(veh_r.cost_class, sol[r][i], tail_v.unwrap_or(veh_r.end))
                    + cx.arc(veh_v.cost_class, sol[v][j], tail_r.unwrap_or(veh_v.end));
                if into - out >= best.as_ref().map_or(0, |b| b.0) {
                    continue;
                }
                let new_r: Vec<NodeId> = sol[r][..=i]
                    .iter()
                    .chain(&sol[v][j + 1..])
                    .copied()
                    .collect();
                let new_v: Vec<NodeId> = sol[v][..=j]
                    .iter()
                    .chain(&sol[r][i + 1..])
                    .copied()
                    .collect();
                let new = match cx.eval(&new_r, VehicleId(r as u32)) {
                    Some(a) => cx.eval(&new_v, VehicleId(v as u32)).map(|b| (a, b)),
                    None => None,
                };
                if let Some((a, b)) = new
                    && a + b < cost[r] + cost[v]
                {
                    best = Some((into - out, v, new_r, new_v, a, b));
                }
            }
        }
    }
    let (_, v, new_r, new_v, a, b) = best?;
    sol[r] = new_r;
    sol[v] = new_v;
    cost[r] = a;
    cost[v] = b;
    Some(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelBuilder;

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
        let mut cx = Search::new(&m);
        let mut cost: Vec<Cost> = (0..2)
            .map(|v| cx.eval(&sol[v], VehicleId(v as u32)).unwrap())
            .collect();
        assert_eq!(cost.iter().sum::<Cost>(), 80);

        let v = try_two_opt_star(&mut cx, &mut sol, &mut cost, 0);
        assert_eq!(v, Some(1));
        assert_eq!(
            sol,
            vec![vec![NodeId(1), NodeId(4)], vec![NodeId(2), NodeId(3)]]
        );
        assert_eq!(cost.iter().sum::<Cost>(), 68);
        // A local optimum of this neighborhood: a second call finds nothing.
        assert_eq!(try_two_opt_star(&mut cx, &mut sol, &mut cost, 0), None);
    }
}
