use super::{RouteEval, Scratch, candidate_vehicles, with_front};
use crate::eval::Routes;
use crate::model::Model;
use crate::types::{Cost, NodeId, VehicleId};

/// Move `u` out of route `r` to its first improving position anywhere,
/// including back into `r`. Returns the receiving vehicle.
pub(super) fn try_relocate(
    m: &Model,
    sol: &mut Routes,
    eval: &impl RouteEval,
    cost: &mut [Cost],
    u: NodeId,
    r: usize,
    sx: &mut Scratch,
) -> Option<usize> {
    let at = sol[r].iter().position(|&x| x == u)?;
    sx.without.clone_from(&sol[r]);
    sx.without.remove(at);
    let without_cost = eval(m, &sx.without, VehicleId(r as u32))?;

    candidate_vehicles(m, sol, &mut sx.vehicles);
    for &v in sx.vehicles.iter() {
        let base = if v == r { &sx.without } else { &sol[v] };
        with_front(base, u, &mut sx.candidate);
        for pos in 0..=base.len() {
            if pos > 0 {
                sx.candidate.swap(pos - 1, pos);
            }
            let Some(c) = eval(m, &sx.candidate, VehicleId(v as u32)) else {
                continue;
            };
            let delta = if v == r {
                c - cost[r]
            } else {
                (without_cost - cost[r]) + (c - cost[v])
            };
            if delta < 0 {
                if v == r {
                    sol[r].clone_from(&sx.candidate);
                    cost[r] = c;
                } else {
                    sol[r].clone_from(&sx.without);
                    cost[r] = without_cost;
                    sol[v].clone_from(&sx.candidate);
                    cost[v] = c;
                }
                return Some(v);
            }
        }
    }
    None
}

/// Trade `u` for a customer on another route, first improving pair wins.
/// Returns the other vehicle.
///
/// Relocate cannot reach these moves when both routes are near capacity: moving
/// a node either way overflows, and only an even trade fits. On the X set that
/// is most of the search space, since the reference fleet is sized tight.
pub(super) fn try_swap(
    m: &Model,
    sol: &mut Routes,
    eval: &impl RouteEval,
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
            // Not `zip`: it evaluates route `v` even when `r` is infeasible.
            #[allow(clippy::manual_option_zip)]
            let new = eval(m, &sol[r], VehicleId(r as u32))
                .and_then(|a| eval(m, &sol[v], VehicleId(v as u32)).map(|b| (a, b)));
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
pub(super) fn try_two_opt(
    m: &Model,
    sol: &mut Routes,
    eval: &impl RouteEval,
    cost: &mut [Cost],
    r: usize,
) -> bool {
    let len = sol[r].len();
    for i in 0..len {
        for j in i + 1..len {
            sol[r][i..=j].reverse();
            match eval(m, &sol[r], VehicleId(r as u32)) {
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

/// A candidate 2-opt* move: arc-cost gain, the other vehicle, the two
/// rebuilt routes, and their costs under `eval`.
type TwoOptStarMove = (Cost, usize, Vec<NodeId>, Vec<NodeId>, Cost, Cost);

/// Inter-route 2-opt*: cut one arc in route `r` and one in another route,
/// trade the tails. Returns the other vehicle.
///
/// This is the operator that changes the customer-to-route partition in
/// chunks: relocate and swap shift one customer at a time, which cannot undo
/// a bad layout once routes fill up. The delta is O(1) arc arithmetic — only
/// the two cut arcs and the two reconnecting arcs change — and `eval`
/// runs only on improving candidates, to confirm capacity.
///
/// Best-improvement, unlike the cheaper operators: a tail swap commits many
/// customers at once, so taking the first improving cut drags the descent
/// into noticeably worse local optima (measured on X-n143-k7).
///
/// The delta ranks candidates by true arc cost even under guided local
/// search; penalties only steer which candidates `eval` accepts.
pub(super) fn try_two_opt_star(
    m: &Model,
    sol: &mut Routes,
    eval: &impl RouteEval,
    cost: &mut [Cost],
    r: usize,
) -> Option<usize> {
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
                let out = m.eval(veh_r.cost_class, sol[r][i], tail_r.unwrap_or(veh_r.end))
                    + m.eval(veh_v.cost_class, sol[v][j], tail_v.unwrap_or(veh_v.end));
                let into = m.eval(veh_r.cost_class, sol[r][i], tail_v.unwrap_or(veh_r.end))
                    + m.eval(veh_v.cost_class, sol[v][j], tail_r.unwrap_or(veh_v.end));
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
                #[allow(clippy::manual_option_zip)]
                let new = eval(m, &new_r, VehicleId(r as u32))
                    .and_then(|a| eval(m, &new_v, VehicleId(v as u32)).map(|b| (a, b)));
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
