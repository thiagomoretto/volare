//! The search context: everything a running solve owns, and the vocabulary
//! its algorithms speak.
//!
//! [`Model`] is the problem; `Search` is the attempt at it. One `Search` per
//! solve, per thread. It holds the objective the operators are minimizing,
//! the caches evaluation needs, and the buffer moves are built in.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::ops::Range;

use crate::model::Model;
use crate::types::{Cost, NodeId, VehicleId};

/// A solve in progress, bound to the model it is solving.
///
/// The borrow is the contract: buffers here are sized to one model, so a
/// `Search` cannot be pointed at another. Every field is owned by the call,
/// so one model still backs any number of concurrent solves — one `Search`
/// each.
///
/// # What algorithms get
///
/// [`eval`](Self::eval) prices any route. [`eval_splice`](Self::eval_splice)
/// prices a route with one stretch replaced, which is the shape every move
/// has: an insertion replaces nothing, a removal replaces a stretch with
/// nothing, a segment move and a tail trade replace one stretch with another.
/// [`arc`](Self::arc) prices a single arc, so an operator that ranks
/// candidates by arc arithmetic ranks them on the same objective acceptance
/// uses.
///
/// # Who owns which buffer
///
/// `Search` owns the buffers evaluation consumes; the caller owns the ones it
/// holds *across* an evaluation. The borrow checker draws that line, not
/// taste: a slice borrowed out of `Search` cannot be handed back to a method
/// that takes `&mut self`.
pub struct Search<'m> {
    m: &'m Model,
    /// Weight on one penalty. No penalties recorded is true cost.
    lambda: Cost,
    penalties: HashMap<u64, Cost, BuildHasherDefault<ArcHasher>>,
    /// Where each node sits in the route being evaluated, valid only for
    /// nodes stamped with the current `epoch`. Empty unless the model orders
    /// anything.
    pos: Vec<u32>,
    stamp: Vec<u32>,
    epoch: u32,
    /// The route `eval_splice` builds. Private: handing it out is what would
    /// make the `&mut self` methods uncallable.
    buf: Vec<NodeId>,
    has_precedence: bool,
}

/// Breaks the build if a `Search` ever stops being movable between threads.
/// Multi-start hands one per worker; a stray `Rc` would take that away
/// silently.
const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<Search<'static>>();
};

impl<'m> Search<'m> {
    /// Allocates nothing the model does not ask for: an unordered model never
    /// pays for the precedence index. Cheap enough to build one per restart.
    pub fn new(m: &'m Model) -> Self {
        let has_precedence = m.has_precedence();
        let index_len = if has_precedence { m.node_count() } else { 0 };
        Search {
            m,
            lambda: 0,
            penalties: HashMap::default(),
            pos: vec![0; index_len],
            stamp: vec![0; index_len],
            epoch: 0,
            buf: Vec::new(),
            has_precedence,
        }
    }

    /// Borrowed from the model, not from `self`, so a vehicle or dimension
    /// read from here stays live across an `&mut self` call.
    #[inline]
    pub fn model(&self) -> &'m Model {
        self.m
    }

    /// Cost of `route` on `v` under the current objective, or `None` if it is
    /// infeasible.
    ///
    /// The general form: the caller owns the slice, so any move shape works.
    /// Feasibility is a full forward pass, as in [`crate::eval::eval_route`].
    pub fn eval(&mut self, route: &[NodeId], v: VehicleId) -> Option<Cost> {
        if route.is_empty() {
            return Some(0);
        }
        let m = self.m;
        let veh = m.vehicle(v);

        if !veh.forbidden.is_empty() && route.iter().any(|&n| veh.forbids(n)) {
            return None;
        }

        // A dropped node's window or ordering must not block dropping it.
        if m.unserved_vehicle() != Some(v) {
            if self.has_precedence && !self.precedence_holds(route) {
                return None;
            }
            for d in m.dimensions() {
                let cap = d.max_cumul[v.index()];
                let mut cumul = d.start_cumul.max(d.lower_bound[veh.start.index()]);
                if cumul > cap {
                    return None;
                }
                let mut prev = veh.start;
                for &node in route.iter().chain(std::iter::once(&veh.end)) {
                    // Late is infeasible before the clamp; early waits via the clamp.
                    let arrive = cumul + m.eval(d.transit, prev, node);
                    if arrive > d.upper_bound[node.index()] {
                        return None;
                    }
                    cumul = arrive.max(d.lower_bound[node.index()]);
                    if cumul > cap {
                        return None;
                    }
                    prev = node;
                }
            }
        }

        // One walk, not two: the penalty rides along with the arc it is on.
        let mut cost = 0;
        let mut prev = veh.start;
        for &node in route.iter().chain(std::iter::once(&veh.end)) {
            cost += self.arc(veh.cost_class, prev, node);
            prev = node;
        }
        Some(cost)
    }

    /// [`eval`](Self::eval) of `route` with `route[range]` replaced by `repl`,
    /// without the caller allocating.
    ///
    /// Every move is this shape. Insert at `pos` is `pos..pos` with one node,
    /// remove at `at` is `at..at + 1` with nothing, an or-opt segment move is
    /// a stretch either way, a 2-opt\* tail trade is the tail of one route
    /// replaced by the tail of another. `repl` may point into `route` itself.
    ///
    /// This is also the only signature a future incremental evaluation can
    /// hide behind: the prefix before `range` and the suffix after it are
    /// exactly what a cached route makes free.
    pub fn eval_splice(
        &mut self,
        route: &[NodeId],
        range: Range<usize>,
        repl: &[NodeId],
        v: VehicleId,
    ) -> Option<Cost> {
        // Out of `self` and back: `eval` wants `&mut self`, which it cannot
        // have while the buffer it is reading is still a field.
        let mut buf = std::mem::take(&mut self.buf);
        buf.clear();
        buf.extend_from_slice(&route[..range.start]);
        buf.extend_from_slice(repl);
        buf.extend_from_slice(&route[range.end..]);
        let cost = self.eval(&buf, v);
        self.buf = buf;
        cost
    }

    /// The route the last [`eval_splice`](Self::eval_splice) priced.
    ///
    /// Probe, then commit what the probe accepted without rebuilding it. Only
    /// valid until the next `eval_splice`.
    #[inline]
    pub fn spliced(&self) -> &[NodeId] {
        &self.buf
    }

    /// Cost of one arc under the current objective.
    ///
    /// An operator whose delta is arc arithmetic must price its arcs here.
    /// Reading `Model::eval` directly ranks candidates on true cost while
    /// acceptance runs on the penalized one, and the two then disagree about
    /// which move is best.
    #[inline]
    pub fn arc(&self, class: usize, a: NodeId, b: NodeId) -> Cost {
        let base = self.m.eval(class, a, b);
        if self.penalties.is_empty() {
            return base;
        }
        base + self.lambda * self.penalty(a, b)
    }

    /// Weight one penalty carries, in arc-cost units.
    #[inline]
    pub fn lambda(&self) -> Cost {
        self.lambda
    }

    pub fn set_lambda(&mut self, lambda: Cost) {
        self.lambda = lambda;
    }

    /// How many times arc `{a, b}` has been penalized.
    #[inline]
    pub fn penalty(&self, a: NodeId, b: NodeId) -> Cost {
        self.penalties.get(&arc_key(a, b)).copied().unwrap_or(0)
    }

    /// Make arc `{a, b}` look worse to every later evaluation. The objective
    /// moves; true cost, from [`crate::eval::eval_routes`], does not.
    pub fn penalize(&mut self, a: NodeId, b: NodeId) {
        *self.penalties.entry(arc_key(a, b)).or_insert(0) += 1;
    }

    /// Stamped positions: no clearing between routes, so a lookup is one
    /// compare instead of a scan of everything already placed.
    fn precedence_holds(&mut self, route: &[NodeId]) -> bool {
        if self.epoch == u32::MAX {
            self.stamp.fill(0);
            self.epoch = 0;
        }
        self.epoch += 1;
        let epoch = self.epoch;
        for (i, &n) in route.iter().enumerate() {
            self.pos[n.index()] = i as u32;
            self.stamp[n.index()] = epoch;
        }
        route.iter().enumerate().all(|(i, &n)| {
            self.m
                .successors(n)
                .iter()
                .all(|s| self.stamp[s.index()] != epoch || self.pos[s.index()] as usize > i)
        })
    }
}

/// Arcs are penalized as unordered pairs, packed into one `u64`. 2-opt reverses
/// whole segments, so a directed key would make a reversal silently change the
/// penalty it carries.
#[inline]
fn arc_key(a: NodeId, b: NodeId) -> u64 {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    (lo.0 as u64) << 32 | hi.0 as u64
}

/// Hasher for the penalty table, and for nothing else.
///
/// That map is read once per arc — the hottest line in the project. The
/// default `RandomState` runs SipHash there, which a profile showed costing
/// more than evaluating the arc it was pricing. The keys are packed node
/// pairs, not adversarial input, so one multiply is enough.
///
/// The map is only ever read by key, never iterated, so nothing downstream can
/// observe the change in hash order.
#[derive(Default)]
pub struct ArcHasher(u64);

impl Hasher for ArcHasher {
    #[inline]
    fn write_u64(&mut self, n: u64) {
        // Fibonacci hashing puts the entropy in the high bits; hashbrown picks
        // its bucket from the low ones, so fold the high half back down.
        let h = n.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        self.0 = h ^ (h >> 32);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, _: &[u8]) {
        unreachable!("ArcHasher only ever hashes a packed arc key");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::eval_route;
    use crate::model::ModelBuilder;

    /// Capacity, a window, a forbid and an ordering, so one model exercises
    /// every arm of the feasibility pass.
    fn mixed_model() -> Model {
        let dist = |a: NodeId, b: NodeId| (a.0 as i64 - b.0 as i64).abs() * 10;
        let mut b = ModelBuilder::new(9);
        let cost = b.cost_class(dist);
        let v0 = b.vehicle(NodeId(0), NodeId(0), cost);
        b.vehicle(NodeId(0), NodeId(0), cost);
        b.forbid(v0, NodeId(7));
        b.dimension(
            "load",
            |_from, to| if to == NodeId(0) { 0 } else { 1 },
            vec![5, 5],
        );
        b.dimension("time", dist, vec![400, 400]);
        b.cumul_bounds("time", NodeId(4), 30, 120);
        b.precede(NodeId(2), NodeId(6));
        b.precede(NodeId(3), NodeId(1));
        b.build()
    }

    /// `Search::eval` caches what `eval_route` recomputes — stamped positions
    /// instead of a rescan — so the two are separate implementations of one
    /// answer. Only a differential check keeps them from drifting apart.
    #[test]
    fn search_eval_agrees_with_the_stateless_reference() {
        let m = mixed_model();
        let mut cx = Search::new(&m);
        let mut state = 0x243F_6A88_85A3_08D3u64;
        let mut rng = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let mut route = Vec::new();
        for _ in 0..20_000 {
            route.clear();
            for n in 1..m.node_count() as u32 {
                if rng() & 1 == 0 {
                    route.push(NodeId(n));
                }
            }
            // Shuffle, so orderings that violate precedence show up too.
            for i in (1..route.len()).rev() {
                route.swap(i, (rng() % (i as u64 + 1)) as usize);
            }
            for v in 0..m.vehicle_count() as u32 {
                let v = VehicleId(v);
                assert_eq!(
                    cx.eval(&route, v),
                    eval_route(&m, &route, v),
                    "route {route:?} on vehicle {}",
                    v.0
                );
            }
        }
    }

    /// A penalized arc costs more, and only through `arc`: an operator reading
    /// `Model::eval` would still see the true cost and rank against it.
    #[test]
    fn penalties_move_the_objective_and_not_the_truth() {
        let m = mixed_model();
        let mut cx = Search::new(&m);
        let route = [NodeId(1), NodeId(2)];
        let v = VehicleId(1);
        let true_cost = cx.eval(&route, v).expect("feasible");

        cx.set_lambda(7);
        cx.penalize(NodeId(0), NodeId(1));
        cx.penalize(NodeId(0), NodeId(1));

        assert_eq!(cx.penalty(NodeId(1), NodeId(0)), 2, "arcs are unordered");
        assert_eq!(cx.eval(&route, v), Some(true_cost + 14));
        assert_eq!(
            eval_route(&m, &route, v),
            Some(true_cost),
            "the stateless reference never sees a penalty"
        );
    }

    /// The splice is the move vocabulary: each shape must equal the route it
    /// describes, and `spliced` must hand that route back for committing.
    #[test]
    fn splice_covers_every_move_shape() {
        let m = mixed_model();
        let mut cx = Search::new(&m);
        let v = VehicleId(1);
        let route = [NodeId(1), NodeId(2), NodeId(3), NodeId(4)];

        let cases: [(Range<usize>, &[NodeId], &[NodeId]); 4] = [
            // insert, remove, segment swap, tail replaced
            (
                2..2,
                &[NodeId(5)],
                &[NodeId(1), NodeId(2), NodeId(5), NodeId(3), NodeId(4)],
            ),
            (1..2, &[], &[NodeId(1), NodeId(3), NodeId(4)]),
            (
                1..3,
                &[NodeId(6), NodeId(5)],
                &[NodeId(1), NodeId(6), NodeId(5), NodeId(4)],
            ),
            (2..4, &[NodeId(8)], &[NodeId(1), NodeId(2), NodeId(8)]),
        ];
        for (range, repl, want) in cases {
            let spliced = cx.eval_splice(&route, range.clone(), repl, v);
            assert_eq!(cx.spliced(), want, "splice {range:?} built the wrong route");
            assert_eq!(spliced, eval_route(&m, want, v), "splice {range:?} cost");
        }
    }
}
