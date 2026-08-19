use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::model::Model;
use crate::types::{Cost, NodeId, VehicleId};

/// Routes indexed by vehicle. Each holds the visits *between* the vehicle's
/// start and end nodes, terminals excluded.
pub type Routes = Vec<Vec<NodeId>>;

/// Arc penalties for guided local search, and the weight one penalty carries.
///
/// Owned by the search that is running, never by the `Model`: the model stays a
/// pure description of the problem, shareable by `&` across concurrent solves.
/// `Penalties::NONE` is the true-cost evaluation every other caller wants, and
/// a penalty cannot outlive the search that made it.
pub struct Penalties {
    lambda: Cost,
    map: HashMap<u64, Cost, BuildHasherDefault<ArcHasher>>,
}

impl Penalties {
    /// No penalties and no weight: `eval_route` reports true cost.
    pub const NONE: Penalties = Penalties {
        lambda: 0,
        map: HashMap::with_hasher(BuildHasherDefault::new()),
    };

    /// An empty table whose penalties each cost `lambda`.
    pub fn new(lambda: Cost) -> Self {
        Penalties {
            lambda,
            map: HashMap::default(),
        }
    }

    /// Weight applied to each penalty. Zero skips the lookup in the cost loop.
    #[inline]
    pub fn lambda(&self) -> Cost {
        self.lambda
    }

    /// How many times arc `{a, b}` has been penalized.
    #[inline]
    pub fn arc(&self, a: NodeId, b: NodeId) -> Cost {
        self.map.get(&arc_key(a, b)).copied().unwrap_or(0)
    }

    pub fn penalize(&mut self, a: NodeId, b: NodeId) {
        *self.map.entry(arc_key(a, b)).or_insert(0) += 1;
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

/// Hasher for `Penalties`, and for nothing else.
///
/// Under GLS, `eval_route` reads that map once per arc — it is the hottest line
/// in the project. The default `RandomState` runs SipHash there, which a
/// profile showed costing more than evaluating the arc it was pricing. The keys
/// are packed node pairs, not adversarial input, so one multiply is enough.
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

/// Cost of running `route` on vehicle `v`, or `None` if it is infeasible.
///
/// The central primitive: construction and every operator route through here.
/// Pass `&Penalties::NONE` for true cost; a guided local search passes its own
/// table and gets the penalized cost it is descending on.
///
/// Feasibility is a full forward pass, O(route length), recomputed on every
/// call. That is the deliberate ceiling; to lift it, cache cumul prefixes
/// per route.
pub fn eval_route(m: &Model, p: &Penalties, route: &[NodeId], v: VehicleId) -> Option<Cost> {
    // ponytail: an unused vehicle is free, it never leaves the depot.
    if route.is_empty() {
        return Some(0);
    }
    let veh = m.vehicle(v);

    for d in m.dimensions() {
        let cap = d.capacity[v.index()];
        let mut cumul = d.start_cumul.max(d.lower_bound[veh.start.index()]);
        if cumul > cap {
            return None;
        }
        let mut prev = veh.start;
        for &node in route.iter().chain(std::iter::once(&veh.end)) {
            cumul = (cumul + m.eval(d.transit, prev, node)).max(d.lower_bound[node.index()]);
            if cumul > cap {
                return None;
            }
            prev = node;
        }
    }

    // Guided local search rides on this loop and nothing else: the penalty is
    // cost-only, so the cumul pass above stays true feasibility. `lambda == 0`
    // is the `Penalties::NONE` case and skips the lookup entirely.
    let lambda = p.lambda();
    let mut cost = 0;
    let mut prev = veh.start;
    for &node in route.iter().chain(std::iter::once(&veh.end)) {
        cost += m.eval(veh.cost_class, prev, node);
        if lambda != 0 {
            cost += lambda * p.arc(prev, node);
        }
        prev = node;
    }
    Some(cost)
}

/// True cost of every route, or `None` if any is infeasible.
///
/// Always true cost: nothing ranks solutions by a penalized number, so this
/// takes no `Penalties` at all.
pub fn eval_routes(m: &Model, sol: &Routes) -> Option<Cost> {
    (0..sol.len()).try_fold(0, |acc, v| {
        Some(acc + eval_route(m, &Penalties::NONE, &sol[v], VehicleId(v as u32))?)
    })
}

/// Every non-terminal node visited exactly once.
pub fn visits_all_nodes(m: &Model, sol: &Routes) -> bool {
    let mut seen = vec![0u32; m.node_count()];
    for route in sol {
        for &n in route {
            seen[n.index()] += 1;
        }
    }
    (0..m.node_count()).all(|i| {
        let expected = if m.is_terminal(NodeId(i as u32)) {
            0
        } else {
            1
        };
        seen[i] == expected
    })
}
