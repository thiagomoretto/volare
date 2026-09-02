use crate::types::{Cost, NodeId, VehicleId};

/// One table, two uses: cost classes and dimension transits.
///
/// `Send + Sync` is what keeps `Model` movable between threads. Without it
/// a caller cannot hand a model to a worker pool or an async task, which is
/// the normal way to serve one.
pub type Evaluator = Box<dyn Fn(NodeId, NodeId) -> i64 + Send + Sync>;

/// Breaks the build if `Model` ever stops being thread-safe. A stray
/// non-`Send` capture in an evaluator is easy to add and invisible until a
/// caller tries to move a model onto a worker thread.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Model>();
};

pub struct Vehicle {
    pub start: NodeId,
    pub end: NodeId,
    pub cost_class: usize,
    /// Empty until the first `forbid`, so the common case costs one
    /// `is_empty` in `eval_route`.
    pub forbidden: Box<[u64]>,
}

impl Vehicle {
    #[inline]
    pub fn forbids(&self, n: NodeId) -> bool {
        self.forbidden
            .get(n.index() / 64)
            .is_some_and(|w| w >> (n.index() % 64) & 1 == 1)
    }
}

/// A quantity accumulated along a route by a transit, bounded per vehicle
/// and per node.
///
/// [`max_cumul`](Self::max_cumul) bounds what a vehicle can carry or endure.
/// [`lower_bound`](Self::lower_bound) and [`upper_bound`](Self::upper_bound)
/// form a window per node: an early arrival waits, a late one is infeasible.
/// The two upper bounds check at different moments. `upper_bound` checks the
/// arrival before the wait, `max_cumul` after it, so waiting counts against
/// the vehicle but not against the node. Neither expresses the other. With
/// negative transits (pickup and delivery) `max_cumul` binds at the mid-route
/// peak, where an end-node bound would not.
pub struct Dimension {
    pub name: String,
    /// Index into the evaluator table.
    pub transit: usize,
    pub max_cumul: Vec<i64>,
    pub start_cumul: i64,
    pub lower_bound: Vec<i64>,
    pub upper_bound: Vec<i64>,
}

/// A solved-for-once description of the problem: nodes, arc costs, dimensions,
/// fleet. Immutable for the whole life of a search, so one model serves any
/// number of concurrent solves by `&`. Search state — including guided local
/// search penalties — lives in the search, never here.
pub struct Model {
    node_count: usize,
    evaluators: Vec<Evaluator>,
    dimensions: Vec<Dimension>,
    vehicles: Vec<Vehicle>,
    unserved: Option<VehicleId>,
    /// Successors by node; empty when no ordering was declared.
    precedence: Vec<Vec<NodeId>>,
}

impl Model {
    #[inline]
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    #[inline]
    pub fn vehicle_count(&self) -> usize {
        self.vehicles.len()
    }

    #[inline]
    pub fn vehicle(&self, v: VehicleId) -> &Vehicle {
        &self.vehicles[v.index()]
    }

    #[inline]
    pub fn dimensions(&self) -> &[Dimension] {
        &self.dimensions
    }

    /// The gate in `eval_route`: a model without ordering pays one
    /// `is_empty` per call and never walks the route twice.
    #[inline]
    pub(crate) fn has_precedence(&self) -> bool {
        !self.precedence.is_empty()
    }

    /// Nodes that must be served after `n` when a route holds both.
    #[inline]
    pub fn successors(&self, n: NodeId) -> &[NodeId] {
        self.precedence.get(n.index()).map_or(&[], Vec::as_slice)
    }

    #[inline]
    pub fn eval(&self, idx: usize, from: NodeId, to: NodeId) -> i64 {
        (self.evaluators[idx])(from, to)
    }

    pub fn is_terminal(&self, n: NodeId) -> bool {
        self.vehicles.iter().any(|v| v.start == n || v.end == n)
    }

    /// The vehicle collecting dropped nodes: always the last one, its route
    /// is the unserved set.
    #[inline]
    pub fn unserved_vehicle(&self) -> Option<VehicleId> {
        self.unserved
    }
}

#[derive(Default)]
pub struct ModelBuilder {
    node_count: usize,
    evaluators: Vec<Evaluator>,
    dimensions: Vec<Dimension>,
    vehicles: Vec<Vehicle>,
    drops: Vec<(NodeId, Cost)>,
    precedence: Vec<Vec<NodeId>>,
}

impl ModelBuilder {
    pub fn new(node_count: usize) -> Self {
        ModelBuilder {
            node_count,
            ..Default::default()
        }
    }

    /// The returned index is the cost class; vehicles sharing a cost model
    /// share one closure.
    pub fn cost_class(
        &mut self,
        f: impl Fn(NodeId, NodeId) -> i64 + Send + Sync + 'static,
    ) -> usize {
        self.evaluators.push(Box::new(f));
        self.evaluators.len() - 1
    }

    pub fn vehicle(&mut self, start: NodeId, end: NodeId, cost_class: usize) -> VehicleId {
        assert!(cost_class < self.evaluators.len(), "unknown cost class");
        self.vehicles.push(Vehicle {
            start,
            end,
            cost_class,
            forbidden: Box::default(),
        });
        VehicleId(self.vehicles.len() as u32 - 1)
    }

    /// `max_cumul` is per vehicle, so add vehicles first.
    pub fn dimension(
        &mut self,
        name: &str,
        transit: impl Fn(NodeId, NodeId) -> i64 + Send + Sync + 'static,
        max_cumul: Vec<i64>,
    ) -> &mut Self {
        self.evaluators.push(Box::new(transit));
        self.dimensions.push(Dimension {
            name: name.to_string(),
            transit: self.evaluators.len() - 1,
            max_cumul,
            start_cumul: 0,
            lower_bound: vec![0; self.node_count],
            upper_bound: vec![i64::MAX; self.node_count],
        });
        self
    }

    /// Set the window `[lb, ub]` on the cumul of dimension `name` at node `n`.
    pub fn cumul_bounds(&mut self, name: &str, n: NodeId, lb: i64, ub: i64) {
        assert!(n.index() < self.node_count, "node out of range");
        let d = self
            .dimensions
            .iter_mut()
            .find(|d| d.name == name)
            .expect("unknown dimension");
        d.lower_bound[n.index()] = lb;
        d.upper_bound[n.index()] = ub;
    }

    /// Construction fails loudly if a node ends up forbidden on every
    /// vehicle.
    pub fn forbid(&mut self, v: VehicleId, n: NodeId) {
        assert!(n.index() < self.node_count, "forbidden node out of range");
        let words = self.node_count.div_ceil(64);
        let veh = &mut self.vehicles[v.index()];
        if veh.forbidden.is_empty() {
            veh.forbidden = vec![0; words].into_boxed_slice();
        }
        veh.forbidden[n.index() / 64] |= 1 << (n.index() % 64);
    }

    /// Node `n` may be left unserved for `penalty`, added to the total cost.
    /// Undeclared nodes stay mandatory.
    pub fn allow_drop(&mut self, n: NodeId, penalty: Cost) {
        assert!(n.index() < self.node_count, "dropped node out of range");
        self.drops.push((n, penalty));
    }

    /// Serve `before` ahead of `after` on any route that holds both.
    ///
    /// Ordering only: a pair split across two vehicles is unconstrained.
    /// Pinning both to one vehicle is a separate constraint, and the two
    /// together are what a pickup-and-delivery pair needs.
    ///
    /// Cycles are legal — `a` before `b` before `a` just means the two never
    /// share a route — so nothing is checked for one here.
    pub fn precede(&mut self, before: NodeId, after: NodeId) {
        assert!(
            before.index() < self.node_count && after.index() < self.node_count,
            "precedence node out of range"
        );
        assert!(before != after, "node {} cannot precede itself", before.0);
        if self.precedence.is_empty() {
            self.precedence = vec![Vec::new(); self.node_count];
        }
        self.precedence[before.index()].push(after);
    }

    pub fn build(mut self) -> Model {
        assert!(!self.vehicles.is_empty(), "model has no vehicles");

        // Charging the penalty on the incoming arc makes the sink's route
        // cost sum(penalties) under any permutation.
        let unserved = if self.drops.is_empty() {
            None
        } else {
            let mut penalty = vec![0; self.node_count];
            let mut forbidden = vec![!0u64; self.node_count.div_ceil(64)];
            for (n, p) in self.drops.drain(..) {
                penalty[n.index()] = p;
                forbidden[n.index() / 64] &= !(1 << (n.index() % 64));
            }
            let depot = self.vehicles[0].start;
            let cost_class = self.cost_class(
                move |_from, to| {
                    if to == depot { 0 } else { penalty[to.index()] }
                },
            );
            let id = VehicleId(self.vehicles.len() as u32);
            self.vehicles.push(Vehicle {
                start: depot,
                end: depot,
                cost_class,
                forbidden: forbidden.into_boxed_slice(),
            });
            for d in &mut self.dimensions {
                d.max_cumul.push(i64::MAX);
            }
            Some(id)
        };

        for d in &self.dimensions {
            assert_eq!(
                d.max_cumul.len(),
                self.vehicles.len(),
                "dimension `{}` has {} cumul limits for {} vehicles — add vehicles before dimensions",
                d.name,
                d.max_cumul.len(),
                self.vehicles.len()
            );
        }
        let model = Model {
            node_count: self.node_count,
            evaluators: self.evaluators,
            dimensions: self.dimensions,
            vehicles: self.vehicles,
            unserved,
            precedence: self.precedence,
        };

        // A terminal never appears in a route, so such a pair could never fire.
        for (i, succ) in model.precedence.iter().enumerate() {
            if succ.is_empty() {
                continue;
            }
            for &n in std::iter::once(&NodeId(i as u32)).chain(succ) {
                assert!(
                    !model.is_terminal(n),
                    "precedence on node {} is vacuous: it is a vehicle terminal",
                    n.0
                );
            }
        }
        model
    }
}
