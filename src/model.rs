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

pub struct Dimension {
    pub name: String,
    /// Index into the evaluator table.
    pub transit: usize,
    /// Upper bound on the cumul, per vehicle: load capacity, max route
    /// distance, max duration — whatever the transit accumulates.
    pub max_cumul: Vec<i64>,
    pub start_cumul: i64,
    /// Per-node lower bound on the cumul. All zeros until time windows exist;
    /// the `max` in the cumul pass is what makes filling this in enough.
    pub lower_bound: Vec<i64>,
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
    /// Vehicle collecting dropped nodes, if any `allow_drop` was declared.
    /// Its route *is* the unserved set, so the solver's core invariant —
    /// every node routed exactly once — survives drops untouched.
    unserved: Option<VehicleId>,
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

    #[inline]
    pub fn eval(&self, idx: usize, from: NodeId, to: NodeId) -> i64 {
        (self.evaluators[idx])(from, to)
    }

    pub fn is_terminal(&self, n: NodeId) -> bool {
        self.vehicles.iter().any(|v| v.start == n || v.end == n)
    }

    /// The vehicle collecting dropped nodes; always the last one, and its
    /// route is exactly the unserved set. `None` unless the builder saw an
    /// `allow_drop`.
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
        });
        self
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

    /// Node `n` may be left unserved; doing so adds `penalty` to the total
    /// cost. Nodes never declared stay mandatory: the unserved vehicle is
    /// built forbidden on everything and each `allow_drop` clears one bit.
    pub fn allow_drop(&mut self, n: NodeId, penalty: Cost) {
        assert!(n.index() < self.node_count, "dropped node out of range");
        self.drops.push((n, penalty));
    }

    pub fn build(mut self) -> Model {
        assert!(!self.vehicles.is_empty(), "model has no vehicles");

        // The unserved sink materializes here, not in `allow_drop`: the
        // penalty closure captures a plain `Vec` (no shared state on the
        // eval hot path), and cumul limits are padded after every dimension
        // exists, so call order never matters. Its arc cost charges each
        // node's penalty on the incoming arc, so the route costs the same
        // `sum(penalties)` under any permutation.
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
        Model {
            node_count: self.node_count,
            evaluators: self.evaluators,
            dimensions: self.dimensions,
            vehicles: self.vehicles,
            unserved,
        }
    }
}
