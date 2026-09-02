# volare

[![CI](https://github.com/thiagomoretto/volare/actions/workflows/ci.yml/badge.svg)](https://github.com/thiagomoretto/volare/actions/workflows/ci.yml)

Vehicle routing solver in Rust, with no dependencies.

Describe a fleet, a set of stops and the limits each vehicle has to respect.
volare builds a first solution with cheapest insertion, then improves it with
local search. Readers for CVRPLIB and Solomon VRPTW files and a benchmark
runner are included.

> Early days. CVRP, hard time windows, optional nodes, per-vehicle node
> exclusion and in-route ordering work today. No soft windows, pickup and
> delivery or multi depot yet.

## Install

```toml
[dependencies]
volare = "0.2"
```

Requires Rust 1.85 or later. To build from source:

```sh
git clone https://github.com/thiagomoretto/volare.git
cd volare
cargo build --release
```

## Usage

```rust
use volare::{Construct, Improve, ModelBuilder, NodeId, solve};

let coords: [(f64, f64); 5] = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0), (40.0, 0.0)];

let mut b = ModelBuilder::new(coords.len());

let cost = b.cost_class(move |from, to| {
    let (p, q) = (coords[from.index()], coords[to.index()]);
    (p.0 - q.0).hypot(p.1 - q.1).round() as i64
});

// Two vehicles, both starting and ending at node 0. Add vehicles before
// dimensions: cumul limits are indexed by vehicle.
b.vehicle(NodeId(0), NodeId(0), cost);
b.vehicle(NodeId(0), NodeId(0), cost);

// One unit of demand per stop, three units of room per vehicle.
b.dimension("demand", |_from, to| if to == NodeId(0) { 0 } else { 1 }, vec![3, 3]);

let model = b.build();
let sol = solve(&model, Construct::CheapestInsertion, Improve::Gls { iters: 200 });

println!("{:?} costs {}", sol.routes, sol.cost);
```

Arc costs are closures, so distances can come from coordinates, a precomputed
matrix or a live routing service. The solver itself never sees a coordinate.

`solve_with` takes the same arguments plus a callback, if you want to watch the
search progress. `search_log()` is a ready made one that prints to stderr.

Full API docs with `cargo doc --open`.

## Constraints

A dimension is a quantity that accumulates along a route: load, time,
distance. `max_cumul` bounds it per vehicle. `cumul_bounds` bounds it per
node, which is what makes a time window.

```rust
// Time: travel on the arc plus service at the node we leave. No vehicle
// limit, so the windows do all the work.
b.dimension("time", move |from, to| travel(from, to) + service(from), vec![i64::MAX; 2]);

// Hard window at node 3. Arriving after 90 is infeasible; arriving before
// 30 makes the vehicle wait until 30.
b.cumul_bounds("time", NodeId(3), 30, 90);

// Vehicle 0 may not serve node 4: no permit, no cold chain, whatever the
// reason. Construction panics if a node ends up forbidden on every vehicle.
b.forbid(VehicleId(0), NodeId(4));

// Node 5 may be left unserved, for 500 added to the total cost. Nodes you
// do not declare stay mandatory.
b.allow_drop(NodeId(5), 500);
```

After the solve, `sol.unserved(&model)` lists the dropped nodes. Their
penalties are already inside `sol.cost`.

The two upper bounds are not the same test. `cumul_bounds` checks the arrival
*before* any wait; `max_cumul` checks it *after*. So waiting counts against
the vehicle's endurance but never against the node's window, and neither
bound expresses the other.

## What is in the box

| | |
| --- | --- |
| Construction | cheapest insertion |
| Improvement | hill climb, or guided local search on top of it |
| Operators | relocate, swap, 2-opt, 2-opt* |
| Constraints | per-vehicle cumul limits, hard windows per node, per-vehicle node exclusion, optional nodes with a drop penalty |
| Input | CVRPLIB `EUC_2D` files, Solomon VRPTW files with the DIMACS metric |

## Bring your own algorithm

`Model` is the problem and never changes during a solve. `Search` is the
attempt at it: one per solve, per thread, holding the objective, the caches
evaluation needs, and the buffer moves are built in.

```rust
let mut cx = Search::new(&model);

// Price any route.
let c = cx.eval(&route, vehicle);

// Price a route with one stretch replaced — the shape every move has.
let insert  = cx.eval_splice(&route, pos..pos, &[node], vehicle);
let remove  = cx.eval_splice(&route, at..at + 1, &[], vehicle);
let segment = cx.eval_splice(&route, i..j, &run, vehicle);

// Commit what the probe accepted, without rebuilding it.
route.clear();
route.extend_from_slice(cx.spliced());

// Price one arc on the objective in force, for operators that rank by
// arc arithmetic rather than by whole routes.
let a = cx.arc(vehicle_cost_class, from, to);
```

`Search` owns the buffers evaluation consumes; the caller owns the ones it
holds *across* an evaluation. Guided local search is built on the same public
surface — `set_lambda` and `penalize` move the objective, while `eval_routes`
still reports true cost.

[`examples/custom_operator.rs`](examples/custom_operator.rs) writes a whole
operator — trading two stops on one route — against nothing else.

## Benchmarks

Mean gap against the best known cost across the 43 CVRPLIB X instances with n up
to 300, measured at commit `b0995dc`:

| Strategy | Mean gap |
| --- | --- |
| cheapest insertion | 25.2% |
| hill climb | 9.5% |
| guided local search, 300 rounds | 5.5% |

```sh
cargo run --release --bin bench -- X-n              # hill climb, about 2 seconds
cargo run --release --bin bench -- X-n --gls=300    # about 3 minutes
```

Drop the `X-n` filter and the run also takes in the five Belgium XL instances,
3k to 11k nodes each, which is a much longer wait.

`baseline.csv` pins the hill climb gap for each X instance, and the runner exits
non-zero if any instance regresses by more than two points.

`--scenario` swaps the plain CVRP for a constrained variant on the same
instances and reports the cost delta against the unconstrained solve:

```sh
cargo run --release --bin bench -- X-n --scenario=forbid  # per-vehicle exclusions
cargo run --release --bin bench -- X-n --scenario=drop    # optional nodes
cargo run --release --bin bench -- X-n --scenario=tw      # hard time windows
```

## Development

```sh
cargo test --release
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

`tests/oracle.rs` and `tests/solomon_oracle.rs` re-evaluate every published best
known solution, CVRP and VRPTW, and check each reproduces the published cost.
That catches the rounding and indexing mistakes that would otherwise surface as
a gap percentage which looks plausible and means nothing. For the VRPTW set it
also pins the window semantics against someone else's answers.

## License

Apache 2.0, see [LICENSE](LICENSE).
