# volare

[![CI](https://github.com/thiagomoretto/volare/actions/workflows/ci.yml/badge.svg)](https://github.com/thiagomoretto/volare/actions/workflows/ci.yml)

Capacitated vehicle routing solver in Rust, with no dependencies.

Describe a fleet, a set of stops and a capacity per vehicle. volare builds a
first solution with cheapest insertion, then improves it with local search. A
CVRPLIB reader and a benchmark runner are included.

> Early days. Plain CVRP works today. No time windows, pickup and delivery or
> multi depot yet.

## Install

```toml
[dependencies]
volare = "0.1"
```

Requires Rust 1.85 or later. To build from source:

```sh
git clone git@github.com:thiagomoretto/volare.git
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
// dimensions: capacities are indexed by vehicle.
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

## What is in the box

| | |
| --- | --- |
| Construction | cheapest insertion |
| Improvement | hill climb, or guided local search on top of it |
| Operators | relocate, swap, 2-opt, 2-opt* |
| Input | CVRPLIB files with the `EUC_2D` metric |

## Benchmarks

Mean gap against the best known cost across the 43 CVRPLIB X instances with n up
to 300, measured at commit `2836fcb`:

| Strategy | Mean gap |
| --- | --- |
| cheapest insertion | 25.2% |
| hill climb | 9.5% |
| guided local search, 300 rounds | 5.4% |

```sh
cargo run --release --bin bench                 # hill climb, about 4 seconds
cargo run --release --bin bench -- --gls=300    # about 3 minutes
```

`baseline.csv` pins the hill climb gap for each instance, and the runner exits
non-zero if any instance regresses by more than two points.

## Development

```sh
cargo test --release
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

`tests/oracle.rs` re-evaluates every published best known solution and checks it
reproduces the published cost. That catches the rounding and indexing mistakes
that would otherwise surface as a gap percentage which looks plausible and means
nothing.

## License

Apache 2.0, see [LICENSE](LICENSE).
