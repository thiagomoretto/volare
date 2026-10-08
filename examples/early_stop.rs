//! Stop a long guided search once it stalls. Run with:
//!
//! ```sh
//! cargo run --release --example early_stop
//! ```
//!
//! The callback handed to `solve_with` sees every `SearchEvent` and can end
//! the search by returning `ControlFlow::Break(())`. The rule below is one
//! choice among many: stop when no improvement of at least `MIN_GAIN_PCT`
//! has landed for `PATIENCE`. Set `MIN_GAIN_PCT` to 0 to count any
//! improvement at all.

use std::ops::ControlFlow;
use std::time::{Duration, Instant};

use volare::{Construct, Cost, Improve, ModelBuilder, NodeId, SearchEvent, search_log, solve_with};

const PATIENCE: Duration = Duration::from_secs(2);
const MIN_GAIN_PCT: Cost = 1;

fn main() {
    // Random stops on a 1000 x 1000 square, from a fixed seed.
    let mut seed: u64 = 42;
    let mut next = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 33) as f64 / (1u64 << 31) as f64 * 1000.0
    };
    let n = 201;
    let coords: Vec<(f64, f64)> = (0..n).map(|_| (next(), next())).collect();
    let demands: Vec<i64> = (0..n)
        .map(|i| if i == 0 { 0 } else { 1 + i as i64 % 9 })
        .collect();

    let mut b = ModelBuilder::new(n);
    let cost = b.cost_class(move |from, to| {
        let (p, q) = (coords[from.index()], coords[to.index()]);
        (p.0 - q.0).hypot(p.1 - q.1).round() as i64
    });
    let fleet = 40;
    for _ in 0..fleet {
        b.vehicle(NodeId(0), NodeId(0), cost);
    }
    b.dimension(
        "demand",
        move |_from, to| demands[to.index()],
        vec![50; fleet],
    );
    let model = b.build();

    let started = Instant::now();
    let mut log = search_log();
    // Time and cost of the last improvement big enough to count.
    let mut mark: Option<(Instant, Cost)> = None;
    let monitor = |event| {
        let _ = log(event);
        if let SearchEvent::GuidedBest { cost, .. } = event {
            match mark {
                Some((_, at)) if cost * 100 > at * (100 - MIN_GAIN_PCT) => {}
                _ => mark = Some((Instant::now(), cost)),
            }
        }
        match mark {
            Some((at, _)) if at.elapsed() > PATIENCE => ControlFlow::Break(()),
            _ => ControlFlow::Continue(()),
        }
    };

    // No round limit: the monitor alone decides when to stop.
    let sol = solve_with(
        &model,
        Construct::CheapestInsertion,
        Improve::Gls { iters: usize::MAX },
        monitor,
    );

    println!(
        "stopped after {:.1}s: no gain of {MIN_GAIN_PCT}% or more in {}s",
        started.elapsed().as_secs_f64(),
        PATIENCE.as_secs(),
    );
    println!("total cost: {}", sol.cost);
}
