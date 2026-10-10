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

use std::fs;
use std::ops::ControlFlow;
use std::path::Path;
use std::time::{Duration, Instant};

use volare::cvrplib::{Instance, cvrp_model};
use volare::{Construct, Cost, Improve, SearchEvent, search_log, solve_with};

const PATIENCE: Duration = Duration::from_secs(2);
const MIN_GAIN_PCT: Cost = 1;

fn main() {
    let vrp = Path::new(env!("CARGO_MANIFEST_DIR")).join("instances/X-n101-k25.vrp");
    let inst = Instance::parse(&fs::read_to_string(&vrp).unwrap());
    let model = cvrp_model(&inst, inst.coords.len() - 1);

    let started = Instant::now();
    let mut log = search_log();
    // Time and cost of the last improvement big enough to count.
    let mut mark: Option<(Instant, Cost)> = None;
    let monitor = |event| {
        let _ = log(event);
        if let SearchEvent::Best { cost, .. } = event {
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
