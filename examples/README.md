# Examples

Each example is self-contained and runs against the volare library itself:

```sh
cargo run --release --example <name>
```

| Example | What it shows |
| --- | --- |
| [`simple_cvrp`](simple_cvrp.rs) | Model a small CVRP from coordinates, solve it, print the routes and total cost |
| [`forbidden_nodes`](forbidden_nodes.rs) | Block a vehicle from serving a customer with `forbid`, and see the solver route around it |
| [`precedence`](precedence.rs) | Order two stops within a route with `precede`, and see a pair split across vehicles go unordered |
| [`custom_operator`](custom_operator.rs) | Write your own operator — trading two stops on one route — against `Search` alone |
