# Examples

Each example is self-contained and runs against the volare library itself:

```sh
cargo run --release --example <name>
```

| Example | What it shows |
| --- | --- |
| [`simple_cvrp`](simple_cvrp.rs) | Model a small CVRP from coordinates, solve it, print the routes and total cost |
| [`forbidden_nodes`](forbidden_nodes.rs) | Block a vehicle from serving a customer with `forbid`, and see the solver route around it |
