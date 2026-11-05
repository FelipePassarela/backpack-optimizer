# Backpack Optimizer

Python package for maximizing the value of items in a backpack subject to a volume
limit. Includes Exhaustive Search, Hill Climbing, and Simulated Annealing optimizers.

Requires Python 3.14 or newer and uv.

## Quick start

```sh
uv sync
uv run backpack-optimizer
```

The example runs repeated optimizations and prints a comparison of the results.

## Compare optimizers

Create an instance, choose the optimizers, and set the number of repetitions:

```python
from backpack_optimizer import Experiment
from backpack_optimizer.models import Backpack, Dimensions, Instance, Item
from backpack_optimizer.optimizers import (
    DumbOptimizer,
    ExhaustiveSearchOptimizer,
    HillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
)

instance = Instance(
    backpack=Backpack(max_volume=5),
    available_items=[
        Item(value=10, dimensions=Dimensions(1, 1, 5)),
        Item(value=6, dimensions=Dimensions(1, 1, 3)),
        Item(value=5, dimensions=Dimensions(1, 1, 2)),
    ],
)
experiment = Experiment(
    instance=instance,
    optimizers=[
        ExhaustiveSearchOptimizer(instance, max_items=20),
        HillClimbingOptimizer(instance, neighborhood_size=3),
        SimulatedAnnealingOptimizer(instance, cooling_rate=0.95, seed=42),
        DumbOptimizer(instance),
    ],
    n_repeats=30,
)
result = experiment.run()

print(result.to_dataframe())
best_solution = result.best_run.best_solution
```

The comparison summarizes solution quality, consistency, and execution time.
Quality statistics consider valid solutions only. `best_run` identifies the
optimizer with the highest mean value, using execution time to break ties.

## Parameter validation

Optimizers and `Experiment` validate scalar parameters with Pydantic at
construction and when public parameter attributes are assigned new values.
Invalid parameters raise `pydantic.ValidationError` (a subclass of `ValueError`);
a rejected assignment leaves the previous configuration unchanged.

| Component | Parameter rules |
| --- | --- |
| Exhaustive Search | `max_items` is an integer >= 0 (default: 20). |
| Hill Climbing | `neighborhood_size` is an integer >= 0 (default: 3); zero produces an empty neighborhood. |
| Simulated Annealing | Temperatures are finite and positive, with `initial_temperature > minimum_temperature`; `cooling_rate` is finite and strictly between 0 and 1; iteration counts are positive integers; `seed` is an integer or `None`. |
| Experiment | `n_repeats` is a positive integer (default: 30). |

Integer parameters reject booleans, floats, and strings. Temperature and cooling
parameters accept integers and floats, rejecting booleans, strings, NaN, and
infinity. Instance checks, such as item availability and matching optimizer
instances, remain separate from parameter validation. Updating search parameters
does not reset the Simulated Annealing random generator.

## Exact reference for small instances

`ExhaustiveSearchOptimizer` checks all `2**n` subsets, including the empty
solution, and returns a feasible solution with maximum value for the same
instance. It selects items by position, so entries with equal attributes each
remain available once. Values and volumes use the same arithmetic and capacity
check as `Solution`, without rounding or conversion to integers.

Use this optimizer as an exact reference for checking heuristics on small
instances. Its cost is exponential: `O(n * 2**n)` time and `O(n)` auxiliary
memory. Each repetition in an `Experiment` performs the full search again.

`max_items` defaults to **20** and accepts non-negative integers, excluding
booleans. Zero allows only an empty instance. Invalid limits raise
`pydantic.ValidationError`; instances above the limit raise `ValueError` at
construction. The item count is checked again before each search in case the
item list or limit changed. Increase the limit explicitly only
for small instances where the exponential cost is acceptable; a rejected search
never returns a partial result as an optimum. If no subset satisfies the model's
capacity check, `optimize()` raises `ValueError`.

## Development

Run the tests:

```sh
uv run pytest
```

Build the package:

```sh
uv build
```
