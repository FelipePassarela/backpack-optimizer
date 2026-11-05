# Backpack Optimizer

Python package for maximizing the value of items in a backpack subject to a volume
limit. Includes Hill Climbing and Simulated Annealing optimizers.

Requires Python 3.14 or newer and uv.

## Development

Install the package and development dependencies:

```sh
uv sync
```

Run the example using either entry point:

```sh
uv run backpack-optimizer
uv run python -m backpack_optimizer
```

Run the tests:

```sh
uv run pytest
```

Build the source distribution and wheel in `dist/`:

```sh
uv build
```

## Experiments

Compare configured optimizers on the same `Instance`:

```python
from backpack_optimizer import Experiment
from backpack_optimizer.models import Backpack, Dimensions, Instance, Item
from backpack_optimizer.optimizers import (
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
        HillClimbingOptimizer(instance, neighborhood_size=3),
        SimulatedAnnealingOptimizer(instance, max_iterations=100, seed=42),
    ],
    n_repeats=30,
)
result = experiment.run()

print(result.to_dataframe().to_string(index=False, float_format="{:.4f}".format))
print(result.executions_to_dataframe())
best_solution = result.best_run.best_solution
```

Each optimizer entry has a numeric ID, so different configurations of the same
algorithm remain separate. `to_dataframe()` ranks entries by descending mean value,
breaking ties by ascending mean execution time. `best_run` uses the same rule.
The summary includes best, worst, mean and median value, value and volume standard
deviations, execution times in seconds, and the percentage of valid solutions.
Standard deviations describe the population of observed repetitions (`ddof=0`),
so a single valid repetition has zero deviation.

Solutions over capacity or using unavailable items (including excess duplicates)
remain in `executions_to_dataframe()` but do not contribute to value and volume
statistics. Timing includes every repetition and measures only `optimize()`.
If an optimizer produces no valid solutions, its value and volume statistics are
`NaN` and its `best_solution` is `None`. If none of the optimizers produces a valid
solution, accessing `best_run` raises `ValueError`. Optimizer errors propagate to
the caller; optimizers and returned solutions must use the supplied instance.

Calling `run()` again returns a fresh result without accumulating repetitions.
It reuses optimizer objects and their random state. Simulated Annealing's `seed`
initializes its random generator; Hill Climbing uses Python's global `random`
generator. Running the experiment does not reset either generator.

## Structure

- `backpack_optimizer/models/`: backpack, item, instance, and solution models.
- `backpack_optimizer/optimizers/`: optimizers and shared neighborhood generation.
- `backpack_optimizer/experiment/`: experiment execution and result classes, each in its own module.
- `backpack_optimizer/__main__.py`: runnable example.
- `tests/experiment/`: experiment tests grouped by responsibility, with shared fixtures.
- `tests/optimizers/`: optimizer behavior tests.
