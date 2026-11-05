# Backpack Optimizer

Python package for maximizing the value of items in a backpack subject to a volume
limit. Includes Hill Climbing and Simulated Annealing optimizers.

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
        HillClimbingOptimizer(instance),
        SimulatedAnnealingOptimizer(instance),
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

## Development

Run the tests:

```sh
uv run pytest
```

Build the package:

```sh
uv build
```
