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

## Structure

- `backpack_optimizer/models/`: backpack, item, instance, and solution models.
- `backpack_optimizer/optimizers/`: optimizers and shared neighborhood generation.
- `backpack_optimizer/main.py`: runnable example.
- `tests/optimizers/`: optimizer behavior tests.
