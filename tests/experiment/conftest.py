import pytest

from backpack_optimizer import Experiment
from backpack_optimizer.models import Backpack, Dimensions, Instance, Item, Solution

from .helpers import SequenceOptimizer, set_durations


@pytest.fixture
def instance() -> Instance:
    return Instance(
        backpack=Backpack(max_volume=5),
        available_items=[
            Item(value=2, dimensions=Dimensions(1, 1, 1)),
            Item(value=8, dimensions=Dimensions(1, 1, 3)),
            Item(value=1000, dimensions=Dimensions(1, 1, 6)),
        ],
    )


@pytest.fixture
def known_solutions(instance: Instance) -> list[Solution]:
    return [
        Solution(items=[], instance=instance),
        Solution(items=[instance.available_items[0]], instance=instance),
        Solution(items=[instance.available_items[1]], instance=instance),
    ]


@pytest.fixture
def known_experiment(
    instance: Instance,
    known_solutions: list[Solution],
    monkeypatch: pytest.MonkeyPatch,
) -> Experiment:
    first = SequenceOptimizer(instance, known_solutions)
    second = SequenceOptimizer(instance, [known_solutions[1]])
    set_durations(monkeypatch, 1, 2, 3, 0.5, 0.5, 0.5)
    return Experiment(instance, [first, second], n_repeats=3)
