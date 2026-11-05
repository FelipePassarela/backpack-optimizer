import pytest

from backpack_optimizer import Experiment
from backpack_optimizer.models import Instance, Solution

from .helpers import SequenceOptimizer


@pytest.mark.parametrize("n_repeats", [0, -1, 1.5, True, False, "2", None])
def test_rejects_invalid_repeat_counts(instance: Instance, n_repeats: object) -> None:
    optimizer = SequenceOptimizer(instance, [Solution(items=[], instance=instance)])

    with pytest.raises(ValueError, match="n_repeats must be a positive integer"):
        Experiment(instance, [optimizer], n_repeats=n_repeats)


def test_rejects_empty_optimizer_iterables(instance: Instance) -> None:
    with pytest.raises(ValueError, match="at least one optimizer"):
        Experiment(instance, iter(()))


def test_rejects_non_optimizer_entries(instance: Instance) -> None:
    with pytest.raises(TypeError, match="Optimizer instances"):
        Experiment(instance, [object()])


def test_rejects_an_optimizer_bound_to_another_instance(instance: Instance) -> None:
    other = Instance(instance.backpack, instance.available_items.copy())
    optimizer = SequenceOptimizer(other, [Solution(items=[], instance=other)])

    with pytest.raises(ValueError, match="experiment instance"):
        Experiment(instance, [optimizer])


def test_rejects_a_returned_solution_for_another_instance(instance: Instance) -> None:
    other = Instance(instance.backpack, instance.available_items.copy())
    optimizer = SequenceOptimizer(instance, [Solution(items=[], instance=other)])

    with pytest.raises(ValueError, match="solution for another instance"):
        Experiment(instance, [optimizer], n_repeats=1).run()
