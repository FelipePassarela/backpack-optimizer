import math

import pytest

from backpack_optimizer import Experiment
from backpack_optimizer.models import Backpack, Dimensions, Instance, Item, Solution

from .helpers import SequenceOptimizer, set_durations


def test_invalid_solutions_are_retained_but_excluded_from_quality_statistics(
    instance: Instance, monkeypatch: pytest.MonkeyPatch
) -> None:
    valid = Solution(items=[instance.available_items[0]], instance=instance)
    invalid = Solution(items=[instance.available_items[2]], instance=instance)
    optimizer = SequenceOptimizer(instance, [valid, invalid])
    set_durations(monkeypatch, 1, 3)

    result = Experiment(instance, [optimizer], n_repeats=2).run()

    run = result.best_run
    assert run.mean_value == run.median_value == run.worst_value == 2
    assert run.mean_volume == 1
    assert run.std_value == run.std_volume == 0
    assert run.best_solution is valid
    assert run.n_repeats == 2
    assert run.n_valid == 1
    assert run.valid_percentage == 50
    assert run.mean_time == 2
    assert run.total_time == 4
    assert result.executions_to_dataframe()["Valid"].tolist() == [True, False]
    assert result.executions_to_dataframe()["Value"].tolist() == [2, 1000]
    assert result.to_dataframe()["Valid (%)"].tolist() == [50]


def test_optimizer_without_valid_solutions_ranks_last(instance: Instance) -> None:
    invalid = SequenceOptimizer(
        instance, [Solution(items=[instance.available_items[2]], instance=instance)]
    )
    valid = SequenceOptimizer(instance, [Solution(items=[], instance=instance)])

    result = Experiment(instance, [invalid, valid], n_repeats=1).run()

    run = result.run_results[0]
    assert run.best_solution is None
    assert math.isnan(run.mean_value)
    assert math.isnan(run.std_value)
    assert math.isnan(run.mean_volume)
    assert math.isnan(run.std_volume)
    assert math.isnan(run.median_value)
    assert math.isnan(run.worst_value)
    assert run.valid_percentage == 0
    assert result.best_run.optimizer is valid
    assert result.to_dataframe()["Optimizer ID"].tolist() == [2, 1]


def test_best_run_reports_when_no_optimizer_found_a_valid_solution(
    instance: Instance,
) -> None:
    optimizer = SequenceOptimizer(
        instance, [Solution(items=[instance.available_items[2]], instance=instance)]
    )

    result = Experiment(instance, [optimizer], n_repeats=1).run()

    assert result.to_dataframe()["Best Value"].isna().all()
    with pytest.raises(ValueError, match="No valid solutions"):
        _ = result.best_run


@pytest.mark.parametrize("excess_duplicates", [False, True])
def test_solutions_cannot_use_unavailable_items_or_excess_duplicates(
    instance: Instance, excess_duplicates: bool
) -> None:
    items = (
        [instance.available_items[0]] * 2
        if excess_duplicates
        else [Item(value=999, dimensions=Dimensions(1, 1, 1))]
    )
    solution = Solution(items=items, instance=instance)
    assert solution.is_valid()
    optimizer = SequenceOptimizer(instance, [solution])

    result = Experiment(instance, [optimizer], n_repeats=1).run()

    assert result.run_results[0].n_valid == 0
    assert result.run_results[0].best_solution is None
    assert result.executions_to_dataframe()["Valid"].tolist() == [False]


def test_available_duplicate_items_remain_valid() -> None:
    item = Item(value=2, dimensions=Dimensions(1, 1, 1))
    instance = Instance(backpack=Backpack(max_volume=2), available_items=[item, item])
    solution = Solution(items=[item, item], instance=instance)
    optimizer = SequenceOptimizer(instance, [solution])

    result = Experiment(instance, [optimizer], n_repeats=1).run()

    assert result.best_run.best_solution is solution
    assert result.best_run.mean_value == 4
    assert result.best_run.valid_percentage == 100
