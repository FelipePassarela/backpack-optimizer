from collections import Counter

import pytest

from backpack_optimizer import Experiment
from backpack_optimizer.models import Backpack, Dimensions, Instance, Item, Solution
from backpack_optimizer.optimizers import ExhaustiveSearchOptimizer, HillClimbingOptimizer


def test_finds_better_combination_than_the_most_valuable_item() -> None:
    items = [
        Item(value=10, dimensions=Dimensions(1, 1, 5)),
        Item(value=6, dimensions=Dimensions(1, 1, 3)),
        Item(value=5, dimensions=Dimensions(1, 1, 2)),
    ]
    instance = Instance(backpack=Backpack(max_volume=5), available_items=items.copy())

    solution = ExhaustiveSearchOptimizer(instance).optimize()

    assert solution.items == items[1:]
    assert solution.value == 11
    assert solution.volume == 5
    assert solution.instance is instance
    assert solution.is_valid()
    assert Counter(solution.items) <= Counter(items)
    assert instance.available_items == items


def test_finds_optimum_when_value_per_volume_greedy_fails() -> None:
    items = [
        Item(value=60, dimensions=Dimensions(1, 1, 10)),
        Item(value=100, dimensions=Dimensions(1, 1, 20)),
        Item(value=120, dimensions=Dimensions(1, 1, 30)),
    ]
    instance = Instance(backpack=Backpack(max_volume=50), available_items=items)
    greedy_items = []
    for item in sorted(
        items, key=lambda item: item.value / item.dimensions.volume, reverse=True
    ):
        candidate = Solution(items=[*greedy_items, item], instance=instance)
        if candidate.is_valid():
            greedy_items.append(item)
    greedy = Solution(items=greedy_items, instance=instance)

    solution = ExhaustiveSearchOptimizer(instance).optimize()

    assert greedy.value == 160
    assert solution.value == 220
    assert solution.items == items[1:]
    assert solution.is_valid()


@pytest.mark.parametrize("volumes", [[], [6, 7]])
def test_returns_empty_solution_when_no_item_fits(volumes: list[int]) -> None:
    instance = Instance(
        backpack=Backpack(max_volume=5),
        available_items=[
            Item(value=10, dimensions=Dimensions(1, 1, volume)) for volume in volumes
        ],
    )

    solution = ExhaustiveSearchOptimizer(instance).optimize()

    assert solution.items == []
    assert solution.value == 0
    assert solution.volume == 0
    assert solution.instance is instance
    assert solution.is_valid()


def test_preserves_fractional_values_and_volumes() -> None:
    items = [
        Item(value=1.05, dimensions=Dimensions(1, 1, 1.5)),
        Item(value=0.55, dimensions=Dimensions(1, 1, 0.75)),
        Item(value=0.6, dimensions=Dimensions(1, 1, 0.5)),
        Item(value=100, dimensions=Dimensions(1, 1, 1.75)),
    ]
    instance = Instance(backpack=Backpack(max_volume=1.5), available_items=items)

    solution = ExhaustiveSearchOptimizer(instance).optimize()

    assert solution.items == items[1:3]
    assert solution.value == pytest.approx(1.15)
    assert solution.volume == 1.25
    assert solution.is_valid()


def test_uses_the_models_capacity_check_without_float_tolerance() -> None:
    items = [
        Item(value=1, dimensions=Dimensions(1, 1, 0.1)),
        Item(value=2, dimensions=Dimensions(1, 1, 0.2)),
    ]
    instance = Instance(backpack=Backpack(max_volume=0.3), available_items=items)
    assert not Solution(items=items, instance=instance).is_valid()

    solution = ExhaustiveSearchOptimizer(instance).optimize()

    assert solution.items == [items[1]]
    assert solution.is_valid()


@pytest.mark.parametrize("same_object", [False, True])
@pytest.mark.parametrize("capacity", [2, 3])
def test_preserves_equal_items_by_position_without_exceeding_availability(
    same_object: bool, capacity: int
) -> None:
    first = Item(value=10, dimensions=Dimensions(1, 1, 1))
    second = first if same_object else Item(value=10, dimensions=Dimensions(1, 1, 1))
    items = [first, second]
    instance = Instance(
        backpack=Backpack(max_volume=capacity), available_items=items.copy()
    )

    solution = ExhaustiveSearchOptimizer(instance).optimize()

    assert solution.items == items
    assert solution.items[0] is first
    assert solution.items[1] is second
    assert solution.value == 20
    assert solution.instance is instance
    assert solution.is_valid()
    assert Counter(solution.items) == Counter(items)
    assert solution.items is not instance.available_items
    assert all(
        actual is expected
        for actual, expected in zip(instance.available_items, items, strict=True)
    )


def test_empty_solution_beats_non_positive_values() -> None:
    instance = Instance(
        backpack=Backpack(max_volume=5),
        available_items=[
            Item(value=-1, dimensions=Dimensions(1, 1, 1)),
            Item(value=0, dimensions=Dimensions(1, 1, 1)),
        ],
    )

    solution = ExhaustiveSearchOptimizer(instance).optimize()

    assert solution.items == []
    assert solution.is_valid()


def test_selects_zero_volume_items_at_zero_capacity() -> None:
    item = Item(value=10, dimensions=Dimensions(1, 1, 0))
    instance = Instance(backpack=Backpack(max_volume=0), available_items=[item, item])

    solution = ExhaustiveSearchOptimizer(instance).optimize()

    assert solution.items == [item, item]
    assert solution.value == 20
    assert solution.is_valid()


def test_checks_combinations_with_negative_volume_accepted_by_the_model() -> None:
    items = [
        Item(value=10, dimensions=Dimensions(1, 1, 6)),
        Item(value=-1, dimensions=Dimensions(1, 1, -1)),
    ]
    instance = Instance(backpack=Backpack(max_volume=5), available_items=items)

    solution = ExhaustiveSearchOptimizer(instance).optimize()

    assert solution.items == items
    assert solution.value == 9
    assert solution.is_valid()


def test_can_find_a_negative_value_optimum_when_empty_solution_is_infeasible() -> None:
    item = Item(value=-1, dimensions=Dimensions(1, 1, -1))
    instance = Instance(backpack=Backpack(max_volume=-1), available_items=[item])

    solution = ExhaustiveSearchOptimizer(instance).optimize()

    assert solution.items == [item]
    assert solution.is_valid()


def test_raises_when_no_subset_is_feasible() -> None:
    instance = Instance(backpack=Backpack(max_volume=-1), available_items=[])

    with pytest.raises(ValueError, match="No feasible solution"):
        ExhaustiveSearchOptimizer(instance).optimize()


@pytest.mark.parametrize(
    "max_items", [-1, 1.5, 2.0, True, False, "2", None, float("inf"), float("nan")]
)
def test_rejects_invalid_limits(max_items: object) -> None:
    instance = Instance(backpack=Backpack(max_volume=5), available_items=[])

    with pytest.raises(ValueError, match="max_items must be a non-negative integer"):
        ExhaustiveSearchOptimizer(instance, max_items=max_items)


def test_zero_limit_allows_an_empty_instance() -> None:
    instance = Instance(backpack=Backpack(max_volume=5), available_items=[])

    solution = ExhaustiveSearchOptimizer(instance, max_items=0).optimize()

    assert solution.items == []
    assert solution.is_valid()


@pytest.mark.parametrize("max_items", [0, 1])
def test_rejects_oversized_instances_at_construction(max_items: int) -> None:
    item = Item(value=10, dimensions=Dimensions(1, 1, 1))
    instance = Instance(backpack=Backpack(max_volume=5), available_items=[item, item])

    with pytest.raises(ValueError, match=f"2 items, exceeding max_items={max_items}"):
        ExhaustiveSearchOptimizer(instance, max_items=max_items)


def test_default_limit_rejects_more_than_twenty_items() -> None:
    item = Item(value=10, dimensions=Dimensions(1, 1, 1))
    instance = Instance(backpack=Backpack(max_volume=5), available_items=[item] * 21)

    with pytest.raises(ValueError, match="21 items, exceeding max_items=20"):
        ExhaustiveSearchOptimizer(instance)


def test_accepts_an_instance_exactly_at_the_configured_limit() -> None:
    item = Item(value=10, dimensions=Dimensions(1, 1, 1))
    instance = Instance(backpack=Backpack(max_volume=5), available_items=[item, item])

    solution = ExhaustiveSearchOptimizer(instance, max_items=2).optimize()

    assert solution.items == [item, item]
    assert solution.is_valid()


def test_rechecks_the_limit_before_search_if_the_item_list_grows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = Item(value=10, dimensions=Dimensions(1, 1, 1))
    instance = Instance(backpack=Backpack(max_volume=5), available_items=[item])
    optimizer = ExhaustiveSearchOptimizer(instance, max_items=1)
    instance.available_items.append(item)

    def fail_if_search_starts(*args: object) -> None:
        pytest.fail("Oversized instances must be rejected before enumeration")

    monkeypatch.setattr(
        "backpack_optimizer.optimizers.exhaustive_search.combinations",
        fail_if_search_starts,
    )

    with pytest.raises(ValueError, match="2 items, exceeding max_items=1"):
        optimizer.optimize()


def test_runs_in_experiment_as_an_exact_reference_without_a_neighborhood() -> None:
    instance = Instance(
        backpack=Backpack(max_volume=5),
        available_items=[
            Item(value=10, dimensions=Dimensions(1, 1, 5)),
            Item(value=6, dimensions=Dimensions(1, 1, 3)),
            Item(value=5, dimensions=Dimensions(1, 1, 2)),
        ],
    )
    exact = ExhaustiveSearchOptimizer(instance)
    heuristic = HillClimbingOptimizer(instance)
    experiment = Experiment(instance, [exact, heuristic], n_repeats=2)

    result = experiment.run()

    exact_run = result.run_results[0]
    assert exact_run.optimizer is exact
    assert exact_run.instance is instance
    assert exact_run.n_valid == 2
    assert exact_run.mean_value == 11
    assert exact_run.std_value == 0
    assert all(execution.solution.value == 11 for execution in exact_run.executions)
    assert result.executions_to_dataframe()["Valid"].all()
    assert result.best_run.best_solution.value == 11
