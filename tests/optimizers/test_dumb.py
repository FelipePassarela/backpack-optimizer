import pytest

from backpack_optimizer.models import Backpack, Dimensions, Instance, Item
from backpack_optimizer.optimizers import DumbOptimizer


@pytest.mark.parametrize("skip_oversized", [False, True])
def test_selects_only_the_first_fitting_item_in_original_order(
    skip_oversized: bool,
) -> None:
    first_fitting = Item(value=10, dimensions=Dimensions(1, 1, 2))
    more_valuable = Item(value=100, dimensions=Dimensions(1, 1, 1))
    items = [first_fitting, more_valuable]
    if skip_oversized:
        items.insert(0, Item(value=1000, dimensions=Dimensions(1, 1, 6)))
    instance = Instance(backpack=Backpack(max_volume=5), available_items=items.copy())

    solution = DumbOptimizer(instance).optimize()

    assert solution.items == [first_fitting]
    assert solution.instance is instance
    assert solution.is_valid()
    assert instance.available_items == items


def test_accepts_an_item_that_exactly_fills_the_backpack() -> None:
    exact_fit = Item(value=10, dimensions=Dimensions(1, 1, 5))
    smaller = Item(value=100, dimensions=Dimensions(1, 1, 1))
    instance = Instance(
        backpack=Backpack(max_volume=5), available_items=[exact_fit, smaller]
    )

    solution = DumbOptimizer(instance).optimize()

    assert solution.items == [exact_fit]
    assert solution.is_valid()


@pytest.mark.parametrize("volumes", [[], [6, 7]])
def test_returns_empty_solution_when_no_item_fits(volumes: list[int]) -> None:
    instance = Instance(
        backpack=Backpack(max_volume=5),
        available_items=[
            Item(value=10, dimensions=Dimensions(1, 1, volume)) for volume in volumes
        ],
    )

    solution = DumbOptimizer(instance).optimize()

    assert solution.items == []
    assert solution.instance is instance
    assert solution.is_valid()
