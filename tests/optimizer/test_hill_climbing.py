from collections import Counter

from backpack import Backpack
from dimensions import Dimensions
from instance import Instance
from item import Item
from optimizer.hill_climbing import HillClimbingOptimizer


def test_empty_instance_returns_empty_solution() -> None:
    instance = Instance(backpack=Backpack(max_volume=10), available_items=[])

    solution = HillClimbingOptimizer(instance).optimize()

    assert solution.items == []
    assert solution.volume == 0


def test_returns_empty_solution_when_no_item_fits() -> None:
    instance = Instance(
        backpack=Backpack(max_volume=100),
        available_items=[Item(value=7.5, dimensions=Dimensions(5, 5, 10))],
    )

    solution = HillClimbingOptimizer(instance).optimize()

    assert solution.items == []


def test_adds_items_from_empty_solution_up_to_capacity() -> None:
    items = [
        Item(value=10, dimensions=Dimensions(1, 1, 3)),
        Item(value=7, dimensions=Dimensions(1, 1, 2)),
    ]
    instance = Instance(backpack=Backpack(max_volume=5), available_items=items.copy())

    solution = HillClimbingOptimizer(instance).optimize()

    assert solution.value == 17
    assert solution.volume == 5
    assert Counter(solution.items) == Counter(items)
    assert instance.available_items == items


def test_can_select_identical_items_as_separate_entries() -> None:
    item = Item(value=10, dimensions=Dimensions(1, 1, 1))
    instance = Instance(backpack=Backpack(max_volume=2), available_items=[item, item])

    solution = HillClimbingOptimizer(instance).optimize()

    assert len(solution.items) == 2
    assert solution.value == 20
    assert solution.volume == 2


def test_optimizer_maximizes_value_without_rewarding_volume() -> None:
    valuable = Item(value=10, dimensions=Dimensions(1, 1, 1))
    bulky = Item(value=8, dimensions=Dimensions(1, 1, 5))
    instance = Instance(
        backpack=Backpack(max_volume=5), available_items=[bulky, valuable]
    )

    solution = HillClimbingOptimizer(instance).optimize()

    assert solution.items == [valuable]
    assert solution.value == 10
    assert solution.volume == 1


def test_optimizer_rejects_items_over_capacity() -> None:
    feasible = Item(value=10, dimensions=Dimensions(1, 1, 1))
    oversized = Item(value=1000, dimensions=Dimensions(1, 1, 6))
    instance = Instance(
        backpack=Backpack(max_volume=5), available_items=[oversized, feasible]
    )

    solution = HillClimbingOptimizer(instance).optimize()

    assert solution.items == [feasible]
    assert solution.value == 10
    assert solution.volume <= instance.backpack.max_volume
