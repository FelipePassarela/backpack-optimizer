from collections import Counter

import pytest

from backpack_optimizer.models import Backpack, Dimensions, Instance, Item, Solution
from backpack_optimizer.optimizers import SimulatedAnnealingOptimizer


def test_empty_instance_returns_empty_solution() -> None:
    instance = Instance(backpack=Backpack(max_volume=10), available_items=[])

    solution = SimulatedAnnealingOptimizer(instance).optimize()

    assert solution.items == []
    assert solution.value == 0
    assert solution.volume == 0


def test_returns_empty_solution_when_no_item_fits() -> None:
    instance = Instance(
        backpack=Backpack(max_volume=5),
        available_items=[Item(value=100, dimensions=Dimensions(1, 1, 6))],
    )

    solution = SimulatedAnnealingOptimizer(instance).optimize()

    assert solution.items == []


def test_adds_items_up_to_capacity_without_mutating_input() -> None:
    items = [
        Item(value=10, dimensions=Dimensions(1, 1, 3)),
        Item(value=7, dimensions=Dimensions(1, 1, 2)),
    ]
    instance = Instance(backpack=Backpack(max_volume=5), available_items=items.copy())
    optimizer = SimulatedAnnealingOptimizer(instance, seed=42, max_iterations=100)

    solution = optimizer.optimize()

    assert solution.value == 17
    assert solution.volume == 5
    assert Counter(solution.items) == Counter(items)
    assert instance.available_items == items


def test_can_select_identical_items_as_separate_entries() -> None:
    item = Item(value=10, dimensions=Dimensions(1, 1, 1))
    instance = Instance(backpack=Backpack(max_volume=2), available_items=[item, item])

    solution = SimulatedAnnealingOptimizer(instance, seed=42, max_iterations=100).optimize()

    assert solution.items == [item, item]
    assert solution.value == 20
    assert solution.volume == 2


def test_neighborhood_preserves_multiplicity_and_filters_over_capacity() -> None:
    item = Item(value=10, dimensions=Dimensions(1, 1, 2))
    oversized = Item(value=1000, dimensions=Dimensions(1, 1, 6))
    items = [item, item, oversized]
    instance = Instance(backpack=Backpack(max_volume=4), available_items=items.copy())
    current_solution = Solution(items=[item], instance=instance)

    neighbors = SimulatedAnnealingOptimizer(instance).create_neighborhood(current_solution)

    assert Counter(tuple(neighbor.items) for neighbor in neighbors) == Counter(
        [(item, item), ()]
    )
    assert all(neighbor.is_valid() and neighbor.instance is instance for neighbor in neighbors)
    assert current_solution.items == [item]
    assert instance.available_items == items


def test_seed_reproduces_solution() -> None:
    instance = Instance(
        backpack=Backpack(max_volume=5),
        available_items=[
            Item(value=10, dimensions=Dimensions(1, 1, 5)),
            Item(value=6, dimensions=Dimensions(1, 1, 3)),
            Item(value=5, dimensions=Dimensions(1, 1, 2)),
            Item(value=1000, dimensions=Dimensions(1, 1, 6)),
        ],
    )
    first = SimulatedAnnealingOptimizer(instance, seed=42, max_iterations=50)
    second = SimulatedAnnealingOptimizer(instance, seed=42, max_iterations=50)
    first_solution = first.optimize()
    second_solution = second.optimize()

    assert first_solution == second_solution
    assert first_solution.is_valid()
    assert Counter(first_solution.items) <= Counter(instance.available_items)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("initial_temperature", 0),
        ("initial_temperature", -1),
        ("initial_temperature", float("inf")),
        ("initial_temperature", float("nan")),
        ("initial_temperature", 0.1),
        ("initial_temperature", 0.01),
        ("minimum_temperature", 0),
        ("minimum_temperature", -1),
        ("minimum_temperature", float("inf")),
        ("minimum_temperature", float("nan")),
        ("minimum_temperature", 100),
        ("minimum_temperature", 101),
        ("cooling_rate", 0),
        ("cooling_rate", -0.1),
        ("cooling_rate", 1),
        ("cooling_rate", 1.1),
        ("cooling_rate", float("inf")),
        ("cooling_rate", float("nan")),
        ("iterations_per_temperature", 0),
        ("iterations_per_temperature", -1),
        ("iterations_per_temperature", 1.5),
        ("iterations_per_temperature", True),
        ("max_iterations", 0),
        ("max_iterations", -1),
        ("max_iterations", 1.5),
        ("max_iterations", False),
    ],
)
def test_rejects_invalid_parameters(name: str, value: float | bool) -> None:
    instance = Instance(backpack=Backpack(max_volume=5), available_items=[])

    with pytest.raises(ValueError):
        SimulatedAnnealingOptimizer(instance, **{name: value})
