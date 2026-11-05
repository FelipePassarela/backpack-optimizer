from collections import Counter

import pytest
from pydantic import ValidationError

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
    ("name", "value", "error_type"),
    [
        ("initial_temperature", 0, "greater_than"),
        ("initial_temperature", -1, "greater_than"),
        ("initial_temperature", float("inf"), "finite_number"),
        ("initial_temperature", float("nan"), "finite_number"),
        ("initial_temperature", 0.1, "value_error"),
        ("initial_temperature", 0.01, "value_error"),
        ("initial_temperature", True, "float_type"),
        ("initial_temperature", "100", "float_type"),
        ("minimum_temperature", 0, "greater_than"),
        ("minimum_temperature", -1, "greater_than"),
        ("minimum_temperature", float("inf"), "finite_number"),
        ("minimum_temperature", float("nan"), "finite_number"),
        ("minimum_temperature", 100, "value_error"),
        ("minimum_temperature", 101, "value_error"),
        ("minimum_temperature", False, "float_type"),
        ("minimum_temperature", "0.1", "float_type"),
        ("cooling_rate", 0, "greater_than"),
        ("cooling_rate", -0.1, "greater_than"),
        ("cooling_rate", 1, "less_than"),
        ("cooling_rate", 1.1, "less_than"),
        ("cooling_rate", float("inf"), "finite_number"),
        ("cooling_rate", float("nan"), "finite_number"),
        ("cooling_rate", True, "float_type"),
        ("cooling_rate", "0.95", "float_type"),
        ("iterations_per_temperature", 0, "greater_than"),
        ("iterations_per_temperature", -1, "greater_than"),
        ("iterations_per_temperature", 1.5, "int_type"),
        ("iterations_per_temperature", True, "int_type"),
        ("iterations_per_temperature", 2.0, "int_type"),
        ("iterations_per_temperature", "100", "int_type"),
        ("max_iterations", 0, "greater_than"),
        ("max_iterations", -1, "greater_than"),
        ("max_iterations", 1.5, "int_type"),
        ("max_iterations", False, "int_type"),
        ("max_iterations", 100.0, "int_type"),
        ("max_iterations", "100", "int_type"),
        ("seed", True, "int_type"),
        ("seed", 42.0, "int_type"),
        ("seed", "42", "int_type"),
    ],
)
def test_rejects_invalid_parameters(name: str, value: object, error_type: str) -> None:
    instance = Instance(backpack=Backpack(max_volume=5), available_items=[])

    with pytest.raises(ValidationError) as exc_info:
        SimulatedAnnealingOptimizer(instance, **{name: value})

    error = exc_info.value.errors()[0]
    assert error["type"] == error_type
    assert error["loc"] == (() if error_type == "value_error" else (name,))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("initial_temperature", 0.1),
        ("minimum_temperature", 100),
        ("cooling_rate", float("nan")),
        ("iterations_per_temperature", True),
        ("max_iterations", 0),
    ],
)
def test_invalid_assignment_preserves_the_previous_parameters(
    name: str, value: object
) -> None:
    instance = Instance(backpack=Backpack(max_volume=5), available_items=[])
    optimizer = SimulatedAnnealingOptimizer(instance)
    defaults = {
        "initial_temperature": 100.0,
        "minimum_temperature": 0.1,
        "cooling_rate": 0.95,
        "iterations_per_temperature": 100,
        "max_iterations": 20_000,
    }

    with pytest.raises(ValidationError):
        setattr(optimizer, name, value)

    assert {name: getattr(optimizer, name) for name in defaults} == defaults


def test_valid_assignments_are_used_by_the_search_without_resetting_the_rng() -> None:
    item = Item(value=10, dimensions=Dimensions(1, 1, 1))
    instance = Instance(backpack=Backpack(max_volume=5), available_items=[item])
    optimizer = SimulatedAnnealingOptimizer(instance, seed=42, max_iterations=50)
    optimizer.optimize()
    generator = optimizer._random
    previous_state = generator.getstate()

    optimizer.initial_temperature = 200
    optimizer.minimum_temperature = 1
    optimizer.cooling_rate = 0.9
    optimizer.iterations_per_temperature = 25
    optimizer.max_iterations = 50

    assert optimizer.initial_temperature == 200
    assert optimizer.minimum_temperature == 1
    assert optimizer.cooling_rate == 0.9
    assert optimizer.iterations_per_temperature == 25
    assert optimizer.max_iterations == 50
    assert optimizer._random is generator
    assert generator.getstate() == previous_state
    assert optimizer.optimize().items == [item]
    assert generator.getstate() != previous_state
