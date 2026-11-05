import math

import pytest

from backpack_optimizer import Experiment
from backpack_optimizer.models import Solution


def test_aggregates_solution_value_statistics(
    known_experiment: Experiment, known_solutions: list[Solution]
) -> None:
    optimizer_result = known_experiment.run().optimizer_results[0]

    assert optimizer_result.mean_value == pytest.approx(10 / 3)
    assert optimizer_result.std_value == pytest.approx(math.sqrt(104 / 9))
    assert optimizer_result.median_value == 2
    assert optimizer_result.worst_value == 0
    assert optimizer_result.best_solution is known_solutions[2]


def test_aggregates_solution_volume_statistics(known_experiment: Experiment) -> None:
    optimizer_result = known_experiment.run().optimizer_results[0]

    assert optimizer_result.mean_volume == pytest.approx(4 / 3)
    assert optimizer_result.std_volume == pytest.approx(math.sqrt(14 / 9))


def test_aggregates_optimization_times(known_experiment: Experiment) -> None:
    optimizer_result = known_experiment.run().optimizer_results[0]

    assert optimizer_result.mean_time == 2
    assert optimizer_result.std_time == pytest.approx(math.sqrt(2 / 3))
    assert optimizer_result.total_time == 6


def test_preserves_individual_solutions_and_validity_counts(
    known_experiment: Experiment, known_solutions: list[Solution]
) -> None:
    optimizer_result = known_experiment.run().optimizer_results[0]

    assert [
        execution.solution for execution in optimizer_result.executions
    ] == known_solutions
    assert optimizer_result.n_repeats == optimizer_result.n_valid == 3
    assert optimizer_result.valid_percentage == 100
