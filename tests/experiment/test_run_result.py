import math

import pytest

from backpack_optimizer import Experiment
from backpack_optimizer.models import Solution


def test_aggregates_solution_value_statistics(
    known_experiment: Experiment, known_solutions: list[Solution]
) -> None:
    run = known_experiment.run().run_results[0]

    assert run.mean_value == pytest.approx(10 / 3)
    assert run.std_value == pytest.approx(math.sqrt(104 / 9))
    assert run.median_value == 2
    assert run.worst_value == 0
    assert run.best_solution is known_solutions[2]


def test_aggregates_solution_volume_statistics(known_experiment: Experiment) -> None:
    run = known_experiment.run().run_results[0]

    assert run.mean_volume == pytest.approx(4 / 3)
    assert run.std_volume == pytest.approx(math.sqrt(14 / 9))


def test_aggregates_optimization_times(known_experiment: Experiment) -> None:
    run = known_experiment.run().run_results[0]

    assert run.mean_time == 2
    assert run.std_time == pytest.approx(math.sqrt(2 / 3))
    assert run.total_time == 6


def test_preserves_individual_solutions_and_validity_counts(
    known_experiment: Experiment, known_solutions: list[Solution]
) -> None:
    run = known_experiment.run().run_results[0]

    assert [execution.solution for execution in run.executions] == known_solutions
    assert run.n_repeats == run.n_valid == 3
    assert run.valid_percentage == 100
