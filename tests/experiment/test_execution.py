import pytest

from backpack_optimizer import Experiment
from backpack_optimizer.models import Instance, Solution
from backpack_optimizer.optimizers import (
    HillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
)

from .helpers import SequenceOptimizer


def test_runs_each_optimizer_requested_number_of_times(
    known_experiment: Experiment,
) -> None:
    assert all(optimizer.calls == 0 for optimizer in known_experiment.optimizers)

    result = known_experiment.run()

    assert len(result.optimizer_results) == 2
    for optimizer_result, optimizer in zip(
        result.optimizer_results, known_experiment.optimizers, strict=True
    ):
        assert optimizer.calls == 3
        assert optimizer_result.instance is known_experiment.instance
        assert optimizer_result.optimizer is optimizer


def test_uses_the_updated_repeat_count(instance: Instance) -> None:
    optimizer = SequenceOptimizer(instance, [Solution(items=[], instance=instance)])
    experiment = Experiment(instance, [optimizer], n_repeats=1)

    experiment.n_repeats = 3
    result = experiment.run()

    assert experiment.n_repeats == 3
    assert optimizer.calls == 3
    assert result.optimizer_results[0].n_repeats == 3
    assert result.optimizer_results[0].n_valid == 3


def test_repeated_runs_support_generator_input_without_accumulating_results(
    instance: Instance,
) -> None:
    optimizer = SequenceOptimizer(
        instance,
        [
            Solution(items=[], instance=instance),
            Solution(items=[instance.available_items[0]], instance=instance),
            Solution(items=[instance.available_items[1]], instance=instance),
        ],
    )
    experiment = Experiment(instance, iter([optimizer]), n_repeats=2)

    first = experiment.run()
    second = experiment.run()

    assert optimizer.calls == 4
    assert len(first.executions_to_dataframe()) == 2
    assert len(second.executions_to_dataframe()) == 2
    assert first.optimizer_results[0].mean_value == 1
    assert second.optimizer_results[0].mean_value == 4
    assert second.executions_to_dataframe()["Repetition"].tolist() == [1, 2]


def test_optimizer_errors_propagate(
    instance: Instance, monkeypatch: pytest.MonkeyPatch
) -> None:
    optimizer = SequenceOptimizer(instance, [Solution(items=[], instance=instance)])

    def fail() -> Solution:
        raise RuntimeError("optimizer failed")

    monkeypatch.setattr(optimizer, "optimize", fail)

    with pytest.raises(RuntimeError, match="optimizer failed"):
        Experiment(instance, [optimizer], n_repeats=1).run()


def test_experiment_with_existing_optimizers(instance: Instance) -> None:
    optimizers = [
        HillClimbingOptimizer(instance),
        SimulatedAnnealingOptimizer(instance, seed=42, max_iterations=50),
    ]

    result = Experiment(instance, optimizers, n_repeats=2).run()

    assert len(result.executions_to_dataframe()) == 4
    assert result.executions_to_dataframe()["Valid"].all()
    assert (result.to_dataframe()["Valid (%)"] == 100).all()
    assert result.best_optimizer_result.best_solution.is_valid()
    assert all(
        optimizer_result.mean_time >= 0 for optimizer_result in result.optimizer_results
    )
