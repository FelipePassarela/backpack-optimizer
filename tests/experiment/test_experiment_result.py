import pytest

from backpack_optimizer import Experiment
from backpack_optimizer.models import Instance, Solution

from .helpers import SequenceOptimizer, set_durations


def test_summary_identifies_optimizer_entries(known_experiment: Experiment) -> None:
    summary = known_experiment.run().to_dataframe()

    assert summary["Optimizer ID"].tolist() == [1, 2]
    assert summary["Optimizer"].tolist() == ["SequenceOptimizer"] * 2
    assert summary["Valid Repeats"].tolist() == [3, 3]


def test_summary_exports_aggregated_statistics(known_experiment: Experiment) -> None:
    summary = known_experiment.run().to_dataframe()

    assert summary["Mean Value"].tolist() == pytest.approx([10 / 3, 2])
    assert summary["Median Value"].tolist() == [2, 2]
    assert summary["Best Value"].tolist() == [8, 2]
    assert summary["Worst Value"].tolist() == [0, 2]
    assert summary["Total Time (s)"].tolist() == [6, 1.5]


def test_best_run_has_highest_mean_value(known_experiment: Experiment) -> None:
    result = known_experiment.run()

    assert result.best_run is result.run_results[0]


def test_exports_each_repetition(known_experiment: Experiment) -> None:
    executions = known_experiment.run().executions_to_dataframe()

    assert executions["Optimizer ID"].tolist() == [1, 1, 1, 2, 2, 2]
    assert executions["Repetition"].tolist() == [1, 2, 3, 1, 2, 3]
    assert executions["Value"].tolist() == [0, 2, 8, 2, 2, 2]
    assert executions["Volume"].tolist() == [0, 1, 3, 1, 1, 1]
    assert executions["Time (s)"].tolist() == [1, 2, 3, 0.5, 0.5, 0.5]
    assert executions["Valid"].all()


def test_ranking_uses_mean_value_then_time_for_same_algorithm_entries(
    instance: Instance, monkeypatch: pytest.MonkeyPatch
) -> None:
    high_value = Solution(items=[instance.available_items[1]], instance=instance)
    low_value = Solution(items=[instance.available_items[0]], instance=instance)
    optimizers = [
        SequenceOptimizer(instance, [high_value]),
        SequenceOptimizer(instance, [high_value]),
        SequenceOptimizer(instance, [low_value]),
    ]
    set_durations(monkeypatch, 2, 1, 0.25)

    result = Experiment(instance, optimizers, n_repeats=1).run()

    assert result.best_run.optimizer is optimizers[1]
    assert result.to_dataframe()["Optimizer ID"].tolist() == [2, 1, 3]
    for run in result.run_results:
        assert run.std_value == run.std_volume == run.std_time == 0
