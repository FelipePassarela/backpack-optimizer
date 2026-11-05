import pytest

from backpack_optimizer.models import Instance, Solution
from backpack_optimizer.optimizers import Optimizer


class SequenceOptimizer(Optimizer):
    def __init__(self, instance: Instance, solutions: list[Solution]) -> None:
        super().__init__(instance)
        self.solutions = solutions
        self.calls = 0

    def optimize(self) -> Solution:
        solution = self.solutions[self.calls % len(self.solutions)]
        self.calls += 1
        return solution

    def create_neighborhood(self, current_solution: Solution) -> list[Solution]:
        return []


def set_durations(monkeypatch: pytest.MonkeyPatch, *durations: float) -> None:
    timestamps = []
    current = 0.0
    for duration in durations:
        timestamps.extend([current, current + duration])
        current += duration
    clock = iter(timestamps)
    monkeypatch.setattr(
        "backpack_optimizer.experiment.experiment.perf_counter", lambda: next(clock)
    )
