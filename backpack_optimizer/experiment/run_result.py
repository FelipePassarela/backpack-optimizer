from dataclasses import dataclass

import numpy as np

from backpack_optimizer.experiment.optimization_run import OptimizationRun
from backpack_optimizer.models import Instance, Solution
from backpack_optimizer.optimizers import Optimizer


@dataclass
class RunResult:
    """Aggregate results for one optimizer entry in an experiment.

    Value and volume statistics include only valid solutions. Time statistics
    include all repetitions. Standard deviations use the population definition.
    """

    instance: Instance
    optimizer: Optimizer
    mean_value: float
    std_value: float
    mean_volume: float
    std_volume: float
    best_solution: Solution | None
    n_repeats: int
    optimizer_id: int
    executions: list[OptimizationRun]

    @property
    def n_valid(self) -> int:
        return sum(execution.is_valid for execution in self.executions)

    @property
    def valid_percentage(self) -> float:
        return 100.0 * self.n_valid / self.n_repeats

    @property
    def median_value(self) -> float:
        values = [
            execution.solution.value
            for execution in self.executions
            if execution.is_valid
        ]
        return float(np.median(values)) if values else float("nan")

    @property
    def worst_value(self) -> float:
        return min(
            (
                execution.solution.value
                for execution in self.executions
                if execution.is_valid
            ),
            default=float("nan"),
        )

    @property
    def mean_time(self) -> float:
        return float(np.mean([run.elapsed_seconds for run in self.executions]))

    @property
    def std_time(self) -> float:
        return float(np.std([run.elapsed_seconds for run in self.executions]))

    @property
    def total_time(self) -> float:
        return sum(run.elapsed_seconds for run in self.executions)
