from collections import Counter
from collections.abc import Iterable
from time import perf_counter
from typing import Any

import numpy as np

from backpack_optimizer.config import ExperimentConfig
from backpack_optimizer.experiment.experiment_result import ExperimentResult
from backpack_optimizer.experiment.optimization_run import OptimizationRun
from backpack_optimizer.experiment.optimizer_result import OptimizerResult
from backpack_optimizer.models import Instance, Item, Solution
from backpack_optimizer.optimizers import Optimizer


class Experiment:
    def __init__(
        self, instance: Instance, optimizers: Iterable[Optimizer], n_repeats: int = 30
    ) -> None:
        self.instance = instance
        self.optimizers = tuple(optimizers)
        self._config = ExperimentConfig(n_repeats=n_repeats)

        self._validate_optimizers()

    @property
    def n_repeats(self) -> int:
        return self._config.n_repeats

    @n_repeats.setter
    def n_repeats(self, value: int) -> None:
        self._config = self._config.with_updates(n_repeats=value)

    def _validate_optimizers(self) -> None:
        if not self.optimizers:
            raise ValueError("optimizers must contain at least one optimizer")
        for optimizer in self.optimizers:
            if not isinstance(optimizer, Optimizer):
                raise TypeError("optimizers must contain Optimizer instances")
            if optimizer.instance is not self.instance:
                raise ValueError("All optimizers must use the experiment instance")

    def run(self) -> ExperimentResult:
        """Return fresh results, reusing the supplied optimizers and their RNG state."""
        optimizer_results = []
        for index, optimizer in enumerate(self.optimizers, start=1):
            optimizer_result = self._run_optimizer(optimizer, index)
            optimizer_results.append(optimizer_result)
        return ExperimentResult(optimizer_results=optimizer_results)

    def _run_optimizer(
        self, optimizer: Optimizer, optimizer_id: int
    ) -> OptimizerResult:
        executions = []
        available_items = Counter(self.instance.available_items)

        for repetition in range(1, self.n_repeats + 1):
            start = perf_counter()
            solution = optimizer.optimize()
            elapsed_seconds = perf_counter() - start

            self._check_inconsistent_instances(solution)

            optimization_run = OptimizationRun(
                repetition=repetition,
                solution=solution,
                elapsed_seconds=elapsed_seconds,
                is_valid=_is_optimization_run_valid(available_items, solution),
            )
            executions.append(optimization_run)

        return self._create_optimizer_result(executions, optimizer, optimizer_id)

    def _check_inconsistent_instances(self, solution: Solution) -> None:
        if solution.instance is not self.instance:
            raise ValueError("Optimizer returned a solution for another instance")

    def _create_optimizer_result(
        self, executions: list[Any], optimizer: Optimizer, optimizer_id: int
    ) -> OptimizerResult:
        valid_solutions = [
            execution.solution for execution in executions if execution.is_valid
        ]
        values = [solution.value for solution in valid_solutions]
        volumes = [solution.volume for solution in valid_solutions]

        return OptimizerResult(
            instance=self.instance,
            optimizer=optimizer,
            mean_value=float(np.mean(values)) if values else float("nan"),
            std_value=float(np.std(values)) if values else float("nan"),
            mean_volume=float(np.mean(volumes)) if volumes else float("nan"),
            std_volume=float(np.std(volumes)) if volumes else float("nan"),
            best_solution=max(valid_solutions, key=lambda sol: sol.value, default=None),
            n_repeats=self.n_repeats,
            optimizer_id=optimizer_id,
            executions=executions,
        )


def _is_optimization_run_valid(
    available_items: Counter[Item], solution: Solution
) -> bool:
    return solution.is_valid() and Counter(solution.items) <= available_items
