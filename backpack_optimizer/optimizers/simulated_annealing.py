import math
import random

from backpack_optimizer.config import SimulatedAnnealingConfig
from backpack_optimizer.models import Instance, Solution
from backpack_optimizer.optimizers.base import Optimizer
from backpack_optimizer.optimizers.neighborhood import create_valid_neighborhood


class SimulatedAnnealingOptimizer(Optimizer):
    def __init__(
        self,
        instance: Instance,
        initial_temperature: float = 100.0,
        minimum_temperature: float = 0.1,
        cooling_rate: float = 0.95,
        iterations_per_temperature: int = 100,
        max_iterations: int = 20_000,
        seed: int | None = None,
    ) -> None:
        super().__init__(instance)

        self._config = SimulatedAnnealingConfig(
            initial_temperature=initial_temperature,
            minimum_temperature=minimum_temperature,
            cooling_rate=cooling_rate,
            iterations_per_temperature=iterations_per_temperature,
            max_iterations=max_iterations,
            seed=seed,
        )
        self._random = random.Random(self._config.seed)

    @property
    def initial_temperature(self) -> float:
        return self._config.initial_temperature

    @initial_temperature.setter
    def initial_temperature(self, value: float) -> None:
        self._config = self._config.with_updates(initial_temperature=value)

    @property
    def minimum_temperature(self) -> float:
        return self._config.minimum_temperature

    @minimum_temperature.setter
    def minimum_temperature(self, value: float) -> None:
        self._config = self._config.with_updates(minimum_temperature=value)

    @property
    def cooling_rate(self) -> float:
        return self._config.cooling_rate

    @cooling_rate.setter
    def cooling_rate(self, value: float) -> None:
        self._config = self._config.with_updates(cooling_rate=value)

    @property
    def iterations_per_temperature(self) -> int:
        return self._config.iterations_per_temperature

    @iterations_per_temperature.setter
    def iterations_per_temperature(self, value: int) -> None:
        self._config = self._config.with_updates(iterations_per_temperature=value)

    @property
    def max_iterations(self) -> int:
        return self._config.max_iterations

    @max_iterations.setter
    def max_iterations(self, value: int) -> None:
        self._config = self._config.with_updates(max_iterations=value)

    def optimize(self) -> Solution:
        current_solution = Solution(items=[], instance=self.instance)
        best_solution = current_solution
        temperature = self.initial_temperature
        iterations = 0

        while (
            temperature > self.minimum_temperature and iterations < self.max_iterations
        ):
            attempts = self._attempts_for_temperature(iterations)

            for _ in range(attempts):
                neighbors = self.create_neighborhood(current_solution)
                if not neighbors:
                    return best_solution

                candidate = self._random.choice(neighbors)
                delta = candidate.value - current_solution.value
                best_solution, current_solution = self._update_solutions(
                    best_solution, candidate, current_solution, delta, temperature
                )

                iterations += 1

            temperature *= self.cooling_rate

        return best_solution

    def _update_solutions(
        self,
        best_solution: Solution,
        candidate: Solution,
        current_solution: Solution,
        delta: float,
        temperature: float,
    ) -> tuple[Solution, Solution]:
        if delta >= 0 or self._random.random() < math.exp(delta / temperature):
            current_solution = candidate
            if current_solution.value > best_solution.value:
                best_solution = current_solution
        return best_solution, current_solution

    def _attempts_for_temperature(self, completed_iterations: int) -> int:
        return min(
            self.iterations_per_temperature, self.max_iterations - completed_iterations
        )

    def create_neighborhood(self, current_solution: Solution) -> list[Solution]:
        return create_valid_neighborhood(
            current_solution, self.instance.available_items
        )
