import math
import random

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

        self.initial_temperature = initial_temperature
        self.minimum_temperature = minimum_temperature
        self.cooling_rate = cooling_rate
        self.iterations_per_temperature = iterations_per_temperature
        self.max_iterations = max_iterations
        self._validate_params()
        self._random = random.Random(seed)

    def _validate_params(self) -> None:
        for name, value in (
            ("initial_temperature", self.initial_temperature),
            ("minimum_temperature", self.minimum_temperature),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.initial_temperature <= self.minimum_temperature:
            raise ValueError("initial_temperature must exceed minimum_temperature")
        if not math.isfinite(self.cooling_rate) or not 0 < self.cooling_rate < 1:
            raise ValueError("cooling_rate must be finite and between 0 and 1")
        for name, value in (
            ("iterations_per_temperature", self.iterations_per_temperature),
            ("max_iterations", self.max_iterations),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

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
