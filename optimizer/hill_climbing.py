import random

from instance import Instance
from optimizer.base import Optimizer
from optimizer.neighborhood import create_valid_neighborhood
from solution import Solution


class HillClimbingOptimizer(Optimizer):
    def __init__(self, instance: Instance, neighborhood_size: int = 3) -> None:
        super().__init__(instance)
        self.neighborhood_size = neighborhood_size

    def optimize(self) -> Solution:
        current_solution = Solution(items=[], instance=self.instance)

        while True:
            neighbors = self.create_neighborhood(current_solution)
            elite = _find_elite(neighbors)
            if elite is None or elite.value <= current_solution.value:
                break
            current_solution = elite

        return current_solution

    def create_neighborhood(self, current_solution: Solution) -> list[Solution]:
        neighborhood = create_valid_neighborhood(
            current_solution, self.instance.available_items
        )
        return _random_neighborhood_subsample(neighborhood, self.neighborhood_size)


def _find_elite(neighbors: list[Solution]) -> Solution | None:
    feasible = (solution for solution in neighbors if solution.is_valid())
    return max(feasible, key=lambda sol: sol.value, default=None)


def _random_neighborhood_subsample(
    neighborhood: list[Solution], subsample_size: int
) -> list[Solution]:
    return random.sample(neighborhood, k=min(subsample_size, len(neighborhood)))
