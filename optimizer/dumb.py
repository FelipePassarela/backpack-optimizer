from optimizer.base import Optimizer
from solution import Solution


class DumbOptimizer(Optimizer):
    def optimize(self) -> Solution:
        return Solution(items=self.instance.available_items[:3], instance=self.instance)

    def create_neighborhood(self, current_solution: Solution) -> list[Solution]:
        pass
