from backpack_optimizer.models import Solution
from backpack_optimizer.optimizers.base import Optimizer


class DumbOptimizer(Optimizer):
    def optimize(self) -> Solution:
        return Solution(items=self.instance.available_items[:3], instance=self.instance)

    def create_neighborhood(self, current_solution: Solution) -> list[Solution]:
        pass
