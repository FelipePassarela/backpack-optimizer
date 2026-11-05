from backpack_optimizer.models import Solution
from backpack_optimizer.optimizers.base import Optimizer


class DumbOptimizer(Optimizer):
    def optimize(self) -> Solution:
        for item in self.instance.available_items:
            solution = Solution(items=[item], instance=self.instance)
            if solution.is_valid():
                return solution

        return Solution(items=[], instance=self.instance)

    def create_neighborhood(self, current_solution: Solution) -> list[Solution]:
        pass
