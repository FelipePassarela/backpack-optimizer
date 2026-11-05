import random

from instance import Instance
from item import Item
from optimizer import Optimizer
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
        addition_candidates = _create_addition_candidates(
            current_solution, self.instance.available_items
        )
        removal_candidates = _create_removal_candidates(current_solution)
        candidates = addition_candidates + removal_candidates
        neighborhood = _create_valid_solutions(candidates)

        return _random_neighborhood_subsample(neighborhood, self.neighborhood_size)


def _create_addition_candidates(
    current_solution: Solution, available_items: list[Item]
) -> list[Solution]:
    remaining_items = available_items.copy()
    for item in current_solution.items:
        remaining_items.remove(item)

    return [
        Solution(
            items=[*current_solution.items, item], instance=current_solution.instance
        )
        for item in remaining_items
    ]


def _create_removal_candidates(current_solution: Solution) -> list[Solution]:
    selected_items = current_solution.items
    return [
        Solution(
            items=selected_items[:index] + selected_items[index + 1 :],
            instance=current_solution.instance,
        )
        for index in range(len(selected_items))
    ]


def _create_valid_solutions(candidates: list[Solution]) -> list[Solution]:
    return [candidate for candidate in candidates if candidate.is_valid()]


def _find_elite(neighbors: list[Solution]) -> Solution | None:
    feasible = (solution for solution in neighbors if solution.is_valid())
    return max(feasible, key=lambda sol: sol.value, default=None)


def _random_neighborhood_subsample(
    neighborhood: list[Solution], subsample_size: int
) -> list[Solution]:
    return random.sample(neighborhood, k=min(subsample_size, len(neighborhood)))
