from collections.abc import Iterable, Iterator
from itertools import combinations

from backpack_optimizer.models import Instance, Solution
from backpack_optimizer.optimizers.base import Optimizer


class ExhaustiveSearchOptimizer(Optimizer):
    """Find an optimum by visiting every subset, including the empty subset.

    Items are selected by position, so equal items remain separate entries.
    Search takes O(n * 2**n) time and O(n) auxiliary space. Use only for small
    instances; max_items defaults to 20 and must be a non-negative integer.
    Oversized instances raise ValueError before enumeration, never returning
    a partial result. If no subset is feasible, optimize() raises ValueError.
    """

    def __init__(self, instance: Instance, max_items: int = 20) -> None:
        super().__init__(instance)
        self.max_items = max_items
        self._validate_params()

    def _validate_params(self) -> None:
        if (
            isinstance(self.max_items, bool)
            or not isinstance(self.max_items, int)
            or self.max_items < 0
        ):
            raise ValueError("max_items must be a non-negative integer")

        item_count = len(self.instance.available_items)
        if item_count > self.max_items:
            raise ValueError(
                f"Instance has {item_count} items, exceeding max_items={self.max_items}"
            )

    def optimize(self) -> Solution:
        self._validate_params()
        return _find_best_solution(self._generate_candidates())

    def _generate_candidates(self) -> Iterator[Solution]:
        items = self.instance.available_items

        for size in range(len(items) + 1):
            for indices in combinations(range(len(items)), size):
                selected_items = [items[index] for index in indices]
                yield Solution(items=selected_items, instance=self.instance)


def _find_best_solution(candidates: Iterable[Solution]) -> Solution:
    feasible = (candidate for candidate in candidates if candidate.is_valid())
    best_solution = max(feasible, key=lambda solution: solution.value, default=None)
    if best_solution is None:
        raise ValueError("No feasible solution exists for this instance")
    return best_solution
