from dataclasses import dataclass

from backpack_optimizer.models import Solution


@dataclass(frozen=True)
class OptimizationRun:
    """The solution and elapsed optimization time for one repetition."""

    repetition: int
    solution: Solution
    elapsed_seconds: float
    is_valid: bool
