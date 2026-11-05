from dataclasses import dataclass

from backpack_optimizer.models.dimensions import Dimensions


@dataclass(frozen=True)
class Item:
    value: float
    dimensions: Dimensions
