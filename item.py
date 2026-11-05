from dataclasses import dataclass

from dimensions import Dimensions


@dataclass(frozen=True)
class Item:
    value: float
    dimensions: Dimensions
