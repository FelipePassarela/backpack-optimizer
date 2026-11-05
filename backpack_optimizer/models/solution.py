from dataclasses import dataclass
from typing import List

import pandas as pd

from backpack_optimizer.models.instance import Instance
from backpack_optimizer.models.item import Item


@dataclass(frozen=True)
class Solution:
    items: List[Item]
    instance: Instance

    def to_dataframe(self) -> pd.DataFrame:
        data = []
        for index, item in enumerate(self.items, start=1):
            data.append(
                {
                    "Item": index,
                    "Value": item.value,
                    "Width": item.dimensions.width,
                    "Height": item.dimensions.height,
                    "Depth": item.dimensions.depth,
                    "Volume": item.dimensions.volume,
                }
            )
        return pd.DataFrame(data)

    @property
    def value(self) -> float:
        return sum(item.value for item in self.items)

    @property
    def volume(self) -> float:
        return sum(item.dimensions.volume for item in self.items)

    def is_valid(self) -> bool:
        max_volume = self.instance.backpack.max_volume
        return self.volume <= max_volume
