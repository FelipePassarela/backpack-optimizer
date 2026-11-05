from dataclasses import dataclass
from typing import List

import pandas as pd

from backpack_optimizer.models.backpack import Backpack
from backpack_optimizer.models.item import Item


@dataclass(frozen=True)
class Instance:
    backpack: Backpack
    available_items: List[Item]

    def to_dataframe(self) -> pd.DataFrame:
        data = []
        for index, item in enumerate(self.available_items, start=1):
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
