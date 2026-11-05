from dataclasses import dataclass


@dataclass(frozen=True)
class Backpack:
    max_volume: float
