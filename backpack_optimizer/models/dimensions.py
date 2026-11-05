from dataclasses import dataclass


@dataclass(frozen=True)
class Dimensions:
    width: float
    height: float
    depth: float

    @property
    def volume(self) -> float:
        return self.width * self.height * self.depth
