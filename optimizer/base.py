from abc import ABC, abstractmethod

from instance import Instance
from solution import Solution


class Optimizer(ABC):
    def __init__(self, instance: Instance) -> None:
        self.instance = instance

    @abstractmethod
    def optimize(self) -> Solution: ...

    @abstractmethod
    def create_neighborhood(self, current_solution: Solution) -> list[Solution]: ...
