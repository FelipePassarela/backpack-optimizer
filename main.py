from backpack import Backpack
from dimensions import Dimensions
from instance import Instance
from item import Item
from optimizer.hill_climbing import HillClimbingOptimizer


def main() -> None:
    instance = Instance(
        backpack=Backpack(max_volume=8100),
        available_items=[
            Item(value=90.0, dimensions=Dimensions(width=20, height=20, depth=20)),
            Item(value=76.5, dimensions=Dimensions(width=20, height=15, depth=20)),
            Item(value=59, dimensions=Dimensions(width=15, height=20, depth=15)),
            Item(value=46, dimensions=Dimensions(width=20, height=10, depth=15)),
            Item(value=37.5, dimensions=Dimensions(width=10, height=25, depth=10)),
            Item(value=29, dimensions=Dimensions(width=10, height=7.5, depth=20)),
            Item(value=23, dimensions=Dimensions(width=8, height=12.5, depth=10)),
            Item(value=16.5, dimensions=Dimensions(width=5, height=12, depth=10)),
            Item(value=12, dimensions=Dimensions(width=5, height=8, depth=10)),
            Item(value=7.5, dimensions=Dimensions(width=5, height=5, depth=10)),
        ],
    )
    optimizer = HillClimbingOptimizer(instance=instance)
    solution = optimizer.optimize()

    print("Available items:")
    print(instance.backpack)
    print(instance.to_dataframe())

    print("\nFound solution:")
    print(solution.to_dataframe())
    print(f"Total value: {solution.value}")
    print(f"Total volume: {solution.volume}")


if __name__ == "__main__":
    main()
