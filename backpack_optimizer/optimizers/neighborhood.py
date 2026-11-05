from backpack_optimizer.models import Item, Solution


def create_valid_neighborhood(
    current_solution: Solution, available_items: list[Item]
) -> list[Solution]:
    candidates = _create_addition_candidates(current_solution, available_items)
    candidates += _create_removal_candidates(current_solution)
    return [candidate for candidate in candidates if candidate.is_valid()]


def _create_addition_candidates(
    current_solution: Solution, available_items: list[Item]
) -> list[Solution]:
    remaining_items = available_items.copy()
    for item in current_solution.items:
        remaining_items.remove(item)

    return [
        Solution(
            items=[*current_solution.items, item], instance=current_solution.instance
        )
        for item in remaining_items
    ]


def _create_removal_candidates(current_solution: Solution) -> list[Solution]:
    selected_items = current_solution.items
    return [
        Solution(
            items=selected_items[:index] + selected_items[index + 1 :],
            instance=current_solution.instance,
        )
        for index in range(len(selected_items))
    ]
