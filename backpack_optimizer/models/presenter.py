from backpack_optimizer.models import Solution, Instance


def present_solution(solution: Solution) -> str:
    df = solution.to_dataframe()
    return df.to_string(index=False, float_format="{:.2f}".format)


def present_instance(instance: Instance) -> str:
    lines = [str(instance.backpack), str(instance.to_dataframe())]
    return "\n".join(lines)
