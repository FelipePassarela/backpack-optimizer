from backpack_optimizer.experiment.experiment_result import ExperimentResult
from backpack_optimizer.experiment.optimizer_result import OptimizerResult
from backpack_optimizer.models.presenter import present_solution


def present_experiment_result(result: ExperimentResult) -> str:
    lines = [
        "\nBest optimizer by mean value:",
        present_optimizer_result(result.best_optimizer_result),
        "\nOptimizer comparison:",
        present_optimizer_comparison(result),
    ]
    return "\n".join(lines)


def present_optimizer_result(optimizer_result: OptimizerResult) -> str:
    lines = [
        optimizer_result.optimizer.__class__.__name__,
        f"Mean Value: {optimizer_result.mean_value:.2f}",
        f"Std Value: {optimizer_result.std_value:.2f}",
        f"Mean Volume: {optimizer_result.mean_volume:.2f}",
        f"Std Volume: {optimizer_result.std_volume:.2f}",
        f"Mean Time: {optimizer_result.mean_time:.4f} s",
    ]

    if optimizer_result.best_solution is not None:
        formatted_best_solution = present_solution(optimizer_result.best_solution)
        lines.extend(["Best Solution:", formatted_best_solution])

    return "\n".join(lines)


def present_optimizer_comparison(result: ExperimentResult) -> str:
    return result.to_dataframe().to_string(index=False, float_format="{:.4f}".format)
