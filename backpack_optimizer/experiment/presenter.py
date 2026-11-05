from backpack_optimizer.experiment.experiment_result import ExperimentResult
from backpack_optimizer.experiment.run_result import RunResult
from backpack_optimizer.models.presenter import present_solution


def present_experiment_result(result: ExperimentResult) -> str:
    lines = [
        "\nBest optimizer by mean value:",
        present_run_result(result.best_run),
        "\nOptimizer comparison:",
        present_optimizer_comparison(result),
    ]
    return "\n".join(lines)


def present_run_result(run_result: RunResult) -> str:
    lines = [
        run_result.optimizer.__class__.__name__,
        f"Mean Value: {run_result.mean_value:.2f}",
        f"Std Value: {run_result.std_value:.2f}",
        f"Mean Volume: {run_result.mean_volume:.2f}",
        f"Std Volume: {run_result.std_volume:.2f}",
        f"Mean Time: {run_result.mean_time:.4f} s",
    ]

    if run_result.best_solution is not None:
        formatted_best_solution = present_solution(run_result.best_solution)
        lines.extend(["Best Solution:", formatted_best_solution])

    return "\n".join(lines)


def present_optimizer_comparison(result: ExperimentResult) -> str:
    return result.to_dataframe().to_string(index=False, float_format="{:.4f}".format)
