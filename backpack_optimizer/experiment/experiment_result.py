from dataclasses import dataclass

import pandas as pd

from backpack_optimizer.experiment.optimizer_result import OptimizerResult
from backpack_optimizer.models import Instance


@dataclass
class ExperimentResult:
    optimizer_results: list[OptimizerResult]

    def to_dataframe(self) -> pd.DataFrame:
        """Compare optimizers by descending mean value, then ascending time."""
        data = []
        for optimizer_result in self.optimizer_results:
            data.append(
                {
                    "Optimizer ID": optimizer_result.optimizer_id,
                    "Optimizer": optimizer_result.optimizer.__class__.__name__,
                    "Mean Value": optimizer_result.mean_value,
                    "Median Value": optimizer_result.median_value,
                    "Std Value": optimizer_result.std_value,
                    "Best Value": (
                        optimizer_result.best_solution.value
                        if optimizer_result.best_solution is not None
                        else float("nan")
                    ),
                    "Worst Value": optimizer_result.worst_value,
                    "Mean Volume": optimizer_result.mean_volume,
                    "Std Volume": optimizer_result.std_volume,
                    "Mean Time (s)": optimizer_result.mean_time,
                    "Std Time (s)": optimizer_result.std_time,
                    "Total Time (s)": optimizer_result.total_time,
                    "Num Repeats": optimizer_result.n_repeats,
                    "Valid Repeats": optimizer_result.n_valid,
                    "Valid (%)": optimizer_result.valid_percentage,
                }
            )

        df = (
            pd.DataFrame(data)
            .sort_values(
                ["Mean Value", "Mean Time (s)"],
                ascending=[False, True],
                na_position="last",
            )
            .reset_index(drop=True)
        )
        return df

    def executions_to_dataframe(self) -> pd.DataFrame:
        """Return every repetition, including invalid solutions."""
        return pd.DataFrame(
            [
                {
                    "Optimizer ID": optimizer_result.optimizer_id,
                    "Optimizer": optimizer_result.optimizer.__class__.__name__,
                    "Repetition": execution.repetition,
                    "Value": execution.solution.value,
                    "Volume": execution.solution.volume,
                    "Valid": execution.is_valid,
                    "Time (s)": execution.elapsed_seconds,
                }
                for optimizer_result in self.optimizer_results
                for execution in optimizer_result.executions
            ]
        )

    @property
    def best_optimizer_result(self) -> OptimizerResult:
        """Select the highest mean value, breaking ties by mean execution time."""
        valid_results = [
            result
            for result in self.optimizer_results
            if result.best_solution is not None
        ]
        best = max(
            valid_results,
            key=lambda result: (result.mean_value, -result.mean_time),
            default=None,
        )
        if best is None:
            raise ValueError("No valid solutions were found in this experiment")
        return best

    @property
    def instance(self) -> Instance:
        return self.best_optimizer_result.instance
