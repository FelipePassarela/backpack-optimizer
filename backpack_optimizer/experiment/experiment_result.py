from dataclasses import dataclass

import pandas as pd

from backpack_optimizer.experiment.run_result import RunResult
from backpack_optimizer.models import Instance


@dataclass
class ExperimentResult:
    run_results: list[RunResult]

    def to_dataframe(self) -> pd.DataFrame:
        """Compare optimizers by descending mean value, then ascending time."""
        data = []
        for run in self.run_results:
            data.append(
                {
                    "Optimizer ID": run.optimizer_id,
                    "Optimizer": run.optimizer.__class__.__name__,
                    "Mean Value": run.mean_value,
                    "Median Value": run.median_value,
                    "Std Value": run.std_value,
                    "Best Value": (
                        run.best_solution.value
                        if run.best_solution is not None
                        else float("nan")
                    ),
                    "Worst Value": run.worst_value,
                    "Mean Volume": run.mean_volume,
                    "Std Volume": run.std_volume,
                    "Mean Time (s)": run.mean_time,
                    "Std Time (s)": run.std_time,
                    "Total Time (s)": run.total_time,
                    "Num Repeats": run.n_repeats,
                    "Valid Repeats": run.n_valid,
                    "Valid (%)": run.valid_percentage,
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
                    "Optimizer ID": run.optimizer_id,
                    "Optimizer": run.optimizer.__class__.__name__,
                    "Repetition": execution.repetition,
                    "Value": execution.solution.value,
                    "Volume": execution.solution.volume,
                    "Valid": execution.is_valid,
                    "Time (s)": execution.elapsed_seconds,
                }
                for run in self.run_results
                for execution in run.executions
            ]
        )

    @property
    def best_run(self) -> RunResult:
        """Select the highest mean value, breaking ties by mean execution time."""
        valid_runs = [run for run in self.run_results if run.best_solution is not None]
        best = max(
            valid_runs, key=lambda run: (run.mean_value, -run.mean_time), default=None
        )
        if best is None:
            raise ValueError("No valid solutions were found in this experiment")
        return best

    @property
    def instance(self) -> Instance:
        return self.best_run.instance
