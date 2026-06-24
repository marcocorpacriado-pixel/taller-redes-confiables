"""Experiment orchestration helpers for the unified MVP.

This module turns the multi-seed Pareto idea from the original procedural
script into a small, testable orchestration layer. It does not train models by
itself; callers provide a function that runs one seed/lambda experiment.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pandas as pd


class ExperimentError(ValueError):
    """Raised when an experiment configuration or result table is invalid."""


@dataclass(frozen=True, slots=True)
class MultiSeedParetoConfig:
    """Configuration for a multi-seed fairness Pareto experiment."""

    seeds: tuple[int, ...] = (42, 123, 7)
    lambda_values: tuple[float, ...] = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0)
    metric_columns: tuple[str, ...] = ("auc", "abs_rho", "dpd", "eod")

    def __post_init__(self) -> None:
        """Validate experiment grid settings."""

        if not self.seeds:
            raise ExperimentError("At least one seed is required.")
        if not self.lambda_values:
            raise ExperimentError("At least one lambda value is required.")
        if any(lambda_value < 0 for lambda_value in self.lambda_values):
            raise ExperimentError("lambda_values must be non-negative.")
        if not self.metric_columns:
            raise ExperimentError("At least one metric column is required.")


@dataclass(frozen=True, slots=True)
class ExperimentRunResult:
    """One result row from a seed/lambda experiment."""

    seed: int
    lambda_fair: float
    metrics: Mapping[str, float]

    def to_record(self) -> dict[str, float | int]:
        """Convert the result to a flat DataFrame-ready record."""

        record: dict[str, float | int] = {
            "seed": self.seed,
            "lambda_fair": self.lambda_fair,
        }
        for key, value in self.metrics.items():
            record[key] = float(value)
        return record


@dataclass(slots=True)
class MultiSeedParetoRunner:
    """Run a callable over all seed/lambda combinations."""

    config: MultiSeedParetoConfig

    def run(
        self,
        run_single_experiment: Callable[[int, float], Mapping[str, float]],
    ) -> pd.DataFrame:
        """Run the configured grid and return one row per experiment."""

        records: list[dict[str, float | int]] = []
        for seed in self.config.seeds:
            for lambda_fair in self.config.lambda_values:
                metrics = run_single_experiment(seed, lambda_fair)
                self._validate_metrics(metrics)
                records.append(
                    ExperimentRunResult(
                        seed=seed,
                        lambda_fair=lambda_fair,
                        metrics=metrics,
                    ).to_record()
                )
        return pd.DataFrame.from_records(records)

    def _validate_metrics(self, metrics: Mapping[str, float]) -> None:
        """Validate that one experiment produced the configured metrics."""

        missing = [
            metric for metric in self.config.metric_columns if metric not in metrics
        ]
        if missing:
            raise ExperimentError(
                "Experiment result is missing metrics: " + ", ".join(missing)
            )


@dataclass(slots=True)
class MultiSeedParetoSummarizer:
    """Aggregate seed-level Pareto rows into mean/std summary tables."""

    metric_columns: tuple[str, ...] = ("auc", "abs_rho", "dpd", "eod")

    def summarize(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Summarize metrics by lambda with mean, std, and run count."""

        self._validate_results(results_df)
        aggregations: dict[str, list[str]] = {
            metric: ["mean", "std"] for metric in self.metric_columns
        }
        aggregations["seed"] = ["count"]

        summary = results_df.groupby("lambda_fair", as_index=False).agg(aggregations)
        summary.columns = self._flatten_columns(summary.columns)
        summary = summary.rename(columns={"seed_count": "n_runs"})
        return summary.sort_values("lambda_fair").reset_index(drop=True)

    def _validate_results(self, results_df: pd.DataFrame) -> None:
        """Validate the seed-level result table."""

        if results_df.empty:
            raise ExperimentError("Results table is empty.")
        required = {"seed", "lambda_fair", *self.metric_columns}
        missing = sorted(required.difference(results_df.columns))
        if missing:
            raise ExperimentError(
                "Results table is missing columns: " + ", ".join(missing)
            )

    @staticmethod
    def _flatten_columns(columns: pd.Index) -> list[str]:
        """Flatten pandas aggregation MultiIndex columns."""

        flattened: list[str] = []
        for column in columns:
            if not isinstance(column, tuple):
                flattened.append(str(column))
                continue
            base, suffix = column
            if suffix:
                flattened.append(f"{base}_{suffix}")
            else:
                flattened.append(str(base))
        return flattened
