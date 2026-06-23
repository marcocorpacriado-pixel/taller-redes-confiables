"""Reporting helpers for the unified trustworthy credit package.

These classes turn already computed datasets and result tables into compact
tables for notebooks, reports, and presentations. They do not train models or
change any model output.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class ReportingError(ValueError):
    """Raised when a reporting input table is invalid."""


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Validate that all required columns are present."""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ReportingError("Missing required columns: " + ", ".join(missing))


def _first_existing_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    """Return the first candidate column present in a DataFrame."""

    for column in candidates:
        if column in frame.columns:
            return column
    raise ReportingError(
        "Missing required column. Expected one of: " + ", ".join(candidates)
    )


@dataclass(slots=True)
class DatasetOverviewReporter:
    """Build compact EDA summaries inspired by the Javi notebook."""

    def target_summary(
        self, frame: pd.DataFrame, target_col: str = "TARGET"
    ) -> pd.DataFrame:
        """Return class counts and rates for the target column."""

        _require_columns(frame, (target_col,))
        summary = (
            frame[target_col]
            .value_counts(dropna=False)
            .rename_axis(target_col)
            .reset_index(name="count")
        )
        summary["rate"] = summary["count"] / len(frame)
        return summary.sort_values(target_col).reset_index(drop=True)

    def gender_summary(
        self,
        frame: pd.DataFrame,
        gender_col: str = "CODE_GENDER",
        target_col: str = "TARGET",
    ) -> pd.DataFrame:
        """Return applicant share and default rate by gender."""

        _require_columns(frame, (gender_col, target_col))
        grouped = (
            frame.groupby(gender_col, dropna=False)
            .agg(count=(target_col, "size"), target_rate=(target_col, "mean"))
            .reset_index()
        )
        grouped["share"] = grouped["count"] / len(frame)
        return grouped

    def ext_source_missingness(
        self,
        frame: pd.DataFrame,
        ext_columns: tuple[str, ...] = (
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
        ),
    ) -> pd.DataFrame:
        """Return missingness rates for EXT_SOURCE variables."""

        _require_columns(frame, ext_columns)
        return pd.DataFrame(
            {
                "feature": list(ext_columns),
                "missing_rate": [frame[column].isna().mean() for column in ext_columns],
            }
        )


@dataclass(slots=True)
class ResultsTableFormatter:
    """Format model-comparison and Pareto tables for reporting."""

    model_columns: tuple[str, ...] = ("modelo", "model", "Modelo")
    lambda_columns: tuple[str, ...] = ("lambda_fair", "lambda", "alpha")
    auc_columns: tuple[str, ...] = ("val_auc", "auc", "test_auc")
    fairness_columns: tuple[str, ...] = (
        "val_abs_rho",
        "abs_rho",
        "dpd",
        "dp",
        "dp_gap",
        "fairness",
    )

    def base_vs_fair_table(
        self,
        metrics_df: pd.DataFrame,
        decimals: int = 4,
    ) -> pd.DataFrame:
        """Return a clean base-vs-FAIR table with common metric columns."""

        if metrics_df.empty:
            raise ReportingError("Metrics table is empty.")

        model_col = _first_existing_column(metrics_df, self.model_columns)
        preferred = (
            "auc",
            "pr_auc",
            "f1",
            "recall",
            "precision",
            "abs_rho",
            "dpd",
            "eod",
        )
        columns = [model_col] + [column for column in preferred if column in metrics_df]
        if len(columns) == 1:
            raise ReportingError("Metrics table has no known reportable metrics.")

        table = metrics_df[columns].copy()
        numeric_columns = table.select_dtypes(include="number").columns
        table[numeric_columns] = table[numeric_columns].round(decimals)
        return table.rename(columns={model_col: "model"})

    def pareto_summary_table(
        self,
        pareto_df: pd.DataFrame,
        decimals: int = 4,
    ) -> pd.DataFrame:
        """Return lambda, AUC, fairness, and fairness reduction summary."""

        if pareto_df.empty:
            raise ReportingError("Pareto table is empty.")

        lambda_col = _first_existing_column(pareto_df, self.lambda_columns)
        auc_col = _first_existing_column(pareto_df, self.auc_columns)
        fairness_col = _first_existing_column(pareto_df, self.fairness_columns)

        table = pareto_df[[lambda_col, auc_col, fairness_col]].copy()
        table = table.sort_values(lambda_col).reset_index(drop=True)
        baseline = table[fairness_col].iloc[0]
        if baseline == 0:
            table["fairness_reduction"] = 0.0
        else:
            table["fairness_reduction"] = 1.0 - table[fairness_col] / baseline

        table = table.round(decimals)
        return table.rename(
            columns={
                lambda_col: "lambda",
                auc_col: "auc",
                fairness_col: "fairness",
            }
        )


@dataclass(slots=True)
class UncertaintyNarrativeReporter:
    """Build uncertainty summaries for the final narrative."""

    uncertainty_columns: tuple[str, ...] = ("uncertainty", "var", "variance")
    target_columns: tuple[str, ...] = ("y_true", "TARGET", "target")
    ext_null_columns: tuple[str, ...] = (
        "EXT_NULL_COUNT",
        "n_ext_missing",
        "n_missing",
    )

    def summary_by_target(self, uncertainty_df: pd.DataFrame) -> pd.DataFrame:
        """Return count, mean, and median uncertainty by observed target."""

        uncertainty_col = _first_existing_column(
            uncertainty_df, self.uncertainty_columns
        )
        target_col = _first_existing_column(uncertainty_df, self.target_columns)
        self._validate_uncertainty(uncertainty_df, uncertainty_col)

        return (
            uncertainty_df.groupby(target_col)[uncertainty_col]
            .agg(count="size", mean_uncertainty="mean", median_uncertainty="median")
            .reset_index()
        )

    def summary_by_ext_null_count(self, uncertainty_df: pd.DataFrame) -> pd.DataFrame:
        """Return uncertainty summary by raw EXT_NULL_COUNT values."""

        uncertainty_col = _first_existing_column(
            uncertainty_df, self.uncertainty_columns
        )
        ext_col = _first_existing_column(uncertainty_df, self.ext_null_columns)
        self._validate_uncertainty(uncertainty_df, uncertainty_col)
        self._validate_ext_null_count(uncertainty_df, ext_col)

        return (
            uncertainty_df.groupby(ext_col)[uncertainty_col]
            .agg(count="size", mean_uncertainty="mean", median_uncertainty="median")
            .reset_index()
            .sort_values(ext_col)
        )

    @staticmethod
    def _validate_uncertainty(frame: pd.DataFrame, uncertainty_col: str) -> None:
        """Reject constant uncertainty, matching the MVP defensive checks."""

        if frame.empty:
            raise ReportingError("Uncertainty table is empty.")
        if frame[uncertainty_col].nunique(dropna=True) <= 1:
            raise ReportingError("Uncertainty is constant.")

    @staticmethod
    def _validate_ext_null_count(frame: pd.DataFrame, ext_col: str) -> None:
        """Ensure EXT_NULL_COUNT remains a raw semantic audit feature."""

        values = set(frame[ext_col].dropna().astype(int).unique().tolist())
        invalid = values.difference({0, 1, 2, 3})
        if invalid:
            raise ReportingError(
                "EXT_NULL_COUNT contains invalid values: "
                + ", ".join(str(value) for value in sorted(invalid))
            )
