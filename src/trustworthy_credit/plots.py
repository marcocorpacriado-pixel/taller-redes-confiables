"""Plotting utilities for the unified trustworthy credit package.

These classes adapt the useful visual ideas from the existing project into a
small object-oriented API. They only consume already computed tables or history
objects; they do not train models or recalculate results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class PlottingError(ValueError):
    """Raised when a plotting input table is missing required information."""


def _first_existing_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    """Return the first candidate column present in a DataFrame."""

    for column in candidates:
        if column in frame.columns:
            return column
    raise PlottingError(
        "Missing required column. Expected one of: " + ", ".join(candidates)
    )


def _save_if_requested(fig: plt.Figure, output_path: str | Path | None) -> None:
    """Save a figure only when an output path is provided."""

    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


@dataclass(slots=True)
class ParetoPlotter:
    """Plot the accuracy-fairness trade-off from a Pareto result table."""

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

    def plot_auc_vs_fairness(
        self,
        pareto_df: pd.DataFrame,
        output_path: str | Path | None = None,
        title: str = "Pareto: accuracy vs fairness",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Create a scatter/line Pareto plot and optionally save it."""

        if pareto_df.empty:
            raise PlottingError("Pareto table is empty.")

        lambda_col = _first_existing_column(pareto_df, self.lambda_columns)
        auc_col = _first_existing_column(pareto_df, self.auc_columns)
        fairness_col = _first_existing_column(pareto_df, self.fairness_columns)

        plot_df = pareto_df.sort_values(fairness_col).copy()

        fig, ax = plt.subplots(figsize=(9, 6))
        sns.lineplot(
            data=plot_df,
            x=fairness_col,
            y=auc_col,
            marker="o",
            color="#4C78A8",
            ax=ax,
        )

        for _, row in plot_df.iterrows():
            ax.annotate(
                f"lambda={row[lambda_col]:g}",
                (row[fairness_col], row[auc_col]),
                textcoords="offset points",
                xytext=(8, 7),
                fontsize=9,
            )

        ax.set_title(title)
        ax.set_xlabel(fairness_col)
        ax.set_ylabel(auc_col)
        ax.grid(alpha=0.3)
        fig.tight_layout()

        _save_if_requested(fig, output_path)
        return fig, ax


@dataclass(slots=True)
class TrainingCurvePlotter:
    """Plot train/validation loss and AUC curves from history dictionaries."""

    def plot_base_vs_fair(
        self,
        base_history: Mapping[str, list[float]] | Any,
        fair_history: Mapping[str, list[float]] | Any,
        output_path: str | Path | None = None,
        title: str = "Training curves: base vs FAIR",
    ) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
        """Plot validation loss and validation AUC for base and FAIR models."""

        base = self._history_mapping(base_history)
        fair = self._history_mapping(fair_history)
        for history_name, history in (("base", base), ("fair", fair)):
            self._require_history_columns(history, history_name)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        fig.suptitle(title)

        self._plot_metric(axes[0], base, fair, "val_loss", "Validation loss")
        self._plot_metric(axes[1], base, fair, "val_auc", "Validation AUC")

        fig.tight_layout()
        _save_if_requested(fig, output_path)
        return fig, (axes[0], axes[1])

    @staticmethod
    def _history_mapping(history: Mapping[str, list[float]] | Any) -> Mapping[str, Any]:
        """Accept either a Keras History object or a plain mapping."""

        if hasattr(history, "history"):
            return history.history
        return history

    @staticmethod
    def _require_history_columns(history: Mapping[str, Any], history_name: str) -> None:
        """Validate the minimal history columns used by this plotter."""

        missing = [column for column in ("val_loss", "val_auc") if column not in history]
        if missing:
            raise PlottingError(
                f"{history_name} history is missing columns: {', '.join(missing)}"
            )

    @staticmethod
    def _plot_metric(
        ax: plt.Axes,
        base: Mapping[str, Any],
        fair: Mapping[str, Any],
        metric: str,
        ylabel: str,
    ) -> None:
        """Plot one metric for base and FAIR histories."""

        ax.plot(range(1, len(base[metric]) + 1), base[metric], label="Base")
        ax.plot(range(1, len(fair[metric]) + 1), fair[metric], label="FAIR")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()


@dataclass(slots=True)
class UncertaintyPlotter:
    """Plot M2 uncertainty distributions and data-quality relationships."""

    uncertainty_columns: tuple[str, ...] = ("uncertainty", "var", "variance")
    target_columns: tuple[str, ...] = ("y_true", "TARGET", "target")
    ext_null_columns: tuple[str, ...] = (
        "EXT_NULL_COUNT",
        "n_ext_missing",
        "n_missing",
    )

    def plot_by_target(
        self,
        uncertainty_df: pd.DataFrame,
        output_path: str | Path | None = None,
        title: str = "Uncertainty by observed target",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot uncertainty by target using a robust boxplot."""

        uncertainty_col = _first_existing_column(
            uncertainty_df, self.uncertainty_columns
        )
        target_col = _first_existing_column(uncertainty_df, self.target_columns)
        self._validate_uncertainty(uncertainty_df, uncertainty_col)

        plot_df = uncertainty_df[[target_col, uncertainty_col]].copy()
        plot_df[target_col] = plot_df[target_col].map(
            {0: "Good payer (0)", 1: "Bad payer (1)"}
        ).fillna(plot_df[target_col].astype(str))

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=plot_df, x=target_col, y=uncertainty_col, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Observed class")
        ax.set_ylabel(uncertainty_col)
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()

        _save_if_requested(fig, output_path)
        return fig, ax

    def plot_by_ext_null_count(
        self,
        uncertainty_df: pd.DataFrame,
        output_path: str | Path | None = None,
        title: str = "Uncertainty vs missing EXT_SOURCE count",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot uncertainty by the semantic raw EXT_NULL_COUNT audit feature."""

        uncertainty_col = _first_existing_column(
            uncertainty_df, self.uncertainty_columns
        )
        ext_col = _first_existing_column(uncertainty_df, self.ext_null_columns)
        self._validate_uncertainty(uncertainty_df, uncertainty_col)
        self._validate_ext_null_count(uncertainty_df, ext_col)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=uncertainty_df, x=ext_col, y=uncertainty_col, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Missing EXT_SOURCE variables")
        ax.set_ylabel(uncertainty_col)
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()

        _save_if_requested(fig, output_path)
        return fig, ax

    @staticmethod
    def _validate_uncertainty(frame: pd.DataFrame, uncertainty_col: str) -> None:
        """Reject empty or constant uncertainty before drawing misleading plots."""

        if frame.empty:
            raise PlottingError("Uncertainty table is empty.")
        if frame[uncertainty_col].nunique(dropna=True) <= 1:
            raise PlottingError("Uncertainty is constant; plot would be misleading.")

    @staticmethod
    def _validate_ext_null_count(frame: pd.DataFrame, ext_col: str) -> None:
        """Validate that EXT_NULL_COUNT keeps its semantic raw values."""

        values = set(frame[ext_col].dropna().astype(int).unique().tolist())
        invalid = values.difference({0, 1, 2, 3})
        if invalid:
            raise PlottingError(
                "EXT_NULL_COUNT contains non-semantic values: "
                + ", ".join(str(value) for value in sorted(invalid))
            )
