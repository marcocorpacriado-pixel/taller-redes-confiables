"""Tests for unified reporting and plotting utilities."""

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from src.trustworthy_credit.plots import ParetoPlotter, PlottingError, UncertaintyPlotter
from src.trustworthy_credit.reporting import (
    DatasetOverviewReporter,
    ReportingError,
    ResultsTableFormatter,
    UncertaintyNarrativeReporter,
)


def test_dataset_overview_reporter_summarizes_core_eda_tables() -> None:
    """Dataset summaries should cover target, gender, and EXT_SOURCE quality."""

    frame = pd.DataFrame(
        {
            "TARGET": [0, 0, 1, 1],
            "CODE_GENDER": ["M", "F", "F", "M"],
            "EXT_SOURCE_1": [0.1, None, 0.3, None],
            "EXT_SOURCE_2": [0.2, 0.4, None, 0.5],
            "EXT_SOURCE_3": [None, None, 0.6, 0.7],
        }
    )

    reporter = DatasetOverviewReporter()

    target = reporter.target_summary(frame)
    gender = reporter.gender_summary(frame)
    missing = reporter.ext_source_missingness(frame)

    assert target["count"].sum() == 4
    assert set(gender["CODE_GENDER"]) == {"F", "M"}
    assert missing["missing_rate"].tolist() == [0.5, 0.25, 0.5]


def test_results_table_formatter_builds_base_vs_fair_and_pareto_tables() -> None:
    """Formatter should produce compact report-ready result tables."""

    formatter = ResultsTableFormatter()
    metrics = pd.DataFrame(
        {
            "modelo": ["Base final", "FAIR final"],
            "auc": [0.743631, 0.738022],
            "pr_auc": [0.222811, 0.218571],
            "abs_rho": [0.097065, 0.008850],
            "unused": [1, 2],
        }
    )
    pareto = pd.DataFrame(
        {
            "lambda_fair": [0.0, 1.0, 5.0],
            "val_auc": [0.7459, 0.7429, 0.7402],
            "val_abs_rho": [0.0984, 0.0257, 0.0098],
        }
    )

    table = formatter.base_vs_fair_table(metrics)
    pareto_table = formatter.pareto_summary_table(pareto)

    assert table.columns.tolist() == ["model", "auc", "pr_auc", "abs_rho"]
    assert pareto_table.columns.tolist() == [
        "lambda",
        "auc",
        "fairness",
        "fairness_reduction",
    ]
    assert pareto_table.loc[2, "fairness_reduction"] > 0.9


def test_uncertainty_reporter_validates_semantic_ext_null_count() -> None:
    """Uncertainty summaries should reject scaled EXT_NULL_COUNT values."""

    reporter = UncertaintyNarrativeReporter()
    valid = pd.DataFrame(
        {
            "y_true": [0, 0, 1, 1],
            "uncertainty": [0.20, 0.30, 0.55, 0.65],
            "EXT_NULL_COUNT": [0, 1, 2, 3],
        }
    )
    invalid = valid.copy()
    invalid.loc[0, "EXT_NULL_COUNT"] = -1

    by_target = reporter.summary_by_target(valid)
    by_ext = reporter.summary_by_ext_null_count(valid)

    assert by_target["count"].tolist() == [2, 2]
    assert by_ext["EXT_NULL_COUNT"].tolist() == [0, 1, 2, 3]
    with pytest.raises(ReportingError, match="invalid values"):
        reporter.summary_by_ext_null_count(invalid)


def test_uncertainty_reporter_rejects_constant_uncertainty() -> None:
    """Constant uncertainty would recreate the old invalid artifact."""

    reporter = UncertaintyNarrativeReporter()
    constant = pd.DataFrame(
        {
            "y_true": [0, 1, 0],
            "uncertainty": [0.0, 0.0, 0.0],
            "EXT_NULL_COUNT": [0, 1, 2],
        }
    )

    with pytest.raises(ReportingError, match="constant"):
        reporter.summary_by_target(constant)


def test_plotters_create_figures_from_synthetic_tables() -> None:
    """Plotters should return Matplotlib figures without needing real artifacts."""

    pareto = pd.DataFrame(
        {
            "lambda_fair": [0.0, 1.0, 5.0],
            "val_auc": [0.7459, 0.7429, 0.7402],
            "val_abs_rho": [0.0984, 0.0257, 0.0098],
        }
    )
    uncertainty = pd.DataFrame(
        {
            "y_true": [0, 0, 1, 1],
            "uncertainty": [0.20, 0.30, 0.55, 0.65],
            "EXT_NULL_COUNT": [0, 1, 2, 3],
        }
    )

    pareto_fig, pareto_ax = ParetoPlotter().plot_auc_vs_fairness(pareto)
    target_fig, target_ax = UncertaintyPlotter().plot_by_target(uncertainty)
    ext_fig, ext_ax = UncertaintyPlotter().plot_by_ext_null_count(uncertainty)

    assert pareto_ax.get_xlabel() == "val_abs_rho"
    assert target_ax.get_ylabel() == "uncertainty"
    assert ext_ax.get_xlabel() == "Missing EXT_SOURCE variables"

    pareto_fig.clf()
    target_fig.clf()
    ext_fig.clf()


def test_uncertainty_plotter_rejects_invalid_inputs() -> None:
    """Plotting should fail fast for misleading uncertainty artifacts."""

    constant = pd.DataFrame(
        {
            "y_true": [0, 1, 0],
            "uncertainty": [0.0, 0.0, 0.0],
            "EXT_NULL_COUNT": [0, 1, 2],
        }
    )

    with pytest.raises(PlottingError, match="constant"):
        UncertaintyPlotter().plot_by_target(constant)
