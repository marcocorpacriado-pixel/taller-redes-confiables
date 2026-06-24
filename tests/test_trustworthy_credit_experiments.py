"""Tests for unified experiment orchestration helpers."""

import pandas as pd
import pytest

from src.trustworthy_credit.experiments import (
    ExperimentError,
    ExperimentRunResult,
    MultiSeedParetoConfig,
    MultiSeedParetoRunner,
    MultiSeedParetoSummarizer,
)


def test_multiseed_config_rejects_invalid_grids() -> None:
    """Invalid experiment grids should fail before launching work."""

    with pytest.raises(ExperimentError, match="seed"):
        MultiSeedParetoConfig(seeds=())
    with pytest.raises(ExperimentError, match="lambda"):
        MultiSeedParetoConfig(lambda_values=())
    with pytest.raises(ExperimentError, match="non-negative"):
        MultiSeedParetoConfig(lambda_values=(-1.0,))
    with pytest.raises(ExperimentError, match="metric"):
        MultiSeedParetoConfig(metric_columns=())


def test_experiment_run_result_flattens_metrics() -> None:
    """One run result should convert to a flat record."""

    record = ExperimentRunResult(
        seed=42,
        lambda_fair=1.0,
        metrics={"auc": 0.74, "abs_rho": 0.02},
    ).to_record()

    assert record == {
        "seed": 42,
        "lambda_fair": 1.0,
        "auc": 0.74,
        "abs_rho": 0.02,
    }


def test_multiseed_runner_executes_all_seed_lambda_combinations() -> None:
    """Runner should call the provided experiment function for every grid point."""

    config = MultiSeedParetoConfig(
        seeds=(1, 2),
        lambda_values=(0.0, 5.0),
        metric_columns=("auc", "abs_rho"),
    )
    calls: list[tuple[int, float]] = []

    def fake_experiment(seed: int, lambda_fair: float) -> dict[str, float]:
        calls.append((seed, lambda_fair))
        return {
            "auc": 0.75 - 0.001 * lambda_fair + seed * 0.0001,
            "abs_rho": 0.10 / (1.0 + lambda_fair),
        }

    results = MultiSeedParetoRunner(config).run(fake_experiment)

    assert calls == [(1, 0.0), (1, 5.0), (2, 0.0), (2, 5.0)]
    assert results.shape == (4, 4)
    assert results["lambda_fair"].tolist() == [0.0, 5.0, 0.0, 5.0]


def test_multiseed_runner_rejects_missing_metrics() -> None:
    """Runner should fail when one experiment does not return required metrics."""

    config = MultiSeedParetoConfig(
        seeds=(1,),
        lambda_values=(0.0,),
        metric_columns=("auc", "abs_rho"),
    )

    with pytest.raises(ExperimentError, match="abs_rho"):
        MultiSeedParetoRunner(config).run(lambda seed, lam: {"auc": 0.74})


def test_multiseed_summarizer_aggregates_mean_std_and_counts() -> None:
    """Summarizer should produce one row per lambda with robust statistics."""

    results = pd.DataFrame(
        {
            "seed": [1, 2, 1, 2],
            "lambda_fair": [0.0, 0.0, 5.0, 5.0],
            "auc": [0.75, 0.74, 0.73, 0.72],
            "abs_rho": [0.10, 0.08, 0.02, 0.01],
        }
    )

    summary = MultiSeedParetoSummarizer(
        metric_columns=("auc", "abs_rho")
    ).summarize(results)

    assert summary["lambda_fair"].tolist() == [0.0, 5.0]
    assert summary["n_runs"].tolist() == [2, 2]
    assert summary.loc[0, "auc_mean"] == pytest.approx(0.745)
    assert summary.loc[1, "abs_rho_mean"] == pytest.approx(0.015)
    assert "auc_std" in summary.columns
    assert "abs_rho_std" in summary.columns


def test_multiseed_summarizer_rejects_invalid_results_table() -> None:
    """Summarizer should reject empty or incomplete result tables."""

    summarizer = MultiSeedParetoSummarizer(metric_columns=("auc", "abs_rho"))

    with pytest.raises(ExperimentError, match="empty"):
        summarizer.summarize(pd.DataFrame())
    with pytest.raises(ExperimentError, match="abs_rho"):
        summarizer.summarize(
            pd.DataFrame({"seed": [1], "lambda_fair": [0.0], "auc": [0.74]})
        )
