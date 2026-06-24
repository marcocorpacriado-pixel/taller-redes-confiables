"""Tests for unified experiment orchestration helpers."""

import json

import numpy as np
import pandas as pd
import pytest

from src.trustworthy_credit.experiments import (
    ExperimentError,
    FeatureAblationReporter,
    ExperimentRunResult,
    MCDropoutArtifactReporter,
    ModelProgressionReporter,
    ModelProgressionSpec,
    MultiSeedParetoArtifactReporter,
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


def test_model_progression_reporter_summarizes_history_json(tmp_path) -> None:
    """Model progression reporter should read Keras histories without training."""

    history_path = tmp_path / "M_test_history.json"
    history_path.write_text(
        json.dumps(
            {
                "auc": [0.60, 0.70, 0.69],
                "val_auc": [0.55, 0.72, 0.71],
                "loss": [0.90, 0.80, 0.75],
                "val_loss": [0.95, 0.82, 0.83],
            }
        ),
        encoding="utf-8",
    )
    spec = ModelProgressionSpec(
        model_id="MX",
        model_name="Test model",
        technical_idea="Synthetic history for unit testing.",
        history_filename=history_path.name,
        reported_val_auc=0.72,
        reported_test_auc=0.73,
        n_params=123,
    )

    summary = ModelProgressionReporter(
        checkpoints_dir=tmp_path,
        specs=(spec,),
    ).summarize()

    assert summary.loc[0, "model_id"] == "MX"
    assert summary.loc[0, "history_found"]
    assert summary.loc[0, "epochs"] == 3
    assert summary.loc[0, "best_epoch"] == 2
    assert summary.loc[0, "history_best_val_auc"] == pytest.approx(0.72)
    assert summary.loc[0, "reported_test_auc"] == pytest.approx(0.73)
    assert summary.loc[0, "test_auc_gain_vs_m0"] == pytest.approx(0.0)


def test_multiseed_artifact_reporter_standardizes_aggregated_csv(tmp_path) -> None:
    """Saved aggregated Pareto artifacts should be normalized for reporting."""

    artifact_path = tmp_path / "pareto_v2_results.csv"
    pd.DataFrame(
        {
            "lambda": [0.0, 1.0],
            "auc": [0.746, 0.742],
            "auc_std": [0.001, 0.002],
            "dp": [0.030, 0.003],
            "dp_std": [0.001, 0.001],
            "mean_F": [0.07, 0.075],
            "mean_M": [0.10, 0.078],
        }
    ).to_csv(artifact_path, index=False)

    summary = MultiSeedParetoArtifactReporter().summarize(artifact_path)

    assert summary["lambda_fair"].tolist() == [0.0, 1.0]
    assert summary.loc[0, "auc_mean"] == pytest.approx(0.746)
    assert summary.loc[1, "abs_rho_mean"] == pytest.approx(0.003)
    assert summary.loc[1, "fairness_reduction"] == pytest.approx(0.9)
    assert summary.loc[1, "auc_delta_vs_baseline"] == pytest.approx(-0.004)
    assert summary.loc[0, "n_runs"] == 3


def test_mc_dropout_artifact_reporter_summarizes_saved_arrays(tmp_path) -> None:
    """MC Dropout reporter should summarize saved mean/variance arrays."""

    np.save(tmp_path / "mc_fair_mean.npy", np.array([0.1, 0.2, 0.3]))
    np.save(tmp_path / "mc_fair_var.npy", np.array([0.01, 0.02, 0.04]))

    reporter = MCDropoutArtifactReporter(checkpoints_dir=tmp_path)
    summary = reporter.saved_array_summary(prefixes=("mc_fair",))
    audited = reporter.audited_summary()

    assert summary.loc[0, "artifact"] == "mc_fair"
    assert summary.loc[0, "n_samples"] == 3
    assert summary.loc[0, "median_prediction"] == pytest.approx(0.2)
    assert summary.loc[0, "median_variance"] == pytest.approx(0.02)
    assert "target_1_to_0_variance_ratio" in audited.columns
    assert audited.loc[audited["model"] == "FAIR lambda=1.0", "target_1_to_0_variance_ratio"].iloc[0] > 1.0


def test_feature_ablation_reporter_reports_auc_gain() -> None:
    """Feature ablation reporter should quantify the 12-to-42-feature lift."""

    summary = FeatureAblationReporter().summarize()

    assert set(summary.columns) >= {
        "model",
        "auc_12_features",
        "auc_42_features",
        "auc_gain_42_vs_12",
    }
    assert (summary["auc_gain_42_vs_12"] > 0).all()
