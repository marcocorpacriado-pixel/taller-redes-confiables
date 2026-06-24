"""Tests for executable squared-DP fairness experiments."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.trustworthy_credit.fairness_experiments import (
    FairnessExperimentDataset,
    FairnessExperimentError,
    SquaredDPModelFactory,
    SquaredDPSweepConfig,
    SquaredDPSweepRunner,
)


def _synthetic_fairness_dataset(n_samples: int = 96) -> FairnessExperimentDataset:
    """Create deterministic binary data with target and sensitive variation."""

    rng = np.random.default_rng(123)
    X = rng.normal(size=(n_samples, 6)).astype(np.float32)
    sensitive = np.tile(np.array([0, 1], dtype=int), n_samples // 2)
    score = 1.3 * X[:, 0] - 0.8 * X[:, 1] + 0.35 * sensitive
    target = (score > np.median(score)).astype(int)

    return FairnessExperimentDataset(
        X_train=X[:56],
        y_train=target[:56],
        s_train=sensitive[:56],
        X_val=X[56:76],
        y_val=target[56:76],
        s_val=sensitive[56:76],
        X_test=X[76:],
        y_test=target[76:],
        s_test=sensitive[76:],
    )


def test_squared_dp_sweep_config_rejects_invalid_values() -> None:
    """Invalid sweep settings should fail before training."""

    with pytest.raises(FairnessExperimentError, match="alpha"):
        SquaredDPSweepConfig(alphas=(-1.0,))
    with pytest.raises(FairnessExperimentError, match="epochs"):
        SquaredDPSweepConfig(epochs=0)
    with pytest.raises(FairnessExperimentError, match="dropout"):
        SquaredDPSweepConfig(dropout=1.0)
    with pytest.raises(FairnessExperimentError, match="threshold_strategy"):
        SquaredDPSweepConfig(threshold_strategy="bad")  # type: ignore[arg-type]


def test_fairness_experiment_dataset_rejects_bad_shapes() -> None:
    """Dataset validation should catch shape mismatches."""

    dataset = _synthetic_fairness_dataset()

    with pytest.raises(FairnessExperimentError, match="same number"):
        FairnessExperimentDataset(
            X_train=dataset.X_train,
            y_train=dataset.y_train[:-1],
            s_train=dataset.s_train,
            X_val=dataset.X_val,
            y_val=dataset.y_val,
            s_val=dataset.s_val,
            X_test=dataset.X_test,
            y_test=dataset.y_test,
            s_test=dataset.s_test,
        )


def test_fairness_experiment_dataset_requires_binary_sensitive_groups() -> None:
    """Sensitive attributes should be binary and contain both groups."""

    dataset = _synthetic_fairness_dataset()
    all_one_sensitive = np.ones_like(dataset.s_train)

    with pytest.raises(FairnessExperimentError, match="both"):
        FairnessExperimentDataset(
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            s_train=all_one_sensitive,
            X_val=dataset.X_val,
            y_val=dataset.y_val,
            s_val=dataset.s_val,
            X_test=dataset.X_test,
            y_test=dataset.y_test,
            s_test=dataset.s_test,
        )


def test_squared_dp_model_factory_builds_compiled_model() -> None:
    """Factory should build a compiled probability model for one alpha."""

    config = SquaredDPSweepConfig(epochs=1, batch_size=8, hidden_units=(8,))
    model = SquaredDPModelFactory(config).build(input_dim=6, alpha=2.0)

    assert model.name == "squared_dp_alpha_2p0"
    assert model.output_shape == (None, 1)
    assert model.loss is not None


def test_squared_dp_sweep_runner_trains_and_writes_artifacts(tmp_path) -> None:
    """Runner should train one model per alpha and persist results."""

    dataset = _synthetic_fairness_dataset()
    config = SquaredDPSweepConfig(
        alphas=(0.0, 2.0),
        epochs=2,
        batch_size=16,
        learning_rate=1e-2,
        hidden_units=(12, 6),
        dropout=0.0,
        early_stopping_patience=2,
        threshold_strategy="fixed",
        fixed_threshold=0.5,
        seed=7,
    )
    runner = SquaredDPSweepRunner(config=config)

    results = runner.run(dataset, output_dir=tmp_path, verbose=0)

    assert results["alpha"].tolist() == [0.0, 2.0]
    assert set(results.columns) >= {
        "val_auc",
        "val_abs_rho",
        "val_dp_gap",
        "test_auc",
        "test_abs_rho",
        "test_dpd",
        "test_eod",
        "test_f1",
    }
    assert results["test_auc"].between(0.0, 1.0).all()
    assert results["test_pr_auc"].between(0.0, 1.0).all()
    assert results["threshold"].eq(0.5).all()

    csv_path = tmp_path / "squared_dp_sweep_results.csv"
    assert csv_path.exists()
    for alpha in ("0p0", "2p0"):
        history_path = tmp_path / f"squared_dp_history_alpha_{alpha}.json"
        assert history_path.exists()
        history = json.loads(history_path.read_text(encoding="utf-8"))
        assert "val_loss" in history


def test_squared_dp_sweep_runner_accepts_class_weights(tmp_path) -> None:
    """Class weights should be converted to sample weights for augmented targets."""

    dataset = _synthetic_fairness_dataset()
    config = SquaredDPSweepConfig(
        alphas=(0.0,),
        epochs=1,
        batch_size=16,
        hidden_units=(8,),
        dropout=0.0,
        threshold_strategy="fixed",
        seed=9,
    )

    results = SquaredDPSweepRunner(config=config).run(
        dataset,
        class_weight={0: 1.0, 1: 2.0},
        output_dir=tmp_path,
        verbose=0,
    )

    assert results.shape[0] == 1
    assert results.loc[0, "alpha"] == pytest.approx(0.0)


def test_squared_dp_sweep_runner_rejects_incomplete_class_weights() -> None:
    """Missing class weights should fail with an explicit message."""

    dataset = _synthetic_fairness_dataset()
    config = SquaredDPSweepConfig(
        alphas=(0.0,),
        epochs=1,
        batch_size=16,
        hidden_units=(8,),
    )

    with pytest.raises(FairnessExperimentError, match="missing labels"):
        SquaredDPSweepRunner(config=config).run(
            dataset,
            class_weight={0: 1.0},
            verbose=0,
        )
