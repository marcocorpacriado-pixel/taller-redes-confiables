"""Tests for executable M0-M6 model progression utilities."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.trustworthy_credit.model_progression import (
    DEFAULT_PROGRESSION_SPECS,
    DebtRatioSaturationLayer,
    ExtSourceIndexLayer,
    ModelProgressionError,
    ModelProgressionRunner,
    ModelProgressionTrainingConfig,
    ProgressionDataset,
    ProgressionFeatureIndices,
    ProgressionModelFactory,
)


def _synthetic_dataset(n_samples: int = 72) -> ProgressionDataset:
    """Build a deterministic binary dataset with custom-branch columns."""

    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, 8)).astype(np.float32)

    # Required semantic columns for M4-M6.
    X[:, 0] = np.clip(rng.normal(loc=0.35, scale=0.12, size=n_samples), 0.02, 1.2)
    X[:, 1:4] = rng.uniform(0.0, 1.0, size=(n_samples, 3))

    score = 1.5 * X[:, 0] - 1.2 * X[:, 1] + 0.5 * X[:, 4]
    threshold = np.median(score)
    y = (score > threshold).astype(int)

    return ProgressionDataset(
        X_train=X[:40],
        y_train=y[:40],
        X_val=X[40:56],
        y_val=y[40:56],
        X_test=X[56:],
        y_test=y[56:],
    )


def test_training_config_rejects_invalid_values() -> None:
    """Invalid training settings should fail before Keras is invoked."""

    with pytest.raises(ModelProgressionError, match="epochs"):
        ModelProgressionTrainingConfig(epochs=0)
    with pytest.raises(ModelProgressionError, match="dropout"):
        ModelProgressionTrainingConfig(dropout_rates=(1.0, 0.2))
    with pytest.raises(ModelProgressionError, match="reduce_lr_factor"):
        ModelProgressionTrainingConfig(reduce_lr_factor=1.0)


def test_feature_indices_resolve_semantic_columns() -> None:
    """Feature names should resolve the custom-branch columns by meaning."""

    feature_names = [
        "DEBT_RATIO",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "AMT_CREDIT",
    ]

    indices = ProgressionFeatureIndices.from_feature_names(feature_names)

    assert indices.require_debt_ratio() == 0
    assert indices.require_ext_sources() == (1, 2, 3)


def test_feature_indices_raise_clear_errors_when_missing() -> None:
    """Custom architectures should fail clearly when required columns are absent."""

    indices = ProgressionFeatureIndices.from_feature_names(["AMT_CREDIT"])

    with pytest.raises(ModelProgressionError, match="DEBT_RATIO"):
        indices.require_debt_ratio()
    with pytest.raises(ModelProgressionError, match="EXT_SOURCE"):
        indices.require_ext_sources()


def test_factory_builds_all_progression_models() -> None:
    """The factory should build every M0-M6 architecture with stable names."""

    config = ModelProgressionTrainingConfig(epochs=1, batch_size=8)
    factory = ProgressionModelFactory(config)
    indices = ProgressionFeatureIndices(debt_ratio_idx=0, ext_source_idxs=(1, 2, 3))

    for spec in DEFAULT_PROGRESSION_SPECS:
        model = factory.build(spec, input_dim=8, feature_indices=indices)
        assert model.name == spec.model_id
        assert model.output_shape == (None, 1)
        assert model.count_params() > 0

    m6 = factory.build(DEFAULT_PROGRESSION_SPECS[-1], input_dim=8, feature_indices=indices)
    assert isinstance(m6.get_layer("debt_saturation"), DebtRatioSaturationLayer)
    assert isinstance(m6.get_layer("ext_source_index"), ExtSourceIndexLayer)


def test_runner_trains_selected_models_and_writes_histories(tmp_path) -> None:
    """Runner should train selected models and return a compact metric table."""

    dataset = _synthetic_dataset()
    config = ModelProgressionTrainingConfig(
        epochs=2,
        batch_size=16,
        learning_rate=1e-2,
        early_stopping_patience=2,
        reduce_lr_patience=1,
        seed=123,
    )
    runner = ModelProgressionRunner(config=config)
    indices = ProgressionFeatureIndices(debt_ratio_idx=0, ext_source_idxs=(1, 2, 3))

    results = runner.run(
        dataset,
        feature_indices=indices,
        model_ids=("M0", "M4", "M6"),
        output_dir=tmp_path,
        verbose=0,
    )

    assert results["model_id"].tolist() == ["M0", "M4", "M6"]
    assert set(results.columns) >= {
        "best_val_auc",
        "test_auc",
        "test_pr_auc",
        "test_precision",
        "test_recall",
        "test_f1",
        "history_path",
    }
    assert results["test_auc"].between(0.0, 1.0).all()
    assert results["best_epoch"].ge(1).all()

    for model_id in ("M0", "M4", "M6"):
        history_path = tmp_path / f"{model_id}_history.json"
        assert history_path.exists()
        history = json.loads(history_path.read_text(encoding="utf-8"))
        assert "val_auc" in history


def test_runner_rejects_unknown_model_id() -> None:
    """Unknown model identifiers should fail before training starts."""

    runner = ModelProgressionRunner(
        config=ModelProgressionTrainingConfig(epochs=1, batch_size=8)
    )

    with pytest.raises(ModelProgressionError, match="Unknown"):
        runner.run(_synthetic_dataset(), model_ids=("MX",))
