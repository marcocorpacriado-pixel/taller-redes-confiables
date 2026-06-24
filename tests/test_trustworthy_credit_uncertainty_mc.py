"""Tests for MC Dropout uncertainty utilities."""

import numpy as np
import pytest
import tensorflow as tf

from src.trustworthy_credit.uncertainty_mc import (
    MCDropoutConfig,
    MCDropoutError,
    MCDropoutSummaryBuilder,
    MCDropoutUncertaintyEstimator,
)


def _build_dropout_model(input_dim: int = 3) -> tf.keras.Model:
    """Create a tiny stochastic model for MC Dropout tests."""

    tf.keras.utils.set_random_seed(123)
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(4, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs)


def _build_dual_dropout_model(input_dim: int = 3) -> tf.keras.Model:
    """Create a tiny dual-input stochastic model."""

    tf.keras.utils.set_random_seed(123)
    features = tf.keras.Input(shape=(input_dim,), name="features")
    sensitive = tf.keras.Input(shape=(1,), name="sensitive")
    x = tf.keras.layers.Concatenate()([features, sensitive])
    x = tf.keras.layers.Dense(4, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model([features, sensitive], outputs)


def test_mc_dropout_config_rejects_invalid_values() -> None:
    """Invalid MC Dropout runtime settings should fail fast."""

    with pytest.raises(MCDropoutError, match="n_passes"):
        MCDropoutConfig(n_passes=1)
    with pytest.raises(MCDropoutError, match="batch_size"):
        MCDropoutConfig(batch_size=0)
    with pytest.raises(MCDropoutError, match="threshold"):
        MCDropoutConfig(threshold=1.5)


def test_mc_dropout_estimator_returns_expected_shapes() -> None:
    """Estimator should produce one prediction, variance, and label per sample."""

    model = _build_dropout_model()
    X = np.ones((6, 3), dtype=np.float32)
    estimator = MCDropoutUncertaintyEstimator(
        MCDropoutConfig(n_passes=5, batch_size=2, threshold=0.5, random_seed=42)
    )

    result = estimator.predict(model, X)

    assert result.mean_proba.shape == (6,)
    assert result.variance.shape == (6,)
    assert result.pred_label.shape == (6,)
    assert result.all_passes is not None
    assert result.all_passes.shape == (6, 5)
    assert np.all(result.variance >= 0.0)


def test_mc_dropout_estimator_supports_dual_inputs() -> None:
    """Estimator should support the dual-input shape used by FAIR models."""

    model = _build_dual_dropout_model()
    X = np.ones((5, 3), dtype=np.float32)
    sensitive = np.array([0, 1, 0, 1, 1], dtype=np.float32)
    estimator = MCDropoutUncertaintyEstimator(
        MCDropoutConfig(n_passes=4, batch_size=2, random_seed=42)
    )

    result = estimator.predict(model, X, sensitive=sensitive)

    assert result.mean_proba.shape == (5,)
    assert result.all_passes is not None
    assert result.all_passes.shape == (5, 4)


def test_mc_dropout_summary_builder_returns_target_and_ext_tables() -> None:
    """Summary builder should aggregate variance by target and EXT_NULL_COUNT."""

    result = MCDropoutUncertaintyEstimator(
        MCDropoutConfig(n_passes=4, batch_size=3, random_seed=42)
    ).predict(_build_dropout_model(), np.ones((6, 3), dtype=np.float32))
    builder = MCDropoutSummaryBuilder()

    by_target = builder.summary_by_target(result, y_true=np.array([0, 0, 1, 1, 0, 1]))
    by_ext = builder.summary_by_ext_null_count(
        result,
        ext_null_count=np.array([0, 1, 2, 3, 1, 2]),
    )
    frame = builder.to_frame(
        result,
        y_true=np.array([0, 0, 1, 1, 0, 1]),
        ext_null_count=np.array([0, 1, 2, 3, 1, 2]),
    )

    assert by_target["y_true"].tolist() == [0, 1]
    assert by_ext["EXT_NULL_COUNT"].tolist() == [0, 1, 2, 3]
    assert frame.columns.tolist() == [
        "mc_mean_proba",
        "mc_variance",
        "mc_pred_label",
        "y_true",
        "EXT_NULL_COUNT",
    ]


def test_mc_dropout_summary_builder_rejects_invalid_ext_null_count() -> None:
    """Scaled or impossible EXT_NULL_COUNT values should be rejected."""

    result = MCDropoutUncertaintyEstimator(
        MCDropoutConfig(n_passes=3, batch_size=3, random_seed=42)
    ).predict(_build_dropout_model(), np.ones((3, 3), dtype=np.float32))
    builder = MCDropoutSummaryBuilder()

    with pytest.raises(MCDropoutError, match="invalid values"):
        builder.summary_by_ext_null_count(result, ext_null_count=np.array([0, -1, 2]))
    with pytest.raises(MCDropoutError, match="integer values"):
        builder.summary_by_ext_null_count(result, ext_null_count=np.array([0, 1.5, 2]))
