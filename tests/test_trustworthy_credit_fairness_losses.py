"""Tests for optional squared Demographic Parity FAIR losses."""

import numpy as np
import pytest
import tensorflow as tf

from src.trustworthy_credit.fairness_losses import (
    FairnessLossError,
    SquaredDemographicParityConfig,
    SquaredDemographicParityLoss,
    make_augmented_fair_targets,
)


def _manual_bce(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return mean binary cross-entropy for small deterministic arrays."""

    clipped = np.clip(y_pred, 1e-7, 1.0 - 1e-7)
    return float(
        np.mean(
            -(
                y_true * np.log(clipped)
                + (1.0 - y_true) * np.log(1.0 - clipped)
            )
        )
    )


def test_squared_dp_config_rejects_invalid_values() -> None:
    """Invalid loss settings should fail early."""

    with pytest.raises(FairnessLossError, match="alpha"):
        SquaredDemographicParityConfig(alpha=-1.0)
    with pytest.raises(FairnessLossError, match="eps"):
        SquaredDemographicParityConfig(eps=0.0)


def test_make_augmented_fair_targets_builds_two_column_tensor() -> None:
    """Helper should combine target and sensitive arrays for Keras losses."""

    augmented = make_augmented_fair_targets(
        target=tf.constant([0.0, 1.0, 1.0]),
        sensitive=tf.constant([0.0, 1.0, 0.0]),
    )

    assert augmented.shape == (3, 2)
    np.testing.assert_allclose(
        augmented.numpy(),
        np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    )


def test_alpha_zero_matches_binary_crossentropy() -> None:
    """With alpha=0 the optional FAIR loss should reduce to normal BCE."""

    y = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    s = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    p = np.array([0.1, 0.8, 0.3, 0.7], dtype=np.float32).reshape(-1, 1)
    augmented = make_augmented_fair_targets(y, s)

    loss = SquaredDemographicParityLoss(
        SquaredDemographicParityConfig(alpha=0.0)
    )

    assert float(loss(augmented, p).numpy()) == pytest.approx(_manual_bce(y, p.ravel()))


def test_squared_dp_penalty_increases_loss_when_group_means_differ() -> None:
    """A prediction gap between sensitive groups should add a positive penalty."""

    y = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32)
    s = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32)
    p = tf.constant([[0.20], [0.30], [0.80], [0.90]], dtype=tf.float32)
    augmented = make_augmented_fair_targets(y, s)

    base_loss = SquaredDemographicParityLoss(
        SquaredDemographicParityConfig(alpha=0.0)
    )
    fair_loss = SquaredDemographicParityLoss(
        SquaredDemographicParityConfig(alpha=2.0)
    )
    components = fair_loss.components(augmented, p)

    assert components.demographic_parity_gap.numpy() == pytest.approx(0.60)
    assert components.fairness_penalty.numpy() == pytest.approx(2.0 * 0.60**2)
    assert fair_loss(augmented, p).numpy() > base_loss(augmented, p).numpy()


def test_squared_dp_penalty_is_zero_when_group_means_match() -> None:
    """Equal group means should produce no fairness penalty."""

    y = tf.constant([0.0, 1.0, 0.0, 1.0], dtype=tf.float32)
    s = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32)
    p = tf.constant([[0.20], [0.80], [0.20], [0.80]], dtype=tf.float32)
    augmented = make_augmented_fair_targets(y, s)
    loss = SquaredDemographicParityLoss(SquaredDemographicParityConfig(alpha=5.0))

    components = loss.components(augmented, p)

    assert components.demographic_parity_gap.numpy() == pytest.approx(0.0)
    assert components.fairness_penalty.numpy() == pytest.approx(0.0)
    assert components.total_loss.numpy() == pytest.approx(
        components.binary_crossentropy.numpy()
    )


def test_squared_dp_loss_handles_single_group_batches() -> None:
    """Single-group batches cannot estimate DP gap, so the FAIR term is zero."""

    y = tf.constant([0.0, 1.0, 1.0], dtype=tf.float32)
    s = tf.constant([1.0, 1.0, 1.0], dtype=tf.float32)
    p = tf.constant([[0.20], [0.70], [0.90]], dtype=tf.float32)
    augmented = make_augmented_fair_targets(y, s)
    loss = SquaredDemographicParityLoss(SquaredDemographicParityConfig(alpha=10.0))

    components = loss.components(augmented, p)

    assert np.isfinite(float(components.total_loss.numpy()))
    assert components.demographic_parity_gap.numpy() == pytest.approx(0.0)
    assert components.fairness_penalty.numpy() == pytest.approx(0.0)


def test_squared_dp_loss_rejects_non_augmented_targets() -> None:
    """The loss should clearly reject y_true without a sensitive column."""

    loss = SquaredDemographicParityLoss()

    with pytest.raises(FairnessLossError, match="two columns"):
        loss(tf.constant([[0.0], [1.0]]), tf.constant([[0.2], [0.8]]))


def test_squared_dp_loss_round_trips_keras_config() -> None:
    """The optional loss should support Keras-style serialization."""

    original = SquaredDemographicParityLoss(
        SquaredDemographicParityConfig(alpha=3.0, eps=1e-6, from_logits=True),
        name="custom_squared_dp",
    )

    rebuilt = SquaredDemographicParityLoss.from_config(original.get_config())

    assert rebuilt.name == "custom_squared_dp"
    assert rebuilt.config.alpha == pytest.approx(3.0)
    assert rebuilt.config.eps == pytest.approx(1e-6)
    assert rebuilt.config.from_logits is True
