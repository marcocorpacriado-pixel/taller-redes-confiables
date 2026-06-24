"""Alternative FAIR losses for methodological comparison.

The validated MVP keeps its main FAIR training path unchanged. This module
adds a compact squared Demographic Parity loss inspired by the comparative
notebook analysis. It is intentionally isolated so it can be tested and used as
an optional experiment without replacing the canonical Pearson-based penalty.
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


class FairnessLossError(ValueError):
    """Raised when a FAIR loss receives invalid configuration or tensors."""


@dataclass(frozen=True, slots=True)
class SquaredDemographicParityConfig:
    """Configuration for the squared Demographic Parity loss.

    Args:
        alpha: Non-negative weight for the squared fairness penalty.
        eps: Numerical stabilizer used in clipping and safe denominators.
        from_logits: Whether predictions are logits instead of probabilities.
    """

    alpha: float = 1.0
    eps: float = 1e-7
    from_logits: bool = False

    def __post_init__(self) -> None:
        """Validate numeric settings before the loss is used."""

        if self.alpha < 0.0:
            raise FairnessLossError("alpha must be non-negative.")
        if self.eps <= 0.0:
            raise FairnessLossError("eps must be positive.")


@dataclass(frozen=True, slots=True)
class SquaredDemographicParityComponents:
    """Tensor components produced by the squared DP loss."""

    binary_crossentropy: tf.Tensor
    demographic_parity_gap: tf.Tensor
    fairness_penalty: tf.Tensor
    total_loss: tf.Tensor


class SquaredDemographicParityLoss(tf.keras.losses.Loss):
    """Binary cross-entropy plus squared Demographic Parity gap.

    The loss expects an augmented target tensor with two columns:

    - column 0: binary label ``TARGET``.
    - column 1: binary sensitive attribute, e.g. encoded ``CODE_GENDER``.

    Its objective is:

    ``mean(BCE(y, p)) + alpha * (E[p | s=1] - E[p | s=0])^2``.

    If a mini-batch contains only one sensitive group, the fairness term is set
    to zero for that batch because the group gap cannot be estimated reliably.
    This keeps optional experiments stable without fabricating a denominator.
    """

    def __init__(
        self,
        config: SquaredDemographicParityConfig | None = None,
        name: str = "squared_demographic_parity_loss",
    ) -> None:
        """Initialize the Keras loss."""

        self.config = config or SquaredDemographicParityConfig()
        super().__init__(name=name)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Return the scalar total loss for one batch."""

        return self.components(y_true=y_true, y_pred=y_pred).total_loss

    def components(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> SquaredDemographicParityComponents:
        """Return BCE, DP gap, FAIR penalty, and total loss tensors."""

        target, sensitive = self._split_augmented_target(y_true)
        probability = self._prediction_probability(y_pred)

        bce_per_row = -(
            target * tf.math.log(probability)
            + (1.0 - target) * tf.math.log(1.0 - probability)
        )
        bce = tf.reduce_mean(bce_per_row)

        gap = self._demographic_parity_gap(probability, sensitive)
        penalty = tf.cast(self.config.alpha, tf.float32) * tf.square(gap)
        total = bce + penalty
        return SquaredDemographicParityComponents(
            binary_crossentropy=bce,
            demographic_parity_gap=gap,
            fairness_penalty=penalty,
            total_loss=total,
        )

    def _prediction_probability(self, y_pred: tf.Tensor) -> tf.Tensor:
        """Convert predictions to clipped probabilities."""

        prediction = tf.cast(y_pred, tf.float32)
        if self.config.from_logits:
            prediction = tf.sigmoid(prediction)
        return tf.clip_by_value(
            prediction,
            clip_value_min=self.config.eps,
            clip_value_max=1.0 - self.config.eps,
        )

    @staticmethod
    def _split_augmented_target(y_true: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Split augmented ``[target, sensitive]`` targets."""

        augmented = tf.cast(y_true, tf.float32)
        if augmented.shape.rank is not None and augmented.shape[-1] is not None:
            if int(augmented.shape[-1]) < 2:
                raise FairnessLossError(
                    "SquaredDemographicParityLoss expects y_true with at least "
                    "two columns: target and sensitive."
                )
        target = augmented[..., 0:1]
        sensitive = augmented[..., 1:2]
        return target, sensitive

    def _demographic_parity_gap(
        self,
        probability: tf.Tensor,
        sensitive: tf.Tensor,
    ) -> tf.Tensor:
        """Compute ``E[p|s=1] - E[p|s=0]`` with single-group protection."""

        sensitive = tf.cast(sensitive > 0.5, tf.float32)
        group_1 = sensitive
        group_0 = 1.0 - sensitive

        count_1 = tf.reduce_sum(group_1)
        count_0 = tf.reduce_sum(group_0)
        has_both_groups = tf.logical_and(count_1 > 0.0, count_0 > 0.0)

        mean_1 = tf.reduce_sum(probability * group_1) / (count_1 + self.config.eps)
        mean_0 = tf.reduce_sum(probability * group_0) / (count_0 + self.config.eps)
        gap = mean_1 - mean_0
        return tf.where(has_both_groups, gap, tf.zeros_like(gap))

    def get_config(self) -> dict[str, float | bool | str]:
        """Return Keras-serializable loss configuration."""

        base_config = super().get_config()
        base_config.update(
            {
                "alpha": self.config.alpha,
                "eps": self.config.eps,
                "from_logits": self.config.from_logits,
            }
        )
        return base_config

    @classmethod
    def from_config(cls, config: dict[str, float | bool | str]):
        """Rebuild the loss from a Keras configuration dictionary."""

        config_copy = dict(config)
        name = str(config_copy.pop("name", "squared_demographic_parity_loss"))
        config_copy.pop("reduction", None)
        return cls(
            config=SquaredDemographicParityConfig(
                alpha=float(config_copy.pop("alpha", 1.0)),
                eps=float(config_copy.pop("eps", 1e-7)),
                from_logits=bool(config_copy.pop("from_logits", False)),
            ),
            name=name,
        )


def make_augmented_fair_targets(
    target: tf.Tensor,
    sensitive: tf.Tensor,
) -> tf.Tensor:
    """Build a two-column target tensor for squared DP experiments."""

    target_tensor = tf.reshape(tf.cast(target, tf.float32), (-1, 1))
    sensitive_tensor = tf.reshape(tf.cast(sensitive, tf.float32), (-1, 1))
    if target_tensor.shape[0] is not None and sensitive_tensor.shape[0] is not None:
        if target_tensor.shape[0] != sensitive_tensor.shape[0]:
            raise FairnessLossError("target and sensitive must have the same length.")
    return tf.concat([target_tensor, sensitive_tensor], axis=1)


__all__ = [
    "FairnessLossError",
    "SquaredDemographicParityComponents",
    "SquaredDemographicParityConfig",
    "SquaredDemographicParityLoss",
    "make_augmented_fair_targets",
]
