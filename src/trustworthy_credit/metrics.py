"""Reusable thresholds and metrics for the Home Credit trustworthy MVP.

This module implements Block 8. It centralizes all threshold-dependent and
threshold-free evaluation utilities so Blocks 7, 9, 11 and 12 do not copy-paste
metric logic.

The central rule is:

    probabilities first, threshold selected on validation, binary metrics after.

No function in this module selects a threshold from test data. The caller must
pass the validation probabilities when choosing a threshold and then reuse the
chosen value on test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class MetricsError(ValueError):
    """Raised when a metric or threshold cannot be computed safely.

    Args:
        message: Human-readable explanation of the metric failure.

    Returns:
        None.

    Raises:
        This exception is raised by Block 8 utilities when arrays are
        misaligned, thresholds are invalid or validation data is unsuitable.
    """


@dataclass(frozen=True)
class ProbabilityMetrics:
    """Threshold-free metrics computed from predicted probabilities.

    Args:
        roc_auc: ROC-AUC, or NaN when undefined.
        pr_auc: Average precision / PR-AUC, or NaN when undefined.
        abs_rho: Absolute Pearson correlation between probabilities and the
            sensitive variable.

    Returns:
        Immutable probability-metric container.

    Raises:
        None.
    """

    roc_auc: float
    pr_auc: float
    abs_rho: float

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        """Convert metrics to a flat dictionary.

        Args:
            prefix: Optional prefix used by result tables, for example `val_`.

        Returns:
            Dictionary with stable metric names.

        Raises:
            None.
        """

        return {
            f"{prefix}roc_auc": self.roc_auc,
            f"{prefix}pr_auc": self.pr_auc,
            f"{prefix}abs_rho": self.abs_rho,
        }


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    """Classification metrics computed after applying a threshold.

    Args:
        accuracy: Fraction of correct binary decisions.
        precision: Positive predictive value.
        recall: True positive rate.
        f1: Harmonic mean of precision and recall.
        threshold: Threshold used to create binary labels.

    Returns:
        Immutable binary-metric container.

    Raises:
        None.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    threshold: float

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        """Convert metrics to a flat dictionary.

        Args:
            prefix: Optional prefix used by result tables.

        Returns:
            Dictionary with stable metric names.

        Raises:
            None.
        """

        return {
            f"{prefix}accuracy": self.accuracy,
            f"{prefix}precision": self.precision,
            f"{prefix}recall": self.recall,
            f"{prefix}f1": self.f1,
            f"{prefix}threshold": self.threshold,
        }


@dataclass(frozen=True)
class FairnessMetrics:
    """Fairness metrics computed from binary decisions.

    Args:
        demographic_parity_difference: Difference in positive decision rates
            between sensitive groups.
        equalized_odds_difference: Maximum of TPR and FPR differences between
            sensitive groups.

    Returns:
        Immutable fairness-metric container.

    Raises:
        None.
    """

    demographic_parity_difference: float
    equalized_odds_difference: float

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        """Convert metrics to a flat dictionary.

        Args:
            prefix: Optional prefix used by result tables.

        Returns:
            Dictionary with stable metric names.

        Raises:
            None.
        """

        return {
            f"{prefix}dpd": self.demographic_parity_difference,
            f"{prefix}eod": self.equalized_odds_difference,
        }


@dataclass(frozen=True)
class ThresholdSelectionResult:
    """Metadata returned when selecting a validation threshold.

    Args:
        threshold: Chosen threshold clipped to [0, 1].
        criterion: Name of the selection criterion.
        score: Criterion score at the selected threshold.

    Returns:
        Immutable threshold selection result.

    Raises:
        None.
    """

    threshold: float
    criterion: str
    score: float


@dataclass(frozen=True)
class BootstrapInterval:
    """Bootstrap estimate and percentile confidence interval.

    Args:
        mean: Mean metric value across bootstrap samples.
        lower: Lower percentile bound.
        upper: Upper percentile bound.
        n_bootstrap: Number of bootstrap samples used.

    Returns:
        Immutable bootstrap interval.

    Raises:
        None.
    """

    mean: float
    lower: float
    upper: float
    n_bootstrap: int

    def to_dict(self, metric_name: str) -> dict[str, float]:
        """Convert interval to named dictionary columns.

        Args:
            metric_name: Base metric name.

        Returns:
            Dictionary with mean, lower and upper fields.

        Raises:
            None.
        """

        return {
            f"{metric_name}_mean": self.mean,
            f"{metric_name}_ci_low": self.lower,
            f"{metric_name}_ci_high": self.upper,
        }


class MetricInputValidator:
    """Validate metric input arrays.

    Args:
        None.

    Returns:
        Validator object.

    Raises:
        None.
    """

    def as_binary_target(self, y_true: np.ndarray) -> np.ndarray:
        """Validate and return a flat binary target array.

        Args:
            y_true: Candidate target array.

        Returns:
            Flat integer array containing only 0 and 1.

        Raises:
            MetricsError: If values are not binary.
        """

        # Flatten because sklearn and fairlearn metrics expect one-dimensional
        # aligned arrays.
        target = np.asarray(y_true).reshape(-1).astype(int)

        # The project is a binary classification task. Anything else indicates
        # a data-contract or preprocessing bug.
        values = set(np.unique(target).tolist())
        if not values.issubset({0, 1}):
            raise MetricsError("y_true must contain only 0/1 values.")

        return target

    def as_probability(self, y_proba: np.ndarray) -> np.ndarray:
        """Validate and return a flat probability array.

        Args:
            y_proba: Candidate probability array.

        Returns:
            Flat float array.

        Raises:
            MetricsError: If probabilities are non-finite.
        """

        proba = np.asarray(y_proba, dtype="float64").reshape(-1)

        # Non-finite probabilities would make all downstream metrics unreliable.
        if not np.isfinite(proba).all():
            raise MetricsError("y_proba contains non-finite values.")

        return proba

    def as_sensitive(self, sensitive: np.ndarray) -> np.ndarray:
        """Validate and return a flat binary sensitive array.

        Args:
            sensitive: Candidate sensitive array.

        Returns:
            Flat integer array containing only 0 and 1.

        Raises:
            MetricsError: If sensitive values are not binary.
        """

        sens = np.asarray(sensitive).reshape(-1).astype(int)
        values = set(np.unique(sens).tolist())
        if not values.issubset({0, 1}):
            raise MetricsError("sensitive must contain only 0/1 values.")

        return sens

    def assert_same_length(self, *arrays: np.ndarray) -> None:
        """Validate all arrays have the same length.

        Args:
            *arrays: Arrays to compare by first dimension after flattening.

        Returns:
            None.

        Raises:
            MetricsError: If lengths differ.
        """

        lengths = [np.asarray(array).reshape(-1).shape[0] for array in arrays]
        if len(set(lengths)) > 1:
            raise MetricsError(f"Metric arrays have different lengths: {lengths}.")

    def assert_two_classes(self, y_true: np.ndarray) -> None:
        """Validate that target has both binary classes.

        Args:
            y_true: Binary target array.

        Returns:
            None.

        Raises:
            MetricsError: If only one class is present.
        """

        target = self.as_binary_target(y_true)
        if set(np.unique(target).tolist()) != {0, 1}:
            raise MetricsError("Metric requires both target classes.")


class ThresholdSelector:
    """Select thresholds from validation probabilities.

    Args:
        validator: Optional metric input validator.

    Returns:
        Threshold selector object.

    Raises:
        None during initialization.
    """

    def __init__(self, validator: MetricInputValidator | None = None) -> None:
        """Initialize the selector.

        Args:
            validator: Optional validator dependency.

        Returns:
            None.

        Raises:
            None.
        """

        self._validator = validator or MetricInputValidator()

    def choose_youden(self, y_true: np.ndarray, y_proba: np.ndarray) -> ThresholdSelectionResult:
        """Choose threshold by Youden's J statistic.

        Args:
            y_true: Validation target array.
            y_proba: Validation probabilities.

        Returns:
            ThresholdSelectionResult with clipped threshold and J score.

        Raises:
            MetricsError: If arrays are invalid or contain one target class.
        """

        target = self._validator.as_binary_target(y_true)
        proba = self._validator.as_probability(y_proba)
        self._validator.assert_same_length(target, proba)
        self._validator.assert_two_classes(target)

        # roc_curve returns FPR, TPR and candidate thresholds.
        fpr, tpr, thresholds = roc_curve(target, proba)

        # Youden's J balances sensitivity and specificity.
        scores = tpr - fpr
        best_index = int(np.argmax(scores))

        # sklearn includes an artificial threshold above max(y_score); clipping
        # keeps the value safe for probability arrays.
        threshold = float(np.clip(thresholds[best_index], 0.0, 1.0))

        return ThresholdSelectionResult(
            threshold=threshold,
            criterion="youden_j",
            score=float(scores[best_index]),
        )

    def choose_f1(self, y_true: np.ndarray, y_proba: np.ndarray) -> ThresholdSelectionResult:
        """Choose threshold that maximizes validation F1.

        Args:
            y_true: Validation target array.
            y_proba: Validation probabilities.

        Returns:
            ThresholdSelectionResult with best F1 threshold.

        Raises:
            MetricsError: If arrays are invalid.
        """

        target = self._validator.as_binary_target(y_true)
        proba = self._validator.as_probability(y_proba)
        self._validator.assert_same_length(target, proba)

        # Candidate thresholds from unique probabilities keep the search exact
        # for the observed validation scores without an arbitrary grid.
        candidates = np.unique(np.clip(proba, 0.0, 1.0))

        # Include boundaries so degenerate models still produce a valid result.
        candidates = np.unique(np.concatenate([np.array([0.0, 1.0]), candidates]))

        best_threshold = 0.5
        best_score = -1.0

        for threshold in candidates:
            labels = ThresholdApplier().apply(proba, float(threshold))
            score = float(f1_score(target, labels, zero_division=0))
            if score > best_score:
                best_threshold = float(threshold)
                best_score = score

        return ThresholdSelectionResult(
            threshold=float(np.clip(best_threshold, 0.0, 1.0)),
            criterion="max_f1",
            score=best_score,
        )


class ThresholdApplier:
    """Apply a fixed threshold to predicted probabilities.

    Args:
        validator: Optional metric input validator.

    Returns:
        Threshold applier object.

    Raises:
        None during initialization.
    """

    def __init__(self, validator: MetricInputValidator | None = None) -> None:
        """Initialize the applier.

        Args:
            validator: Optional validator dependency.

        Returns:
            None.

        Raises:
            None.
        """

        self._validator = validator or MetricInputValidator()

    def apply(self, y_proba: np.ndarray, threshold: float) -> np.ndarray:
        """Convert probabilities into binary labels.

        Args:
            y_proba: Predicted probabilities.
            threshold: Decision threshold in [0, 1].

        Returns:
            Integer labels where probability >= threshold becomes 1.

        Raises:
            MetricsError: If threshold is outside [0, 1].
        """

        if not 0.0 <= float(threshold) <= 1.0:
            raise MetricsError("threshold must be in [0, 1].")

        proba = self._validator.as_probability(y_proba)
        return (proba >= float(threshold)).astype(int)


class AbsolutePearsonCorrelation:
    """Compute absolute Pearson correlation for fairness auditing.

    Args:
        eps: Small denominator guard.

    Returns:
        Metric object.

    Raises:
        None during initialization.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        """Initialize the correlation metric.

        Args:
            eps: Positive denominator guard.

        Returns:
            None.

        Raises:
            MetricsError: If eps is not positive.
        """

        if eps <= 0:
            raise MetricsError("eps must be positive.")

        self._eps = float(eps)
        self._validator = MetricInputValidator()

    def compute(self, y_proba: np.ndarray, sensitive: np.ndarray) -> float:
        """Compute absolute Pearson correlation.

        Args:
            y_proba: Predicted probabilities.
            sensitive: Binary sensitive values aligned with predictions.

        Returns:
            Absolute Pearson correlation. Returns 0 when either variable has
            near-zero variance.

        Raises:
            MetricsError: If arrays are invalid or misaligned.
        """

        proba = self._validator.as_probability(y_proba)
        sens = self._validator.as_sensitive(sensitive).astype("float64")
        self._validator.assert_same_length(proba, sens)

        proba_centered = proba - proba.mean()
        sens_centered = sens - sens.mean()

        numerator = float(np.mean(proba_centered * sens_centered))
        denominator = float(
            np.sqrt(np.mean(proba_centered**2) * np.mean(sens_centered**2))
        )

        if denominator <= self._eps:
            return 0.0

        return abs(numerator / denominator)


class ProbabilityMetricCalculator:
    """Calculate threshold-free probability metrics.

    Args:
        correlation_metric: Optional absolute Pearson metric.

    Returns:
        Calculator object.

    Raises:
        None during initialization.
    """

    def __init__(
        self,
        correlation_metric: AbsolutePearsonCorrelation | None = None,
    ) -> None:
        """Initialize the calculator.

        Args:
            correlation_metric: Optional correlation metric dependency.

        Returns:
            None.

        Raises:
            None.
        """

        self._validator = MetricInputValidator()
        self._correlation_metric = correlation_metric or AbsolutePearsonCorrelation()

    def calculate(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        sensitive: np.ndarray,
    ) -> ProbabilityMetrics:
        """Calculate ROC-AUC, PR-AUC and abs rho.

        Args:
            y_true: Binary target array.
            y_proba: Predicted probabilities.
            sensitive: Binary sensitive array.

        Returns:
            ProbabilityMetrics container.

        Raises:
            MetricsError: If arrays are invalid or misaligned.
        """

        target = self._validator.as_binary_target(y_true)
        proba = self._validator.as_probability(y_proba)
        sens = self._validator.as_sensitive(sensitive)
        self._validator.assert_same_length(target, proba, sens)

        return ProbabilityMetrics(
            roc_auc=self._safe_roc_auc(target, proba),
            pr_auc=self._safe_pr_auc(target, proba),
            abs_rho=self._correlation_metric.compute(proba, sens),
        )

    def _safe_roc_auc(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Compute ROC-AUC with a single-class guard.

        Args:
            y_true: Binary target array.
            y_proba: Predicted probabilities.

        Returns:
            ROC-AUC or NaN if undefined.

        Raises:
            None.
        """

        if len(np.unique(y_true)) < 2:
            return float("nan")

        return float(roc_auc_score(y_true, y_proba))

    def _safe_pr_auc(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Compute PR-AUC / average precision with a single-class guard.

        Args:
            y_true: Binary target array.
            y_proba: Predicted probabilities.

        Returns:
            Average precision or NaN if undefined.

        Raises:
            None.
        """

        if len(np.unique(y_true)) < 2:
            return float("nan")

        return float(average_precision_score(y_true, y_proba))


class BinaryClassificationMetricCalculator:
    """Calculate binary classification metrics after thresholding.

    Args:
        threshold_applier: Optional threshold applier dependency.

    Returns:
        Calculator object.

    Raises:
        None during initialization.
    """

    def __init__(
        self,
        threshold_applier: ThresholdApplier | None = None,
    ) -> None:
        """Initialize the calculator.

        Args:
            threshold_applier: Optional threshold applier dependency.

        Returns:
            None.

        Raises:
            None.
        """

        self._validator = MetricInputValidator()
        self._threshold_applier = threshold_applier or ThresholdApplier()

    def calculate(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float,
    ) -> BinaryClassificationMetrics:
        """Calculate accuracy, precision, recall and F1.

        Args:
            y_true: Binary target array.
            y_proba: Predicted probabilities.
            threshold: Fixed decision threshold.

        Returns:
            BinaryClassificationMetrics container.

        Raises:
            MetricsError: If arrays are invalid or misaligned.
        """

        target = self._validator.as_binary_target(y_true)
        labels = self._threshold_applier.apply(y_proba, threshold)
        self._validator.assert_same_length(target, labels)

        return BinaryClassificationMetrics(
            accuracy=float(accuracy_score(target, labels)),
            precision=float(precision_score(target, labels, zero_division=0)),
            recall=float(recall_score(target, labels, zero_division=0)),
            f1=float(f1_score(target, labels, zero_division=0)),
            threshold=float(threshold),
        )


class FairnessMetricCalculator:
    """Calculate group fairness metrics from binary decisions.

    Args:
        threshold_applier: Optional threshold applier dependency.

    Returns:
        Calculator object.

    Raises:
        None during initialization.
    """

    def __init__(
        self,
        threshold_applier: ThresholdApplier | None = None,
    ) -> None:
        """Initialize the calculator.

        Args:
            threshold_applier: Optional threshold applier dependency.

        Returns:
            None.

        Raises:
            None.
        """

        self._validator = MetricInputValidator()
        self._threshold_applier = threshold_applier or ThresholdApplier()

    def calculate(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        sensitive: np.ndarray,
        threshold: float,
    ) -> FairnessMetrics:
        """Calculate DPD and EOD at a fixed threshold.

        Args:
            y_true: Binary target array.
            y_proba: Predicted probabilities.
            sensitive: Binary sensitive values.
            threshold: Fixed decision threshold selected on validation.

        Returns:
            FairnessMetrics container.

        Raises:
            MetricsError: If arrays are invalid or misaligned.
        """

        target = self._validator.as_binary_target(y_true)
        sens = self._validator.as_sensitive(sensitive)
        labels = self._threshold_applier.apply(y_proba, threshold)
        self._validator.assert_same_length(target, labels, sens)

        dpd = demographic_parity_difference(
            y_true=target,
            y_pred=labels,
            sensitive_features=sens,
        )
        eod = equalized_odds_difference(
            y_true=target,
            y_pred=labels,
            sensitive_features=sens,
        )

        return FairnessMetrics(
            demographic_parity_difference=float(dpd),
            equalized_odds_difference=float(eod),
        )


class BootstrapMetricCalculator:
    """Compute bootstrap confidence intervals for metrics.

    Args:
        n_bootstrap: Number of bootstrap samples.
        confidence_level: Confidence level, for example 0.95.
        random_state: Random seed.

    Returns:
        Bootstrap calculator object.

    Raises:
        MetricsError: If bootstrap settings are invalid.
    """

    def __init__(
        self,
        n_bootstrap: int = 500,
        confidence_level: float = 0.95,
        random_state: int = 42,
    ) -> None:
        """Initialize the bootstrap calculator.

        Args:
            n_bootstrap: Number of bootstrap resamples.
            confidence_level: Central confidence mass.
            random_state: Random seed.

        Returns:
            None.

        Raises:
            MetricsError: If settings are invalid.
        """

        if n_bootstrap <= 0:
            raise MetricsError("n_bootstrap must be positive.")

        if not 0.0 < confidence_level < 1.0:
            raise MetricsError("confidence_level must be in (0, 1).")

        self._n_bootstrap = int(n_bootstrap)
        self._confidence_level = float(confidence_level)
        self._random_state = int(random_state)
        self._validator = MetricInputValidator()

    def compute(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        sensitive: np.ndarray,
        metric_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    ) -> BootstrapInterval:
        """Compute a bootstrap interval for a metric.

        Args:
            y_true: Binary target array.
            y_proba: Predicted probabilities.
            sensitive: Binary sensitive array.
            metric_fn: Callable with signature `(y, proba, sensitive) -> float`.

        Returns:
            BootstrapInterval with mean and percentile bounds.

        Raises:
            MetricsError: If arrays are invalid or metric_fn returns no values.
        """

        target = self._validator.as_binary_target(y_true)
        proba = self._validator.as_probability(y_proba)
        sens = self._validator.as_sensitive(sensitive)
        self._validator.assert_same_length(target, proba, sens)

        rng = np.random.default_rng(self._random_state)
        n_rows = target.shape[0]
        values: list[float] = []

        for _ in range(self._n_bootstrap):
            indices = rng.integers(0, n_rows, size=n_rows)
            value = float(metric_fn(target[indices], proba[indices], sens[indices]))

            # Metrics like ROC-AUC can be NaN in bootstrap samples with one
            # target class. Dropping NaNs keeps the interval meaningful.
            if np.isfinite(value):
                values.append(value)

        if not values:
            raise MetricsError("Bootstrap metric produced no finite values.")

        alpha = 1.0 - self._confidence_level
        lower_pct = 100.0 * (alpha / 2.0)
        upper_pct = 100.0 * (1.0 - alpha / 2.0)
        values_array = np.asarray(values, dtype="float64")

        return BootstrapInterval(
            mean=float(np.mean(values_array)),
            lower=float(np.percentile(values_array, lower_pct)),
            upper=float(np.percentile(values_array, upper_pct)),
            n_bootstrap=self._n_bootstrap,
        )


def choose_threshold_youden(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Convenience function returning the Youden threshold only.

    Args:
        y_true: Validation target array.
        y_proba: Validation probabilities.

    Returns:
        Threshold clipped to [0, 1].

    Raises:
        MetricsError: If threshold selection is invalid.
    """

    return ThresholdSelector().choose_youden(y_true, y_proba).threshold


def apply_threshold(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    """Convenience function applying a threshold to probabilities.

    Args:
        y_proba: Predicted probabilities.
        threshold: Decision threshold in [0, 1].

    Returns:
        Binary labels.

    Raises:
        MetricsError: If threshold or probabilities are invalid.
    """

    return ThresholdApplier().apply(y_proba, threshold)


def absolute_pearson_correlation(
    y_proba: np.ndarray,
    sensitive: np.ndarray,
) -> float:
    """Convenience function for absolute Pearson correlation.

    Args:
        y_proba: Predicted probabilities.
        sensitive: Binary sensitive values.

    Returns:
        Absolute Pearson correlation.

    Raises:
        MetricsError: If arrays are invalid.
    """

    return AbsolutePearsonCorrelation().compute(y_proba, sensitive)


def classification_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> BinaryClassificationMetrics:
    """Convenience function for binary classification metrics.

    Args:
        y_true: Binary target array.
        y_proba: Predicted probabilities.
        threshold: Fixed decision threshold.

    Returns:
        BinaryClassificationMetrics container.

    Raises:
        MetricsError: If arrays are invalid.
    """

    return BinaryClassificationMetricCalculator().calculate(y_true, y_proba, threshold)


def fairness_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    sensitive: np.ndarray,
    threshold: float,
) -> FairnessMetrics:
    """Convenience function for DPD and EOD.

    Args:
        y_true: Binary target array.
        y_proba: Predicted probabilities.
        sensitive: Binary sensitive values.
        threshold: Fixed decision threshold.

    Returns:
        FairnessMetrics container.

    Raises:
        MetricsError: If arrays are invalid.
    """

    return FairnessMetricCalculator().calculate(y_true, y_proba, sensitive, threshold)


def bootstrap_metric(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    sensitive: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    *,
    n_bootstrap: int = 500,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> BootstrapInterval:
    """Convenience function for bootstrap confidence intervals.

    Args:
        y_true: Binary target array.
        y_proba: Predicted probabilities.
        sensitive: Binary sensitive array.
        metric_fn: Callable with signature `(y, proba, sensitive) -> float`.
        n_bootstrap: Number of bootstrap samples.
        confidence_level: Confidence level.
        random_state: Random seed.

    Returns:
        BootstrapInterval.

    Raises:
        MetricsError: If inputs or bootstrap settings are invalid.
    """

    return BootstrapMetricCalculator(
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
    ).compute(y_true, y_proba, sensitive, metric_fn)


__all__ = [
    "absolute_pearson_correlation",
    "AbsolutePearsonCorrelation",
    "apply_threshold",
    "BinaryClassificationMetricCalculator",
    "BinaryClassificationMetrics",
    "bootstrap_metric",
    "BootstrapInterval",
    "BootstrapMetricCalculator",
    "choose_threshold_youden",
    "classification_metrics",
    "fairness_metrics",
    "FairnessMetricCalculator",
    "FairnessMetrics",
    "MetricInputValidator",
    "MetricsError",
    "ProbabilityMetricCalculator",
    "ProbabilityMetrics",
    "ThresholdApplier",
    "ThresholdSelectionResult",
    "ThresholdSelector",
]
