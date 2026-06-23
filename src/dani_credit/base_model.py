"""Baseline neural model for the Home Credit MVP.

This module implements Block 4. It builds and trains a first Keras MLP without
FAIR loss. The goal is to verify that the processed data pipeline works and to
obtain an initial predictive and fairness audit reference.

The final base model used in the report will later be rebuilt with the custom
architecture and `lambda_fair=0`. This module is the Block 4 sanity-check
baseline and shared training utility foundation.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

from .callbacks import FairnessLogger
from .preprocessing import ProcessedSplitDataset


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class BaseModelError(ValueError):
    """Raised when baseline model training or evaluation cannot proceed.

    Args:
        message: Human-readable explanation of the baseline modelling failure.

    Returns:
        None.

    Raises:
        This exception is raised by Block 4 classes when input arrays or
        configuration values violate baseline model assumptions.
    """


@dataclass(frozen=True)
class ReproducibilityConfig:
    """Configuration for deterministic-ish model training.

    Args:
        seed: Seed used for Python, NumPy and TensorFlow.
        enable_tensorflow_determinism: Whether to request deterministic
            TensorFlow operations when supported by the installed version.
        python_hash_seed_env: Environment variable value for `PYTHONHASHSEED`.
        tensorflow_deterministic_ops_env: Environment variable value for
            `TF_DETERMINISTIC_OPS`.

    Returns:
        Immutable reproducibility configuration.

    Raises:
        None.
    """

    # A single project seed makes split, initialization and training less noisy.
    seed: int = 42

    # TensorFlow determinism is useful but not always available on all versions
    # or hardware backends, so the manager handles failures gracefully.
    enable_tensorflow_determinism: bool = True

    # PYTHONHASHSEED reduces hash-order nondeterminism in Python internals.
    python_hash_seed_env: str = "42"

    # TF_DETERMINISTIC_OPS asks TensorFlow kernels to prefer deterministic
    # implementations when possible.
    tensorflow_deterministic_ops_env: str = "1"


@dataclass(frozen=True)
class BaseModelConfig:
    """Configuration for the Block 4 baseline MLP.

    Args:
        hidden_units: Dense layer widths.
        activation: Activation function used in hidden dense layers.
        dropout: Dropout rate after each hidden dense layer.
        learning_rate: Adam optimizer learning rate.
        gradient_clipnorm: Global gradient norm clipping value for Adam.
        loss: Keras loss used for the baseline.
        batch_size: Batch size for training.
        epochs: Maximum number of epochs; EarlyStopping may stop earlier.
        early_stopping_monitor: Metric monitored by EarlyStopping.
        early_stopping_mode: Direction used by EarlyStopping.
        early_stopping_patience: Epochs without improvement before stopping.
        reduce_lr_monitor: Metric monitored by ReduceLROnPlateau.
        reduce_lr_mode: Direction used by ReduceLROnPlateau.
        reduce_lr_factor: Multiplicative learning-rate reduction.
        reduce_lr_patience: Epochs without improvement before reducing LR.
        min_learning_rate: Minimum learning rate allowed by ReduceLROnPlateau.
        early_stopping_restore_best_weights: Whether EarlyStopping restores the
            best validation-monitor weights.
        provisional_threshold: Temporary threshold used only for initial
            validation diagnostics.

    Returns:
        Immutable model/training configuration.

    Raises:
        None.
    """

    # The initial baseline follows the documented architecture: 128 -> 64.
    hidden_units: tuple[int, ...] = (128, 64)

    # ELU usually behaves well for tabular neural nets after robust scaling.
    activation: str = "elu"

    # Dropout helps reduce overfitting in the sanity-check MLP.
    dropout: float = 0.2

    # Starting point used consistently in the MVP.
    learning_rate: float = 1e-3

    # Clip gradients to reduce the chance of unstable updates.
    gradient_clipnorm: float = 1.0

    # BCE is the baseline probabilistic binary classification loss.
    loss: str = "binary_crossentropy"

    # Home Credit is large; 1024 is efficient and stable for tabular batches.
    batch_size: int = 1024

    # EarlyStopping usually stops before this maximum.
    epochs: int = 100

    # AUC is the primary monitoring metric because the dataset is imbalanced.
    early_stopping_monitor: str = "val_auc"

    # Higher AUC is better.
    early_stopping_mode: str = "max"

    # Patience follows the controlled config from the planning docs.
    early_stopping_patience: int = 10

    # LR reduction watches validation loss because it reflects optimization.
    reduce_lr_monitor: str = "val_loss"

    # Lower validation loss is better.
    reduce_lr_mode: str = "min"

    # Halve the learning rate when the loss plateaus.
    reduce_lr_factor: float = 0.5

    # Slightly shorter than EarlyStopping so LR gets a chance to help first.
    reduce_lr_patience: int = 5

    # Avoid learning rates that become numerically meaningless.
    min_learning_rate: float = 1e-6

    # Restoring the best validation checkpoint is the default, but exposing the
    # flag keeps tuner experiments and future ablations explicit.
    early_stopping_restore_best_weights: bool = True

    # This is only provisional; Block 8 selects the real threshold on validation.
    provisional_threshold: float = 0.5


@dataclass(frozen=True)
class BaseValidationMetrics:
    """Validation metrics for the baseline model.

    Args:
        auc: ROC-AUC on validation probabilities.
        pr_auc: Average precision / PR-AUC on validation probabilities.
        accuracy: Accuracy using the provisional threshold.
        precision: Precision using the provisional threshold.
        recall: Recall using the provisional threshold.
        f1: F1 using the provisional threshold.
        abs_rho: Absolute Pearson correlation between prediction and sensitive.
        threshold: Threshold used for binary diagnostics.

    Returns:
        Immutable validation metrics object.

    Raises:
        None.
    """

    # Probability-ranking metric. NaN if validation has one class only.
    auc: float

    # PR-AUC is useful with minority positive class.
    pr_auc: float

    # Binary diagnostics below use a provisional threshold only.
    accuracy: float
    precision: float
    recall: float
    f1: float

    # Fairness audit metric for the baseline.
    abs_rho: float

    # The threshold is stored to avoid ambiguity in reports.
    threshold: float

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to a dictionary.

        Args:
            None.

        Returns:
            Dictionary suitable for CSV rows or logs.

        Raises:
            None.
        """

        # Keep key names aligned with later result tables.
        return {
            "auc": self.auc,
            "pr_auc": self.pr_auc,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "abs_rho": self.abs_rho,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class BaseTrainingArtifacts:
    """Paths where optional baseline artifacts are stored.

    Args:
        history_csv: Path for the Keras training history CSV.
        validation_predictions_csv: Path for validation predictions.
        model_path: Optional path for the saved Keras model.

    Returns:
        Immutable artifact path configuration.

    Raises:
        None.
    """

    # Training history supports the mandatory loss-curve figure later.
    history_csv: Path = _PROJECT_ROOT / "results/tables/base_training_history.csv"

    # Validation predictions support quick audits and future plotting.
    validation_predictions_csv: Path = (
        _PROJECT_ROOT / "results/tables/base_val_predictions.csv"
    )

    # Model saving is optional because results/models is intentionally ignored.
    model_path: Path | None = _PROJECT_ROOT / "results/models/base_mlp.keras"


@dataclass(frozen=True)
class BaseTrainingResult:
    """Complete result returned by baseline training.

    Args:
        model: Trained Keras model.
        history: Keras History object returned by `fit`.
        class_weight: Class weights used during training.
        validation_metrics: Validation diagnostic metrics.
        validation_predictions: DataFrame with y_true, y_proba and sensitive.

    Returns:
        Immutable container with trained model and diagnostics.

    Raises:
        None.
    """

    # The trained model can be reused for predictions and artifact saving.
    model: tf.keras.Model

    # Keras History stores loss/metric curves.
    history: tf.keras.callbacks.History

    # Class weights make the imbalance treatment explicit and reproducible.
    class_weight: dict[int, float]

    # Metrics summarize initial validation performance and fairness audit.
    validation_metrics: BaseValidationMetrics

    # Row-level validation predictions are useful for debugging and later plots.
    validation_predictions: pd.DataFrame


class ReproducibilityManager:
    """Apply reproducibility settings for baseline training.

    Args:
        config: Reproducibility configuration.

    Returns:
        Manager object able to apply seeds and TensorFlow determinism settings.

    Raises:
        None during initialization.
    """

    def __init__(self, config: ReproducibilityConfig | None = None) -> None:
        """Initialize the reproducibility manager.

        Args:
            config: Optional reproducibility configuration.

        Returns:
            None.

        Raises:
            None.
        """

        # Default config keeps all blocks using seed 42 unless explicitly
        # overridden in tests.
        self._config = config or ReproducibilityConfig()

    def apply(self) -> None:
        """Apply Python, NumPy and TensorFlow seed settings.

        Args:
            None.

        Returns:
            None.

        Raises:
            None. TensorFlow determinism failures are intentionally ignored
            because support can depend on backend and version.
        """

        # PYTHONHASHSEED must normally be set before interpreter startup, but
        # setting it here still documents and partially controls the run.
        os.environ["PYTHONHASHSEED"] = self._config.python_hash_seed_env

        # Ask TensorFlow for deterministic operations where possible.
        os.environ["TF_DETERMINISTIC_OPS"] = (
            self._config.tensorflow_deterministic_ops_env
        )

        # Python's own RNG can affect shuffling or helper utilities.
        random.seed(self._config.seed)

        # NumPy RNG affects any local array sampling.
        np.random.seed(self._config.seed)

        # TensorFlow RNG affects weight initialization and dropout.
        tf.random.set_seed(self._config.seed)

        # Some TensorFlow versions expose an explicit determinism switch.
        if self._config.enable_tensorflow_determinism:
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception:
                # Determinism support can be unavailable; the training should
                # still proceed, just with a note in documentation.
                pass


class TrainingArrayValidator:
    """Validate processed arrays before Keras training.

    Args:
        None.

    Returns:
        Validator instance.

    Raises:
        None.
    """

    def validate(self, data: ProcessedSplitDataset) -> None:
        """Raise if processed data is not suitable for baseline training.

        Args:
            data: Processed split dataset from Block 2.

        Returns:
            None.

        Raises:
            BaseModelError: If shapes, NaNs or binary labels are invalid.
        """

        # Check that each X split has matching y and sensitive lengths.
        self._assert_same_length("train", data.X_train, data.y_train, data.s_train)
        self._assert_same_length("validation", data.X_val, data.y_val, data.s_val)
        self._assert_same_length("test", data.X_test, data.y_test, data.s_test)

        # NaNs in X would break Keras training or make gradients meaningless.
        self._assert_no_nan("X_train", data.X_train)
        self._assert_no_nan("X_val", data.X_val)
        self._assert_no_nan("X_test", data.X_test)

        # Targets must stay binary after preprocessing.
        self._assert_binary("y_train", data.y_train)
        self._assert_binary("y_val", data.y_val)
        self._assert_binary("y_test", data.y_test)

        # Sensitive values are not used for training here, but evaluation needs
        # them to be binary.
        self._assert_binary("s_train", data.s_train)
        self._assert_binary("s_val", data.s_val)
        self._assert_binary("s_test", data.s_test)

    def _assert_same_length(
        self,
        name: str,
        X: np.ndarray,
        y: np.ndarray,
        s: np.ndarray,
    ) -> None:
        """Validate one split has aligned X, y and s lengths.

        Args:
            name: Split name for the error message.
            X: Feature matrix.
            y: Target array.
            s: Sensitive array.

        Returns:
            None.

        Raises:
            BaseModelError: If lengths differ.
        """

        # X rows must align with target and sensitive arrays row-by-row.
        if not (X.shape[0] == y.shape[0] == s.shape[0]):
            raise BaseModelError(f"{name} X/y/s lengths are not aligned.")

    def _assert_no_nan(self, name: str, array: np.ndarray) -> None:
        """Validate an array contains no NaN values.

        Args:
            name: Array name for the error message.
            array: Array to validate.

        Returns:
            None.

        Raises:
            BaseModelError: If NaN values are present.
        """

        # Neural networks cannot learn reliably from unhandled NaNs.
        if np.isnan(array).any():
            raise BaseModelError(f"{name} contains NaN values.")

    def _assert_binary(self, name: str, array: np.ndarray) -> None:
        """Validate an array contains only binary 0/1 values.

        Args:
            name: Array name for the error message.
            array: Array to validate.

        Returns:
            None.

        Raises:
            BaseModelError: If values outside 0/1 are found.
        """

        # Convert through int-compatible comparison while preserving unique
        # values for the error check.
        values = set(np.unique(array).astype(int).tolist())

        # y and s should both be binary at this stage.
        if not values.issubset({0, 1}):
            raise BaseModelError(f"{name} must contain only 0/1 values.")


class ClassWeightCalculator:
    """Compute class weights for imbalanced binary classification.

    Args:
        None.

    Returns:
        Calculator instance.

    Raises:
        None.
    """

    def compute(self, y_train: np.ndarray) -> dict[int, float]:
        """Compute sklearn balanced class weights.

        Args:
            y_train: Binary training target array.

        Returns:
            Dictionary compatible with Keras `model.fit(class_weight=...)`.

        Raises:
            BaseModelError: If one of the binary classes is absent.
        """

        # Keras expects class ids as dictionary keys.
        classes = np.array([0, 1])

        # Compute weights only after confirming both classes exist in train.
        observed_classes = set(np.unique(y_train.astype(int)).tolist())

        # If a class is absent, balanced weights and model training are not
        # meaningful for this binary task.
        if observed_classes != {0, 1}:
            raise BaseModelError("y_train must contain both classes 0 and 1.")

        # sklearn implements the standard balanced weight formula.
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train.astype(int),
        )

        # Convert numpy floats to plain Python floats for clean logs/JSON.
        return {
            0: float(weights[0]),
            1: float(weights[1]),
        }


class BaseMLPModelBuilder:
    """Build the Block 4 baseline MLP model.

    Args:
        config: Baseline model configuration.

    Returns:
        Builder object able to create compiled Keras models.

    Raises:
        None during initialization.
    """

    def __init__(self, config: BaseModelConfig | None = None) -> None:
        """Initialize the baseline MLP builder.

        Args:
            config: Optional model configuration.

        Returns:
            None.

        Raises:
            None.
        """

        # Config injection keeps the architecture reproducible and testable.
        self._config = config or BaseModelConfig()

    @property
    def config(self) -> BaseModelConfig:
        """Return the model configuration.

        Args:
            None.

        Returns:
            BaseModelConfig used by the builder.

        Raises:
            None.
        """

        # The dataclass is frozen, so exposing it is safe.
        return self._config

    def build(self, input_dim: int) -> tf.keras.Model:
        """Build and compile the baseline MLP.

        Args:
            input_dim: Number of processed feature columns.

        Returns:
            Compiled Keras model.

        Raises:
            BaseModelError: If input dimension is not positive.
        """

        # A model with no features is invalid and likely indicates upstream
        # preprocessing failure.
        if input_dim <= 0:
            raise BaseModelError("input_dim must be positive.")

        # The model input represents the processed numeric feature vector.
        inputs = tf.keras.Input(shape=(input_dim,), name="features")

        # Hidden representation starts as the raw input tensor.
        x = inputs

        # Build one Dense + Dropout block per configured hidden layer.
        for layer_index, units in enumerate(self._config.hidden_units):
            # Dense layer learns non-linear interactions among tabular features.
            x = tf.keras.layers.Dense(
                units,
                activation=self._config.activation,
                name=f"dense_{layer_index}",
            )(x)

            # Dropout regularizes the small sanity-check network.
            x = tf.keras.layers.Dropout(
                self._config.dropout,
                name=f"dropout_{layer_index}",
            )(x)

        # Sigmoid output gives P(TARGET=1 | X).
        outputs = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            name="prob",
        )(x)

        # Functional API keeps the model explicit and easy to extend later.
        model = tf.keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="base_mlp",
        )

        # Adam is the default optimizer for the MVP, with gradient clipping for
        # extra stability.
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self._config.learning_rate,
            clipnorm=self._config.gradient_clipnorm,
        )

        # Compile with BCE and the metrics documented in Block 4.
        model.compile(
            optimizer=optimizer,
            loss=self._config.loss,
            metrics=[
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
                tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

        return model


class TrainingCallbackFactory:
    """Create Keras callbacks for baseline training.

    Args:
        config: Baseline model configuration.

    Returns:
        Factory object able to produce fresh callback lists.

    Raises:
        None during initialization.
    """

    def __init__(self, config: BaseModelConfig | None = None) -> None:
        """Initialize the callback factory.

        Args:
            config: Optional baseline configuration.

        Returns:
            None.

        Raises:
            None.
        """

        # Callbacks use the same config as the model/trainer.
        self._config = config or BaseModelConfig()

    def build(self) -> list[tf.keras.callbacks.Callback]:
        """Build a fresh list of Keras callbacks.

        Args:
            None.

        Returns:
            List containing EarlyStopping and ReduceLROnPlateau.

        Raises:
            None.
        """

        # EarlyStopping avoids overfitting and restores the best validation AUC
        # weights.
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=self._config.early_stopping_monitor,
            mode=self._config.early_stopping_mode,
            patience=self._config.early_stopping_patience,
            restore_best_weights=self._config.early_stopping_restore_best_weights,
        )

        # ReduceLROnPlateau lowers the learning rate when validation loss stalls.
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=self._config.reduce_lr_monitor,
            mode=self._config.reduce_lr_mode,
            factor=self._config.reduce_lr_factor,
            patience=self._config.reduce_lr_patience,
            min_lr=self._config.min_learning_rate,
        )

        return [early_stopping, reduce_lr]


class AbsolutePearsonCorrelation:
    """Compute absolute Pearson correlation for fairness auditing.

    Args:
        eps: Small positive constant to avoid division by zero.

    Returns:
        Metric object able to compute `abs(corr(pred, sensitive))`.

    Raises:
        None during initialization.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        """Initialize the correlation metric.

        Args:
            eps: Numerical stability constant.

        Returns:
            None.

        Raises:
            None.
        """

        # eps prevents division by zero when predictions or sensitive values
        # have almost no variance.
        self._eps = eps

    def compute(self, predictions: np.ndarray, sensitive: np.ndarray) -> float:
        """Compute absolute Pearson correlation.

        Args:
            predictions: Predicted probabilities.
            sensitive: Binary sensitive values aligned with predictions.

        Returns:
            Absolute Pearson correlation as a float.

        Raises:
            BaseModelError: If lengths do not match.
        """

        # Flatten to make metric robust to shape (n,) or (n, 1).
        pred = np.asarray(predictions, dtype="float64").reshape(-1)

        # Sensitive values also become flat float64 for stable arithmetic.
        sens = np.asarray(sensitive, dtype="float64").reshape(-1)

        # Alignment is required for any row-level correlation.
        if pred.shape[0] != sens.shape[0]:
            raise BaseModelError("predictions and sensitive lengths differ.")

        # Center both variables before computing covariance.
        pred_centered = pred - pred.mean()
        sens_centered = sens - sens.mean()

        # Numerator is empirical covariance without Bessel correction.
        numerator = float(np.mean(pred_centered * sens_centered))

        # Denominator is product of standard deviations.
        denominator = float(
            np.sqrt(np.mean(pred_centered**2) * np.mean(sens_centered**2))
        )

        # If either variable has no variance, dependence cannot be estimated;
        # returning 0 is operationally safer than NaN for logging.
        if denominator <= self._eps:
            return 0.0

        return abs(numerator / denominator)


class BaseValidationEvaluator:
    """Evaluate the baseline model on validation data.

    Args:
        correlation_metric: Optional absolute Pearson correlation metric.

    Returns:
        Evaluator object.

    Raises:
        None during initialization.
    """

    def __init__(
        self,
        correlation_metric: AbsolutePearsonCorrelation | None = None,
    ) -> None:
        """Initialize the validation evaluator.

        Args:
            correlation_metric: Optional metric object for fairness auditing.

        Returns:
            None.

        Raises:
            None.
        """

        # Dependency injection lets tests replace the metric if needed.
        self._correlation_metric = correlation_metric or AbsolutePearsonCorrelation()

    def predict_probabilities(
        self,
        model: tf.keras.Model,
        X: np.ndarray,
    ) -> np.ndarray:
        """Predict probabilities with a trained model.

        Args:
            model: Trained Keras model.
            X: Feature matrix.

        Returns:
            Flat numpy array with predicted probabilities.

        Raises:
            None.
        """

        # Keras returns shape (n, 1) for a sigmoid binary output.
        probabilities = model.predict(X, verbose=0)

        # Flatten to a consistent shape for sklearn metrics.
        return np.asarray(probabilities).reshape(-1)

    def evaluate(
        self,
        model: tf.keras.Model,
        X_val: np.ndarray,
        y_val: np.ndarray,
        s_val: np.ndarray,
        threshold: float,
    ) -> tuple[BaseValidationMetrics, pd.DataFrame]:
        """Evaluate a trained baseline model on validation data.

        Args:
            model: Trained Keras model.
            X_val: Validation feature matrix.
            y_val: Validation target array.
            s_val: Validation sensitive array.
            threshold: Provisional binary threshold.

        Returns:
            Tuple with validation metrics and row-level predictions DataFrame.

        Raises:
            BaseModelError: If array lengths are inconsistent.
        """

        # Generate predicted probabilities from the trained model.
        probabilities = self.predict_probabilities(model, X_val)

        # Validate all arrays align before computing metrics.
        if not (len(probabilities) == len(y_val) == len(s_val)):
            raise BaseModelError("Validation predictions, y and s lengths differ.")

        # Binary labels use a provisional threshold only. Block 8 will replace
        # this with a validation-selected threshold.
        labels = (probabilities >= threshold).astype(int)

        # Probability metrics can fail if y_val has one class only in tiny tests.
        auc = self._safe_roc_auc(y_val, probabilities)

        # PR-AUC has the same single-class risk.
        pr_auc = self._safe_pr_auc(y_val, probabilities)

        # Binary metrics use zero_division=0 so degenerate predictions are logged
        # rather than crashing the baseline run.
        metrics = BaseValidationMetrics(
            auc=auc,
            pr_auc=pr_auc,
            accuracy=float(accuracy_score(y_val, labels)),
            precision=float(precision_score(y_val, labels, zero_division=0)),
            recall=float(recall_score(y_val, labels, zero_division=0)),
            f1=float(f1_score(y_val, labels, zero_division=0)),
            abs_rho=self._correlation_metric.compute(probabilities, s_val),
            threshold=float(threshold),
        )

        # Store row-level validation predictions for audits and plotting.
        predictions = pd.DataFrame(
            {
                "y_true": y_val.astype(float),
                "y_proba": probabilities.astype(float),
                "y_pred_label": labels.astype(int),
                "sensitive": s_val.astype(float),
                "threshold": float(threshold),
            }
        )

        return metrics, predictions

    def _safe_roc_auc(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Compute ROC-AUC, returning NaN if undefined.

        Args:
            y_true: Binary target array.
            y_proba: Predicted probabilities.

        Returns:
            ROC-AUC or NaN when only one class is present.

        Raises:
            None.
        """

        # ROC-AUC is undefined when validation has a single class.
        if len(np.unique(y_true.astype(int))) < 2:
            return float("nan")

        return float(roc_auc_score(y_true, y_proba))

    def _safe_pr_auc(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Compute PR-AUC, returning NaN if undefined.

        Args:
            y_true: Binary target array.
            y_proba: Predicted probabilities.

        Returns:
            Average precision / PR-AUC or NaN when only one class is present.

        Raises:
            None.
        """

        # Average precision is not meaningful with one class only in validation.
        if len(np.unique(y_true.astype(int))) < 2:
            return float("nan")

        return float(average_precision_score(y_true, y_proba))


class BaseModelArtifactSaver:
    """Persist baseline training artifacts to disk.

    Args:
        artifacts: Artifact path configuration.

    Returns:
        Saver object.

    Raises:
        None during initialization.
    """

    def __init__(self, artifacts: BaseTrainingArtifacts | None = None) -> None:
        """Initialize the artifact saver.

        Args:
            artifacts: Optional path configuration.

        Returns:
            None.

        Raises:
            None.
        """

        # Default paths match the documentation and repository structure.
        self._artifacts = artifacts or BaseTrainingArtifacts()

    def save(
        self,
        result: BaseTrainingResult,
        *,
        save_model: bool = False,
    ) -> None:
        """Save history, validation predictions and optionally the model.

        Args:
            result: Baseline training result.
            save_model: Whether to save the Keras model to `model_path`.

        Returns:
            None.

        Raises:
            OSError: If any artifact path cannot be written.
            ValueError: If model saving is requested but no model path exists.
        """

        # History is always useful for loss curves.
        self.save_history(result.history)

        # Validation predictions are always useful for audits.
        self.save_validation_predictions(result.validation_predictions)

        # Model saving is optional because model files are ignored by Git.
        if save_model:
            self.save_model(result.model)

    def save_history(self, history: tf.keras.callbacks.History) -> Path:
        """Save Keras history as CSV.

        Args:
            history: Keras History object.

        Returns:
            Path to the written CSV.

        Raises:
            OSError: If the CSV cannot be written.
        """

        # Ensure the results/tables directory exists.
        self._artifacts.history_csv.parent.mkdir(parents=True, exist_ok=True)

        # Keras stores metrics as lists in history.history.
        pd.DataFrame(history.history).to_csv(
            self._artifacts.history_csv,
            index=False,
        )

        return self._artifacts.history_csv

    def save_validation_predictions(self, predictions: pd.DataFrame) -> Path:
        """Save validation predictions as CSV.

        Args:
            predictions: DataFrame with y_true, y_proba, labels and sensitive.

        Returns:
            Path to the written CSV.

        Raises:
            OSError: If the CSV cannot be written.
        """

        # Ensure the results/tables directory exists.
        self._artifacts.validation_predictions_csv.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Save without index because row order already matches validation order.
        predictions.to_csv(
            self._artifacts.validation_predictions_csv,
            index=False,
        )

        return self._artifacts.validation_predictions_csv

    def save_model(self, model: tf.keras.Model) -> Path:
        """Save the trained Keras model.

        Args:
            model: Trained Keras model.

        Returns:
            Path to the saved model.

        Raises:
            ValueError: If model path is not configured.
        """

        # The path is optional by design.
        if self._artifacts.model_path is None:
            raise ValueError("No model_path configured for BaseTrainingArtifacts.")

        # Ensure results/models exists.
        self._artifacts.model_path.parent.mkdir(parents=True, exist_ok=True)

        # Keras v3-compatible .keras format is used by the configured path.
        model.save(self._artifacts.model_path)

        return self._artifacts.model_path


class BaseModelTrainer:
    """Train the Block 4 baseline model end to end.

    Args:
        config: Optional baseline model configuration.
        reproducibility_manager: Optional reproducibility manager.
        array_validator: Optional processed-array validator.
        class_weight_calculator: Optional class-weight calculator.
        model_builder: Optional Keras model builder.
        callback_factory: Optional Keras callback factory.
        evaluator: Optional validation evaluator.
        artifact_saver: Optional artifact saver.

    Returns:
        Trainer object.

    Raises:
        None during initialization.
    """

    def __init__(
        self,
        config: BaseModelConfig | None = None,
        reproducibility_manager: ReproducibilityManager | None = None,
        array_validator: TrainingArrayValidator | None = None,
        class_weight_calculator: ClassWeightCalculator | None = None,
        model_builder: BaseMLPModelBuilder | None = None,
        callback_factory: TrainingCallbackFactory | None = None,
        evaluator: BaseValidationEvaluator | None = None,
        artifact_saver: BaseModelArtifactSaver | None = None,
    ) -> None:
        """Initialize the baseline trainer.

        Args:
            config: Optional baseline configuration.
            reproducibility_manager: Optional seed/determinism manager.
            array_validator: Optional processed data validator.
            class_weight_calculator: Optional class-weight calculator.
            model_builder: Optional model builder.
            callback_factory: Optional callback factory.
            evaluator: Optional validation evaluator.
            artifact_saver: Optional artifact saver.

        Returns:
            None.

        Raises:
            None.
        """

        # Store the shared configuration.
        self._config = config or BaseModelConfig()

        # Each dependency has a default implementation but can be swapped in
        # tests, keeping the class SOLID-friendly.
        self._reproducibility_manager = (
            reproducibility_manager or ReproducibilityManager()
        )
        self._array_validator = array_validator or TrainingArrayValidator()
        self._class_weight_calculator = (
            class_weight_calculator or ClassWeightCalculator()
        )
        self._model_builder = model_builder or BaseMLPModelBuilder(self._config)
        self._callback_factory = callback_factory or TrainingCallbackFactory(
            self._config
        )
        self._evaluator = evaluator or BaseValidationEvaluator()
        self._artifact_saver = artifact_saver or BaseModelArtifactSaver()

    @property
    def config(self) -> BaseModelConfig:
        """Return the baseline training configuration.

        Args:
            None.

        Returns:
            BaseModelConfig used by this trainer.

        Raises:
            None.
        """

        return self._config

    def train(
        self,
        data: ProcessedSplitDataset,
        *,
        save_artifacts: bool = True,
        save_model: bool = False,
        verbose: int = 1,
    ) -> BaseTrainingResult:
        """Train the baseline model and evaluate it on validation.

        Args:
            data: Processed split dataset from Blocks 2 and 3.
            save_artifacts: Whether to save history and validation predictions.
            save_model: Whether to save the Keras model.
            verbose: Verbosity passed to `model.fit`.

        Returns:
            BaseTrainingResult with trained model, history, weights and metrics.

        Raises:
            BaseModelError: If processed arrays are invalid.
        """

        # Apply seeds before model construction so initialization is controlled.
        self._reproducibility_manager.apply()

        # Validate arrays before TensorFlow sees them.
        self._array_validator.validate(data)

        # Class weights are computed only on train, never validation/test.
        class_weight = self._class_weight_calculator.compute(data.y_train)

        # The input dimension is the processed feature count.
        input_dim = int(data.X_train.shape[1])

        # Build a fresh compiled baseline model.
        model = self._model_builder.build(input_dim=input_dim)

        # Build fresh callbacks; callbacks should not be reused across fits.
        callbacks = self._callback_factory.build()

        # Log validation fairness once per epoch. This prepares histories for
        # Pareto/tuning plots before FAIR loss is implemented.
        callbacks.append(
            FairnessLogger(
                X_val=data.X_val,
                s_val=data.s_val,
                include_sensitive_input=False,
            )
        )

        # Fit the model using validation only for monitoring.
        history = model.fit(
            data.X_train,
            data.y_train,
            validation_data=(data.X_val, data.y_val),
            epochs=self._config.epochs,
            batch_size=self._config.batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=verbose,
        )

        # Evaluate validation predictions and fairness audit metric.
        validation_metrics, validation_predictions = self._evaluator.evaluate(
            model=model,
            X_val=data.X_val,
            y_val=data.y_val,
            s_val=data.s_val,
            threshold=self._config.provisional_threshold,
        )

        # Package everything in an immutable result container.
        result = BaseTrainingResult(
            model=model,
            history=history,
            class_weight=class_weight,
            validation_metrics=validation_metrics,
            validation_predictions=validation_predictions,
        )

        # Persist CSV artifacts by default to support later figures and audits.
        if save_artifacts:
            self._artifact_saver.save(result, save_model=save_model)

        return result


__all__ = [
    "AbsolutePearsonCorrelation",
    "BaseMLPModelBuilder",
    "BaseModelArtifactSaver",
    "BaseModelConfig",
    "BaseModelError",
    "BaseModelTrainer",
    "BaseTrainingArtifacts",
    "BaseTrainingResult",
    "BaseValidationEvaluator",
    "BaseValidationMetrics",
    "ClassWeightCalculator",
    "ReproducibilityConfig",
    "ReproducibilityManager",
    "TrainingArrayValidator",
    "TrainingCallbackFactory",
]
