"""Reusable Keras callbacks for model auditing.

This module is introduced in Block 5.5 to make fairness diagnostics available
during training, not only after training. The immediate need is `val_abs_rho`,
which later blocks use for the Pareto frontier and for comparing lambda values.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf


class CallbackError(ValueError):
    """Raised when a project callback receives invalid data.

    Args:
        message: Human-readable explanation of the callback configuration error.

    Returns:
        None.

    Raises:
        This exception is raised by callback constructors when validation arrays
        are not aligned or cannot support the requested diagnostic.
    """


class FairnessLogger(tf.keras.callbacks.Callback):
    """Log absolute Pearson correlation between predictions and sensitive data.

    Args:
        X_val: Validation feature matrix.
        s_val: Validation sensitive values aligned with `X_val`.
        include_sensitive_input: Whether the model expects a second input named
            `sensitive`. Use False for base models and True for FAIR models.
        batch_size: Optional batch size used by `model.predict`.
        log_name: Metric key inserted in Keras logs.
        eps: Numerical tolerance used for zero-variance checks.

    Returns:
        Keras callback that writes `logs[log_name]` at each epoch end.

    Raises:
        CallbackError: If validation arrays are empty, misaligned or contain
        invalid sensitive values.
    """

    def __init__(
        self,
        X_val: np.ndarray,
        s_val: np.ndarray,
        *,
        include_sensitive_input: bool = False,
        batch_size: int | None = None,
        log_name: str = "val_abs_rho",
        eps: float = 1e-8,
    ) -> None:
        """Initialize the fairness logger.

        Args:
            X_val: Validation feature matrix.
            s_val: Validation sensitive values.
            include_sensitive_input: Whether to pass sensitive values as a
                named model input.
            batch_size: Optional prediction batch size.
            log_name: Name of the metric stored in Keras logs.
            eps: Numerical stability value.

        Returns:
            None.

        Raises:
            CallbackError: If arrays or configuration are invalid.
        """

        super().__init__()

        # Store feature data as float32 because that is what TensorFlow models
        # receive from the preprocessing pipeline.
        self._X_val = np.asarray(X_val, dtype="float32")

        # Sensitive values are kept as a column for dual-input FAIR models.
        self._s_val = np.asarray(s_val, dtype="float32").reshape(-1, 1)

        # This switch keeps the same callback usable for one-input and
        # two-input models without duplicating fairness logging code.
        self._include_sensitive_input = bool(include_sensitive_input)

        # Keras accepts None here and then chooses its default.
        self._batch_size = batch_size

        # Make the log key configurable so experiments can store variants later.
        self._log_name = str(log_name)

        # eps prevents division by zero when prediction or sensitive variance is
        # essentially zero.
        self._eps = float(eps)

        self._validate()

    def _validate(self) -> None:
        """Validate callback inputs.

        Args:
            None.

        Returns:
            None.

        Raises:
            CallbackError: If validation inputs are unsafe.
        """

        # Validation features must be a 2-D matrix.
        if self._X_val.ndim != 2:
            raise CallbackError("X_val must be a 2-D matrix.")

        # Empty validation data would make correlation meaningless.
        if self._X_val.shape[0] == 0:
            raise CallbackError("X_val cannot be empty.")

        # Row alignment is mandatory for correlation.
        if self._X_val.shape[0] != self._s_val.shape[0]:
            raise CallbackError("X_val and s_val lengths differ.")

        # Sensitive values must be finite binary indicators.
        if not np.isfinite(self._s_val).all():
            raise CallbackError("s_val contains non-finite values.")

        values = set(np.unique(self._s_val.astype(int)).tolist())

        if not values.issubset({0, 1}):
            raise CallbackError("s_val must contain only 0/1 values.")

        # The log name must be non-empty, otherwise Keras history becomes hard
        # to interpret.
        if not self._log_name:
            raise CallbackError("log_name cannot be empty.")

        if self._eps <= 0:
            raise CallbackError("eps must be positive.")

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Compute and store fairness correlation at the end of an epoch.

        Args:
            epoch: Epoch index provided by Keras.
            logs: Mutable Keras logs dictionary.

        Returns:
            None.

        Raises:
            None. Prediction failures are allowed to propagate from Keras.
        """

        # Keras normally passes a dictionary; the fallback keeps the method safe
        # if called manually in tests.
        logs = logs if logs is not None else {}

        # Predict with the input format expected by the current model.
        probabilities = self.model.predict(
            self._model_inputs(),
            batch_size=self._batch_size,
            verbose=0,
        )

        # Flatten the sigmoid output to one probability per validation row.
        pred = np.asarray(probabilities, dtype="float64").reshape(-1)

        # Sensitive values are stored as a flat vector for correlation.
        sens = self._s_val.astype("float64").reshape(-1)

        logs[self._log_name] = self._absolute_pearson(pred, sens)

    def _model_inputs(self) -> np.ndarray | dict[str, np.ndarray]:
        """Build model inputs for one-input or two-input models.

        Args:
            None.

        Returns:
            Either `X_val` directly or a dict with `features` and `sensitive`.

        Raises:
            None.
        """

        if self._include_sensitive_input:
            return {
                "features": self._X_val,
                "sensitive": self._s_val,
            }

        return self._X_val

    def _absolute_pearson(self, predictions: np.ndarray, sensitive: np.ndarray) -> float:
        """Compute absolute Pearson correlation.

        Args:
            predictions: Flat predicted probabilities.
            sensitive: Flat sensitive values.

        Returns:
            Absolute Pearson correlation, or 0.0 if variance is too small.

        Raises:
            CallbackError: If predictions and sensitive values are misaligned.
        """

        if predictions.shape[0] != sensitive.shape[0]:
            raise CallbackError("Prediction and sensitive lengths differ.")

        pred_centered = predictions - predictions.mean()
        sens_centered = sensitive - sensitive.mean()

        numerator = float(np.mean(pred_centered * sens_centered))
        denominator = float(
            np.sqrt(np.mean(pred_centered**2) * np.mean(sens_centered**2))
        )

        if denominator <= self._eps:
            return 0.0

        return float(abs(numerator / denominator))


__all__ = [
    "CallbackError",
    "FairnessLogger",
]
