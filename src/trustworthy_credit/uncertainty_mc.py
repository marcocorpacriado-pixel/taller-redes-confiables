"""Monte Carlo Dropout uncertainty utilities.

MC Dropout is included as a complementary uncertainty signal. The primary MVP
uncertainty remains the M1 -> M2 error-prediction approach implemented in
``src.trustworthy_credit.uncertainty``. This module estimates epistemic
uncertainty by keeping dropout active during inference and measuring prediction
variance across stochastic forward passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf


class MCDropoutError(ValueError):
    """Raised when MC Dropout inputs or outputs are invalid."""


ArrayLike = np.ndarray | Sequence[float]
ModelInput = Any


@dataclass(slots=True)
class MCDropoutConfig:
    """Configuration for MC Dropout inference."""

    n_passes: int = 50
    batch_size: int = 4096
    threshold: float = 0.5
    random_seed: int | None = None

    def __post_init__(self) -> None:
        """Validate MC Dropout runtime settings."""

        if self.n_passes < 2:
            raise MCDropoutError("n_passes must be at least 2.")
        if self.batch_size <= 0:
            raise MCDropoutError("batch_size must be positive.")
        if not 0.0 <= self.threshold <= 1.0:
            raise MCDropoutError("threshold must be in [0, 1].")


@dataclass(slots=True)
class MCDropoutPredictionResult:
    """Predictions produced by MC Dropout inference."""

    mean_proba: np.ndarray
    variance: np.ndarray
    pred_label: np.ndarray
    all_passes: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate result shapes and numeric consistency."""

        self.mean_proba = np.asarray(self.mean_proba, dtype=np.float32).reshape(-1)
        self.variance = np.asarray(self.variance, dtype=np.float32).reshape(-1)
        self.pred_label = np.asarray(self.pred_label, dtype=np.int32).reshape(-1)

        if self.mean_proba.shape != self.variance.shape:
            raise MCDropoutError("mean_proba and variance must have the same shape.")
        if self.pred_label.shape != self.mean_proba.shape:
            raise MCDropoutError("pred_label and mean_proba must have the same shape.")
        if np.any(self.variance < -1e-8):
            raise MCDropoutError("MC Dropout variance cannot be negative.")

        if self.all_passes is not None:
            self.all_passes = np.asarray(self.all_passes, dtype=np.float32)
            expected = (self.mean_proba.shape[0], self.all_passes.shape[1])
            if self.all_passes.shape != expected:
                raise MCDropoutError(
                    "all_passes must have shape (n_samples, n_passes)."
                )


@dataclass(slots=True)
class MCDropoutUncertaintyEstimator:
    """Run stochastic dropout inference and compute predictive variance."""

    config: MCDropoutConfig = field(default_factory=MCDropoutConfig)

    def predict(
        self,
        model: Any,
        X: ModelInput,
        sensitive: ArrayLike | None = None,
        keep_all_passes: bool = True,
    ) -> MCDropoutPredictionResult:
        """Estimate predictive mean and variance with dropout active.

        Parameters
        ----------
        model:
            Keras-like model callable with ``training=True``.
        X:
            Model features. It can be a single array or an already formatted
            Keras input structure.
        sensitive:
            Optional sensitive input for dual-input models. When provided, the
            model input becomes ``[X, sensitive]``.
        keep_all_passes:
            Whether to retain the full stochastic prediction matrix.
        """

        if self.config.random_seed is not None:
            tf.keras.utils.set_random_seed(self.config.random_seed)

        model_input = self._format_model_input(X=X, sensitive=sensitive)
        n_samples = self._infer_n_samples(model_input)
        if n_samples == 0:
            raise MCDropoutError("Cannot run MC Dropout on an empty input.")

        passes = np.empty((n_samples, self.config.n_passes), dtype=np.float32)
        for pass_idx in range(self.config.n_passes):
            passes[:, pass_idx] = self._predict_one_pass(model, model_input, n_samples)

        mean_proba = passes.mean(axis=1)
        variance = passes.var(axis=1)
        pred_label = (mean_proba >= self.config.threshold).astype(np.int32)

        return MCDropoutPredictionResult(
            mean_proba=mean_proba,
            variance=variance,
            pred_label=pred_label,
            all_passes=passes if keep_all_passes else None,
        )

    @staticmethod
    def _format_model_input(X: ModelInput, sensitive: ArrayLike | None) -> ModelInput:
        """Build simple or dual Keras inputs without altering existing structures."""

        if sensitive is None:
            return X
        if isinstance(X, Mapping):
            raise MCDropoutError("sensitive cannot be combined with mapping inputs.")
        return [np.asarray(X), np.asarray(sensitive).reshape(-1, 1)]

    def _predict_one_pass(
        self,
        model: Any,
        model_input: ModelInput,
        n_samples: int,
    ) -> np.ndarray:
        """Run one stochastic forward pass in batches."""

        predictions = np.empty(n_samples, dtype=np.float32)
        for start in range(0, n_samples, self.config.batch_size):
            end = min(start + self.config.batch_size, n_samples)
            batch_input = self._slice_model_input(model_input, start, end)
            batch_pred = model(batch_input, training=True)
            predictions[start:end] = self._flatten_prediction(batch_pred, end - start)
        return predictions

    @staticmethod
    def _flatten_prediction(prediction: Any, expected_length: int) -> np.ndarray:
        """Convert a model output to a one-dimensional probability vector."""

        if isinstance(prediction, (list, tuple)):
            if not prediction:
                raise MCDropoutError("Model returned an empty prediction sequence.")
            prediction = prediction[0]
        values = np.asarray(prediction, dtype=np.float32).reshape(-1)
        if values.shape[0] != expected_length:
            raise MCDropoutError(
                "Model prediction length does not match input batch length."
            )
        return values

    @classmethod
    def _infer_n_samples(cls, model_input: ModelInput) -> int:
        """Infer sample count from a Keras input structure."""

        first = cls._first_array(model_input)
        return int(np.asarray(first).shape[0])

    @classmethod
    def _slice_model_input(cls, model_input: ModelInput, start: int, end: int) -> ModelInput:
        """Slice a Keras input structure along the sample axis."""

        if isinstance(model_input, Mapping):
            return {
                key: np.asarray(value)[start:end]
                for key, value in model_input.items()
            }
        if isinstance(model_input, tuple):
            return tuple(np.asarray(value)[start:end] for value in model_input)
        if isinstance(model_input, list):
            return [np.asarray(value)[start:end] for value in model_input]
        return np.asarray(model_input)[start:end]

    @classmethod
    def _first_array(cls, model_input: ModelInput) -> Any:
        """Return the first array-like object inside a model input structure."""

        if isinstance(model_input, Mapping):
            try:
                return next(iter(model_input.values()))
            except StopIteration as exc:
                raise MCDropoutError("Mapping input cannot be empty.") from exc
        if isinstance(model_input, (list, tuple)):
            if not model_input:
                raise MCDropoutError("Sequence input cannot be empty.")
            return model_input[0]
        return model_input


@dataclass(slots=True)
class MCDropoutSummaryBuilder:
    """Build reporting tables from MC Dropout prediction results."""

    def to_frame(
        self,
        result: MCDropoutPredictionResult,
        y_true: ArrayLike | None = None,
        ext_null_count: ArrayLike | None = None,
    ) -> pd.DataFrame:
        """Create a row-level MC Dropout result table."""

        frame = pd.DataFrame(
            {
                "mc_mean_proba": result.mean_proba,
                "mc_variance": result.variance,
                "mc_pred_label": result.pred_label,
            }
        )
        if y_true is not None:
            y = self._validate_vector(y_true, len(frame), "y_true")
            frame["y_true"] = y
        if ext_null_count is not None:
            ext = self._validate_ext_null_count(ext_null_count, len(frame))
            frame["EXT_NULL_COUNT"] = ext
        return frame

    def summary_by_target(
        self,
        result: MCDropoutPredictionResult,
        y_true: ArrayLike,
    ) -> pd.DataFrame:
        """Summarize MC Dropout variance by observed target."""

        y = self._validate_vector(y_true, result.variance.shape[0], "y_true")
        frame = pd.DataFrame({"y_true": y, "mc_variance": result.variance})
        return (
            frame.groupby("y_true")["mc_variance"]
            .agg(count="size", mean_variance="mean", median_variance="median")
            .reset_index()
        )

    def summary_by_ext_null_count(
        self,
        result: MCDropoutPredictionResult,
        ext_null_count: ArrayLike,
    ) -> pd.DataFrame:
        """Summarize MC Dropout variance by raw EXT_NULL_COUNT."""

        ext = self._validate_ext_null_count(ext_null_count, result.variance.shape[0])
        frame = pd.DataFrame({"EXT_NULL_COUNT": ext, "mc_variance": result.variance})
        return (
            frame.groupby("EXT_NULL_COUNT")["mc_variance"]
            .agg(count="size", mean_variance="mean", median_variance="median")
            .reset_index()
            .sort_values("EXT_NULL_COUNT")
        )

    @staticmethod
    def _validate_vector(values: ArrayLike, expected_length: int, name: str) -> np.ndarray:
        """Validate a one-dimensional vector length."""

        vector = np.asarray(values).reshape(-1)
        if vector.shape[0] != expected_length:
            raise MCDropoutError(
                f"{name} length {vector.shape[0]} does not match "
                f"expected length {expected_length}."
            )
        return vector

    @classmethod
    def _validate_ext_null_count(
        cls,
        values: ArrayLike,
        expected_length: int,
    ) -> np.ndarray:
        """Validate semantic EXT_NULL_COUNT values."""

        vector = cls._validate_vector(values, expected_length, "EXT_NULL_COUNT")
        int_vector = vector.astype(int)
        if not np.allclose(vector, int_vector):
            raise MCDropoutError("EXT_NULL_COUNT must contain integer values.")
        invalid = set(np.unique(int_vector).tolist()).difference({0, 1, 2, 3})
        if invalid:
            raise MCDropoutError(
                "EXT_NULL_COUNT contains invalid values: "
                + ", ".join(str(value) for value in sorted(invalid))
            )
        return int_vector
