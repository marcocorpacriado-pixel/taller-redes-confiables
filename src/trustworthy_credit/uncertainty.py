"""MVP uncertainty estimation with a secondary error model.

This module implements Block 9. It follows the classroom approach:

    M1 -> trained classifier that outputs P(TARGET=1 | X)
    M2 -> secondary regressor that predicts the absolute error of M1

The MVP trains M2 on validation errors, not on in-sample train errors. This is
less strong than the OOF approach of Block 10, but it is honest enough for the
mandatory MVP and produces the uncertainty CSV used by Block 12.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .metrics import apply_threshold
from .models import custom_model_objects
from .preprocessing import ProcessedSplitDataset
from .tuning import DualInputFormatter


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class UncertaintyError(ValueError):
    """Raised when MVP uncertainty estimation cannot proceed safely.

    Args:
        message: Human-readable explanation of the uncertainty failure.

    Returns:
        None.

    Raises:
        This exception is raised by Block 9 classes when inputs, configuration
        values or feature metadata violate the uncertainty pipeline contract.
    """


@dataclass(frozen=True)
class UncertaintyModelConfig:
    """Configuration for the secondary uncertainty model M2.

    Args:
        hidden_units: Dense layer widths for M2.
        activation: Activation used in hidden Dense layers.
        dropout: Dropout rate after hidden layers.
        output_activation: Output activation. `softplus` keeps the prediction
            non-negative without the hard zero-collapse risk of ReLU.
        normalize_inputs: Whether to add a train-adapted Keras normalization
            layer at the front of M2.
        learning_rate: Adam learning rate.
        gradient_clipnorm: Adam clipnorm.
        loss: Regression loss. MVP uses MAE.
        batch_size: Batch size for M2 training and prediction.
        epochs: Maximum M2 epochs.
        internal_validation_size: Fraction of validation-error rows reserved
            for M2 early stopping.
        random_state: Seed for the explicit M2 internal split.
        early_stopping_monitor: Metric monitored by EarlyStopping.
        early_stopping_patience: Epochs without improvement before stopping.
        early_stopping_restore_best_weights: Whether to restore best M2 weights.

    Returns:
        Immutable M2 configuration.

    Raises:
        None.
    """

    # A deliberately small network reduces the risk of M2 overfitting the
    # validation-error target.
    hidden_units: tuple[int, ...] = (32,)

    # ReLU remains acceptable in hidden layers, where a zero activation does not
    # directly force the final uncertainty estimate to be zero.
    activation: str = "relu"

    # Dropout makes the tiny M2 less likely to memorize validation rows.
    dropout: float = 0.3

    # Error is non-negative. Softplus avoids the hard zero output that made the
    # previous ReLU head collapse to constant uncertainty.
    output_activation: str = "softplus"

    # M2 receives processed features plus unscaled financial amounts; this
    # in-model normalization keeps magnitudes comparable and is saved with M2.
    normalize_inputs: bool = True

    # Conservative defaults aligned with earlier Keras models.
    learning_rate: float = 1e-3
    gradient_clipnorm: float = 1.0

    # MAE is robust for absolute-error targets concentrated near zero.
    loss: str = "mae"

    # Same operational batch size as other blocks.
    batch_size: int = 1024
    epochs: int = 100

    # Explicit split, instead of Keras validation_split, keeps the seed visible.
    internal_validation_size: float = 0.2
    random_state: int = 42

    # M2 early stopping monitors its own validation loss.
    early_stopping_monitor: str = "val_loss"
    early_stopping_patience: int = 10
    early_stopping_restore_best_weights: bool = True


@dataclass(frozen=True)
class UncertaintyArtifactPaths:
    """Artifact paths for Block 9 outputs.

    Args:
        project_root: Repository root used to anchor absolute paths.
        tables_dir_name: Relative tables directory.
        models_dir_name: Relative models directory.
        test_predictions_filename: Filename for row-level test uncertainty.
        summary_filename: Filename for uncertainty summary by target.
        history_filename: Filename for M2 training history.
        model_filename: Filename for the saved M2 model.

    Returns:
        Immutable path configuration.

    Raises:
        None.
    """

    # Anchor to project root so notebooks do not write under notebooks/results.
    project_root: Path = _PROJECT_ROOT

    # Keep directory names aligned with previous blocks.
    tables_dir_name: str = "results/tables"
    models_dir_name: str = "results/models"

    # Filenames consumed later by Blocks 11 and 12.
    test_predictions_filename: str = "uncertainty_test.csv"
    summary_filename: str = "uncertainty_summary_by_target.csv"
    history_filename: str = "history_uncertainty_m2.csv"
    model_filename: str = "uncertainty_m2.keras"

    @property
    def tables_directory(self) -> Path:
        """Return absolute tables directory.

        Args:
            None.

        Returns:
            Absolute path to results tables.

        Raises:
            None.
        """

        return self.project_root / self.tables_dir_name

    @property
    def models_directory(self) -> Path:
        """Return absolute models directory.

        Args:
            None.

        Returns:
            Absolute path to results models.

        Raises:
            None.
        """

        return self.project_root / self.models_dir_name

    @property
    def test_predictions_csv(self) -> Path:
        """Return absolute uncertainty test CSV path.

        Args:
            None.

        Returns:
            Absolute path for `uncertainty_test.csv`.

        Raises:
            None.
        """

        return self.tables_directory / self.test_predictions_filename

    @property
    def summary_csv(self) -> Path:
        """Return absolute summary CSV path.

        Args:
            None.

        Returns:
            Absolute path for target-group uncertainty summary.

        Raises:
            None.
        """

        return self.tables_directory / self.summary_filename

    @property
    def history_csv(self) -> Path:
        """Return absolute M2 history CSV path.

        Args:
            None.

        Returns:
            Absolute path for M2 training history.

        Raises:
            None.
        """

        return self.tables_directory / self.history_filename

    @property
    def model_path(self) -> Path:
        """Return absolute M2 model path.

        Args:
            None.

        Returns:
            Absolute path for saved M2 model.

        Raises:
            None.
        """

        return self.models_directory / self.model_filename


@dataclass(frozen=True)
class UncertaintyTrainingData:
    """Training arrays used by the secondary uncertainty model.

    Args:
        Z: Augmented feature matrix `[X_val, y_val_proba]`.
        error: Absolute validation error target.
        y_proba: M1 validation probabilities.

    Returns:
        Immutable uncertainty training data.

    Raises:
        None.
    """

    Z: np.ndarray
    error: np.ndarray
    y_proba: np.ndarray


@dataclass(frozen=True)
class UncertaintyPredictionResult:
    """Row-level uncertainty prediction result on test.

    Args:
        predictions: DataFrame saved as `uncertainty_test.csv`.
        summary: DataFrame saved as `uncertainty_summary_by_target.csv`.
        y_test_proba: M1 test probabilities.
        uncertainty: M2 predicted uncertainty values.

    Returns:
        Immutable test uncertainty result.

    Raises:
        None.
    """

    predictions: pd.DataFrame
    summary: pd.DataFrame
    y_test_proba: np.ndarray
    uncertainty: np.ndarray


@dataclass(frozen=True)
class UncertaintyMVPResult:
    """Complete result returned by the Block 9 MVP runner.

    Args:
        m2_model: Trained uncertainty model.
        history: Keras History from M2 training.
        training_data: Validation-derived M2 training target.
        prediction_result: Test uncertainty predictions and summary.
        artifacts: Artifact paths used by the run.

    Returns:
        Immutable Block 9 result.

    Raises:
        None.
    """

    m2_model: tf.keras.Model
    history: tf.keras.callbacks.History
    training_data: UncertaintyTrainingData
    prediction_result: UncertaintyPredictionResult
    artifacts: UncertaintyArtifactPaths


class DualInputModelPredictor:
    """Predict probabilities from a dual-input FAIR model.

    Args:
        batch_size: Batch size used during model prediction.

    Returns:
        Predictor object.

    Raises:
        None during initialization.
    """

    def __init__(self, batch_size: int = 1024) -> None:
        """Initialize the predictor.

        Args:
            batch_size: Prediction batch size.

        Returns:
            None.

        Raises:
            UncertaintyError: If batch size is not positive.
        """

        if batch_size <= 0:
            raise UncertaintyError("batch_size must be positive.")

        self._batch_size = int(batch_size)
        self._formatter = DualInputFormatter()

    def predict(self, model: tf.keras.Model, X: np.ndarray, sensitive: np.ndarray) -> np.ndarray:
        """Predict flat probabilities with a dual-input model.

        Args:
            model: Trained M1 FAIR model.
            X: Processed feature matrix.
            sensitive: Sensitive vector aligned with X.

        Returns:
            Flat probability array.

        Raises:
            UncertaintyError: If model output length is inconsistent.
        """

        probabilities = model.predict(
            self._formatter.format(X, sensitive),
            batch_size=self._batch_size,
            verbose=0,
        ).reshape(-1)

        if probabilities.shape[0] != X.shape[0]:
            raise UncertaintyError("Prediction length does not match X rows.")

        return probabilities.astype("float64")


class UncertaintyFeatureBuilder:
    """Build augmented features and extract EXT_NULL_COUNT.

    Args:
        ext_null_feature_name: Name of the processed EXT missingness count.

    Returns:
        Feature builder object.

    Raises:
        None during initialization.
    """

    def __init__(self, ext_null_feature_name: str = "EXT_NULL_COUNT") -> None:
        """Initialize the feature builder.

        Args:
            ext_null_feature_name: Feature name used for external-score
                missingness count.

        Returns:
            None.

        Raises:
            UncertaintyError: If feature name is empty.
        """

        if not ext_null_feature_name:
            raise UncertaintyError("ext_null_feature_name cannot be empty.")

        self._ext_null_feature_name = ext_null_feature_name

    def build_augmented(self, X: np.ndarray, y_proba: np.ndarray) -> np.ndarray:
        """Build M2 input matrix `[X, y_proba]`.

        Args:
            X: Processed feature matrix.
            y_proba: M1 probabilities aligned with X.

        Returns:
            Augmented matrix with one extra probability column.

        Raises:
            UncertaintyError: If row counts do not align.
        """

        probabilities = np.asarray(y_proba, dtype="float64").reshape(-1, 1)

        if X.shape[0] != probabilities.shape[0]:
            raise UncertaintyError("X and y_proba lengths are not aligned.")

        return np.column_stack([X, probabilities]).astype("float32")

    def extract_ext_null_count(
        self,
        X: np.ndarray,
        feature_names: tuple[str, ...],
    ) -> np.ndarray:
        """Extract EXT_NULL_COUNT from a processed matrix.

        Args:
            X: Processed feature matrix.
            feature_names: Ordered processed feature names.

        Returns:
            Flat EXT_NULL_COUNT array.

        Raises:
            UncertaintyError: If feature is missing.
        """

        try:
            index = tuple(feature_names).index(self._ext_null_feature_name)
        except ValueError as exc:
            raise UncertaintyError(
                f"{self._ext_null_feature_name} is missing from feature_names."
            ) from exc

        return np.asarray(X[:, index]).reshape(-1)


class UncertaintyTrainingDataBuilder:
    """Build M2 training target from M1 validation errors.

    Args:
        predictor: Optional dual-input predictor.
        feature_builder: Optional uncertainty feature builder.

    Returns:
        Builder object.

    Raises:
        None during initialization.
    """

    def __init__(
        self,
        predictor: DualInputModelPredictor | None = None,
        feature_builder: UncertaintyFeatureBuilder | None = None,
    ) -> None:
        """Initialize the training data builder.

        Args:
            predictor: Optional M1 predictor.
            feature_builder: Optional M2 feature builder.

        Returns:
            None.

        Raises:
            None.
        """

        self._predictor = predictor or DualInputModelPredictor()
        self._feature_builder = feature_builder or UncertaintyFeatureBuilder()

    def build(
        self,
        *,
        m1_model: tf.keras.Model,
        data: ProcessedSplitDataset,
    ) -> UncertaintyTrainingData:
        """Build validation-derived M2 training data.

        Args:
            m1_model: Trained FAIR model selected by validation.
            data: Processed split dataset.

        Returns:
            UncertaintyTrainingData with Z and absolute error target.

        Raises:
            UncertaintyError: If validation predictions are misaligned.
        """

        y_val_proba = self._predictor.predict(m1_model, data.X_val, data.s_val)

        if y_val_proba.shape[0] != data.y_val.shape[0]:
            raise UncertaintyError("Validation probabilities and y_val differ.")

        # Absolute error is the target M2 learns to predict.
        error = np.abs(y_val_proba - data.y_val.reshape(-1)).astype("float32")

        # M2 receives original processed features plus M1 probability.
        Z = self._feature_builder.build_augmented(data.X_val, y_val_proba)

        return UncertaintyTrainingData(
            Z=Z,
            error=error,
            y_proba=y_val_proba,
        )


class UncertaintyInternalSplitter:
    """Split validation-error rows into M2 train/validation subsets.

    Args:
        config: Optional M2 configuration.

    Returns:
        Splitter object.

    Raises:
        None during initialization.
    """

    def __init__(self, config: UncertaintyModelConfig | None = None) -> None:
        """Initialize the splitter.

        Args:
            config: Optional uncertainty model config.

        Returns:
            None.

        Raises:
            UncertaintyError: If split size is invalid.
        """

        self._config = config or UncertaintyModelConfig()

        if not 0.0 < self._config.internal_validation_size < 1.0:
            raise UncertaintyError("internal_validation_size must be in (0, 1).")

    def split(self, Z: np.ndarray, error: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split M2 data with an explicit reproducible seed.

        Args:
            Z: M2 input matrix.
            error: M2 absolute-error target.

        Returns:
            Tuple `(Z_train, Z_val, error_train, error_val)`.

        Raises:
            UncertaintyError: If row counts do not align.
        """

        if Z.shape[0] != error.reshape(-1).shape[0]:
            raise UncertaintyError("Z and error lengths are not aligned.")

        return train_test_split(
            Z,
            error.reshape(-1),
            test_size=self._config.internal_validation_size,
            random_state=self._config.random_state,
            shuffle=True,
        )


class UncertaintyM2ModelBuilder:
    """Build the secondary uncertainty regressor M2.

    Args:
        config: Optional uncertainty model configuration.

    Returns:
        Builder object.

    Raises:
        None during initialization.
    """

    def __init__(self, config: UncertaintyModelConfig | None = None) -> None:
        """Initialize the model builder.

        Args:
            config: Optional M2 config.

        Returns:
            None.

        Raises:
            UncertaintyError: If config is invalid.
        """

        self._config = config or UncertaintyModelConfig()
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the M2 configuration.

        Args:
            None.

        Returns:
            None.

        Raises:
            UncertaintyError: If config is unsafe.
        """

        if not self._config.hidden_units:
            raise UncertaintyError("hidden_units cannot be empty.")

        if any(units <= 0 for units in self._config.hidden_units):
            raise UncertaintyError("hidden_units must contain positive values.")

        if not 0.0 <= self._config.dropout < 1.0:
            raise UncertaintyError("dropout must be in [0, 1).")

        if self._config.learning_rate <= 0:
            raise UncertaintyError("learning_rate must be positive.")

        if self._config.gradient_clipnorm <= 0:
            raise UncertaintyError("gradient_clipnorm must be positive.")

        allowed_output_activations = {"softplus", "linear"}
        if self._config.output_activation not in allowed_output_activations:
            raise UncertaintyError(
                "output_activation must be 'softplus' or 'linear' for M2."
            )

    def build(
        self,
        input_dim: int,
        normalization_data: np.ndarray | None = None,
    ) -> tf.keras.Model:
        """Build and compile M2.

        Args:
            input_dim: Number of augmented M2 input columns.
            normalization_data: M2 training rows used to adapt the optional
                in-model normalization layer.

        Returns:
            Compiled Keras regression model.

        Raises:
            UncertaintyError: If input_dim or normalization data is invalid.
        """

        if input_dim <= 0:
            raise UncertaintyError("input_dim must be positive.")

        inputs = tf.keras.Input(shape=(input_dim,), name="uncertainty_features")
        x = inputs

        if self._config.normalize_inputs:
            if normalization_data is None:
                raise UncertaintyError(
                    "normalization_data is required when normalize_inputs=True."
                )

            normalizer_data = np.asarray(normalization_data, dtype="float32")

            if normalizer_data.ndim != 2 or normalizer_data.shape[1] != input_dim:
                raise UncertaintyError(
                    "normalization_data must be a 2D matrix aligned with input_dim."
                )

            if not np.isfinite(normalizer_data).all():
                raise UncertaintyError("normalization_data contains non-finite values.")

            # Financial inputs can be orders of magnitude larger than the
            # probability column. Keeping normalization inside M2 makes the
            # saved model self-contained for later prediction.
            normalizer = tf.keras.layers.Normalization(
                name="m2_feature_normalization"
            )
            normalizer.adapt(normalizer_data)
            x = normalizer(inputs)

        for layer_index, units in enumerate(self._config.hidden_units):
            x = tf.keras.layers.Dense(
                units,
                activation=self._config.activation,
                name=f"m2_dense_{layer_index}",
            )(x)
            x = tf.keras.layers.Dropout(
                self._config.dropout,
                name=f"m2_dropout_{layer_index}",
            )(x)

        outputs = tf.keras.layers.Dense(
            1,
            activation=self._config.output_activation,
            name="uncertainty",
        )(x)

        model = tf.keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="uncertainty_m2_mvp",
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self._config.learning_rate,
            clipnorm=self._config.gradient_clipnorm,
        )

        model.compile(optimizer=optimizer, loss=self._config.loss)
        return model


class UncertaintyCallbackFactory:
    """Create callbacks for M2 training.

    Args:
        config: Optional uncertainty model config.

    Returns:
        Callback factory object.

    Raises:
        None during initialization.
    """

    def __init__(self, config: UncertaintyModelConfig | None = None) -> None:
        """Initialize the callback factory.

        Args:
            config: Optional M2 config.

        Returns:
            None.

        Raises:
            None.
        """

        self._config = config or UncertaintyModelConfig()

    def build(self) -> list[tf.keras.callbacks.Callback]:
        """Build callbacks for one M2 fit.

        Args:
            None.

        Returns:
            Fresh callback list.

        Raises:
            None.
        """

        return [
            tf.keras.callbacks.EarlyStopping(
                monitor=self._config.early_stopping_monitor,
                patience=self._config.early_stopping_patience,
                restore_best_weights=self._config.early_stopping_restore_best_weights,
            )
        ]


class UncertaintySummaryBuilder:
    """Build summary tables for uncertainty analysis.

    Args:
        None.

    Returns:
        Summary builder object.

    Raises:
        None.
    """

    def build_by_target(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Summarize uncertainty by true TARGET group.

        Args:
            predictions: Row-level uncertainty predictions.

        Returns:
            DataFrame with count, mean, median and IQR by target.

        Raises:
            UncertaintyError: If required columns are missing.
        """

        required = {"y_true", "uncertainty"}
        missing = required.difference(predictions.columns)
        if missing:
            raise UncertaintyError(f"Missing columns for summary: {sorted(missing)}")

        rows: list[dict[str, Any]] = []

        for target_value, group in predictions.groupby("y_true", sort=True):
            uncertainty = group["uncertainty"].astype(float)
            q1 = float(uncertainty.quantile(0.25))
            q3 = float(uncertainty.quantile(0.75))
            rows.append(
                {
                    "y_true": int(target_value),
                    "count": int(group.shape[0]),
                    "uncertainty_mean": float(uncertainty.mean()),
                    "uncertainty_median": float(uncertainty.median()),
                    "uncertainty_q1": q1,
                    "uncertainty_q3": q3,
                    "uncertainty_iqr": q3 - q1,
                }
            )

        return pd.DataFrame(rows)


class UncertaintyPredictionBuilder:
    """Build row-level test uncertainty predictions.

    Args:
        predictor: Optional M1 predictor.
        feature_builder: Optional M2 feature builder.
        summary_builder: Optional summary builder.

    Returns:
        Prediction builder object.

    Raises:
        None during initialization.
    """

    def __init__(
        self,
        predictor: DualInputModelPredictor | None = None,
        feature_builder: UncertaintyFeatureBuilder | None = None,
        summary_builder: UncertaintySummaryBuilder | None = None,
    ) -> None:
        """Initialize the prediction builder.

        Args:
            predictor: Optional dual-input M1 predictor.
            feature_builder: Optional M2 feature builder.
            summary_builder: Optional summary builder.

        Returns:
            None.

        Raises:
            None.
        """

        self._predictor = predictor or DualInputModelPredictor()
        self._feature_builder = feature_builder or UncertaintyFeatureBuilder()
        self._summary_builder = summary_builder or UncertaintySummaryBuilder()

    def build(
        self,
        *,
        m1_model: tf.keras.Model,
        m2_model: tf.keras.Model,
        data: ProcessedSplitDataset,
        threshold: float,
        batch_size: int,
    ) -> UncertaintyPredictionResult:
        """Build test predictions with uncertainty.

        Args:
            m1_model: Trained M1 FAIR classifier.
            m2_model: Trained M2 uncertainty regressor.
            data: Processed split dataset.
            threshold: Validation-selected threshold for M1.
            batch_size: Batch size for M2 prediction.

        Returns:
            UncertaintyPredictionResult with row-level and summary data.

        Raises:
            UncertaintyError: If arrays are misaligned.
        """

        y_test_proba = self._predictor.predict(m1_model, data.X_test, data.s_test)
        Z_test = self._feature_builder.build_augmented(data.X_test, y_test_proba)
        raw_uncertainty = m2_model.predict(
            Z_test,
            batch_size=batch_size,
            verbose=0,
        ).reshape(-1)

        if raw_uncertainty.shape[0] != data.y_test.shape[0]:
            raise UncertaintyError("Uncertainty length does not match y_test.")

        uncertainty = self._clip_and_validate_uncertainty(raw_uncertainty)
        labels = apply_threshold(y_test_proba, threshold)
        ext_null_count = self._validate_ext_null_count(
            data.ext_null_count_test,
            expected_rows=data.y_test.shape[0],
        )

        predictions = pd.DataFrame(
            {
                "SK_ID_CURR": list(data.test_ids),
                "y_true": data.y_test.astype(int),
                "y_proba": y_test_proba.astype(float),
                "y_pred_label": labels.astype(int),
                "sensitive": data.s_test.astype(int),
                "threshold": float(threshold),
                "uncertainty": uncertainty.astype(float),
                "EXT_NULL_COUNT": ext_null_count.astype(int),
            }
        )

        summary = self._summary_builder.build_by_target(predictions)

        return UncertaintyPredictionResult(
            predictions=predictions,
            summary=summary,
            y_test_proba=y_test_proba,
            uncertainty=uncertainty.astype(float),
        )

    def _clip_and_validate_uncertainty(self, uncertainty: np.ndarray) -> np.ndarray:
        """Clip M2 predictions to probability-error range and validate them.

        Args:
            uncertainty: Raw M2 predictions for the test split.

        Returns:
            One-dimensional float array clipped to `[0, 1]`.

        Raises:
            UncertaintyError: If predictions are non-finite or collapse to a
                constant value on a multi-row test split.
        """

        values = np.asarray(uncertainty, dtype="float64").reshape(-1)

        if not np.isfinite(values).all():
            raise UncertaintyError("M2 uncertainty predictions contain non-finite values.")

        clipped = np.clip(values, 0.0, 1.0)

        # A constant uncertainty vector is not defensible for the required
        # distribution plots and usually signals an M2 output-head failure.
        if clipped.shape[0] > 1 and np.unique(np.round(clipped, decimals=8)).size <= 1:
            raise UncertaintyError(
                "M2 produced constant uncertainty; refusing to save invalid artifacts."
            )

        return clipped

    def _validate_ext_null_count(
        self,
        ext_null_count: np.ndarray,
        *,
        expected_rows: int,
    ) -> np.ndarray:
        """Validate raw EXT_NULL_COUNT values before writing uncertainty CSV.

        Args:
            ext_null_count: Raw external-score missing counts from
                `ProcessedSplitDataset`.
            expected_rows: Number of rows expected in the test split.

        Returns:
            One-dimensional integer array with values in `{0, 1, 2, 3}`.

        Raises:
            UncertaintyError: If values are misaligned, missing or scaled.
        """

        values = np.asarray(ext_null_count).reshape(-1)

        if values.shape[0] != expected_rows:
            raise UncertaintyError("EXT_NULL_COUNT length does not match y_test.")

        if pd.isna(values).any():
            raise UncertaintyError("EXT_NULL_COUNT contains missing values.")

        try:
            numeric_values = values.astype("float64", copy=False)
        except ValueError as exc:
            raise UncertaintyError("EXT_NULL_COUNT must be numeric.") from exc

        if not np.all(np.isin(numeric_values, [0.0, 1.0, 2.0, 3.0])):
            raise UncertaintyError(
                "EXT_NULL_COUNT must contain raw integer values 0, 1, 2 or 3."
            )

        return numeric_values.astype("int8", copy=False)


class UncertaintyArtifactWriter:
    """Persist Block 9 artifacts to disk.

    Args:
        artifacts: Optional uncertainty artifact paths.

    Returns:
        Writer object.

    Raises:
        None during initialization.
    """

    def __init__(self, artifacts: UncertaintyArtifactPaths | None = None) -> None:
        """Initialize the writer.

        Args:
            artifacts: Optional artifact paths.

        Returns:
            None.

        Raises:
            None.
        """

        self._artifacts = artifacts or UncertaintyArtifactPaths()

    def save_history(self, history: tf.keras.callbacks.History) -> Path:
        """Save M2 training history.

        Args:
            history: Keras History from M2 training.

        Returns:
            Written CSV path.

        Raises:
            OSError: If the CSV cannot be written.
        """

        self._artifacts.history_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history.history).to_csv(self._artifacts.history_csv, index=False)
        return self._artifacts.history_csv

    def save_predictions(self, predictions: pd.DataFrame) -> Path:
        """Save row-level test uncertainty predictions.

        Args:
            predictions: DataFrame for `uncertainty_test.csv`.

        Returns:
            Written CSV path.

        Raises:
            OSError: If the CSV cannot be written.
        """

        self._artifacts.test_predictions_csv.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(self._artifacts.test_predictions_csv, index=False)
        return self._artifacts.test_predictions_csv

    def save_summary(self, summary: pd.DataFrame) -> Path:
        """Save target-group uncertainty summary.

        Args:
            summary: Summary DataFrame.

        Returns:
            Written CSV path.

        Raises:
            OSError: If the CSV cannot be written.
        """

        self._artifacts.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(self._artifacts.summary_csv, index=False)
        return self._artifacts.summary_csv

    def save_model(self, model: tf.keras.Model) -> Path:
        """Save the trained M2 model.

        Args:
            model: Trained uncertainty regressor.

        Returns:
            Written model path.

        Raises:
            OSError: If model cannot be written.
        """

        self._artifacts.model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(self._artifacts.model_path)
        return self._artifacts.model_path

    def save_all(
        self,
        *,
        result: UncertaintyPredictionResult,
        history: tf.keras.callbacks.History,
        m2_model: tf.keras.Model,
        save_model: bool,
    ) -> None:
        """Save all configured Block 9 artifacts.

        Args:
            result: Prediction result to persist.
            history: M2 training history.
            m2_model: Trained M2 model.
            save_model: Whether to save the `.keras` model.

        Returns:
            None.

        Raises:
            OSError: If any artifact cannot be written.
        """

        self.save_history(history)
        self.save_predictions(result.predictions)
        self.save_summary(result.summary)
        if save_model:
            self.save_model(m2_model)


class FairModelLoader:
    """Load a saved FAIR M1 model.

    Args:
        None.

    Returns:
        Loader object.

    Raises:
        None.
    """

    def load(self, model_path: Path | str) -> tf.keras.Model:
        """Load a saved Keras model with project custom objects.

        Args:
            model_path: Path to a saved `.keras` model.

        Returns:
            Loaded Keras model.

        Raises:
            FileNotFoundError: If path does not exist.
        """

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(path)

        return tf.keras.models.load_model(
            path,
            custom_objects=custom_model_objects(),
        )


class UncertaintyMVPTrainer:
    """Run the complete Block 9 MVP uncertainty pipeline.

    Args:
        config: Optional M2 configuration.
        artifacts: Optional artifact paths.
        training_data_builder: Optional validation-error data builder.
        splitter: Optional internal M2 splitter.
        model_builder: Optional M2 model builder.
        callback_factory: Optional callback factory.
        prediction_builder: Optional test prediction builder.
        writer: Optional artifact writer.

    Returns:
        Trainer object.

    Raises:
        None during initialization.
    """

    def __init__(
        self,
        config: UncertaintyModelConfig | None = None,
        artifacts: UncertaintyArtifactPaths | None = None,
        training_data_builder: UncertaintyTrainingDataBuilder | None = None,
        splitter: UncertaintyInternalSplitter | None = None,
        model_builder: UncertaintyM2ModelBuilder | None = None,
        callback_factory: UncertaintyCallbackFactory | None = None,
        prediction_builder: UncertaintyPredictionBuilder | None = None,
        writer: UncertaintyArtifactWriter | None = None,
    ) -> None:
        """Initialize the MVP uncertainty trainer.

        Args:
            config: Optional M2 config.
            artifacts: Optional artifact paths.
            training_data_builder: Optional M2 data builder.
            splitter: Optional internal splitter.
            model_builder: Optional M2 builder.
            callback_factory: Optional callback factory.
            prediction_builder: Optional test prediction builder.
            writer: Optional artifact writer.

        Returns:
            None.

        Raises:
            None.
        """

        self._config = config or UncertaintyModelConfig()
        self._artifacts = artifacts or UncertaintyArtifactPaths()
        self._training_data_builder = (
            training_data_builder or UncertaintyTrainingDataBuilder()
        )
        self._splitter = splitter or UncertaintyInternalSplitter(self._config)
        self._model_builder = model_builder or UncertaintyM2ModelBuilder(self._config)
        self._callback_factory = callback_factory or UncertaintyCallbackFactory(
            self._config
        )
        self._prediction_builder = prediction_builder or UncertaintyPredictionBuilder()
        self._writer = writer or UncertaintyArtifactWriter(self._artifacts)

    def run(
        self,
        *,
        m1_model: tf.keras.Model,
        data: ProcessedSplitDataset,
        selected_threshold: float,
        save_artifacts: bool = True,
        save_model: bool = True,
        verbose: int = 1,
    ) -> UncertaintyMVPResult:
        """Run M2 training and test uncertainty prediction.

        Args:
            m1_model: Trained FAIR classifier selected by validation.
            data: Processed split dataset.
            selected_threshold: Validation-selected M1 threshold.
            save_artifacts: Whether to write CSV/model artifacts.
            save_model: Whether to save M2 model when saving artifacts.
            verbose: Verbosity passed to M2 `fit`.

        Returns:
            UncertaintyMVPResult with model, history and predictions.

        Raises:
            UncertaintyError: If threshold is invalid or arrays are inconsistent.
        """

        if not 0.0 <= float(selected_threshold) <= 1.0:
            raise UncertaintyError("selected_threshold must be in [0, 1].")

        training_data = self._training_data_builder.build(
            m1_model=m1_model,
            data=data,
        )

        Z_train, Z_val, error_train, error_val = self._splitter.split(
            training_data.Z,
            training_data.error,
        )

        m2_model = self._model_builder.build(
            input_dim=Z_train.shape[1],
            normalization_data=Z_train,
        )

        history = m2_model.fit(
            Z_train,
            error_train,
            validation_data=(Z_val, error_val),
            epochs=self._config.epochs,
            batch_size=self._config.batch_size,
            callbacks=self._callback_factory.build(),
            verbose=verbose,
        )

        prediction_result = self._prediction_builder.build(
            m1_model=m1_model,
            m2_model=m2_model,
            data=data,
            threshold=float(selected_threshold),
            batch_size=self._config.batch_size,
        )

        if save_artifacts:
            self._writer.save_all(
                result=prediction_result,
                history=history,
                m2_model=m2_model,
                save_model=save_model,
            )

        return UncertaintyMVPResult(
            m2_model=m2_model,
            history=history,
            training_data=training_data,
            prediction_result=prediction_result,
            artifacts=self._artifacts,
        )


__all__ = [
    "DualInputModelPredictor",
    "FairModelLoader",
    "UncertaintyArtifactPaths",
    "UncertaintyArtifactWriter",
    "UncertaintyCallbackFactory",
    "UncertaintyError",
    "UncertaintyFeatureBuilder",
    "UncertaintyInternalSplitter",
    "UncertaintyM2ModelBuilder",
    "UncertaintyModelConfig",
    "UncertaintyMVPResult",
    "UncertaintyMVPTrainer",
    "UncertaintyPredictionBuilder",
    "UncertaintyPredictionResult",
    "UncertaintySummaryBuilder",
    "UncertaintyTrainingData",
    "UncertaintyTrainingDataBuilder",
]
