"""Keras Tuner and lambda sweep utilities for the FAIR MVP.

This module implements Block 7. It keeps the search process separate from the
model definitions in `src.models`:

    - Keras Tuner model factory with fixed lambda_fair = 0.5.
    - Common callbacks for dual-input FAIR models.
    - Manual lambda sweep after the best architecture is known.
    - Pareto CSV rows and artifact naming conventions.

The design deliberately reuses `FairCustomModelBuilder`, which itself reuses the
Block 5 probability graph. No model architecture is duplicated in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf

from .base_model import (
    BaseModelConfig,
    ClassWeightCalculator,
    ReproducibilityManager,
    TrainingArrayValidator,
)
from .callbacks import FairnessLogger
from .layers import FinancialRatioIndices
from .models import (
    CustomMLPConfig,
    CustomMLPModelBuilder,
    FairCustomModelBuilder,
    FairModelConfig,
    lambda_slug,
)
from .metrics import (
    BinaryClassificationMetricCalculator,
    MetricsError,
    ProbabilityMetricCalculator,
    ThresholdSelector,
)
from .preprocessing import ProcessedSplitDataset


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TuningError(ValueError):
    """Raised when Block 7 tuning or lambda sweep cannot proceed.

    Args:
        message: Human-readable explanation of the failure.

    Returns:
        None.

    Raises:
        This exception is raised by Block 7 classes when configuration,
        validation data or artifact settings are unsafe.
    """


@dataclass(frozen=True)
class TuningConfig:
    """Configuration for Keras Tuner and the manual lambda sweep.

    Args:
        tuning_lambda_fair: Fixed fairness lambda used during architecture
            search. The project uses 0.5 to put moderate FAIR pressure on the
            tuner.
        lambda_values: Lambda values trained during the controlled sweep.
        max_trials: Maximum Keras Tuner trials.
        executions_per_trial: Number of executions per tuner trial.
        tuner_objective: Keras metric name optimized by the tuner.
        tuner_direction: Optimization direction for the tuner objective.
        overwrite_tuner: Whether Keras Tuner may overwrite a previous search.
        tuner_project_name: Keras Tuner project folder name.
        min_layers: Minimum number of hidden layers considered by the tuner.
        max_layers: Maximum number of hidden layers considered by the tuner.
        units_choices: Candidate hidden-layer widths.
        activation_choices: Candidate hidden activations.
        dropout_min: Minimum dropout rate.
        dropout_max: Maximum dropout rate.
        dropout_step: Step size for dropout search.
        learning_rate_min: Lower learning-rate bound.
        learning_rate_max: Upper learning-rate bound.
        batch_size: Batch size for tuner search and lambda sweep.
        epochs: Maximum epochs for each fit.
        early_stopping_patience: EarlyStopping patience.
        reduce_lr_patience: ReduceLROnPlateau patience.
        reduce_lr_factor: Multiplicative learning-rate reduction.
        min_learning_rate: Floor for ReduceLROnPlateau.
        fair_selection_max_auc_drop: Maximum validation AUC drop allowed when
            selecting the final FAIR candidate by lowest abs rho.

    Returns:
        Immutable Block 7 configuration.

    Raises:
        None.
    """

    # The tuner searches architectures under moderate fairness pressure.
    tuning_lambda_fair: float = 0.5

    # The sweep creates the Pareto family consumed by Blocks 11 and 12.
    lambda_values: tuple[float, ...] = (
        0.0,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
    )

    # Keras Tuner defaults for the real MVP run.
    max_trials: int = 30
    executions_per_trial: int = 1
    tuner_objective: str = "val_auc"
    tuner_direction: str = "max"
    overwrite_tuner: bool = True
    tuner_project_name: str = "fair_credit_mvp"

    # Search space for the dense stack.
    min_layers: int = 1
    max_layers: int = 4
    units_choices: tuple[int, ...] = (64, 128, 256)
    activation_choices: tuple[str, ...] = ("relu", "elu")
    dropout_min: float = 0.0
    dropout_max: float = 0.5
    dropout_step: float = 0.1
    learning_rate_min: float = 1e-4
    learning_rate_max: float = 1e-2

    # Training controls shared by tuner and sweep.
    batch_size: int = 1024
    epochs: int = 100
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-6

    # FAIR selection is validation-only. Test remains untouched until Block 11.
    fair_selection_max_auc_drop: float = 0.02


@dataclass(frozen=True)
class TuningArtifactPaths:
    """Artifact locations for Block 7.

    Args:
        project_root: Repository root used to build absolute paths.
        tuner_dir_name: Directory containing Keras Tuner state.
        tables_dir_name: Directory containing CSV artifacts.
        models_dir_name: Directory containing saved Keras models.
        pareto_filename: Filename for the Pareto results table.

    Returns:
        Immutable path configuration with helper methods.

    Raises:
        None.
    """

    # Anchor every path to project root so notebooks cannot accidentally write
    # into notebooks/results.
    project_root: Path = _PROJECT_ROOT

    # Names remain relative so the repository structure is easy to understand.
    tuner_dir_name: str = "kt_dir"
    tables_dir_name: str = "results/tables"
    models_dir_name: str = "results/models"
    pareto_filename: str = "pareto_results.csv"

    @property
    def tuner_directory(self) -> Path:
        """Return the absolute Keras Tuner directory.

        Args:
            None.

        Returns:
            Absolute path to the tuner directory.

        Raises:
            None.
        """

        return self.project_root / self.tuner_dir_name

    @property
    def tables_directory(self) -> Path:
        """Return the absolute tables directory.

        Args:
            None.

        Returns:
            Absolute path to `results/tables`.

        Raises:
            None.
        """

        return self.project_root / self.tables_dir_name

    @property
    def models_directory(self) -> Path:
        """Return the absolute models directory.

        Args:
            None.

        Returns:
            Absolute path to `results/models`.

        Raises:
            None.
        """

        return self.project_root / self.models_dir_name

    @property
    def pareto_results_csv(self) -> Path:
        """Return the absolute Pareto results path.

        Args:
            None.

        Returns:
            Absolute path to `pareto_results.csv`.

        Raises:
            None.
        """

        return self.tables_directory / self.pareto_filename

    def model_path(self, lambda_fair: float) -> Path:
        """Return the absolute model path for one lambda.

        Args:
            lambda_fair: Lambda value used to build the filename slug.

        Returns:
            Absolute `.keras` model path.

        Raises:
            None.
        """

        return self.models_directory / f"fair_lambda_{lambda_slug(lambda_fair)}.keras"

    def history_path(self, lambda_fair: float) -> Path:
        """Return the absolute history CSV path for one lambda.

        Args:
            lambda_fair: Lambda value used to build the filename slug.

        Returns:
            Absolute history CSV path.

        Raises:
            None.
        """

        return (
            self.tables_directory
            / f"history_fair_lambda_{lambda_slug(lambda_fair)}.csv"
        )

    def to_project_relative(self, path: Path) -> str:
        """Convert an absolute path to a portable project-relative string.

        Args:
            path: Absolute or relative path.

        Returns:
            POSIX-style project-relative path when possible.

        Raises:
            None.
        """

        # CSV artifacts should be portable across machines, so paths inside the
        # CSV are stored relative to the repository when possible.
        try:
            relative = Path(path).resolve().relative_to(self.project_root.resolve())
        except ValueError:
            return Path(path).as_posix()

        return relative.as_posix()


@dataclass(frozen=True)
class ParetoResultRow:
    """One row of `pareto_results.csv`.

    Args:
        lambda_fair: Lambda value used for the trained model.
        val_auc: Validation ROC-AUC.
        val_pr_auc: Validation PR-AUC / average precision.
        val_abs_rho: Absolute Pearson correlation between validation
            probabilities and sensitive values.
        val_threshold: Threshold selected on validation with Youden's J.
        val_accuracy: Validation accuracy at `val_threshold`.
        val_precision: Validation precision at `val_threshold`.
        val_recall: Validation recall at `val_threshold`.
        val_f1: Validation F1 at `val_threshold`.
        epochs_trained: Number of epochs actually run by Keras.
        model_path: Project-relative model path.
        history_path: Project-relative history CSV path.
        selected_for_test: Whether Block 11 should evaluate this model on test.

    Returns:
        Immutable row object.

    Raises:
        None.
    """

    lambda_fair: float
    val_auc: float
    val_pr_auc: float
    val_abs_rho: float
    val_threshold: float
    val_accuracy: float
    val_precision: float
    val_recall: float
    val_f1: float
    epochs_trained: int
    model_path: str
    history_path: str
    selected_for_test: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert the row to a CSV-friendly dictionary.

        Args:
            None.

        Returns:
            Dictionary with stable Pareto CSV column names.

        Raises:
            None.
        """

        return {
            "lambda_fair": self.lambda_fair,
            "val_auc": self.val_auc,
            "val_pr_auc": self.val_pr_auc,
            "val_abs_rho": self.val_abs_rho,
            "val_threshold": self.val_threshold,
            "val_accuracy": self.val_accuracy,
            "val_precision": self.val_precision,
            "val_recall": self.val_recall,
            "val_f1": self.val_f1,
            "epochs_trained": self.epochs_trained,
            "model_path": self.model_path,
            "history_path": self.history_path,
            "selected_for_test": self.selected_for_test,
        }


@dataclass(frozen=True)
class LambdaSweepResult:
    """Result returned by the manual lambda sweep.

    Args:
        rows: Pareto rows after selection flags are applied.
        pareto_csv: Path to the saved Pareto table.
        class_weight: Class weights used during all fits.

    Returns:
        Immutable sweep result.

    Raises:
        None.
    """

    rows: tuple[ParetoResultRow, ...]
    pareto_csv: Path
    class_weight: dict[int, float]


@dataclass(frozen=True)
class TunerSearchResult:
    """Result returned by the Keras Tuner search.

    Args:
        tuner: Fitted Keras Tuner instance.
        best_hyperparameters: Best HyperParameters object.
        best_config: CustomMLPConfig reconstructed from best hyperparameters.
        ratio_indices: Financial indices used by the tuned models.
        class_weight: Class weights used during tuner search.

    Returns:
        Immutable tuner search result.

    Raises:
        None.
    """

    tuner: kt.BayesianOptimization
    best_hyperparameters: kt.HyperParameters
    best_config: CustomMLPConfig
    ratio_indices: FinancialRatioIndices
    class_weight: dict[int, float]


class DualInputFormatter:
    """Format arrays for dual-input Keras FAIR models.

    Args:
        None.

    Returns:
        Formatter instance.

    Raises:
        None.
    """

    def format(self, X: np.ndarray, sensitive: np.ndarray) -> dict[str, np.ndarray]:
        """Return Keras input dictionary for a dual-input model.

        Args:
            X: Processed feature matrix.
            sensitive: Binary sensitive array aligned with `X`.

        Returns:
            Dictionary with keys `features` and `sensitive`.

        Raises:
            TuningError: If row counts are not aligned.
        """

        # TensorFlow models expect the sensitive input as a column vector.
        sensitive_column = np.asarray(sensitive).reshape(-1, 1)

        # Row alignment is essential because fairness is computed per batch row.
        if X.shape[0] != sensitive_column.shape[0]:
            raise TuningError("X and sensitive lengths are not aligned.")

        return {
            "features": X,
            "sensitive": sensitive_column,
        }


class FairTuningCallbackFactory:
    """Create callbacks for dual-input FAIR training.

    Args:
        config: Block 7 tuning configuration.

    Returns:
        Factory object able to create fresh callback lists per fit.

    Raises:
        None during initialization.
    """

    def __init__(self, config: TuningConfig | None = None) -> None:
        """Initialize the callback factory.

        Args:
            config: Optional tuning configuration.

        Returns:
            None.

        Raises:
            None.
        """

        self._config = config or TuningConfig()

    def build(self, *, X_val: np.ndarray, s_val: np.ndarray) -> list[tf.keras.callbacks.Callback]:
        """Build callbacks for one FAIR model fit.

        Args:
            X_val: Validation feature matrix.
            s_val: Validation sensitive values.

        Returns:
            Fresh callback list with EarlyStopping, ReduceLROnPlateau and
            FairnessLogger.

        Raises:
            CallbackError: Propagated if FairnessLogger validation fails.
        """

        # EarlyStopping monitors validation AUC because AUC is the primary
        # metric for an imbalanced credit-default model.
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=self._config.early_stopping_patience,
            restore_best_weights=True,
        )

        # ReduceLROnPlateau watches validation loss, which reflects optimization
        # quality even when AUC plateaus.
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=self._config.reduce_lr_factor,
            patience=self._config.reduce_lr_patience,
            min_lr=self._config.min_learning_rate,
        )

        # Dual-input FAIR models require include_sensitive_input=True. This is
        # the most common failure mode in Blocks 6-7, so the flag is hardcoded.
        fairness_logger = FairnessLogger(
            X_val=X_val,
            s_val=s_val,
            include_sensitive_input=True,
        )

        return [early_stopping, reduce_lr, fairness_logger]


class FairTunerBuildFunctionFactory:
    """Create Keras Tuner build functions for Block 7.

    Args:
        input_dim: Number of processed input features.
        ratio_indices: Financial column positions resolved from feature names.
        config: Optional tuning configuration.

    Returns:
        Factory object whose `build` method is compatible with Keras Tuner.

    Raises:
        TuningError: If the search configuration is invalid.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        ratio_indices: FinancialRatioIndices,
        config: TuningConfig | None = None,
    ) -> None:
        """Initialize the build-function factory.

        Args:
            input_dim: Number of processed features.
            ratio_indices: Financial indices used by custom ratio layers.
            config: Optional tuning configuration.

        Returns:
            None.

        Raises:
            TuningError: If values are unsafe.
        """

        self._input_dim = int(input_dim)
        self._ratio_indices = ratio_indices
        self._config = config or TuningConfig()
        self._validate()

    def _validate(self) -> None:
        """Validate the tuner build-function configuration.

        Args:
            None.

        Returns:
            None.

        Raises:
            TuningError: If the search space is invalid.
        """

        if self._input_dim <= 0:
            raise TuningError("input_dim must be positive.")

        if self._config.min_layers <= 0:
            raise TuningError("min_layers must be positive.")

        if self._config.max_layers < self._config.min_layers:
            raise TuningError("max_layers must be >= min_layers.")

        if not self._config.units_choices:
            raise TuningError("units_choices cannot be empty.")

        if any(units <= 0 for units in self._config.units_choices):
            raise TuningError("units_choices must contain positive values.")

        if not self._config.activation_choices:
            raise TuningError("activation_choices cannot be empty.")

        if self._config.tuning_lambda_fair < 0:
            raise TuningError("tuning_lambda_fair must be non-negative.")

    def build(self, hp: kt.HyperParameters) -> tf.keras.Model:
        """Build one FAIR model for a Keras Tuner trial.

        Args:
            hp: Keras Tuner hyperparameter object.

        Returns:
            Compiled dual-input FAIR model.

        Raises:
            TuningError: If the sampled hyperparameters are invalid.
        """

        # Sample the number of layers first.
        n_layers = hp.Int(
            "n_layers",
            min_value=self._config.min_layers,
            max_value=self._config.max_layers,
        )

        # Define all possible unit hyperparameters every time. This avoids a
        # fragile conditional search space in BayesianOptimization.
        all_units = tuple(
            hp.Choice(f"units_{index}", list(self._config.units_choices))
            for index in range(self._config.max_layers)
        )

        # Only the first n_layers widths are active in this concrete trial.
        hidden_units = all_units[:n_layers]

        # Activation and dropout are shared across hidden layers for a compact
        # and defensible MVP search space.
        activation = hp.Choice(
            "activation",
            list(self._config.activation_choices),
        )

        dropout = hp.Float(
            "dropout",
            min_value=self._config.dropout_min,
            max_value=self._config.dropout_max,
            step=self._config.dropout_step,
        )

        # Learning rate is searched on a log scale, which is standard for Adam.
        learning_rate = hp.Float(
            "learning_rate",
            min_value=self._config.learning_rate_min,
            max_value=self._config.learning_rate_max,
            sampling="log",
        )

        # Build the predictive architecture config from sampled hyperparameters.
        custom_config = CustomMLPConfig(
            hidden_units=tuple(int(units) for units in hidden_units),
            activation=str(activation),
            dropout=float(dropout),
            learning_rate=float(learning_rate),
        )

        # The custom builder owns the financial backbone.
        custom_builder = CustomMLPModelBuilder(custom_config)

        # The FAIR builder adds only sensitive input and add_loss penalty.
        fair_builder = FairCustomModelBuilder(
            custom_builder=custom_builder,
            fair_config=FairModelConfig(lambda_fair=self._config.tuning_lambda_fair),
        )

        return fair_builder.build(
            input_dim=self._input_dim,
            ratio_indices=self._ratio_indices,
        ).model


class BestHyperparameterExtractor:
    """Convert Keras Tuner HyperParameters into `CustomMLPConfig`.

    Args:
        config: Optional tuning configuration defining max layer count.

    Returns:
        Extractor object.

    Raises:
        None during initialization.
    """

    def __init__(self, config: TuningConfig | None = None) -> None:
        """Initialize the extractor.

        Args:
            config: Optional tuning configuration.

        Returns:
            None.

        Raises:
            None.
        """

        self._config = config or TuningConfig()

    def extract(self, hp: kt.HyperParameters) -> CustomMLPConfig:
        """Build a CustomMLPConfig from best tuner hyperparameters.

        Args:
            hp: Best Keras Tuner HyperParameters object.

        Returns:
            CustomMLPConfig for the fixed architecture used in lambda sweep.

        Raises:
            TuningError: If required hyperparameters are missing.
        """

        try:
            n_layers = int(hp.get("n_layers"))
            activation = str(hp.get("activation"))
            dropout = float(hp.get("dropout"))
            learning_rate = float(hp.get("learning_rate"))
            hidden_units = tuple(
                int(hp.get(f"units_{index}")) for index in range(n_layers)
            )
        except Exception as exc:
            raise TuningError("Best hyperparameters are incomplete.") from exc

        return CustomMLPConfig(
            hidden_units=hidden_units,
            activation=activation,
            dropout=dropout,
            learning_rate=learning_rate,
        )


class FairKerasTunerFactory:
    """Create configured Keras Tuner instances.

    Args:
        config: Optional tuning configuration.
        artifacts: Optional artifact path configuration.

    Returns:
        Factory object able to create BayesianOptimization tuners.

    Raises:
        None during initialization.
    """

    def __init__(
        self,
        config: TuningConfig | None = None,
        artifacts: TuningArtifactPaths | None = None,
    ) -> None:
        """Initialize the tuner factory.

        Args:
            config: Optional tuning configuration.
            artifacts: Optional path configuration.

        Returns:
            None.

        Raises:
            None.
        """

        self._config = config or TuningConfig()
        self._artifacts = artifacts or TuningArtifactPaths()

    def create(self, build_fn: Any) -> kt.BayesianOptimization:
        """Create a BayesianOptimization tuner.

        Args:
            build_fn: Callable accepted by Keras Tuner.

        Returns:
            Configured Keras Tuner instance.

        Raises:
            None.
        """

        # Ensure the tuner directory exists before Keras Tuner writes metadata.
        self._artifacts.tuner_directory.mkdir(parents=True, exist_ok=True)

        return kt.BayesianOptimization(
            build_fn,
            objective=kt.Objective(
                self._config.tuner_objective,
                direction=self._config.tuner_direction,
            ),
            max_trials=self._config.max_trials,
            executions_per_trial=self._config.executions_per_trial,
            overwrite=self._config.overwrite_tuner,
            directory=str(self._artifacts.tuner_directory),
            project_name=self._config.tuner_project_name,
        )


class ValidationThresholdSelector:
    """Select a binary threshold from validation probabilities.

    Args:
        None.

    Returns:
        Threshold selector instance.

    Raises:
        None.
    """

    def choose_youden(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Choose threshold by Youden's J statistic.

        Args:
            y_true: Binary validation target.
            y_proba: Validation predicted probabilities.

        Returns:
            Threshold clipped to [0, 1].

        Raises:
            TuningError: If validation contains a single target class.
        """

        # Delegate to the Block 8 shared selector so Keras Tuner, final
        # evaluation and notebooks all use the exact same threshold logic.
        try:
            return ThresholdSelector().choose_youden(y_true, y_proba).threshold
        except MetricsError as exc:
            raise TuningError(
                "Cannot choose threshold with one validation class."
            ) from exc


class ValidationParetoEvaluator:
    """Evaluate one trained lambda model on validation data.

    Args:
        threshold_selector: Optional threshold selector.
        probability_calculator: Optional threshold-free metric calculator.
        binary_calculator: Optional threshold-dependent metric calculator.

    Returns:
        Evaluator object.

    Raises:
        None during initialization.
    """

    def __init__(
        self,
        threshold_selector: ValidationThresholdSelector | None = None,
        probability_calculator: ProbabilityMetricCalculator | None = None,
        binary_calculator: BinaryClassificationMetricCalculator | None = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            threshold_selector: Optional validation threshold selector.
            probability_calculator: Optional threshold-free metric calculator.
            binary_calculator: Optional threshold-dependent metric calculator.

        Returns:
            None.

        Raises:
            None.
        """

        self._threshold_selector = threshold_selector or ValidationThresholdSelector()
        self._probability_calculator = (
            probability_calculator or ProbabilityMetricCalculator()
        )
        self._binary_calculator = (
            binary_calculator or BinaryClassificationMetricCalculator()
        )
        self._formatter = DualInputFormatter()

    def evaluate(
        self,
        *,
        model: tf.keras.Model,
        data: ProcessedSplitDataset,
        lambda_fair: float,
        history: tf.keras.callbacks.History,
        model_path: str,
        history_path: str,
    ) -> ParetoResultRow:
        """Evaluate a trained model and produce one Pareto row.

        Args:
            model: Trained dual-input FAIR model.
            data: Processed split dataset.
            lambda_fair: Lambda value used by the model.
            history: Keras training history.
            model_path: Project-relative model path.
            history_path: Project-relative history path.

        Returns:
            ParetoResultRow with validation metrics.

        Raises:
            TuningError: If validation arrays are inconsistent.
        """

        # Predict validation probabilities with both model inputs.
        probabilities = model.predict(
            self._formatter.format(data.X_val, data.s_val),
            verbose=0,
            batch_size=1024,
        ).reshape(-1)

        # Validate row alignment before computing any metric.
        if not (len(probabilities) == len(data.y_val) == len(data.s_val)):
            raise TuningError("Validation predictions, y and sensitive differ.")

        # Probability metrics are threshold-free and now come from Block 8.
        probability_metrics = self._probability_calculator.calculate(
            data.y_val,
            probabilities,
            data.s_val,
        )

        # Binary metrics require a validation-selected threshold.
        threshold = self._threshold_selector.choose_youden(data.y_val, probabilities)
        binary_metrics = self._binary_calculator.calculate(
            data.y_val,
            probabilities,
            threshold,
        )

        return ParetoResultRow(
            lambda_fair=float(lambda_fair),
            val_auc=probability_metrics.roc_auc,
            val_pr_auc=probability_metrics.pr_auc,
            val_abs_rho=probability_metrics.abs_rho,
            val_threshold=threshold,
            val_accuracy=binary_metrics.accuracy,
            val_precision=binary_metrics.precision,
            val_recall=binary_metrics.recall,
            val_f1=binary_metrics.f1,
            epochs_trained=self._epochs_trained(history),
            model_path=model_path,
            history_path=history_path,
            selected_for_test=False,
        )

    def _epochs_trained(self, history: tf.keras.callbacks.History) -> int:
        """Return number of epochs represented in a Keras history.

        Args:
            history: Keras History object.

        Returns:
            Epoch count.

        Raises:
            None.
        """

        # Any metric list has one value per epoch. If history is empty, return 0
        # instead of failing while assembling diagnostics.
        if not history.history:
            return 0

        first_metric = next(iter(history.history.values()))
        return int(len(first_metric))

class ParetoModelSelector:
    """Mark base and validation-selected FAIR models for Block 11.

    Args:
        max_auc_drop: Maximum validation AUC loss allowed when choosing the
            lowest-correlation FAIR model.

    Returns:
        Selector object.

    Raises:
        None during initialization.
    """

    def __init__(self, max_auc_drop: float = 0.02) -> None:
        """Initialize the selector.

        Args:
            max_auc_drop: Allowed AUC drop relative to lambda=0.

        Returns:
            None.

        Raises:
            TuningError: If max_auc_drop is negative.
        """

        if max_auc_drop < 0:
            raise TuningError("max_auc_drop must be non-negative.")

        self._max_auc_drop = float(max_auc_drop)

    def select(self, rows: Sequence[ParetoResultRow]) -> tuple[ParetoResultRow, ...]:
        """Return rows with `selected_for_test` flags applied.

        Args:
            rows: Pareto rows from the lambda sweep.

        Returns:
            Tuple of rows with lambda=0 and one FAIR candidate selected.

        Raises:
            TuningError: If no rows are provided or lambda=0 is absent.
        """

        if not rows:
            raise TuningError("Cannot select from an empty Pareto row list.")

        # The controlled base final must be present in every valid sweep.
        base_rows = [row for row in rows if float(row.lambda_fair) == 0.0]
        if not base_rows:
            raise TuningError("Lambda sweep must include lambda_fair=0.0.")

        base = base_rows[0]

        # Nonzero lambdas are FAIR candidates.
        fair_candidates = [row for row in rows if float(row.lambda_fair) != 0.0]

        # If no FAIR candidate exists, only the base can be selected.
        if not fair_candidates:
            selected_lambdas = {0.0}
            return self._mark_selected(rows, selected_lambdas)

        # Prefer models that improve abs_rho without losing too much AUC.
        auc_floor = base.val_auc - self._max_auc_drop
        viable = [
            row
            for row in fair_candidates
            if np.isnan(base.val_auc) or np.isnan(row.val_auc) or row.val_auc >= auc_floor
        ]

        # In normal runs, choose the viable candidate with lowest abs_rho. If
        # all candidates fail the AUC floor, choose the highest-AUC FAIR model.
        pool = viable or fair_candidates
        chosen_fair = min(pool, key=lambda row: (row.val_abs_rho, -row.val_auc))

        selected_lambdas = {0.0, float(chosen_fair.lambda_fair)}
        return self._mark_selected(rows, selected_lambdas)

    def _mark_selected(
        self,
        rows: Sequence[ParetoResultRow],
        selected_lambdas: set[float],
    ) -> tuple[ParetoResultRow, ...]:
        """Copy rows while setting selected flags.

        Args:
            rows: Original rows.
            selected_lambdas: Lambda values to mark for test evaluation.

        Returns:
            New tuple of ParetoResultRow objects.

        Raises:
            None.
        """

        return tuple(
            ParetoResultRow(
                **{
                    **row.to_dict(),
                    "selected_for_test": float(row.lambda_fair) in selected_lambdas,
                }
            )
            for row in rows
        )


class TrainingArtifactWriter:
    """Write histories, models and Pareto results to disk.

    Args:
        artifacts: Optional Block 7 artifact paths.

    Returns:
        Writer object.

    Raises:
        None during initialization.
    """

    def __init__(self, artifacts: TuningArtifactPaths | None = None) -> None:
        """Initialize the writer.

        Args:
            artifacts: Optional artifact path configuration.

        Returns:
            None.

        Raises:
            None.
        """

        self._artifacts = artifacts or TuningArtifactPaths()

    def save_history(
        self,
        history: tf.keras.callbacks.History,
        path: Path,
    ) -> Path:
        """Save one Keras history to CSV.

        Args:
            history: Keras History object.
            path: Destination CSV path.

        Returns:
            Written path.

        Raises:
            OSError: If the CSV cannot be written.
        """

        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history.history).to_csv(path, index=False)
        return path

    def save_model(self, model: tf.keras.Model, path: Path) -> Path:
        """Save one Keras model.

        Args:
            model: Trained model.
            path: Destination `.keras` path.

        Returns:
            Written model path.

        Raises:
            OSError: If the model cannot be written.
        """

        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        return path

    def save_pareto(self, rows: Sequence[ParetoResultRow]) -> Path:
        """Save Pareto rows to the configured CSV.

        Args:
            rows: Rows to write.

        Returns:
            Written Pareto CSV path.

        Raises:
            OSError: If the CSV cannot be written.
        """

        self._artifacts.pareto_results_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row.to_dict() for row in rows]).to_csv(
            self._artifacts.pareto_results_csv,
            index=False,
        )
        return self._artifacts.pareto_results_csv


class FairKerasTunerRunner:
    """Run Keras Tuner search for the FAIR architecture.

    Args:
        config: Optional tuning configuration.
        artifacts: Optional artifact paths.
        callback_factory: Optional callback factory.
        class_weight_calculator: Optional class-weight calculator.
        array_validator: Optional processed-array validator.
        reproducibility_manager: Optional reproducibility manager.

    Returns:
        Runner object.

    Raises:
        None during initialization.
    """

    def __init__(
        self,
        config: TuningConfig | None = None,
        artifacts: TuningArtifactPaths | None = None,
        callback_factory: FairTuningCallbackFactory | None = None,
        class_weight_calculator: ClassWeightCalculator | None = None,
        array_validator: TrainingArrayValidator | None = None,
        reproducibility_manager: ReproducibilityManager | None = None,
    ) -> None:
        """Initialize the tuner runner.

        Args:
            config: Optional Block 7 config.
            artifacts: Optional artifact paths.
            callback_factory: Optional callback factory.
            class_weight_calculator: Optional class-weight calculator.
            array_validator: Optional array validator.
            reproducibility_manager: Optional reproducibility manager.

        Returns:
            None.

        Raises:
            None.
        """

        self._config = config or TuningConfig()
        self._artifacts = artifacts or TuningArtifactPaths()
        self._callback_factory = callback_factory or FairTuningCallbackFactory(
            self._config
        )
        self._class_weight_calculator = (
            class_weight_calculator or ClassWeightCalculator()
        )
        self._array_validator = array_validator or TrainingArrayValidator()
        self._reproducibility_manager = (
            reproducibility_manager or ReproducibilityManager()
        )
        self._formatter = DualInputFormatter()

    def search(
        self,
        data: ProcessedSplitDataset,
        *,
        verbose: int = 1,
    ) -> TunerSearchResult:
        """Run Keras Tuner search on processed train/validation data.

        Args:
            data: Processed split dataset from Block 2.
            verbose: Verbosity passed to `tuner.search`.

        Returns:
            TunerSearchResult with best hyperparameters and config.

        Raises:
            TuningError: If tuner does not return best hyperparameters.
        """

        # Reapply seeds before building tuner models.
        self._reproducibility_manager.apply()

        # Reuse the established Block 4 array validator.
        self._array_validator.validate(data)

        # Class weights are computed once from training labels.
        class_weight = self._class_weight_calculator.compute(data.y_train)

        # Resolve ratio indices from the processed feature names.
        default_builder = CustomMLPModelBuilder()
        ratio_indices = default_builder.index_resolver.resolve(data.feature_names)

        # Build the tuner-compatible model factory.
        build_factory = FairTunerBuildFunctionFactory(
            input_dim=data.X_train.shape[1],
            ratio_indices=ratio_indices,
            config=self._config,
        )

        # Create Keras Tuner around the build function.
        tuner = FairKerasTunerFactory(
            config=self._config,
            artifacts=self._artifacts,
        ).create(build_factory.build)

        # Callbacks are fresh for the tuner search.
        callbacks = self._callback_factory.build(X_val=data.X_val, s_val=data.s_val)

        # Launch the actual search.
        tuner.search(
            self._formatter.format(data.X_train, data.s_train),
            data.y_train,
            validation_data=(
                self._formatter.format(data.X_val, data.s_val),
                data.y_val,
            ),
            class_weight=class_weight,
            epochs=self._config.epochs,
            batch_size=self._config.batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

        best = tuner.get_best_hyperparameters(num_trials=1)
        if not best:
            raise TuningError("Keras Tuner did not return best hyperparameters.")

        best_hp = best[0]
        best_config = BestHyperparameterExtractor(self._config).extract(best_hp)

        return TunerSearchResult(
            tuner=tuner,
            best_hyperparameters=best_hp,
            best_config=best_config,
            ratio_indices=ratio_indices,
            class_weight=class_weight,
        )


class FairLambdaSweepTrainer:
    """Train the fixed architecture across a lambda grid.

    Args:
        config: Optional tuning configuration.
        artifacts: Optional artifact paths.
        callback_factory: Optional callback factory.
        evaluator: Optional validation evaluator.
        selector: Optional Pareto model selector.
        writer: Optional artifact writer.
        class_weight_calculator: Optional class-weight calculator.
        array_validator: Optional processed-array validator.
        reproducibility_manager: Optional reproducibility manager.

    Returns:
        Trainer object.

    Raises:
        None during initialization.
    """

    def __init__(
        self,
        config: TuningConfig | None = None,
        artifacts: TuningArtifactPaths | None = None,
        callback_factory: FairTuningCallbackFactory | None = None,
        evaluator: ValidationParetoEvaluator | None = None,
        selector: ParetoModelSelector | None = None,
        writer: TrainingArtifactWriter | None = None,
        class_weight_calculator: ClassWeightCalculator | None = None,
        array_validator: TrainingArrayValidator | None = None,
        reproducibility_manager: ReproducibilityManager | None = None,
    ) -> None:
        """Initialize the lambda sweep trainer.

        Args:
            config: Optional Block 7 config.
            artifacts: Optional artifact path config.
            callback_factory: Optional callback factory.
            evaluator: Optional validation evaluator.
            selector: Optional model selector.
            writer: Optional artifact writer.
            class_weight_calculator: Optional class-weight calculator.
            array_validator: Optional array validator.
            reproducibility_manager: Optional reproducibility manager.

        Returns:
            None.

        Raises:
            None.
        """

        self._config = config or TuningConfig()
        self._artifacts = artifacts or TuningArtifactPaths()
        self._callback_factory = callback_factory or FairTuningCallbackFactory(
            self._config
        )
        self._evaluator = evaluator or ValidationParetoEvaluator()
        self._selector = selector or ParetoModelSelector(
            self._config.fair_selection_max_auc_drop
        )
        self._writer = writer or TrainingArtifactWriter(self._artifacts)
        self._class_weight_calculator = (
            class_weight_calculator or ClassWeightCalculator()
        )
        self._array_validator = array_validator or TrainingArrayValidator()
        self._reproducibility_manager = (
            reproducibility_manager or ReproducibilityManager()
        )
        self._formatter = DualInputFormatter()

    def run(
        self,
        *,
        data: ProcessedSplitDataset,
        custom_config: CustomMLPConfig,
        ratio_indices: FinancialRatioIndices | None = None,
        class_weight: dict[int, float] | None = None,
        save_models: bool = True,
        verbose: int = 1,
    ) -> LambdaSweepResult:
        """Run the manual lambda sweep.

        Args:
            data: Processed split dataset.
            custom_config: Fixed architecture config, usually from tuner.
            ratio_indices: Optional pre-resolved financial indices.
            class_weight: Optional precomputed class weights.
            save_models: Whether to save `.keras` models.
            verbose: Verbosity passed to Keras `fit`.

        Returns:
            LambdaSweepResult with selected Pareto rows and CSV path.

        Raises:
            TuningError: If lambda grid is empty.
        """

        if not self._config.lambda_values:
            raise TuningError("lambda_values cannot be empty.")

        # Reset seeds before the controlled sweep.
        self._reproducibility_manager.apply()

        # Validate processed arrays once before multiple fits.
        self._array_validator.validate(data)

        # Compute class weights on train unless the tuner result already
        # provided them.
        weights = class_weight or self._class_weight_calculator.compute(data.y_train)

        # Resolve ratio indices once so every lambda uses identical financial
        # feature positions.
        resolved_indices = ratio_indices or CustomMLPModelBuilder(
            custom_config
        ).index_resolver.resolve(data.feature_names)

        rows: list[ParetoResultRow] = []

        # Iterate in configured order; this makes CSV rows deterministic.
        for lambda_value in self._config.lambda_values:
            row = self._train_one_lambda(
                data=data,
                custom_config=custom_config,
                ratio_indices=resolved_indices,
                lambda_fair=float(lambda_value),
                class_weight=weights,
                save_model=save_models,
                verbose=verbose,
            )
            rows.append(row)

        # Apply validation-only selection flags after all rows are known.
        selected_rows = self._selector.select(rows)

        # Save final Pareto table consumed by Blocks 11 and 12.
        pareto_csv = self._writer.save_pareto(selected_rows)

        return LambdaSweepResult(
            rows=selected_rows,
            pareto_csv=pareto_csv,
            class_weight=weights,
        )

    def _train_one_lambda(
        self,
        *,
        data: ProcessedSplitDataset,
        custom_config: CustomMLPConfig,
        ratio_indices: FinancialRatioIndices,
        lambda_fair: float,
        class_weight: dict[int, float],
        save_model: bool,
        verbose: int,
    ) -> ParetoResultRow:
        """Train and evaluate one lambda model.

        Args:
            data: Processed split dataset.
            custom_config: Fixed architecture config.
            ratio_indices: Shared financial indices.
            lambda_fair: Lambda value for this run.
            class_weight: Class weights used in Keras fit.
            save_model: Whether to save the model file.
            verbose: Keras fit verbosity.

        Returns:
            ParetoResultRow before selection flags are applied.

        Raises:
            OSError: If artifact writing fails.
        """

        # Build a fresh model for this lambda. Reusing model objects would leak
        # weights between lambda experiments.
        custom_builder = CustomMLPModelBuilder(custom_config)
        fair_builder = FairCustomModelBuilder(
            custom_builder=custom_builder,
            fair_config=FairModelConfig(lambda_fair=lambda_fair),
        )
        build_result = fair_builder.build(
            input_dim=data.X_train.shape[1],
            ratio_indices=ratio_indices,
        )
        model = build_result.model

        # Fresh callbacks are required for every Keras fit.
        callbacks = self._callback_factory.build(X_val=data.X_val, s_val=data.s_val)

        # Fit on train, monitor on validation, never touch test.
        history = model.fit(
            self._formatter.format(data.X_train, data.s_train),
            data.y_train,
            validation_data=(
                self._formatter.format(data.X_val, data.s_val),
                data.y_val,
            ),
            class_weight=class_weight,
            epochs=self._config.epochs,
            batch_size=self._config.batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

        # Compute absolute artifact paths from the fixed naming convention.
        model_path_abs = self._artifacts.model_path(lambda_fair)
        history_path_abs = self._artifacts.history_path(lambda_fair)

        # History is always saved because Block 12 needs loss curves.
        self._writer.save_history(history, history_path_abs)

        # Model saving can be disabled in tests to keep them fast and light.
        if save_model:
            self._writer.save_model(model, model_path_abs)

        # CSV rows store relative paths for portability.
        model_path = self._artifacts.to_project_relative(model_path_abs)
        history_path = self._artifacts.to_project_relative(history_path_abs)

        return self._evaluator.evaluate(
            model=model,
            data=data,
            lambda_fair=lambda_fair,
            history=history,
            model_path=model_path,
            history_path=history_path,
        )


__all__ = [
    "BestHyperparameterExtractor",
    "DualInputFormatter",
    "FairKerasTunerFactory",
    "FairKerasTunerRunner",
    "FairLambdaSweepTrainer",
    "FairTunerBuildFunctionFactory",
    "FairTuningCallbackFactory",
    "LambdaSweepResult",
    "ParetoModelSelector",
    "ParetoResultRow",
    "TunerSearchResult",
    "TuningArtifactPaths",
    "TuningConfig",
    "TuningError",
    "TrainingArtifactWriter",
    "ValidationParetoEvaluator",
    "ValidationThresholdSelector",
]
