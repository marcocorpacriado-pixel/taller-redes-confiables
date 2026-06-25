"""Squared FAIR-loss sensitivity experiments for the extras notebook.

The final MVP already uses the FAIR regularizer implemented by
``FairnessPenalty``. That layer adds a squared batch-correlation penalty,
``lambda_fair * rho^2``. This module wraps that model family in a small,
isolated runner so the extras notebook can train a controlled quadratic sweep
without touching the canonical MVP artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import tensorflow as tf

from .base_model import (
    ClassWeightCalculator,
    ReproducibilityConfig,
    ReproducibilityManager,
    TrainingArrayValidator,
)
from .metrics import (
    BinaryClassificationMetricCalculator,
    FairnessMetricCalculator,
    ProbabilityMetricCalculator,
    ThresholdSelector,
)
from .models import (
    CustomMLPConfig,
    CustomMLPModelBuilder,
    FairCustomModelBuilder,
    FairModelConfig,
    lambda_slug,
)
from .preprocessing import HomeCreditMVPPreprocessingPipeline, ProcessedSplitDataset
from .splitting import HomeCreditTrainValTestSplitter, SplitConfig


class FairnessLossExperimentError(ValueError):
    """Raised when a squared FAIR-loss experiment cannot be completed."""


@dataclass(frozen=True)
class SquaredFairnessTrainingConfig:
    """Configuration for the quadratic FAIR sensitivity sweep."""

    alpha_values: tuple[float, ...] = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5)
    seed: int = 42
    epochs: int = 40
    batch_size: int = 1024
    early_stopping_patience: int = 8
    reduce_lr_patience: int = 4
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-6
    hidden_units: tuple[int, ...] = (128, 64)
    activation: str = "elu"
    dropout: float = 0.2
    learning_rate: float = 1e-3
    max_auc_drop_for_selection: float = 0.02
    collapse_auc_threshold: float = 0.55
    collapse_threshold_high: float = 0.99
    collapse_prediction_std_threshold: float = 1e-4
    save_models: bool = True

    def __post_init__(self) -> None:
        alphas = tuple(float(value) for value in self.alpha_values)
        object.__setattr__(self, "alpha_values", alphas)

        if not alphas:
            raise FairnessLossExperimentError("alpha_values cannot be empty.")
        if any(value < 0.0 for value in alphas):
            raise FairnessLossExperimentError("alpha_values must be non-negative.")
        if 0.0 not in set(alphas):
            raise FairnessLossExperimentError(
                "alpha_values must include 0.0 as the quadratic base control."
            )
        if self.epochs <= 0:
            raise FairnessLossExperimentError("epochs must be positive.")
        if self.batch_size <= 0:
            raise FairnessLossExperimentError("batch_size must be positive.")
        if self.early_stopping_patience <= 0:
            raise FairnessLossExperimentError(
                "early_stopping_patience must be positive."
            )
        if self.reduce_lr_patience <= 0:
            raise FairnessLossExperimentError("reduce_lr_patience must be positive.")
        if not self.hidden_units:
            raise FairnessLossExperimentError("hidden_units cannot be empty.")
        if any(units <= 0 for units in self.hidden_units):
            raise FairnessLossExperimentError("hidden_units must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise FairnessLossExperimentError("dropout must be in [0, 1).")
        if self.learning_rate <= 0.0:
            raise FairnessLossExperimentError("learning_rate must be positive.")
        if self.max_auc_drop_for_selection < 0.0:
            raise FairnessLossExperimentError(
                "max_auc_drop_for_selection must be non-negative."
            )

    def custom_model_config(self) -> CustomMLPConfig:
        """Return the fixed custom architecture used by the sweep."""

        return CustomMLPConfig(
            hidden_units=self.hidden_units,
            activation=self.activation,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
        )


@dataclass(frozen=True)
class SquaredFairnessExperimentPaths:
    """Paths written by the quadratic FAIR sensitivity block."""

    sweep_csv: Path
    comparison_csv: Path
    split_report_csv: Path
    histories_dir: Path
    models_dir: Path
    predictions_dir: Path


@dataclass(frozen=True)
class SquaredFairnessRunResult:
    """Result returned by ``SquaredFairnessComparisonRunner.run``."""

    sweep: pd.DataFrame
    comparison: pd.DataFrame
    paths: SquaredFairnessExperimentPaths


class DualInputArrayFormatter:
    """Format arrays for Keras models with features and sensitive inputs."""

    def format(self, X: np.ndarray, sensitive: np.ndarray) -> dict[str, np.ndarray]:
        sensitive_column = np.asarray(sensitive, dtype="float32").reshape(-1, 1)
        if X.shape[0] != sensitive_column.shape[0]:
            raise FairnessLossExperimentError(
                "X and sensitive arrays are not aligned."
            )
        return {
            "features": np.asarray(X, dtype="float32"),
            "sensitive": sensitive_column,
        }


class SquaredFairnessResultInterpreter:
    """Classify and explain quadratic FAIR-loss outcomes."""

    def classify(
        self,
        *,
        alpha: float,
        test_auc: float,
        threshold: float,
        prediction_std: float,
        config: SquaredFairnessTrainingConfig,
    ) -> str:
        """Return a compact status label for one trained model."""

        if float(alpha) == 0.0:
            return "base_cuadratica"

        if (
            np.isfinite(test_auc)
            and test_auc <= config.collapse_auc_threshold
        ) or threshold >= config.collapse_threshold_high:
            return "colapsado"

        if prediction_std <= config.collapse_prediction_std_threshold:
            return "colapsado"

        return "fair_cuadratico"

    def reading(self, *, alpha: float, status: str) -> str:
        """Return a human-readable interpretation for one status."""

        if float(alpha) == 0.0:
            return (
                "Control sin penalizacion FAIR; comprueba que el runner "
                "cuadratico conserva senal predictiva antes de imponer justicia."
            )
        if status == "colapsado":
            return (
                "La penalizacion cuadratica domina el aprendizaje: reduce "
                "dependencia, pero no conserva senal predictiva suficiente."
            )
        return (
            "Penalizacion cuadratica viable; se interpreta por perdida de AUC "
            "frente a la base y mejora de metricas de fairness."
        )


class SquaredFairnessComparisonRunner:
    """Train and evaluate the quadratic FAIR-loss sensitivity sweep."""

    def __init__(
        self,
        *,
        config: SquaredFairnessTrainingConfig | None = None,
        artifacts: Any,
        class_weight_calculator: ClassWeightCalculator | None = None,
        array_validator: TrainingArrayValidator | None = None,
        reproducibility_manager: ReproducibilityManager | None = None,
        interpreter: SquaredFairnessResultInterpreter | None = None,
    ) -> None:
        self.config = config or SquaredFairnessTrainingConfig()
        self.artifacts = artifacts
        self.class_weight_calculator = class_weight_calculator or ClassWeightCalculator()
        self.array_validator = array_validator or TrainingArrayValidator()
        self.reproducibility_manager = reproducibility_manager or ReproducibilityManager(
            ReproducibilityConfig(seed=self.config.seed)
        )
        self.interpreter = interpreter or SquaredFairnessResultInterpreter()
        self.formatter = DualInputArrayFormatter()
        self.probability_calculator = ProbabilityMetricCalculator()
        self.binary_calculator = BinaryClassificationMetricCalculator()
        self.fairness_calculator = FairnessMetricCalculator()
        self.threshold_selector = ThresholdSelector()

    def run(
        self,
        *,
        application_train_path: Path,
        processed: ProcessedSplitDataset | None = None,
        mvp_results_path: Path | None = None,
        verbose: int = 1,
    ) -> SquaredFairnessRunResult:
        """Train the quadratic sweep and write all block artifacts."""

        self._validate_artifact_contract()
        paths = self.experiment_paths()
        self._create_directories(paths)

        data = processed or self.prepare_data(Path(application_train_path), paths)
        self.array_validator.validate(data)

        self.reproducibility_manager.apply()
        class_weight = self.class_weight_calculator.compute(data.y_train)
        custom_config = self.config.custom_model_config()
        ratio_indices = CustomMLPModelBuilder(
            custom_config
        ).index_resolver.resolve(data.feature_names)

        rows: list[dict[str, Any]] = []
        for alpha in self.config.alpha_values:
            row = self._train_one_alpha(
                data=data,
                custom_config=custom_config,
                ratio_indices=ratio_indices,
                alpha=float(alpha),
                class_weight=class_weight,
                paths=paths,
                verbose=verbose,
            )
            rows.append(row)

        sweep = pd.DataFrame(rows)
        sweep = self._mark_selected_rows(sweep)
        sweep.to_csv(paths.sweep_csv, index=False)

        comparison = self._build_comparison_table(
            sweep=sweep,
            mvp_results_path=mvp_results_path,
        )
        comparison.to_csv(paths.comparison_csv, index=False)

        return SquaredFairnessRunResult(
            sweep=sweep,
            comparison=comparison,
            paths=paths,
        )

    def prepare_data(
        self,
        application_train_path: Path,
        paths: SquaredFairnessExperimentPaths | None = None,
    ) -> ProcessedSplitDataset:
        """Prepare the same leakage-safe MVP data used by the final notebook."""

        pipeline = HomeCreditMVPPreprocessingPipeline()
        raw = pipeline.load_raw(application_train_path)
        deterministic = pipeline.apply_deterministic_transforms(raw)
        splitter = HomeCreditTrainValTestSplitter(
            config=SplitConfig(random_state=self.config.seed)
        )
        split = splitter.split(deterministic)
        processed = pipeline.fit_transform_splits(split.raw_splits)

        if paths is not None:
            split.report.to_csv(paths.split_report_csv, index=False)

        return processed

    def experiment_paths(self) -> SquaredFairnessExperimentPaths:
        """Return run-local paths for the quadratic FAIR block."""

        run_dir = Path(self.artifacts.run_dir)
        tables_dir = Path(self.artifacts.tables_dir)
        predictions_dir = Path(self.artifacts.predictions_dir) / "fairness_squared"
        return SquaredFairnessExperimentPaths(
            sweep_csv=tables_dir / "fairness_squared_sweep.csv",
            comparison_csv=tables_dir / "fairness_loss_comparison.csv",
            split_report_csv=tables_dir / "fairness_squared_split_report.csv",
            histories_dir=tables_dir / "fairness_squared_histories",
            models_dir=run_dir / "models" / "fairness_squared",
            predictions_dir=predictions_dir,
        )

    def _train_one_alpha(
        self,
        *,
        data: ProcessedSplitDataset,
        custom_config: CustomMLPConfig,
        ratio_indices: Any,
        alpha: float,
        class_weight: Mapping[int, float],
        paths: SquaredFairnessExperimentPaths,
        verbose: int,
    ) -> dict[str, Any]:
        """Train, evaluate and persist one alpha value."""

        tf.keras.backend.clear_session()
        ReproducibilityManager(ReproducibilityConfig(seed=self.config.seed)).apply()

        custom_builder = CustomMLPModelBuilder(custom_config)
        fair_builder = FairCustomModelBuilder(
            custom_builder=custom_builder,
            fair_config=FairModelConfig(
                lambda_fair=alpha,
                model_name_prefix="squared_fair_alpha",
                fairness_layer_name="squared_fairness_penalty",
            ),
        )
        build = fair_builder.build(
            input_dim=data.X_train.shape[1],
            ratio_indices=ratio_indices,
        )
        model = build.model
        callbacks = self._callbacks()

        history = model.fit(
            self.formatter.format(data.X_train, data.s_train),
            data.y_train,
            validation_data=(
                self.formatter.format(data.X_val, data.s_val),
                data.y_val,
            ),
            class_weight=dict(class_weight),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

        history_path = paths.histories_dir / f"history_squared_alpha_{lambda_slug(alpha)}.csv"
        pd.DataFrame(history.history).to_csv(history_path, index=False)

        model_path = paths.models_dir / f"squared_fair_alpha_{lambda_slug(alpha)}.keras"
        if self.config.save_models:
            model.save(model_path)
            model_path_text = self._to_project_relative(model_path)
        else:
            model_path_text = ""

        val_proba = self._predict(model, data.X_val, data.s_val)
        test_proba = self._predict(model, data.X_test, data.s_test)
        prediction_path = (
            paths.predictions_dir / f"test_predictions_squared_alpha_{lambda_slug(alpha)}.csv"
        )
        self._save_predictions(
            path=prediction_path,
            ids=data.test_ids,
            y_true=data.y_test,
            sensitive=data.s_test,
            y_proba=test_proba,
        )

        threshold = self.threshold_selector.choose_youden(
            data.y_val,
            val_proba,
        ).threshold

        val_probability = self.probability_calculator.calculate(
            data.y_val,
            val_proba,
            data.s_val,
        )
        val_binary = self.binary_calculator.calculate(
            data.y_val,
            val_proba,
            threshold,
        )
        val_fairness = self.fairness_calculator.calculate(
            data.y_val,
            val_proba,
            data.s_val,
            threshold,
        )

        test_probability = self.probability_calculator.calculate(
            data.y_test,
            test_proba,
            data.s_test,
        )
        test_binary = self.binary_calculator.calculate(
            data.y_test,
            test_proba,
            threshold,
        )
        test_fairness = self.fairness_calculator.calculate(
            data.y_test,
            test_proba,
            data.s_test,
            threshold,
        )

        prediction_std = float(np.std(test_proba))
        status = self.interpreter.classify(
            alpha=alpha,
            test_auc=test_probability.roc_auc,
            threshold=threshold,
            prediction_std=prediction_std,
            config=self.config,
        )

        return {
            "modelo": self._model_label(alpha),
            "alpha": float(alpha),
            "status": status,
            "selected_for_comparison": False,
            "epochs_trained": self._epochs_trained(history),
            "threshold": float(threshold),
            "prediction_std": prediction_std,
            "val_auc": val_probability.roc_auc,
            "val_pr_auc": val_probability.pr_auc,
            "val_abs_rho": val_probability.abs_rho,
            "val_dpd": val_fairness.demographic_parity_difference,
            "val_eod": val_fairness.equalized_odds_difference,
            "val_accuracy": val_binary.accuracy,
            "val_precision": val_binary.precision,
            "val_recall": val_binary.recall,
            "val_f1": val_binary.f1,
            "test_auc": test_probability.roc_auc,
            "test_pr_auc": test_probability.pr_auc,
            "test_abs_rho": test_probability.abs_rho,
            "test_dpd": test_fairness.demographic_parity_difference,
            "test_eod": test_fairness.equalized_odds_difference,
            "test_accuracy": test_binary.accuracy,
            "test_precision": test_binary.precision,
            "test_recall": test_binary.recall,
            "test_f1": test_binary.f1,
            "model_path": model_path_text,
            "history_path": self._to_project_relative(history_path),
            "prediction_path": self._to_project_relative(prediction_path),
            "lectura": self.interpreter.reading(alpha=alpha, status=status),
        }

    def _mark_selected_rows(self, sweep: pd.DataFrame) -> pd.DataFrame:
        """Mark alpha=0 and one validation-selected FAIR candidate."""

        result = sweep.copy()
        result["selected_for_comparison"] = False

        base_rows = result[np.isclose(result["alpha"].astype(float), 0.0)]
        if base_rows.empty:
            raise FairnessLossExperimentError("The sweep must contain alpha=0.0.")
        base_index = base_rows.index[0]
        result.loc[base_index, "selected_for_comparison"] = True
        base_val_auc = float(result.loc[base_index, "val_auc"])

        candidates = result[~np.isclose(result["alpha"].astype(float), 0.0)].copy()
        if candidates.empty:
            return result

        non_collapsed = candidates[candidates["status"] != "colapsado"]
        pool = non_collapsed if not non_collapsed.empty else candidates

        auc_floor = base_val_auc - self.config.max_auc_drop_for_selection
        viable = pool[pool["val_auc"].astype(float) >= auc_floor]
        selected_pool = viable if not viable.empty else pool
        selected = selected_pool.sort_values(
            ["val_abs_rho", "val_auc"],
            ascending=[True, False],
        ).iloc[0]
        result.loc[selected.name, "selected_for_comparison"] = True
        return result

    def _build_comparison_table(
        self,
        *,
        sweep: pd.DataFrame,
        mvp_results_path: Path | None,
    ) -> pd.DataFrame:
        """Build Base/Fair MVP versus quadratic Base/Fair comparison."""

        rows: list[dict[str, Any]] = []
        mvp = self._read_mvp_results(mvp_results_path)

        if not mvp.empty:
            for label, experiment, objective in (
                ("Base", "MVP Base 12 features", "referencia predictiva principal"),
                ("FAIR", "MVP FAIR principal", "solucion confiable del MVP"),
            ):
                match = mvp[mvp["modelo"].str.contains(label, case=False, na=False)]
                if match.empty:
                    continue
                row = match.iloc[0]
                rows.append(
                    {
                        "experimento": experiment,
                        "familia_modelo": "red neuronal MVP",
                        "alpha": np.nan,
                        "auc": row["auc"],
                        "pr_auc": row["pr_auc"],
                        "accuracy": row["accuracy"],
                        "precision": row["precision"],
                        "recall": row["recall"],
                        "f1": row["f1"],
                        "abs_rho": row["abs_rho"],
                        "dpd": row["dpd"],
                        "eod": row["eod"],
                        "threshold": row["threshold"],
                        "status": "referencia_mvp",
                        "objetivo": objective,
                        "lectura": "Resultado validado del notebook MVP.",
                    }
                )

        selected = sweep[sweep["selected_for_comparison"].astype(bool)].copy()
        selected = selected.sort_values("alpha")
        for _, row in selected.iterrows():
            rows.append(
                {
                    "experimento": self._comparison_label(float(row["alpha"])),
                    "familia_modelo": "red neuronal con penalizacion cuadratica",
                    "alpha": row["alpha"],
                    "auc": row["test_auc"],
                    "pr_auc": row["test_pr_auc"],
                    "accuracy": row["test_accuracy"],
                    "precision": row["test_precision"],
                    "recall": row["test_recall"],
                    "f1": row["test_f1"],
                    "abs_rho": row["test_abs_rho"],
                    "dpd": row["test_dpd"],
                    "eod": row["test_eod"],
                    "threshold": row["threshold"],
                    "status": row["status"],
                    "objetivo": (
                        "control del runner cuadratico"
                        if float(row["alpha"]) == 0.0
                        else "sensibilidad de fairness cuadratica"
                    ),
                    "lectura": row["lectura"],
                }
            )

        comparison = pd.DataFrame(rows)
        if comparison.empty:
            return comparison

        reference = self._reference_row(comparison)
        for metric in ("auc", "abs_rho", "dpd"):
            comparison[metric] = pd.to_numeric(comparison[metric], errors="coerce")

        comparison["perdida_auc_vs_base"] = reference["auc"] - comparison["auc"]
        comparison["mejora_abs_rho_vs_base"] = (
            reference["abs_rho"] - comparison["abs_rho"]
        )
        comparison["mejora_dpd_vs_base"] = reference["dpd"] - comparison["dpd"]
        return comparison

    def _callbacks(self) -> list[tf.keras.callbacks.Callback]:
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                mode="min",
                factor=self.config.reduce_lr_factor,
                patience=self.config.reduce_lr_patience,
                min_lr=self.config.min_learning_rate,
            ),
        ]

    def _predict(
        self,
        model: tf.keras.Model,
        X: np.ndarray,
        sensitive: np.ndarray,
    ) -> np.ndarray:
        return model.predict(
            self.formatter.format(X, sensitive),
            batch_size=self.config.batch_size,
            verbose=0,
        ).reshape(-1)

    def _save_predictions(
        self,
        *,
        path: Path,
        ids: tuple[Any, ...],
        y_true: np.ndarray,
        sensitive: np.ndarray,
        y_proba: np.ndarray,
    ) -> None:
        frame = pd.DataFrame(
            {
                "SK_ID_CURR": list(ids),
                "TARGET": np.asarray(y_true, dtype="int32"),
                "SENSITIVE": np.asarray(sensitive, dtype="int32"),
                "y_proba": y_proba,
            }
        )
        frame.to_csv(path, index=False)

    def _read_mvp_results(self, mvp_results_path: Path | None) -> pd.DataFrame:
        if mvp_results_path is None:
            return pd.DataFrame()
        path = Path(mvp_results_path)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def _reference_row(self, comparison: pd.DataFrame) -> pd.Series:
        base = comparison[
            comparison["experimento"].str.contains("MVP Base", case=False, na=False)
        ]
        if not base.empty:
            return base.iloc[0]

        quadratic_base = comparison[
            comparison["experimento"].str.contains(
                "Cuadratica base",
                case=False,
                na=False,
            )
        ]
        if not quadratic_base.empty:
            return quadratic_base.iloc[0]

        return comparison.iloc[0]

    def _create_directories(self, paths: SquaredFairnessExperimentPaths) -> None:
        for directory in (
            paths.sweep_csv.parent,
            paths.histories_dir,
            paths.models_dir,
            paths.predictions_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def _validate_artifact_contract(self) -> None:
        run_dir = Path(self.artifacts.run_dir).resolve()
        try:
            run_dir.relative_to((Path(self.artifacts.project_root) / "results" / "extras").resolve())
        except ValueError as exc:
            raise FairnessLossExperimentError(
                "Squared FAIR artifacts must live under results/extras/<run_id>/."
            ) from exc

    def _to_project_relative(self, path: Path) -> str:
        return self.artifacts.to_project_relative(path)

    @staticmethod
    def _epochs_trained(history: tf.keras.callbacks.History) -> int:
        if not history.history:
            return 0
        return int(len(next(iter(history.history.values()))))

    @staticmethod
    def _model_label(alpha: float) -> str:
        if float(alpha) == 0.0:
            return "cuadratica_base_alpha_0"
        return f"cuadratica_fair_alpha_{lambda_slug(alpha)}"

    @staticmethod
    def _comparison_label(alpha: float) -> str:
        if float(alpha) == 0.0:
            return "Cuadratica base alpha=0"
        return f"Cuadratica FAIR alpha={alpha:g}"


__all__ = [
    "DualInputArrayFormatter",
    "FairnessLossExperimentError",
    "SquaredFairnessComparisonRunner",
    "SquaredFairnessExperimentPaths",
    "SquaredFairnessResultInterpreter",
    "SquaredFairnessRunResult",
    "SquaredFairnessTrainingConfig",
]
