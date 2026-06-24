"""Reproducible full-run orchestration for the final MVP.

This module keeps generated artifacts isolated from the validated canonical
results. A full training run writes under:

    results/runs/<run_id>/

and never under the historical `results/tables`, `results/models` or `kt_dir`
locations. The goal is to make the project reproducible without risking the
already defended Pareto curve and uncertainty artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import (
    BinaryClassificationMetricCalculator,
    FairnessMetricCalculator,
    ProbabilityMetricCalculator,
)
from .preprocessing import HomeCreditMVPPreprocessingPipeline, ProcessedSplitDataset
from .splitting import HomeCreditTrainValTestSplitter, SplitConfig, SplitArtifacts
from .tuning import (
    DualInputFormatter,
    FairKerasTunerRunner,
    FairLambdaSweepTrainer,
    LambdaSweepResult,
    TunerSearchResult,
    TuningArtifactPaths,
    TuningConfig,
)
from .uncertainty import (
    FairModelLoader,
    UncertaintyArtifactPaths,
    UncertaintyMVPResult,
    UncertaintyMVPTrainer,
    UncertaintyModelConfig,
)


class ReproducibleRunError(ValueError):
    """Raised when a reproducible run would violate the artifact contract."""


def timestamp_run_id(now: datetime | None = None) -> str:
    """Return a stable filesystem-safe run id.

    Args:
        now: Optional datetime injected by tests.

    Returns:
        Run identifier in `YYYYMMDD_HHMMSS` format.
    """

    value = now or datetime.now()
    return value.strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class ReproducibleRunPaths:
    """Filesystem contract for one isolated reproducible run.

    Args:
        project_root: Repository root.
        run_id: Human-readable run id. If omitted, a timestamp is used.
        runs_dir_name: Project-relative directory where isolated runs live.

    Returns:
        Immutable path bundle used by notebooks and runners.
    """

    project_root: Path
    run_id: str | None = None
    runs_dir_name: str = "results/runs"

    def __post_init__(self) -> None:
        """Validate the run id and root location."""

        object.__setattr__(self, "project_root", Path(self.project_root).resolve())
        object.__setattr__(self, "run_id", self._normalize_run_id(self.run_id))
        self._validate_isolated_root()

    @staticmethod
    def _normalize_run_id(run_id: str | None) -> str:
        """Return a filesystem-safe run id."""

        value = run_id or timestamp_run_id()
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
        cleaned = "".join(char if char in allowed else "-" for char in value.strip())
        cleaned = cleaned.strip("-_")
        if not cleaned:
            raise ReproducibleRunError("run_id cannot be empty.")
        return cleaned

    @property
    def runs_root(self) -> Path:
        """Return the absolute root containing all reproducible runs."""

        return self.project_root / self.runs_dir_name

    @property
    def run_dir(self) -> Path:
        """Return the absolute directory for this run."""

        return self.runs_root / str(self.run_id)

    @property
    def tables_dir(self) -> Path:
        """Return the isolated tables directory."""

        return self.run_dir / "tables"

    @property
    def figures_dir(self) -> Path:
        """Return the isolated figures directory."""

        return self.run_dir / "figures"

    @property
    def models_dir(self) -> Path:
        """Return the isolated model directory."""

        return self.run_dir / "models"

    @property
    def tuner_dir(self) -> Path:
        """Return the isolated Keras Tuner directory."""

        return self.run_dir / "kt_dir"

    @property
    def manifest_path(self) -> Path:
        """Return the run manifest path."""

        return self.run_dir / "run_manifest.csv"

    @property
    def pareto_results_csv(self) -> Path:
        """Return the run-local Pareto CSV path."""

        return self.tables_dir / "pareto_results.csv"

    @property
    def test_results_csv(self) -> Path:
        """Return the run-local test comparison CSV path."""

        return self.tables_dir / "test_results_base_vs_fair.csv"

    @property
    def uncertainty_by_ext_csv(self) -> Path:
        """Return the run-local EXT_NULL_COUNT uncertainty summary path."""

        return self.tables_dir / "uncertainty_by_ext_null_count.csv"

    def create(self) -> None:
        """Create every run directory."""

        for directory in (
            self.tables_dir,
            self.figures_dir,
            self.models_dir,
            self.tuner_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def to_project_relative(self, path: Path) -> str:
        """Return a POSIX project-relative path when possible."""

        try:
            relative = Path(path).resolve().relative_to(self.project_root)
        except ValueError:
            return Path(path).as_posix()
        return relative.as_posix()

    def tuning_artifacts(self) -> TuningArtifactPaths:
        """Build isolated TuningArtifactPaths for this run."""

        return TuningArtifactPaths(
            project_root=self.project_root,
            tuner_dir_name=self.to_project_relative(self.tuner_dir),
            tables_dir_name=self.to_project_relative(self.tables_dir),
            models_dir_name=self.to_project_relative(self.models_dir),
        )

    def uncertainty_artifacts(self) -> UncertaintyArtifactPaths:
        """Build isolated UncertaintyArtifactPaths for this run."""

        return UncertaintyArtifactPaths(
            project_root=self.project_root,
            tables_dir_name=self.to_project_relative(self.tables_dir),
            models_dir_name=self.to_project_relative(self.models_dir),
        )

    def _validate_isolated_root(self) -> None:
        """Reject paths that could overwrite canonical artifacts."""

        run_dir = self.run_dir.resolve()
        required_root = (self.project_root / self.runs_dir_name).resolve()
        try:
            run_dir.relative_to(required_root)
        except ValueError as exc:
            raise ReproducibleRunError(
                "Reproducible runs must live under results/runs/."
            ) from exc

        forbidden = {
            (self.project_root / "results" / "tables").resolve(),
            (self.project_root / "results" / "models").resolve(),
            (self.project_root / "kt_dir").resolve(),
        }
        if run_dir in forbidden:
            raise ReproducibleRunError(
                "Run directory cannot point to canonical or legacy artifact locations."
            )


@dataclass(frozen=True)
class ReproducibleMVPConfig:
    """Configuration for the complete MVP reproducible run."""

    seed: int = 42
    lambda_values: tuple[float, ...] = (
        0.0,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.0,
        5.0,
    )
    max_trials: int = 15
    epochs: int = 50
    batch_size: int = 1024
    tuning_lambda_fair: float = 0.5
    tuner_project_name: str = "fair_credit_reproducible"
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    uncertainty_epochs: int = 50
    uncertainty_hidden_units: tuple[int, ...] = (32,)
    uncertainty_dropout: float = 0.3

    def tuning_config(self) -> TuningConfig:
        """Return the TuningConfig used by tuner and lambda sweep."""

        return TuningConfig(
            tuning_lambda_fair=self.tuning_lambda_fair,
            lambda_values=self.lambda_values,
            max_trials=self.max_trials,
            executions_per_trial=1,
            epochs=self.epochs,
            batch_size=self.batch_size,
            early_stopping_patience=self.early_stopping_patience,
            reduce_lr_patience=self.reduce_lr_patience,
            tuner_project_name=self.tuner_project_name,
            overwrite_tuner=True,
        )

    def uncertainty_config(self) -> UncertaintyModelConfig:
        """Return the M2 uncertainty configuration."""

        return UncertaintyModelConfig(
            hidden_units=self.uncertainty_hidden_units,
            dropout=self.uncertainty_dropout,
            output_activation="softplus",
            normalize_inputs=True,
            batch_size=self.batch_size,
            epochs=self.uncertainty_epochs,
            early_stopping_patience=self.early_stopping_patience,
        )


@dataclass(frozen=True)
class ReproducibleDataResult:
    """Data objects produced before neural training."""

    raw: pd.DataFrame
    split: SplitArtifacts
    processed: ProcessedSplitDataset


@dataclass(frozen=True)
class TestEvaluationBundle:
    """Test evaluation artifacts for the selected base and FAIR models."""

    table: pd.DataFrame
    base_probabilities: np.ndarray
    fair_probabilities: np.ndarray


class ReproducibleMVPWorkflow:
    """Orchestrate the full MVP run into an isolated run directory."""

    def __init__(
        self,
        *,
        project_root: Path,
        paths: ReproducibleRunPaths | None = None,
        config: ReproducibleMVPConfig | None = None,
    ) -> None:
        """Initialize the workflow.

        Args:
            project_root: Repository root.
            paths: Optional run path bundle.
            config: Optional run configuration.
        """

        self.project_root = Path(project_root).resolve()
        self.paths = paths or ReproducibleRunPaths(project_root=self.project_root)
        self.config = config or ReproducibleMVPConfig()
        self.paths.create()

    def prepare_data(self, application_train_path: Path) -> ReproducibleDataResult:
        """Load, split and preprocess Home Credit data without leakage."""

        pipeline = HomeCreditMVPPreprocessingPipeline()
        raw = pipeline.load_raw(application_train_path)
        deterministic = pipeline.apply_deterministic_transforms(raw)
        splitter = HomeCreditTrainValTestSplitter(
            config=SplitConfig(random_state=self.config.seed)
        )
        split = splitter.split(deterministic)
        processed = pipeline.fit_transform_splits(split.raw_splits)

        split.report.to_csv(self.paths.tables_dir / "split_report.csv", index=False)
        return ReproducibleDataResult(raw=raw, split=split, processed=processed)

    def train_tuner_and_sweep(
        self,
        processed: ProcessedSplitDataset,
        *,
        verbose: int = 1,
    ) -> tuple[TunerSearchResult, LambdaSweepResult]:
        """Run Keras Tuner and the controlled lambda sweep."""

        tuning_config = self.config.tuning_config()
        artifacts = self.paths.tuning_artifacts()

        search_result = FairKerasTunerRunner(
            config=tuning_config,
            artifacts=artifacts,
        ).search(processed, verbose=verbose)

        sweep_result = FairLambdaSweepTrainer(
            config=tuning_config,
            artifacts=artifacts,
        ).run(
            data=processed,
            custom_config=search_result.best_config,
            ratio_indices=search_result.ratio_indices,
            class_weight=search_result.class_weight,
            save_models=True,
            verbose=verbose,
        )

        return search_result, sweep_result

    def evaluate_selected_models(
        self,
        processed: ProcessedSplitDataset,
        pareto: pd.DataFrame | None = None,
    ) -> TestEvaluationBundle:
        """Evaluate selected base and FAIR models on the untouched test set."""

        pareto_table = pareto if pareto is not None else pd.read_csv(
            self.paths.pareto_results_csv
        )

        base_rows = pareto_table[
            pareto_table["selected_for_test"].astype(bool)
            & np.isclose(pareto_table["lambda_fair"].astype(float), 0.0)
        ]
        fair_rows = pareto_table[
            pareto_table["selected_for_test"].astype(bool)
            & ~np.isclose(pareto_table["lambda_fair"].astype(float), 0.0)
        ]

        if base_rows.empty or fair_rows.empty:
            raise ReproducibleRunError(
                "Pareto table must select one base row and one FAIR row."
            )

        base_eval = self._evaluate_one_model(
            processed=processed,
            row=base_rows.iloc[0],
            model_name="Base final",
        )
        fair_eval = self._evaluate_one_model(
            processed=processed,
            row=fair_rows.iloc[0],
            model_name="FAIR final",
        )

        table = pd.DataFrame(
            [
                {key: value for key, value in base_eval.items() if key != "y_proba"},
                {key: value for key, value in fair_eval.items() if key != "y_proba"},
            ]
        )
        table.to_csv(self.paths.test_results_csv, index=False)

        return TestEvaluationBundle(
            table=table,
            base_probabilities=base_eval["y_proba"],
            fair_probabilities=fair_eval["y_proba"],
        )

    def train_uncertainty(
        self,
        processed: ProcessedSplitDataset,
        pareto: pd.DataFrame | None = None,
        *,
        verbose: int = 1,
    ) -> UncertaintyMVPResult:
        """Train M2 uncertainty on validation errors and evaluate test uncertainty."""

        pareto_table = pareto if pareto is not None else pd.read_csv(
            self.paths.pareto_results_csv
        )
        fair_rows = pareto_table[
            pareto_table["selected_for_test"].astype(bool)
            & ~np.isclose(pareto_table["lambda_fair"].astype(float), 0.0)
        ]
        if fair_rows.empty:
            raise ReproducibleRunError("No selected FAIR row found for uncertainty.")

        fair_row = fair_rows.iloc[0]
        fair_model = FairModelLoader().load(self.project_root / fair_row["model_path"])

        result = UncertaintyMVPTrainer(
            config=self.config.uncertainty_config(),
            artifacts=self.paths.uncertainty_artifacts(),
        ).run(
            m1_model=fair_model,
            data=processed,
            selected_threshold=float(fair_row["val_threshold"]),
            save_artifacts=True,
            save_model=True,
            verbose=verbose,
        )

        self.write_uncertainty_by_ext_null_count(
            result.prediction_result.predictions
        )
        return result

    def write_uncertainty_by_ext_null_count(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Write uncertainty summary by raw EXT_NULL_COUNT."""

        required = {"EXT_NULL_COUNT", "uncertainty"}
        missing = required.difference(predictions.columns)
        if missing:
            raise ReproducibleRunError(
                f"Missing columns for EXT_NULL_COUNT summary: {sorted(missing)}"
            )

        summary = (
            predictions.groupby("EXT_NULL_COUNT", observed=False)["uncertainty"]
            .agg(
                count="size",
                uncertainty_mean="mean",
                uncertainty_median="median",
                uncertainty_q1=lambda values: values.quantile(0.25),
                uncertainty_q3=lambda values: values.quantile(0.75),
            )
            .reset_index()
        )
        summary["uncertainty_iqr"] = (
            summary["uncertainty_q3"] - summary["uncertainty_q1"]
        )
        summary.to_csv(self.paths.uncertainty_by_ext_csv, index=False)
        return summary

    def write_manifest(self, extra: dict[str, Any] | None = None) -> Path:
        """Write a compact manifest for the run."""

        rows = {
            "run_id": self.paths.run_id,
            "run_dir": self.paths.to_project_relative(self.paths.run_dir),
            "max_trials": self.config.max_trials,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "lambda_values": ",".join(str(value) for value in self.config.lambda_values),
            "uncertainty_epochs": self.config.uncertainty_epochs,
        }
        if extra:
            rows.update(extra)
        pd.DataFrame([rows]).to_csv(self.paths.manifest_path, index=False)
        return self.paths.manifest_path

    def _evaluate_one_model(
        self,
        *,
        processed: ProcessedSplitDataset,
        row: pd.Series,
        model_name: str,
    ) -> dict[str, Any]:
        """Evaluate one selected model row on the test set."""

        model = FairModelLoader().load(self.project_root / row["model_path"])
        proba = DualInputModelPredictorAdapter(self.config.batch_size).predict(
            model=model,
            X=processed.X_test,
            sensitive=processed.s_test,
        )
        threshold = float(row["val_threshold"])
        probability = ProbabilityMetricCalculator().calculate(
            processed.y_test,
            proba,
            processed.s_test,
        )
        binary = BinaryClassificationMetricCalculator().calculate(
            processed.y_test,
            proba,
            threshold,
        )
        fairness = FairnessMetricCalculator().calculate(
            processed.y_test,
            proba,
            processed.s_test,
            threshold,
        )

        return {
            "modelo": model_name,
            "lambda_fair": float(row["lambda_fair"]),
            "threshold": threshold,
            "auc": probability.roc_auc,
            "pr_auc": probability.pr_auc,
            "accuracy": binary.accuracy,
            "precision": binary.precision,
            "recall": binary.recall,
            "f1": binary.f1,
            "abs_rho": probability.abs_rho,
            "dpd": fairness.demographic_parity_difference,
            "eod": fairness.equalized_odds_difference,
            "y_proba": proba,
        }


class DualInputModelPredictorAdapter:
    """Small adapter used to avoid exposing notebook prediction logic."""

    def __init__(self, batch_size: int) -> None:
        """Initialize the adapter."""

        self._batch_size = int(batch_size)
        self._formatter = DualInputFormatter()

    def predict(self, *, model: Any, X: np.ndarray, sensitive: np.ndarray) -> np.ndarray:
        """Predict flat probabilities from a dual-input Keras model."""

        return model.predict(
            self._formatter.format(X, sensitive),
            batch_size=self._batch_size,
            verbose=0,
        ).reshape(-1)


__all__ = [
    "ReproducibleDataResult",
    "ReproducibleMVPConfig",
    "ReproducibleMVPWorkflow",
    "ReproducibleRunError",
    "ReproducibleRunPaths",
    "TestEvaluationBundle",
    "timestamp_run_id",
]
