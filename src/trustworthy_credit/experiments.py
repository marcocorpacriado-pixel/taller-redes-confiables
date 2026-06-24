"""Experiment orchestration helpers for the unified MVP.

This module turns the multi-seed Pareto idea from the original procedural
script into a small, testable orchestration layer. It does not train models by
itself; callers provide a function that runs one seed/lambda experiment.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


class ExperimentError(ValueError):
    """Raised when an experiment configuration or result table is invalid."""


@dataclass(frozen=True, slots=True)
class ModelProgressionSpec:
    """Reference metadata for one audited model in the M0-M6 progression."""

    model_id: str
    model_name: str
    technical_idea: str
    history_filename: str
    reported_val_auc: float
    reported_test_auc: float
    n_params: int


DEFAULT_MODEL_PROGRESSION_SPECS: tuple[ModelProgressionSpec, ...] = (
    ModelProgressionSpec(
        model_id="M0",
        model_name="Regresion logistica",
        technical_idea="Baseline lineal para medir el suelo predictivo.",
        history_filename="M0_LogReg_history.json",
        reported_val_auc=0.7275,
        reported_test_auc=0.7335,
        n_params=13,
    ),
    ModelProgressionSpec(
        model_id="M1",
        model_name="MLP 1 capa",
        technical_idea="Primera no linealidad densa sobre variables tabulares.",
        history_filename="M1_MLP1capa_history.json",
        reported_val_auc=0.7416,
        reported_test_auc=0.7449,
        n_params=897,
    ),
    ModelProgressionSpec(
        model_id="M2",
        model_name="MLP 2 capas",
        technical_idea="Mayor capacidad para interacciones no lineales.",
        history_filename="M2_MLP2capas_history.json",
        reported_val_auc=0.7411,
        reported_test_auc=0.7447,
        n_params=9985,
    ),
    ModelProgressionSpec(
        model_id="M3",
        model_name="MLP + Dropout",
        technical_idea="Regularizacion para reducir sobreajuste en tabular.",
        history_filename="M3_Dropout_history.json",
        reported_val_auc=0.7424,
        reported_test_auc=0.7452,
        n_params=9985,
    ),
    ModelProgressionSpec(
        model_id="M4",
        model_name="Capa financiera custom",
        technical_idea="Inyecta conocimiento de dominio mediante ratio financiero.",
        history_filename="M4_Custom_history.json",
        reported_val_auc=0.7424,
        reported_test_auc=0.7449,
        n_params=9988,
    ),
    ModelProgressionSpec(
        model_id="M5",
        model_name="Custom + scheduler",
        technical_idea="Estabiliza entrenamiento con ajuste dinamico del learning rate.",
        history_filename="M5_ReduceLR_history.json",
        reported_val_auc=0.7428,
        reported_test_auc=0.7449,
        n_params=9988,
    ),
    ModelProgressionSpec(
        model_id="M6",
        model_name="Dual custom",
        technical_idea="Separa ratios financieros, fuentes externas y bloque denso.",
        history_filename="M6_DualCustom_history.json",
        reported_val_auc=0.7421,
        reported_test_auc=0.7451,
        n_params=9991,
    ),
    ModelProgressionSpec(
        model_id="M3 largo",
        model_name="MLP + Dropout largo",
        technical_idea="Controla si entrenar mas epocas mejora frente al M3 compacto.",
        history_filename="M3_largo_history.json",
        reported_val_auc=0.7419,
        reported_test_auc=0.7457,
        n_params=9985,
    ),
)


@dataclass(slots=True)
class ModelProgressionReporter:
    """Read audited M0-M6 histories and build a compact comparison table.

    The reporter consumes already computed JSON histories. It does not rebuild
    or train any model. The reported validation/test AUC values come from the
    audited experiment table used in the technical report, while the JSON files
    provide epoch counts and best validation epochs.
    """

    checkpoints_dir: Path
    specs: tuple[ModelProgressionSpec, ...] = DEFAULT_MODEL_PROGRESSION_SPECS

    def summarize(self) -> pd.DataFrame:
        """Return one row per audited architecture in the M0-M6 progression."""

        records = [self._record_from_spec(spec) for spec in self.specs]
        frame = pd.DataFrame.from_records(records)
        baseline_auc = float(frame.loc[0, "reported_test_auc"])
        frame["test_auc_gain_vs_m0"] = frame["reported_test_auc"] - baseline_auc
        return frame

    def _record_from_spec(self, spec: ModelProgressionSpec) -> dict[str, object]:
        """Build one progression row from metadata plus an optional history file."""

        history_path = self.checkpoints_dir / spec.history_filename
        history = self._load_history(history_path)
        summary = self._summarize_history(history)
        return {
            "model_id": spec.model_id,
            "model_name": spec.model_name,
            "technical_idea": spec.technical_idea,
            "history_file": spec.history_filename,
            "history_found": history_path.exists(),
            "epochs": summary["epochs"],
            "best_epoch": summary["best_epoch"],
            "history_best_val_auc": summary["history_best_val_auc"],
            "history_final_val_auc": summary["history_final_val_auc"],
            "reported_val_auc": spec.reported_val_auc,
            "reported_test_auc": spec.reported_test_auc,
            "n_params": spec.n_params,
        }

    @staticmethod
    def _load_history(history_path: Path) -> Mapping[str, list[float]]:
        """Load a Keras history JSON file, returning an empty mapping if absent."""

        if not history_path.exists():
            return {}
        with history_path.open("r", encoding="utf-8") as handle:
            history = json.load(handle)
        if not isinstance(history, dict):
            raise ExperimentError(f"History file is not a mapping: {history_path}")
        return history

    @staticmethod
    def _summarize_history(
        history: Mapping[str, list[float]],
    ) -> dict[str, float | int]:
        """Extract stable summary values from a Keras history dictionary."""

        if not history:
            return {
                "epochs": 0,
                "best_epoch": 0,
                "history_best_val_auc": float("nan"),
                "history_final_val_auc": float("nan"),
            }
        val_auc = history.get("val_auc", [])
        if not isinstance(val_auc, list) or not val_auc:
            return {
                "epochs": max(
                    (len(value) for value in history.values() if isinstance(value, list)),
                    default=0,
                ),
                "best_epoch": 0,
                "history_best_val_auc": float("nan"),
                "history_final_val_auc": float("nan"),
            }

        val_auc_array = np.asarray(val_auc, dtype=float)
        best_index = int(np.nanargmax(val_auc_array))
        return {
            "epochs": int(len(val_auc_array)),
            "best_epoch": best_index + 1,
            "history_best_val_auc": float(val_auc_array[best_index]),
            "history_final_val_auc": float(val_auc_array[-1]),
        }


@dataclass(frozen=True, slots=True)
class MultiSeedParetoConfig:
    """Configuration for a multi-seed fairness Pareto experiment."""

    seeds: tuple[int, ...] = (42, 123, 7)
    lambda_values: tuple[float, ...] = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0)
    metric_columns: tuple[str, ...] = ("auc", "abs_rho", "dpd", "eod")

    def __post_init__(self) -> None:
        """Validate experiment grid settings."""

        if not self.seeds:
            raise ExperimentError("At least one seed is required.")
        if not self.lambda_values:
            raise ExperimentError("At least one lambda value is required.")
        if any(lambda_value < 0 for lambda_value in self.lambda_values):
            raise ExperimentError("lambda_values must be non-negative.")
        if not self.metric_columns:
            raise ExperimentError("At least one metric column is required.")


@dataclass(frozen=True, slots=True)
class ExperimentRunResult:
    """One result row from a seed/lambda experiment."""

    seed: int
    lambda_fair: float
    metrics: Mapping[str, float]

    def to_record(self) -> dict[str, float | int]:
        """Convert the result to a flat DataFrame-ready record."""

        record: dict[str, float | int] = {
            "seed": self.seed,
            "lambda_fair": self.lambda_fair,
        }
        for key, value in self.metrics.items():
            record[key] = float(value)
        return record


@dataclass(slots=True)
class MultiSeedParetoRunner:
    """Run a callable over all seed/lambda combinations."""

    config: MultiSeedParetoConfig

    def run(
        self,
        run_single_experiment: Callable[[int, float], Mapping[str, float]],
    ) -> pd.DataFrame:
        """Run the configured grid and return one row per experiment."""

        records: list[dict[str, float | int]] = []
        for seed in self.config.seeds:
            for lambda_fair in self.config.lambda_values:
                metrics = run_single_experiment(seed, lambda_fair)
                self._validate_metrics(metrics)
                records.append(
                    ExperimentRunResult(
                        seed=seed,
                        lambda_fair=lambda_fair,
                        metrics=metrics,
                    ).to_record()
                )
        return pd.DataFrame.from_records(records)

    def _validate_metrics(self, metrics: Mapping[str, float]) -> None:
        """Validate that one experiment produced the configured metrics."""

        missing = [
            metric for metric in self.config.metric_columns if metric not in metrics
        ]
        if missing:
            raise ExperimentError(
                "Experiment result is missing metrics: " + ", ".join(missing)
            )


@dataclass(slots=True)
class MultiSeedParetoSummarizer:
    """Aggregate seed-level Pareto rows into mean/std summary tables."""

    metric_columns: tuple[str, ...] = ("auc", "abs_rho", "dpd", "eod")

    def summarize(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Summarize metrics by lambda with mean, std, and run count."""

        self._validate_results(results_df)
        aggregations: dict[str, list[str]] = {
            metric: ["mean", "std"] for metric in self.metric_columns
        }
        aggregations["seed"] = ["count"]

        summary = results_df.groupby("lambda_fair", as_index=False).agg(aggregations)
        summary.columns = self._flatten_columns(summary.columns)
        summary = summary.rename(columns={"seed_count": "n_runs"})
        return summary.sort_values("lambda_fair").reset_index(drop=True)

    def _validate_results(self, results_df: pd.DataFrame) -> None:
        """Validate the seed-level result table."""

        if results_df.empty:
            raise ExperimentError("Results table is empty.")
        required = {"seed", "lambda_fair", *self.metric_columns}
        missing = sorted(required.difference(results_df.columns))
        if missing:
            raise ExperimentError(
                "Results table is missing columns: " + ", ".join(missing)
            )

    @staticmethod
    def _flatten_columns(columns: pd.Index) -> list[str]:
        """Flatten pandas aggregation MultiIndex columns."""

        flattened: list[str] = []
        for column in columns:
            if not isinstance(column, tuple):
                flattened.append(str(column))
                continue
            base, suffix = column
            if suffix:
                flattened.append(f"{base}_{suffix}")
            else:
                flattened.append(str(base))
        return flattened


@dataclass(slots=True)
class MultiSeedParetoArtifactReporter:
    """Load a saved Pareto artifact and standardize it for final reporting."""

    config: MultiSeedParetoConfig = field(default_factory=MultiSeedParetoConfig)

    def summarize(self, artifact_path: str | Path) -> pd.DataFrame:
        """Return a normalized multi-seed Pareto summary table."""

        path = Path(artifact_path)
        if not path.exists():
            raise ExperimentError(f"Pareto artifact does not exist: {path}")

        raw = pd.read_csv(path)
        raw = raw.rename(
            columns={
                "lambda": "lambda_fair",
                "dp": "abs_rho",
                "dp_std": "abs_rho_std",
            }
        )

        aggregated_columns = {
            "lambda_fair",
            "auc",
            "auc_std",
            "abs_rho",
            "abs_rho_std",
        }
        already_summarized_columns = {
            "lambda_fair",
            "auc_mean",
            "auc_std",
            "abs_rho_mean",
            "abs_rho_std",
        }
        seed_level_columns = {"seed", "lambda_fair", "auc", "abs_rho"}

        if already_summarized_columns.issubset(raw.columns):
            summary = raw.copy()
        elif aggregated_columns.issubset(raw.columns):
            summary = raw.rename(
                columns={
                    "auc": "auc_mean",
                    "abs_rho": "abs_rho_mean",
                }
            ).copy()
        elif seed_level_columns.issubset(raw.columns):
            metric_columns = tuple(
                column
                for column in self.config.metric_columns
                if column in raw.columns
            )
            summary = MultiSeedParetoSummarizer(
                metric_columns=metric_columns
            ).summarize(raw)
        else:
            expected = sorted(
                aggregated_columns | already_summarized_columns | seed_level_columns
            )
            raise ExperimentError(
                "Pareto artifact has an unsupported format. Expected columns: "
                + ", ".join(expected)
            )

        if "n_runs" not in summary.columns:
            summary["n_runs"] = len(self.config.seeds)

        summary = summary.sort_values("lambda_fair").reset_index(drop=True)
        baseline_auc = float(summary.loc[0, "auc_mean"])
        baseline_fairness = float(summary.loc[0, "abs_rho_mean"])
        summary["auc_delta_vs_baseline"] = summary["auc_mean"] - baseline_auc
        if baseline_fairness == 0:
            summary["fairness_reduction"] = 0.0
        else:
            summary["fairness_reduction"] = (
                1.0 - summary["abs_rho_mean"] / baseline_fairness
            )
        return summary


@dataclass(frozen=True, slots=True)
class MCDropoutAuditRecord:
    """Audited MC Dropout values reported by the saved experiment artifacts."""

    model: str
    point_auc: float
    mc_mean_auc: float
    variance_target_0: float
    variance_target_1: float
    variance_ext_missing_0: float
    variance_ext_missing_3: float


DEFAULT_MC_DROPOUT_AUDIT_RECORDS: tuple[MCDropoutAuditRecord, ...] = (
    MCDropoutAuditRecord(
        model="M6 base",
        point_auc=0.7451,
        mc_mean_auc=0.7455,
        variance_target_0=0.001422,
        variance_target_1=0.001306,
        variance_ext_missing_0=0.001472,
        variance_ext_missing_3=0.001414,
    ),
    MCDropoutAuditRecord(
        model="FAIR lambda=1.0",
        point_auc=0.7424,
        mc_mean_auc=0.7428,
        variance_target_0=0.000123,
        variance_target_1=0.000301,
        variance_ext_missing_0=0.000134,
        variance_ext_missing_3=0.000241,
    ),
)


@dataclass(slots=True)
class MCDropoutArtifactReporter:
    """Summarize saved MC Dropout arrays and audited uncertainty contrasts."""

    checkpoints_dir: Path
    audit_records: tuple[MCDropoutAuditRecord, ...] = (
        DEFAULT_MC_DROPOUT_AUDIT_RECORDS
    )

    def saved_array_summary(
        self,
        prefixes: tuple[str, ...] = ("mc_m6", "mc_fair", "mc_fair10"),
    ) -> pd.DataFrame:
        """Return descriptive statistics for saved MC mean/variance arrays."""

        records: list[dict[str, float | int | str]] = []
        for prefix in prefixes:
            mean_path = self.checkpoints_dir / f"{prefix}_mean.npy"
            var_path = self.checkpoints_dir / f"{prefix}_var.npy"
            if not mean_path.exists() or not var_path.exists():
                continue
            mean_values = np.load(mean_path)
            variance_values = np.load(var_path)
            if mean_values.shape != variance_values.shape:
                raise ExperimentError(
                    f"MC arrays have different shapes for prefix {prefix}."
                )
            records.append(
                {
                    "artifact": prefix,
                    "n_samples": int(mean_values.size),
                    "mean_prediction": float(np.mean(mean_values)),
                    "median_prediction": float(np.median(mean_values)),
                    "mean_variance": float(np.mean(variance_values)),
                    "median_variance": float(np.median(variance_values)),
                    "p95_variance": float(np.percentile(variance_values, 95)),
                }
            )
        return pd.DataFrame.from_records(records)

    def audited_summary(self) -> pd.DataFrame:
        """Return the audited target/data-quality MC Dropout comparison table."""

        records = []
        for record in self.audit_records:
            target_ratio = (
                record.variance_target_1 / record.variance_target_0
                if record.variance_target_0
                else float("nan")
            )
            ext_missing_lift = (
                record.variance_ext_missing_3 / record.variance_ext_missing_0 - 1.0
                if record.variance_ext_missing_0
                else float("nan")
            )
            records.append(
                {
                    "model": record.model,
                    "point_auc": record.point_auc,
                    "mc_mean_auc": record.mc_mean_auc,
                    "variance_target_0": record.variance_target_0,
                    "variance_target_1": record.variance_target_1,
                    "target_1_to_0_variance_ratio": target_ratio,
                    "variance_ext_missing_0": record.variance_ext_missing_0,
                    "variance_ext_missing_3": record.variance_ext_missing_3,
                    "ext_missing_3_lift_vs_0": ext_missing_lift,
                }
            )
        return pd.DataFrame.from_records(records)


@dataclass(frozen=True, slots=True)
class FeatureAblationRecord:
    """Audited AUC comparison between compact and extended feature sets."""

    model: str
    auc_12_features: float
    auc_42_features: float
    interpretation: str


DEFAULT_FEATURE_ABLATION_RECORDS: tuple[FeatureAblationRecord, ...] = (
    FeatureAblationRecord(
        model="M0 logistic",
        auc_12_features=0.7335,
        auc_42_features=0.7501,
        interpretation="La senal adicional ayuda incluso a un modelo lineal.",
    ),
    FeatureAblationRecord(
        model="M3 dropout",
        auc_12_features=0.7457,
        auc_42_features=0.7555,
        interpretation="La regularizacion aprovecha mejor el espacio extendido.",
    ),
    FeatureAblationRecord(
        model="M6 dual custom",
        auc_12_features=0.7451,
        auc_42_features=0.7550,
        interpretation="El modelo auditable mejora cuando recibe mas informacion.",
    ),
)


@dataclass(slots=True)
class FeatureAblationReporter:
    """Build the audited 12-vs-42-feature comparison table."""

    records: tuple[FeatureAblationRecord, ...] = DEFAULT_FEATURE_ABLATION_RECORDS

    def summarize(self) -> pd.DataFrame:
        """Return AUC gains from the compact to the extended feature setup."""

        frame = pd.DataFrame.from_records(
            {
                "model": record.model,
                "auc_12_features": record.auc_12_features,
                "auc_42_features": record.auc_42_features,
                "interpretation": record.interpretation,
            }
            for record in self.records
        )
        frame["auc_gain_42_vs_12"] = (
            frame["auc_42_features"] - frame["auc_12_features"]
        )
        frame["relative_auc_gain_pct"] = (
            frame["auc_gain_42_vs_12"] / frame["auc_12_features"] * 100.0
        )
        return frame
