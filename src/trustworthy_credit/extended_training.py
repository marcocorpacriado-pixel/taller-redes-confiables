"""Executable training experiments for the extended 42-feature pipeline.

The canonical MVP keeps the compact Dani feature set as the main deliverable.
This module makes the advanced 42-feature path reproducible: it runs the
leakage-safe extended preprocessing, trains selected models, audits predictive
and fairness metrics, and can compare the result against compact MVP tables.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.trustworthy_credit.extended_features import (
    ExtendedFeaturePreprocessingPipeline,
    ExtendedFeatureSelectionConfig,
    ExtendedFeatureSet,
)
from src.trustworthy_credit.metrics import (
    absolute_pearson_correlation,
    fairness_metrics,
)
from src.trustworthy_credit.model_progression import (
    DEFAULT_PROGRESSION_SPECS,
    ModelProgressionTrainingConfig,
    ProgressionDataset,
    ProgressionFeatureIndices,
    ProgressionModelFactory,
    ProgressionModelSpec,
)


class ExtendedTrainingError(ValueError):
    """Raised when the extended-feature training experiment cannot run safely."""


@dataclass(frozen=True, slots=True)
class ExtendedTrainingConfig:
    """Configuration for training models on the extended feature set."""

    model_ids: tuple[str, ...] = ("M0", "M3")
    epochs: int = 30
    batch_size: int = 512
    learning_rate: float = 1e-3
    activation: str = "relu"
    hidden_units: tuple[int, int] = (128, 64)
    m1_units: int = 64
    dropout_rates: tuple[float, float] = (0.30, 0.20)
    early_stopping_patience: int = 8
    reduce_lr_patience: int = 4
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6
    threshold: float = 0.5
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate settings before preprocessing or model training."""

        if not self.model_ids:
            raise ExtendedTrainingError("At least one model_id is required.")
        known_ids = {spec.model_id for spec in DEFAULT_PROGRESSION_SPECS}
        missing = sorted(set(self.model_ids).difference(known_ids))
        if missing:
            raise ExtendedTrainingError(
                "Unknown model_ids: " + ", ".join(missing)
            )
        if self.epochs <= 0:
            raise ExtendedTrainingError("epochs must be positive.")
        if self.batch_size <= 0:
            raise ExtendedTrainingError("batch_size must be positive.")
        if self.learning_rate <= 0:
            raise ExtendedTrainingError("learning_rate must be positive.")
        if self.m1_units <= 0:
            raise ExtendedTrainingError("m1_units must be positive.")
        if len(self.hidden_units) != 2 or any(units <= 0 for units in self.hidden_units):
            raise ExtendedTrainingError("hidden_units must contain two positive widths.")
        if len(self.dropout_rates) != 2:
            raise ExtendedTrainingError("dropout_rates must contain two values.")
        if any(not 0.0 <= rate < 1.0 for rate in self.dropout_rates):
            raise ExtendedTrainingError("dropout rates must be in [0, 1).")
        if self.early_stopping_patience <= 0:
            raise ExtendedTrainingError("early_stopping_patience must be positive.")
        if self.reduce_lr_patience <= 0:
            raise ExtendedTrainingError("reduce_lr_patience must be positive.")
        if not 0.0 < self.reduce_lr_factor < 1.0:
            raise ExtendedTrainingError("reduce_lr_factor must be in (0, 1).")
        if self.min_lr <= 0:
            raise ExtendedTrainingError("min_lr must be positive.")
        if not 0.0 <= self.threshold <= 1.0:
            raise ExtendedTrainingError("threshold must be in [0, 1].")

    def progression_config(self) -> ModelProgressionTrainingConfig:
        """Return the matching configuration for progression model factories."""

        return ModelProgressionTrainingConfig(
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            activation=self.activation,
            hidden_units=self.hidden_units,
            m1_units=self.m1_units,
            dropout_rates=self.dropout_rates,
            early_stopping_patience=self.early_stopping_patience,
            reduce_lr_patience=self.reduce_lr_patience,
            reduce_lr_factor=self.reduce_lr_factor,
            min_lr=self.min_lr,
            threshold=self.threshold,
            seed=self.seed,
        )


@dataclass(frozen=True, slots=True)
class ExtendedTrainingData:
    """Processed extended data plus metadata needed by model builders."""

    feature_set: ExtendedFeatureSet
    progression_dataset: ProgressionDataset
    feature_indices: ProgressionFeatureIndices


@dataclass(frozen=True, slots=True)
class ExtendedTrainingRecord:
    """Flat result row for one model trained on extended features."""

    feature_setup: str
    model_id: str
    model_name: str
    technical_idea: str
    n_features: int
    n_params: int
    epochs: int
    best_epoch: int
    best_val_auc: float
    final_val_auc: float
    test_auc: float
    test_pr_auc: float
    test_precision: float
    test_recall: float
    test_f1: float
    test_abs_rho: float
    test_dpd: float
    test_eod: float
    history_path: str

    def to_dict(self) -> dict[str, float | int | str]:
        """Convert the record into a DataFrame-ready dictionary."""

        return {
            "feature_setup": self.feature_setup,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "technical_idea": self.technical_idea,
            "n_features": self.n_features,
            "n_params": self.n_params,
            "epochs": self.epochs,
            "best_epoch": self.best_epoch,
            "best_val_auc": self.best_val_auc,
            "final_val_auc": self.final_val_auc,
            "test_auc": self.test_auc,
            "test_pr_auc": self.test_pr_auc,
            "test_precision": self.test_precision,
            "test_recall": self.test_recall,
            "test_f1": self.test_f1,
            "test_abs_rho": self.test_abs_rho,
            "test_dpd": self.test_dpd,
            "test_eod": self.test_eod,
            "history_path": self.history_path,
        }


@dataclass(slots=True)
class ExtendedTrainingDatasetBuilder:
    """Build model-ready data from raw Home Credit rows."""

    feature_config: ExtendedFeatureSelectionConfig = field(
        default_factory=ExtendedFeatureSelectionConfig
    )

    def build(self, raw_frame: pd.DataFrame) -> ExtendedTrainingData:
        """Run extended preprocessing and adapt it to progression model inputs."""

        feature_set = ExtendedFeaturePreprocessingPipeline(
            self.feature_config
        ).fit_transform(raw_frame)
        progression_dataset = ProgressionDataset(
            X_train=feature_set.X_train,
            y_train=feature_set.y_train,
            X_val=feature_set.X_val,
            y_val=feature_set.y_val,
            X_test=feature_set.X_test,
            y_test=feature_set.y_test,
        )
        feature_indices = ProgressionFeatureIndices.from_feature_names(
            feature_set.feature_names
        )
        return ExtendedTrainingData(
            feature_set=feature_set,
            progression_dataset=progression_dataset,
            feature_indices=feature_indices,
        )


@dataclass(slots=True)
class ExtendedFeatureExperimentRunner:
    """Train selected progression models on the extended feature set."""

    training_config: ExtendedTrainingConfig = field(
        default_factory=ExtendedTrainingConfig
    )
    dataset_builder: ExtendedTrainingDatasetBuilder = field(
        default_factory=ExtendedTrainingDatasetBuilder
    )
    specs: tuple[ProgressionModelSpec, ...] = DEFAULT_PROGRESSION_SPECS

    def run(
        self,
        raw_frame: pd.DataFrame,
        *,
        class_weight: dict[int, float] | None = None,
        output_dir: str | Path | None = None,
        verbose: int = 0,
    ) -> pd.DataFrame:
        """Preprocess, train selected models and return test metrics."""

        output_path = Path(output_dir) if output_dir is not None else None
        if output_path is not None:
            output_path.mkdir(parents=True, exist_ok=True)

        prepared = self.dataset_builder.build(raw_frame)
        self._write_preprocessing_artifacts(prepared.feature_set, output_path)

        progression_config = self.training_config.progression_config()
        factory = ProgressionModelFactory(progression_config)
        selected_specs = self._select_specs()

        records: list[dict[str, float | int | str]] = []
        for spec in selected_specs:
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(self.training_config.seed)
            model = factory.build(
                spec,
                input_dim=prepared.progression_dataset.input_dim,
                feature_indices=prepared.feature_indices,
            )
            history = model.fit(
                np.asarray(prepared.progression_dataset.X_train, dtype=np.float32),
                np.asarray(prepared.progression_dataset.y_train).reshape(-1),
                validation_data=(
                    np.asarray(prepared.progression_dataset.X_val, dtype=np.float32),
                    np.asarray(prepared.progression_dataset.y_val).reshape(-1),
                ),
                epochs=self.training_config.epochs,
                batch_size=self.training_config.batch_size,
                class_weight=class_weight,
                callbacks=self._callbacks(spec),
                verbose=verbose,
            )
            history_path = self._write_history(output_path, spec, history.history)
            test_prob = model.predict(
                np.asarray(prepared.progression_dataset.X_test, dtype=np.float32),
                verbose=0,
            ).reshape(-1)
            records.append(
                self._build_record(
                    spec=spec,
                    model=model,
                    feature_set=prepared.feature_set,
                    history=history.history,
                    test_prob=test_prob,
                    history_path=history_path,
                ).to_dict()
            )

        results = pd.DataFrame.from_records(records)
        if output_path is not None:
            results.to_csv(output_path / "extended_feature_training_results.csv", index=False)
        return results

    def _select_specs(self) -> tuple[ProgressionModelSpec, ...]:
        """Return selected progression specs preserving the canonical order."""

        wanted = set(self.training_config.model_ids)
        return tuple(spec for spec in self.specs if spec.model_id in wanted)

    def _callbacks(self, spec: ProgressionModelSpec) -> list[tf.keras.callbacks.Callback]:
        """Build callbacks for one extended-feature model."""

        callbacks: list[tf.keras.callbacks.Callback] = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=self.training_config.early_stopping_patience,
                restore_best_weights=True,
                verbose=0,
            )
        ]
        if spec.use_reduce_lr:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_auc",
                    mode="max",
                    patience=self.training_config.reduce_lr_patience,
                    factor=self.training_config.reduce_lr_factor,
                    min_lr=self.training_config.min_lr,
                    verbose=0,
                )
            )
        return callbacks

    @staticmethod
    def _write_preprocessing_artifacts(
        feature_set: ExtendedFeatureSet,
        output_dir: Path | None,
    ) -> None:
        """Persist preprocessing audit tables when requested."""

        if output_dir is None:
            return
        feature_set.audit_table.to_csv(output_dir / "extended_feature_audit.csv", index=False)
        feature_set.split_report.to_csv(output_dir / "extended_split_report.csv", index=False)

    @staticmethod
    def _write_history(
        output_dir: Path | None,
        spec: ProgressionModelSpec,
        history: dict[str, list[float]],
    ) -> str:
        """Persist one training history as JSON when requested."""

        if output_dir is None:
            return ""
        path = output_dir / f"extended_{spec.model_id}_history.json"
        serializable = {
            key: [float(value) for value in values]
            for key, values in history.items()
        }
        path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        return str(path)

    def _build_record(
        self,
        *,
        spec: ProgressionModelSpec,
        model: tf.keras.Model,
        feature_set: ExtendedFeatureSet,
        history: dict[str, list[float]],
        test_prob: np.ndarray,
        history_path: str,
    ) -> ExtendedTrainingRecord:
        """Build one result row for a trained extended-feature model."""

        val_auc = np.asarray(history.get("val_auc", []), dtype=float)
        if val_auc.size == 0:
            raise ExtendedTrainingError(f"History for {spec.model_id} lacks val_auc.")

        best_index = int(np.nanargmax(val_auc))
        y_test = np.asarray(feature_set.y_test).reshape(-1).astype(int)
        s_test = np.asarray(feature_set.s_test).reshape(-1).astype(int)
        labels = (test_prob >= self.training_config.threshold).astype(int)
        fairness = fairness_metrics(
            y_test,
            test_prob,
            s_test,
            self.training_config.threshold,
        )
        return ExtendedTrainingRecord(
            feature_setup="extended",
            model_id=spec.model_id,
            model_name=spec.model_name,
            technical_idea=spec.technical_idea,
            n_features=len(feature_set.feature_names),
            n_params=int(model.count_params()),
            epochs=int(len(val_auc)),
            best_epoch=best_index + 1,
            best_val_auc=float(val_auc[best_index]),
            final_val_auc=float(val_auc[-1]),
            test_auc=float(roc_auc_score(y_test, test_prob)),
            test_pr_auc=float(average_precision_score(y_test, test_prob)),
            test_precision=float(precision_score(y_test, labels, zero_division=0)),
            test_recall=float(recall_score(y_test, labels, zero_division=0)),
            test_f1=float(f1_score(y_test, labels, zero_division=0)),
            test_abs_rho=float(absolute_pearson_correlation(test_prob, s_test)),
            test_dpd=float(fairness.demographic_parity_difference),
            test_eod=float(fairness.equalized_odds_difference),
            history_path=history_path,
        )


@dataclass(slots=True)
class CompactVsExtendedComparator:
    """Compare compact MVP results against extended-feature training results."""

    compact_model_col: str = "model_id"
    extended_model_col: str = "model_id"
    compact_auc_col: str | None = None
    extended_auc_col: str | None = None

    def compare(
        self,
        compact_results: pd.DataFrame,
        extended_results: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return an AUC comparison table between compact and extended runs."""

        if compact_results.empty:
            raise ExtendedTrainingError("compact_results cannot be empty.")
        if extended_results.empty:
            raise ExtendedTrainingError("extended_results cannot be empty.")

        compact_auc_col = self.compact_auc_col or self._detect_auc_column(
            compact_results,
            candidates=("reported_test_auc", "test_auc", "auc"),
        )
        extended_auc_col = self.extended_auc_col or self._detect_auc_column(
            extended_results,
            candidates=("test_auc", "auc", "reported_test_auc"),
        )
        self._require_columns(
            compact_results,
            (self.compact_model_col, compact_auc_col),
            "compact_results",
        )
        self._require_columns(
            extended_results,
            (self.extended_model_col, extended_auc_col),
            "extended_results",
        )

        compact = compact_results[[self.compact_model_col, compact_auc_col]].rename(
            columns={
                self.compact_model_col: "model_id",
                compact_auc_col: "compact_auc",
            }
        )
        extended = extended_results[[self.extended_model_col, extended_auc_col]].rename(
            columns={
                self.extended_model_col: "model_id",
                extended_auc_col: "extended_auc",
            }
        )
        comparison = compact.merge(extended, on="model_id", how="inner")
        if comparison.empty:
            raise ExtendedTrainingError("No shared model_id values to compare.")
        comparison["auc_gain_extended_vs_compact"] = (
            comparison["extended_auc"] - comparison["compact_auc"]
        )
        comparison["relative_auc_gain_pct"] = (
            comparison["auc_gain_extended_vs_compact"] / comparison["compact_auc"] * 100.0
        )
        return comparison.sort_values("model_id").reset_index(drop=True)

    def save_auc_figure(
        self,
        comparison: pd.DataFrame,
        output_path: str | Path,
    ) -> Path:
        """Save a compact-vs-extended AUC bar chart."""

        self._require_columns(
            comparison,
            ("model_id", "compact_auc", "extended_auc"),
            "comparison",
        )
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        x = np.arange(len(comparison))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(
            x - width / 2,
            comparison["compact_auc"],
            width,
            label="Compacto",
            color="#4C78A8",
        )
        ax.bar(
            x + width / 2,
            comparison["extended_auc"],
            width,
            label="Extendido",
            color="#59A14F",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(comparison["model_id"])
        ax.set_ylabel("AUC test")
        ax.set_title("Comparacion compacta vs extendida")
        ax.legend()
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return path

    @staticmethod
    def _detect_auc_column(
        frame: pd.DataFrame,
        *,
        candidates: tuple[str, ...],
    ) -> str:
        """Return the first known AUC column available in a result table."""

        for candidate in candidates:
            if candidate in frame.columns:
                return candidate
        raise ExtendedTrainingError(
            "Could not detect an AUC column. Tried: " + ", ".join(candidates)
        )

    @staticmethod
    def _require_columns(
        frame: pd.DataFrame,
        columns: tuple[str, ...],
        frame_name: str,
    ) -> None:
        """Validate required columns in a DataFrame."""

        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ExtendedTrainingError(
                f"{frame_name} is missing columns: " + ", ".join(missing)
            )


__all__ = [
    "CompactVsExtendedComparator",
    "ExtendedFeatureExperimentRunner",
    "ExtendedTrainingConfig",
    "ExtendedTrainingData",
    "ExtendedTrainingDatasetBuilder",
    "ExtendedTrainingError",
    "ExtendedTrainingRecord",
]
