"""Executable squared-DP FAIR experiments for the unified MVP.

The canonical Dani pipeline keeps its validated Pearson-based FAIR model. This
module adds a real, reproducible comparison inspired by Javi's formulation:

``BinaryCrossentropy + alpha * (E[pred | s=1] - E[pred | s=0]) ** 2``.

The code is deliberately isolated from ``src.dani_credit``. It lets the final
notebook run a secondary sweep without changing the main Pareto, preprocessing
or uncertainty pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

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

from src.trustworthy_credit.fairness_losses import (
    SquaredDemographicParityConfig,
    SquaredDemographicParityLoss,
    make_augmented_fair_targets,
)
from src.trustworthy_credit.metrics import (
    absolute_pearson_correlation,
    choose_threshold_youden,
    fairness_metrics,
)


ThresholdStrategy = Literal["fixed", "youden"]


class FairnessExperimentError(ValueError):
    """Raised when a squared-DP experiment is incorrectly configured."""


@dataclass(frozen=True, slots=True)
class SquaredDPSweepConfig:
    """Configuration for the executable squared-DP alpha sweep."""

    alphas: tuple[float, ...] = (0.0, 5.0, 20.0)
    epochs: int = 30
    batch_size: int = 512
    learning_rate: float = 1e-3
    activation: str = "elu"
    hidden_units: tuple[int, ...] = (128, 64)
    dropout: float = 0.20
    early_stopping_patience: int = 8
    threshold_strategy: ThresholdStrategy = "youden"
    fixed_threshold: float = 0.5
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate sweep settings before launching any training."""

        if not self.alphas:
            raise FairnessExperimentError("At least one alpha is required.")
        if any(alpha < 0.0 for alpha in self.alphas):
            raise FairnessExperimentError("alphas must be non-negative.")
        if self.epochs <= 0:
            raise FairnessExperimentError("epochs must be positive.")
        if self.batch_size <= 0:
            raise FairnessExperimentError("batch_size must be positive.")
        if self.learning_rate <= 0:
            raise FairnessExperimentError("learning_rate must be positive.")
        if not self.hidden_units or any(units <= 0 for units in self.hidden_units):
            raise FairnessExperimentError("hidden_units must contain positive values.")
        if not 0.0 <= self.dropout < 1.0:
            raise FairnessExperimentError("dropout must be in [0, 1).")
        if self.early_stopping_patience <= 0:
            raise FairnessExperimentError("early_stopping_patience must be positive.")
        if self.threshold_strategy not in {"fixed", "youden"}:
            raise FairnessExperimentError("threshold_strategy must be fixed or youden.")
        if not 0.0 <= self.fixed_threshold <= 1.0:
            raise FairnessExperimentError("fixed_threshold must be in [0, 1].")


@dataclass(frozen=True, slots=True)
class FairnessExperimentDataset:
    """Train/validation/test arrays required by the squared-DP sweep."""

    X_train: np.ndarray
    y_train: np.ndarray
    s_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    s_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    s_test: np.ndarray

    def __post_init__(self) -> None:
        """Validate shapes, binary labels and sensitive groups."""

        X_train = np.asarray(self.X_train)
        X_val = np.asarray(self.X_val)
        X_test = np.asarray(self.X_test)
        if X_train.ndim != 2 or X_val.ndim != 2 or X_test.ndim != 2:
            raise FairnessExperimentError("X arrays must be two-dimensional.")
        if X_train.shape[1] != X_val.shape[1] or X_train.shape[1] != X_test.shape[1]:
            raise FairnessExperimentError("All X arrays must have the same feature count.")

        for split_name, X_split, y_split, s_split in (
            ("train", X_train, self.y_train, self.s_train),
            ("val", X_val, self.y_val, self.s_val),
            ("test", X_test, self.y_test, self.s_test),
        ):
            y = np.asarray(y_split).reshape(-1)
            s = np.asarray(s_split).reshape(-1)
            if X_split.shape[0] != y.size or X_split.shape[0] != s.size:
                raise FairnessExperimentError(
                    f"{split_name} arrays must have the same number of rows."
                )
            self._validate_binary_array(y, f"y_{split_name}", require_both=True)
            self._validate_binary_array(s, f"s_{split_name}", require_both=True)

    @staticmethod
    def _validate_binary_array(
        values: np.ndarray,
        name: str,
        *,
        require_both: bool,
    ) -> None:
        """Validate one binary target or sensitive vector."""

        unique = set(np.unique(values).tolist())
        if not unique.issubset({0, 1, 0.0, 1.0}):
            raise FairnessExperimentError(f"{name} must contain binary values.")
        if require_both and len(unique) < 2:
            raise FairnessExperimentError(f"{name} must contain both classes/groups.")

    @property
    def input_dim(self) -> int:
        """Return the number of input features."""

        return int(np.asarray(self.X_train).shape[1])


@dataclass(frozen=True, slots=True)
class SquaredDPSweepRecord:
    """Flat result row for one alpha in the squared-DP sweep."""

    alpha: float
    epochs: int
    best_epoch: int
    best_val_loss: float
    final_val_loss: float
    threshold: float
    val_auc: float
    val_pr_auc: float
    val_abs_rho: float
    val_dp_gap: float
    val_dpd: float
    val_eod: float
    test_auc: float
    test_pr_auc: float
    test_abs_rho: float
    test_dp_gap: float
    test_dpd: float
    test_eod: float
    test_precision: float
    test_recall: float
    test_f1: float
    history_path: str

    def to_dict(self) -> dict[str, float | int | str]:
        """Convert the record to a DataFrame-ready dictionary."""

        return {
            "alpha": self.alpha,
            "epochs": self.epochs,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "final_val_loss": self.final_val_loss,
            "threshold": self.threshold,
            "val_auc": self.val_auc,
            "val_pr_auc": self.val_pr_auc,
            "val_abs_rho": self.val_abs_rho,
            "val_dp_gap": self.val_dp_gap,
            "val_dpd": self.val_dpd,
            "val_eod": self.val_eod,
            "test_auc": self.test_auc,
            "test_pr_auc": self.test_pr_auc,
            "test_abs_rho": self.test_abs_rho,
            "test_dp_gap": self.test_dp_gap,
            "test_dpd": self.test_dpd,
            "test_eod": self.test_eod,
            "test_precision": self.test_precision,
            "test_recall": self.test_recall,
            "test_f1": self.test_f1,
            "history_path": self.history_path,
        }


@dataclass(slots=True)
class SquaredDPModelFactory:
    """Build compiled MLPs for squared-DP fairness sweeps."""

    config: SquaredDPSweepConfig = field(default_factory=SquaredDPSweepConfig)

    def build(self, *, input_dim: int, alpha: float) -> tf.keras.Model:
        """Build one compiled model for a specific fairness alpha."""

        if input_dim <= 0:
            raise FairnessExperimentError("input_dim must be positive.")
        if alpha < 0.0:
            raise FairnessExperimentError("alpha must be non-negative.")

        inputs = tf.keras.Input(shape=(input_dim,), name="features")
        x = inputs
        for layer_index, units in enumerate(self.config.hidden_units):
            x = tf.keras.layers.Dense(
                units,
                activation=self.config.activation,
                name=f"dense_{layer_index}",
            )(x)
            if self.config.dropout > 0:
                x = tf.keras.layers.Dropout(
                    self.config.dropout,
                    name=f"dropout_{layer_index}",
                )(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="prob")(x)

        model = tf.keras.Model(
            inputs=inputs,
            outputs=outputs,
            name=f"squared_dp_alpha_{self._alpha_slug(alpha)}",
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss=SquaredDemographicParityLoss(
                SquaredDemographicParityConfig(alpha=alpha)
            ),
        )
        return model

    @staticmethod
    def _alpha_slug(alpha: float) -> str:
        """Return a filesystem/model-name safe alpha token."""

        return str(float(alpha)).replace(".", "p").replace("-", "m")


@dataclass(slots=True)
class SquaredDPSweepRunner:
    """Run the executable squared-Demographic-Parity alpha sweep."""

    config: SquaredDPSweepConfig = field(default_factory=SquaredDPSweepConfig)
    factory: SquaredDPModelFactory | None = None

    def run(
        self,
        dataset: FairnessExperimentDataset,
        *,
        class_weight: dict[int, float] | None = None,
        output_dir: str | Path | None = None,
        verbose: int = 0,
    ) -> pd.DataFrame:
        """Train one model per alpha and return validation/test metrics."""

        output_path = Path(output_dir) if output_dir is not None else None
        if output_path is not None:
            output_path.mkdir(parents=True, exist_ok=True)

        records: list[dict[str, float | int | str]] = []
        model_factory = self.factory or SquaredDPModelFactory(self.config)
        for alpha in self.config.alphas:
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(self.config.seed)
            model = model_factory.build(input_dim=dataset.input_dim, alpha=alpha)
            history = model.fit(
                np.asarray(dataset.X_train, dtype=np.float32),
                self._augmented_targets(dataset.y_train, dataset.s_train),
                validation_data=(
                    np.asarray(dataset.X_val, dtype=np.float32),
                    self._augmented_targets(dataset.y_val, dataset.s_val),
                ),
                sample_weight=self._sample_weight(dataset.y_train, class_weight),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=self._callbacks(),
                verbose=verbose,
            )

            history_path = self._write_history(output_path, alpha, history.history)
            val_prob = model.predict(
                np.asarray(dataset.X_val, dtype=np.float32),
                verbose=0,
            ).reshape(-1)
            test_prob = model.predict(
                np.asarray(dataset.X_test, dtype=np.float32),
                verbose=0,
            ).reshape(-1)
            threshold = self._threshold(dataset.y_val, val_prob)
            records.append(
                self._build_record(
                    alpha=alpha,
                    history=history.history,
                    threshold=threshold,
                    y_val=dataset.y_val,
                    s_val=dataset.s_val,
                    val_prob=val_prob,
                    y_test=dataset.y_test,
                    s_test=dataset.s_test,
                    test_prob=test_prob,
                    history_path=history_path,
                ).to_dict()
            )

        results = pd.DataFrame.from_records(records)
        if output_path is not None:
            results.to_csv(output_path / "squared_dp_sweep_results.csv", index=False)
        return results

    @staticmethod
    def _augmented_targets(y: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Return a NumPy augmented target matrix for Keras training."""

        return make_augmented_fair_targets(y, s).numpy()

    @staticmethod
    def _sample_weight(
        y: np.ndarray,
        class_weight: dict[int, float] | None,
    ) -> np.ndarray | None:
        """Convert optional class weights to per-row sample weights."""

        if class_weight is None:
            return None
        labels = np.asarray(y).reshape(-1).astype(int)
        missing = sorted(set(labels.tolist()).difference(class_weight))
        if missing:
            raise FairnessExperimentError(
                "class_weight is missing labels: " + ", ".join(map(str, missing))
            )
        return np.asarray([float(class_weight[int(label)]) for label in labels])

    def _callbacks(self) -> list[tf.keras.callbacks.Callback]:
        """Build callbacks for stable optional experiments."""

        return [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=0,
            )
        ]

    def _threshold(self, y_val: np.ndarray, val_prob: np.ndarray) -> float:
        """Select the binary decision threshold for reported FAIR metrics."""

        if self.config.threshold_strategy == "fixed":
            return float(self.config.fixed_threshold)
        return float(choose_threshold_youden(y_val, val_prob))

    @staticmethod
    def _write_history(
        output_dir: Path | None,
        alpha: float,
        history: dict[str, list[float]],
    ) -> str:
        """Persist one Keras history JSON when output_dir is provided."""

        if output_dir is None:
            return ""
        alpha_slug = SquaredDPModelFactory._alpha_slug(alpha)
        path = output_dir / f"squared_dp_history_alpha_{alpha_slug}.json"
        serializable = {
            key: [float(value) for value in values]
            for key, values in history.items()
        }
        path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        return str(path)

    def _build_record(
        self,
        *,
        alpha: float,
        history: dict[str, list[float]],
        threshold: float,
        y_val: np.ndarray,
        s_val: np.ndarray,
        val_prob: np.ndarray,
        y_test: np.ndarray,
        s_test: np.ndarray,
        test_prob: np.ndarray,
        history_path: str,
    ) -> SquaredDPSweepRecord:
        """Build one flat sweep record after training."""

        val_loss = np.asarray(history.get("val_loss", []), dtype=float)
        if val_loss.size == 0:
            raise FairnessExperimentError("Training history lacks val_loss.")
        best_index = int(np.nanargmin(val_loss))
        val_metrics = self._metric_bundle(y_val, s_val, val_prob, threshold)
        test_metrics = self._metric_bundle(y_test, s_test, test_prob, threshold)
        return SquaredDPSweepRecord(
            alpha=float(alpha),
            epochs=int(len(val_loss)),
            best_epoch=best_index + 1,
            best_val_loss=float(val_loss[best_index]),
            final_val_loss=float(val_loss[-1]),
            threshold=float(threshold),
            val_auc=val_metrics["auc"],
            val_pr_auc=val_metrics["pr_auc"],
            val_abs_rho=val_metrics["abs_rho"],
            val_dp_gap=val_metrics["dp_gap"],
            val_dpd=val_metrics["dpd"],
            val_eod=val_metrics["eod"],
            test_auc=test_metrics["auc"],
            test_pr_auc=test_metrics["pr_auc"],
            test_abs_rho=test_metrics["abs_rho"],
            test_dp_gap=test_metrics["dp_gap"],
            test_dpd=test_metrics["dpd"],
            test_eod=test_metrics["eod"],
            test_precision=test_metrics["precision"],
            test_recall=test_metrics["recall"],
            test_f1=test_metrics["f1"],
            history_path=history_path,
        )

    @staticmethod
    def _metric_bundle(
        y_true: np.ndarray,
        sensitive: np.ndarray,
        probability: np.ndarray,
        threshold: float,
    ) -> dict[str, float]:
        """Compute probability, binary and group FAIR metrics."""

        y = np.asarray(y_true).reshape(-1).astype(int)
        s = np.asarray(sensitive).reshape(-1).astype(int)
        p = np.asarray(probability).reshape(-1).astype(float)
        labels = (p >= threshold).astype(int)
        fairness = fairness_metrics(y, p, s, threshold)
        return {
            "auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "precision": float(precision_score(y, labels, zero_division=0)),
            "recall": float(recall_score(y, labels, zero_division=0)),
            "f1": float(f1_score(y, labels, zero_division=0)),
            "abs_rho": float(absolute_pearson_correlation(p, s)),
            "dp_gap": float(SquaredDPSweepRunner._probability_dp_gap(p, s)),
            "dpd": float(fairness.demographic_parity_difference),
            "eod": float(fairness.equalized_odds_difference),
        }

    @staticmethod
    def _probability_dp_gap(probability: np.ndarray, sensitive: np.ndarray) -> float:
        """Return ``E[p|s=1] - E[p|s=0]`` on probabilities."""

        p = np.asarray(probability).reshape(-1).astype(float)
        s = np.asarray(sensitive).reshape(-1).astype(int)
        group_1 = p[s == 1]
        group_0 = p[s == 0]
        if group_1.size == 0 or group_0.size == 0:
            raise FairnessExperimentError("Both sensitive groups are required.")
        return float(np.mean(group_1) - np.mean(group_0))


__all__ = [
    "FairnessExperimentDataset",
    "FairnessExperimentError",
    "SquaredDPModelFactory",
    "SquaredDPSweepConfig",
    "SquaredDPSweepRecord",
    "SquaredDPSweepRunner",
]
