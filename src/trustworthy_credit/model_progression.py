"""Executable M0-M6 model progression for the unified MVP.

The audited report already contains historical M0-M6 results. This module adds
the missing executable backing: a small object-oriented runner that can rebuild,
train and evaluate the progression without touching the validated Dani MVP.

The progression is intentionally conservative:

* M0: logistic baseline.
* M1: one hidden layer MLP.
* M2: two hidden layers MLP.
* M3: M2 plus dropout.
* M4: M3 plus a debt-ratio saturation branch.
* M5: M4 plus ReduceLROnPlateau during training.
* M6: M4 plus an external-source index branch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

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


ProgressionArchitecture = Literal[
    "logistic",
    "mlp_1",
    "mlp_2",
    "mlp_dropout",
    "debt_custom",
    "dual_custom",
]


class ModelProgressionError(ValueError):
    """Raised when the executable model progression cannot run safely."""


@tf.keras.utils.register_keras_serializable(package="TrustworthyCredit")
class DebtRatioSaturationLayer(tf.keras.layers.Layer):
    """Learn a monotonic saturation over a debt-ratio signal.

    The layer implements ``sigmoid(slope * (x - threshold))``. The slope is
    constrained to be non-negative so higher debt ratios cannot reduce the
    saturation signal.
    """

    def __init__(
        self,
        slope_init: float = 10.0,
        threshold_init: float = 0.35,
        **kwargs: Any,
    ) -> None:
        """Create the saturation layer with interpretable initial values."""

        super().__init__(**kwargs)
        self.slope_init = float(slope_init)
        self.threshold_init = float(threshold_init)

    def build(self, input_shape: tf.TensorShape | tuple[Any, ...]) -> None:
        """Create the trainable slope and threshold parameters."""

        self.slope = self.add_weight(
            name="slope",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.slope_init),
            trainable=True,
            constraint=tf.keras.constraints.NonNeg(),
        )
        self.threshold = self.add_weight(
            name="threshold",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(self.threshold_init),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Return a bounded debt-risk signal."""

        x = tf.cast(inputs, self.compute_dtype)
        return tf.keras.activations.sigmoid(self.slope * (x - self.threshold))

    def get_config(self) -> dict[str, Any]:
        """Return a serializable Keras layer configuration."""

        config = super().get_config()
        config.update(
            {
                "slope_init": self.slope_init,
                "threshold_init": self.threshold_init,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="TrustworthyCredit")
class ExtSourceIndexLayer(tf.keras.layers.Layer):
    """Learn a non-negative weighted index over EXT_SOURCE_1/2/3."""

    def build(self, input_shape: tf.TensorShape | tuple[Any, ...]) -> None:
        """Create one non-negative weight per external source plus a bias."""

        shape = tf.TensorShape(input_shape)
        input_dim = shape[-1]
        if input_dim is None:
            raise ModelProgressionError("ExtSourceIndexLayer requires known input_dim.")
        if int(input_dim) != 3:
            raise ModelProgressionError(
                "ExtSourceIndexLayer expects exactly three EXT_SOURCE columns."
            )

        self.source_weights = self.add_weight(
            name="source_weights",
            shape=(3,),
            initializer=tf.keras.initializers.Constant(1.0 / 3.0),
            trainable=True,
            constraint=tf.keras.constraints.NonNeg(),
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Return a bounded external-creditworthiness index."""

        x = tf.cast(inputs, self.compute_dtype)
        weighted_sum = tf.reduce_sum(x * self.source_weights, axis=-1, keepdims=True)
        return tf.keras.activations.sigmoid(weighted_sum + self.bias)


@dataclass(frozen=True, slots=True)
class ProgressionFeatureIndices:
    """Column positions required by the custom M4-M6 branches."""

    debt_ratio_idx: int | None = None
    ext_source_idxs: tuple[int, int, int] | None = None

    @classmethod
    def from_feature_names(cls, feature_names: list[str] | tuple[str, ...]) -> "ProgressionFeatureIndices":
        """Resolve required custom-branch indices from ordered feature names."""

        names = tuple(feature_names)
        debt_ratio_idx = names.index("DEBT_RATIO") if "DEBT_RATIO" in names else None
        ext_names = ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")
        if all(name in names for name in ext_names):
            ext_source_idxs = tuple(names.index(name) for name in ext_names)
        else:
            ext_source_idxs = None
        return cls(
            debt_ratio_idx=debt_ratio_idx,
            ext_source_idxs=ext_source_idxs,  # type: ignore[arg-type]
        )

    def require_debt_ratio(self) -> int:
        """Return the debt-ratio index or raise a clear configuration error."""

        if self.debt_ratio_idx is None:
            raise ModelProgressionError(
                "M4/M5/M6 require a DEBT_RATIO column. "
                "Run them on a feature matrix that includes DEBT_RATIO."
            )
        return self.debt_ratio_idx

    def require_ext_sources(self) -> tuple[int, int, int]:
        """Return the external-source indices or raise a clear error."""

        if self.ext_source_idxs is None:
            raise ModelProgressionError(
                "M6 requires EXT_SOURCE_1, EXT_SOURCE_2 and EXT_SOURCE_3."
            )
        return self.ext_source_idxs


@dataclass(frozen=True, slots=True)
class ProgressionModelSpec:
    """Definition of one architecture in the executable M0-M6 progression."""

    model_id: str
    model_name: str
    technical_idea: str
    architecture: ProgressionArchitecture
    use_reduce_lr: bool = False


DEFAULT_PROGRESSION_SPECS: tuple[ProgressionModelSpec, ...] = (
    ProgressionModelSpec(
        model_id="M0",
        model_name="Logistic baseline",
        technical_idea="Linear floor for the predictive task.",
        architecture="logistic",
    ),
    ProgressionModelSpec(
        model_id="M1",
        model_name="One-layer MLP",
        technical_idea="First non-linear interactions over tabular features.",
        architecture="mlp_1",
    ),
    ProgressionModelSpec(
        model_id="M2",
        model_name="Two-layer MLP",
        technical_idea="More capacity for piecewise non-linear effects.",
        architecture="mlp_2",
    ),
    ProgressionModelSpec(
        model_id="M3",
        model_name="Two-layer MLP plus dropout",
        technical_idea="Regularized dense baseline.",
        architecture="mlp_dropout",
    ),
    ProgressionModelSpec(
        model_id="M4",
        model_name="Debt-ratio custom branch",
        technical_idea="Adds a monotonic saturation over debt ratio.",
        architecture="debt_custom",
    ),
    ProgressionModelSpec(
        model_id="M5",
        model_name="Debt-ratio custom branch plus scheduler",
        technical_idea="Keeps M4 architecture and adds ReduceLROnPlateau.",
        architecture="debt_custom",
        use_reduce_lr=True,
    ),
    ProgressionModelSpec(
        model_id="M6",
        model_name="Dual custom branches",
        technical_idea="Adds debt-ratio and external-source audit branches.",
        architecture="dual_custom",
        use_reduce_lr=True,
    ),
)


@dataclass(frozen=True, slots=True)
class ModelProgressionTrainingConfig:
    """Training configuration shared by all executable progression models."""

    epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 1e-3
    activation: str = "relu"
    hidden_units: tuple[int, int] = (128, 64)
    m1_units: int = 64
    dropout_rates: tuple[float, float] = (0.30, 0.20)
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6
    threshold: float = 0.5
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate settings before any model is built."""

        if self.epochs <= 0:
            raise ModelProgressionError("epochs must be positive.")
        if self.batch_size <= 0:
            raise ModelProgressionError("batch_size must be positive.")
        if self.learning_rate <= 0:
            raise ModelProgressionError("learning_rate must be positive.")
        if self.m1_units <= 0:
            raise ModelProgressionError("m1_units must be positive.")
        if len(self.hidden_units) != 2 or any(units <= 0 for units in self.hidden_units):
            raise ModelProgressionError("hidden_units must contain two positive widths.")
        if len(self.dropout_rates) != 2:
            raise ModelProgressionError("dropout_rates must contain two values.")
        if any(not 0.0 <= rate < 1.0 for rate in self.dropout_rates):
            raise ModelProgressionError("dropout rates must be in [0, 1).")
        if self.early_stopping_patience <= 0:
            raise ModelProgressionError("early_stopping_patience must be positive.")
        if self.reduce_lr_patience <= 0:
            raise ModelProgressionError("reduce_lr_patience must be positive.")
        if not 0.0 < self.reduce_lr_factor < 1.0:
            raise ModelProgressionError("reduce_lr_factor must be in (0, 1).")
        if self.min_lr <= 0:
            raise ModelProgressionError("min_lr must be positive.")
        if not 0.0 <= self.threshold <= 1.0:
            raise ModelProgressionError("threshold must be in [0, 1].")


@dataclass(frozen=True, slots=True)
class ProgressionDataset:
    """Train/validation/test arrays consumed by the progression runner."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    def __post_init__(self) -> None:
        """Validate array shapes and binary labels."""

        X_train = np.asarray(self.X_train)
        X_val = np.asarray(self.X_val)
        X_test = np.asarray(self.X_test)
        y_train = np.asarray(self.y_train).reshape(-1)
        y_val = np.asarray(self.y_val).reshape(-1)
        y_test = np.asarray(self.y_test).reshape(-1)

        if X_train.ndim != 2 or X_val.ndim != 2 or X_test.ndim != 2:
            raise ModelProgressionError("X arrays must be two-dimensional.")
        if X_train.shape[1] != X_val.shape[1] or X_train.shape[1] != X_test.shape[1]:
            raise ModelProgressionError("All X arrays must have the same feature count.")
        if X_train.shape[0] != y_train.size:
            raise ModelProgressionError("X_train and y_train lengths differ.")
        if X_val.shape[0] != y_val.size:
            raise ModelProgressionError("X_val and y_val lengths differ.")
        if X_test.shape[0] != y_test.size:
            raise ModelProgressionError("X_test and y_test lengths differ.")
        for name, y_values in {
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }.items():
            unique = set(np.unique(y_values).tolist())
            if not unique.issubset({0, 1, 0.0, 1.0}):
                raise ModelProgressionError(f"{name} must contain binary labels.")
            if len(unique) < 2:
                raise ModelProgressionError(f"{name} must contain both classes.")

    @property
    def input_dim(self) -> int:
        """Return the number of input features."""

        return int(np.asarray(self.X_train).shape[1])


@dataclass(frozen=True, slots=True)
class ProgressionRunRecord:
    """Flat record returned after training and evaluating one progression model."""

    model_id: str
    model_name: str
    technical_idea: str
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
    history_path: str

    def to_dict(self) -> dict[str, float | int | str]:
        """Convert the record to a DataFrame-ready dictionary."""

        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "technical_idea": self.technical_idea,
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
            "history_path": self.history_path,
        }


@dataclass(slots=True)
class ProgressionModelFactory:
    """Build compiled Keras models for the M0-M6 progression."""

    config: ModelProgressionTrainingConfig = field(
        default_factory=ModelProgressionTrainingConfig
    )

    def build(
        self,
        spec: ProgressionModelSpec,
        *,
        input_dim: int,
        feature_indices: ProgressionFeatureIndices | None = None,
    ) -> tf.keras.Model:
        """Build and compile the model described by ``spec``."""

        if input_dim <= 0:
            raise ModelProgressionError("input_dim must be positive.")

        tf.keras.utils.set_random_seed(self.config.seed)
        inputs = tf.keras.Input(shape=(input_dim,), name="features")

        if spec.architecture == "logistic":
            outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="prob")(inputs)
        elif spec.architecture == "mlp_1":
            x = tf.keras.layers.Dense(
                self.config.m1_units,
                activation=self.config.activation,
                name="dense_1",
            )(inputs)
            outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="prob")(x)
        elif spec.architecture == "mlp_2":
            x = self._dense_backbone(inputs, with_dropout=False)
            outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="prob")(x)
        elif spec.architecture == "mlp_dropout":
            x = self._dense_backbone(inputs, with_dropout=True)
            outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="prob")(x)
        elif spec.architecture == "debt_custom":
            indices = feature_indices or ProgressionFeatureIndices()
            debt_signal = self._debt_branch(inputs, indices)
            x = self._dense_backbone(inputs, with_dropout=True)
            combined = tf.keras.layers.Concatenate(name="concat_dense_debt")(
                [x, debt_signal]
            )
            outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="prob")(
                combined
            )
        elif spec.architecture == "dual_custom":
            indices = feature_indices or ProgressionFeatureIndices()
            debt_signal = self._debt_branch(inputs, indices)
            ext_signal = self._ext_source_branch(inputs, indices)
            x = self._dense_backbone(inputs, with_dropout=True)
            combined = tf.keras.layers.Concatenate(name="concat_dense_custom")(
                [x, debt_signal, ext_signal]
            )
            outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="prob")(
                combined
            )
        else:  # pragma: no cover - safeguarded by Literal typing.
            raise ModelProgressionError(f"Unsupported architecture: {spec.architecture}")

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=spec.model_id)
        self._compile(model)
        return model

    def _dense_backbone(self, inputs: tf.Tensor, *, with_dropout: bool) -> tf.Tensor:
        """Build the shared two-layer dense backbone."""

        first_units, second_units = self.config.hidden_units
        first_dropout, second_dropout = self.config.dropout_rates
        x = tf.keras.layers.Dense(
            first_units,
            activation=self.config.activation,
            name="dense_1",
        )(inputs)
        if with_dropout:
            x = tf.keras.layers.Dropout(first_dropout, name="dropout_1")(x)
        x = tf.keras.layers.Dense(
            second_units,
            activation=self.config.activation,
            name="dense_2",
        )(x)
        if with_dropout:
            x = tf.keras.layers.Dropout(second_dropout, name="dropout_2")(x)
        return x

    @staticmethod
    def _debt_branch(
        inputs: tf.Tensor,
        feature_indices: ProgressionFeatureIndices,
    ) -> tf.Tensor:
        """Build the debt-ratio custom branch."""

        debt_idx = feature_indices.require_debt_ratio()
        debt_col = tf.keras.layers.Lambda(
            lambda tensor: tensor[:, debt_idx : debt_idx + 1],
            output_shape=(1,),
            name="extract_debt_ratio",
        )(inputs)
        return DebtRatioSaturationLayer(name="debt_saturation")(debt_col)

    @staticmethod
    def _ext_source_branch(
        inputs: tf.Tensor,
        feature_indices: ProgressionFeatureIndices,
    ) -> tf.Tensor:
        """Build the EXT_SOURCE custom branch."""

        ext_idxs = feature_indices.require_ext_sources()
        ext_cols = tf.keras.layers.Lambda(
            lambda tensor: tf.gather(tensor, list(ext_idxs), axis=1),
            output_shape=(3,),
            name="extract_ext_sources",
        )(inputs)
        return ExtSourceIndexLayer(name="ext_source_index")(ext_cols)

    def _compile(self, model: tf.keras.Model) -> None:
        """Compile a progression model with common instrumentation."""

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
                tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )


@dataclass(slots=True)
class ModelProgressionRunner:
    """Train and evaluate the executable M0-M6 progression."""

    config: ModelProgressionTrainingConfig = field(
        default_factory=ModelProgressionTrainingConfig
    )
    specs: tuple[ProgressionModelSpec, ...] = DEFAULT_PROGRESSION_SPECS
    factory: ProgressionModelFactory | None = None

    def run(
        self,
        dataset: ProgressionDataset,
        *,
        feature_indices: ProgressionFeatureIndices | None = None,
        model_ids: tuple[str, ...] | None = None,
        class_weight: dict[int, float] | None = None,
        output_dir: str | Path | None = None,
        verbose: int = 0,
    ) -> pd.DataFrame:
        """Train selected progression models and return one row per model."""

        selected_specs = self._select_specs(model_ids)
        output_path = Path(output_dir) if output_dir is not None else None
        if output_path is not None:
            output_path.mkdir(parents=True, exist_ok=True)

        records: list[dict[str, float | int | str]] = []
        model_factory = self.factory or ProgressionModelFactory(self.config)
        for spec in selected_specs:
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(self.config.seed)
            model = model_factory.build(
                spec,
                input_dim=dataset.input_dim,
                feature_indices=feature_indices,
            )
            callbacks = self._callbacks(spec)
            history = model.fit(
                np.asarray(dataset.X_train, dtype=np.float32),
                np.asarray(dataset.y_train).reshape(-1),
                validation_data=(
                    np.asarray(dataset.X_val, dtype=np.float32),
                    np.asarray(dataset.y_val).reshape(-1),
                ),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                class_weight=class_weight,
                callbacks=callbacks,
                verbose=verbose,
            )
            history_path = self._write_history(output_path, spec, history.history)
            y_prob = model.predict(
                np.asarray(dataset.X_test, dtype=np.float32),
                verbose=0,
            ).reshape(-1)
            records.append(
                self._build_record(
                    spec=spec,
                    model=model,
                    history=history.history,
                    y_true=np.asarray(dataset.y_test).reshape(-1),
                    y_prob=y_prob,
                    history_path=history_path,
                ).to_dict()
            )

        return pd.DataFrame.from_records(records)

    def _select_specs(
        self,
        model_ids: tuple[str, ...] | None,
    ) -> tuple[ProgressionModelSpec, ...]:
        """Return specs requested by ``model_ids`` preserving configured order."""

        if model_ids is None:
            return self.specs
        known = {spec.model_id for spec in self.specs}
        missing = sorted(set(model_ids).difference(known))
        if missing:
            raise ModelProgressionError(
                "Unknown progression model ids: " + ", ".join(missing)
            )
        wanted = set(model_ids)
        return tuple(spec for spec in self.specs if spec.model_id in wanted)

    def _callbacks(self, spec: ProgressionModelSpec) -> list[tf.keras.callbacks.Callback]:
        """Build callbacks for one model."""

        callbacks: list[tf.keras.callbacks.Callback] = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=0,
            )
        ]
        if spec.use_reduce_lr:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_auc",
                    mode="max",
                    patience=self.config.reduce_lr_patience,
                    factor=self.config.reduce_lr_factor,
                    min_lr=self.config.min_lr,
                    verbose=0,
                )
            )
        return callbacks

    @staticmethod
    def _write_history(
        output_dir: Path | None,
        spec: ProgressionModelSpec,
        history: dict[str, list[float]],
    ) -> str:
        """Persist one Keras history JSON when an output directory is provided."""

        if output_dir is None:
            return ""
        path = output_dir / f"{spec.model_id}_history.json"
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
        history: dict[str, list[float]],
        y_true: np.ndarray,
        y_prob: np.ndarray,
        history_path: str,
    ) -> ProgressionRunRecord:
        """Evaluate one trained model and build a flat result record."""

        val_auc = np.asarray(history.get("val_auc", []), dtype=float)
        if val_auc.size == 0:
            raise ModelProgressionError(f"Training history for {spec.model_id} lacks val_auc.")

        best_idx = int(np.nanargmax(val_auc))
        y_pred = (y_prob >= self.config.threshold).astype(int)
        return ProgressionRunRecord(
            model_id=spec.model_id,
            model_name=spec.model_name,
            technical_idea=spec.technical_idea,
            n_params=int(model.count_params()),
            epochs=int(len(val_auc)),
            best_epoch=best_idx + 1,
            best_val_auc=float(val_auc[best_idx]),
            final_val_auc=float(val_auc[-1]),
            test_auc=float(roc_auc_score(y_true, y_prob)),
            test_pr_auc=float(average_precision_score(y_true, y_prob)),
            test_precision=float(
                precision_score(y_true, y_pred, zero_division=0)
            ),
            test_recall=float(recall_score(y_true, y_pred, zero_division=0)),
            test_f1=float(f1_score(y_true, y_pred, zero_division=0)),
            history_path=history_path,
        )


__all__ = [
    "DEFAULT_PROGRESSION_SPECS",
    "DebtRatioSaturationLayer",
    "ExtSourceIndexLayer",
    "ModelProgressionError",
    "ModelProgressionRunner",
    "ModelProgressionTrainingConfig",
    "ProgressionDataset",
    "ProgressionFeatureIndices",
    "ProgressionModelFactory",
    "ProgressionModelSpec",
    "ProgressionRunRecord",
]
