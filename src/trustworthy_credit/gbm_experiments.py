"""Gradient boosting experiments for the relational extras notebook."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


class GBMExperimentError(ValueError):
    """Raised when a GBM experiment cannot be run or persisted safely."""


def extras_timestamp_run_id(now: datetime | None = None) -> str:
    """Return a filesystem-safe timestamp run id for extras."""

    value = now or datetime.now()
    return value.strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class GBMExperimentConfig:
    """Shared configuration for out-of-fold GBM experiments."""

    n_folds: int = 5
    seed: int = 42
    n_seeds: int = 1
    early_stopping_rounds: int = 200
    log_period: int = 200
    prediction_threshold: float = 0.5
    lightgbm_params: Mapping[str, Any] | None = None
    xgboost_params: Mapping[str, Any] | None = None

    def resolved_lightgbm_params(self) -> dict[str, Any]:
        """Return LightGBM parameters, using the teammate notebook as default."""

        if self.lightgbm_params is not None:
            return dict(self.lightgbm_params)
        return {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "n_estimators": 10000,
            "learning_rate": 0.01,
            "num_leaves": 48,
            "max_depth": 7,
            "colsample_bytree": 0.25,
            "subsample": 0.80,
            "subsample_freq": 1,
            "reg_alpha": 0.05,
            "reg_lambda": 0.10,
            "min_child_samples": 40,
            "min_child_weight": 30,
            "n_jobs": -1,
            "verbose": -1,
        }

    def resolved_xgboost_params(self) -> dict[str, Any]:
        """Return a conservative XGBoost configuration for the same feature set."""

        if self.xgboost_params is not None:
            return dict(self.xgboost_params)
        return {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "n_estimators": 5000,
            "learning_rate": 0.02,
            "max_depth": 4,
            "min_child_weight": 40,
            "subsample": 0.80,
            "colsample_bytree": 0.35,
            "reg_alpha": 0.05,
            "reg_lambda": 0.10,
            "tree_method": "hist",
            "n_jobs": -1,
        }


@dataclass(frozen=True)
class GBMArtifactPaths:
    """Filesystem contract for one isolated extras run."""

    project_root: Path
    run_id: str | None = None
    extras_dir_name: str = "results/extras"

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_root", Path(self.project_root).resolve())
        object.__setattr__(self, "run_id", self._normalize_run_id(self.run_id))
        self._validate_run_dir()

    @staticmethod
    def _normalize_run_id(run_id: str | None) -> str:
        value = run_id or extras_timestamp_run_id()
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
        cleaned = "".join(char if char in allowed else "-" for char in value.strip())
        cleaned = cleaned.strip("-_")
        if not cleaned:
            raise GBMExperimentError("run_id cannot be empty.")
        return cleaned

    @property
    def extras_root(self) -> Path:
        return self.project_root / self.extras_dir_name

    @property
    def run_dir(self) -> Path:
        return self.extras_root / str(self.run_id)

    @property
    def tables_dir(self) -> Path:
        return self.run_dir / "tables"

    @property
    def predictions_dir(self) -> Path:
        return self.run_dir / "predictions"

    @property
    def figures_dir(self) -> Path:
        return self.run_dir / "figures"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "extras_manifest.csv"

    def create(self) -> None:
        for directory in (self.tables_dir, self.predictions_dir, self.figures_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def to_project_relative(self, path: Path) -> str:
        try:
            relative = Path(path).resolve().relative_to(self.project_root)
        except ValueError:
            return Path(path).as_posix()
        return relative.as_posix()

    def _validate_run_dir(self) -> None:
        run_dir = self.run_dir.resolve()
        allowed_root = (self.project_root / self.extras_dir_name).resolve()
        try:
            run_dir.relative_to(allowed_root)
        except ValueError as exc:
            raise GBMExperimentError("Extras runs must live under results/extras/.") from exc


@dataclass(frozen=True)
class OOFResult:
    """Out-of-fold result returned by a GBM runner."""

    model_name: str
    oof_auc: float
    oof_accuracy: float
    fold_aucs: tuple[float, ...]
    oof_predictions: np.ndarray
    test_predictions: np.ndarray | None
    feature_importance: pd.DataFrame
    params: Mapping[str, Any]

    @property
    def fold_auc_mean(self) -> float:
        return float(np.mean(self.fold_aucs)) if self.fold_aucs else float("nan")

    @property
    def fold_auc_std(self) -> float:
        return float(np.std(self.fold_aucs)) if self.fold_aucs else float("nan")

    def metrics_dict(self) -> dict[str, Any]:
        """Return JSON/CSV-ready summary metrics."""

        return {
            "model_name": self.model_name,
            "oof_auc": float(self.oof_auc),
            "oof_accuracy": float(self.oof_accuracy),
            "fold_auc_mean": self.fold_auc_mean,
            "fold_auc_std": self.fold_auc_std,
            "fold_aucs": [float(value) for value in self.fold_aucs],
            "params": serializable_params(self.params),
        }


class OOFResultWriter:
    """Persist OOF metrics, predictions and feature importances."""

    def write(self, result: OOFResult, paths: GBMArtifactPaths) -> dict[str, Path]:
        paths.create()
        safe_name = safe_slug(result.model_name)
        metrics_path = paths.tables_dir / f"{safe_name}_metrics.json"
        fold_path = paths.tables_dir / f"{safe_name}_fold_metrics.csv"
        importance_path = paths.tables_dir / f"{safe_name}_feature_importance.csv"
        oof_path = paths.predictions_dir / f"{safe_name}_oof.npy"
        test_path = paths.predictions_dir / f"{safe_name}_test.npy"

        metrics_path.write_text(
            json.dumps(result.metrics_dict(), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        pd.DataFrame(
            {
                "fold_number": np.arange(1, len(result.fold_aucs) + 1),
                "fold_auc": list(result.fold_aucs),
            }
        ).to_csv(fold_path, index=False)
        result.feature_importance.to_csv(importance_path, index=False)
        np.save(oof_path, result.oof_predictions)
        written = {
            "metrics": metrics_path,
            "fold_metrics": fold_path,
            "feature_importance": importance_path,
            "oof_predictions": oof_path,
        }
        if result.test_predictions is not None:
            np.save(test_path, result.test_predictions)
            written["test_predictions"] = test_path

        return written


class FeatureImportanceReporter:
    """Create compact feature-importance summaries for the notebook."""

    def top_features(self, result: OOFResult, n: int = 25) -> pd.DataFrame:
        return result.feature_importance.head(n).copy()

    def by_source(self, result: OOFResult) -> pd.DataFrame:
        frame = result.feature_importance.copy()
        frame["source"] = frame["feature"].map(feature_source)
        return (
            frame.groupby("source", as_index=False)["importance"]
            .sum()
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


class _BaseOOFRunner:
    """Shared OOF loop for concrete GBM implementations."""

    model_name: str = "gbm"

    def __init__(
        self,
        config: GBMExperimentConfig | None = None,
        artifacts: GBMArtifactPaths | None = None,
        writer: OOFResultWriter | None = None,
    ) -> None:
        self.config = config or GBMExperimentConfig()
        self.artifacts = artifacts
        self.writer = writer or OOFResultWriter()

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_test: pd.DataFrame | None = None,
    ) -> OOFResult:
        """Train a stratified OOF model and optionally predict official test."""

        self._validate_inputs(X_train, y_train, X_test)
        target = pd.Series(y_train).reset_index(drop=True)
        X = X_train.reset_index(drop=True)
        X_official = X_test.reset_index(drop=True) if X_test is not None else None

        oof = np.zeros(len(X), dtype="float64")
        test_pred = (
            np.zeros(len(X_official), dtype="float64") if X_official is not None else None
        )
        importances = np.zeros(X.shape[1], dtype="float64")
        fold_aucs: list[float] = []
        n_total_models = self.config.n_folds * self.config.n_seeds

        for seed_number in range(self.config.n_seeds):
            seed = self.config.seed + seed_number * 101
            splitter = StratifiedKFold(
                n_splits=self.config.n_folds,
                shuffle=True,
                random_state=seed,
            )
            for fold_number, (train_idx, val_idx) in enumerate(splitter.split(X, target), start=1):
                model = self._build_model(seed)
                X_tr, y_tr = X.iloc[train_idx], target.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], target.iloc[val_idx]
                self._fit_model(model, X_tr, y_tr, X_val, y_val)

                val_pred = self._predict_proba(model, X_val)
                oof[val_idx] += val_pred / self.config.n_seeds
                if X_official is not None and test_pred is not None:
                    test_pred += self._predict_proba(model, X_official) / n_total_models
                importances += self._feature_importance(model, X.columns) / n_total_models

                auc = float(roc_auc_score(y_val, val_pred))
                fold_aucs.append(auc)
                print(
                    f"{self.model_name} seed {seed} fold {fold_number}/{self.config.n_folds} "
                    f"AUC={auc:.5f}"
                )

        result = OOFResult(
            model_name=self.model_name,
            oof_auc=float(roc_auc_score(target, oof)),
            oof_accuracy=float(
                accuracy_score(target, (oof >= self.config.prediction_threshold).astype(int))
            ),
            fold_aucs=tuple(fold_aucs),
            oof_predictions=oof,
            test_predictions=test_pred,
            feature_importance=pd.DataFrame(
                {"feature": X.columns, "importance": importances}
            ).sort_values("importance", ascending=False, ignore_index=True),
            params=self._resolved_params(),
        )
        if self.artifacts is not None:
            self.writer.write(result, self.artifacts)
        return result

    def _validate_inputs(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_test: pd.DataFrame | None,
    ) -> None:
        if self.config.n_folds < 2:
            raise GBMExperimentError("n_folds must be at least 2.")
        if self.config.n_seeds < 1:
            raise GBMExperimentError("n_seeds must be at least 1.")
        if len(X_train) != len(y_train):
            raise GBMExperimentError("X_train and y_train lengths differ.")
        if "TARGET" in X_train.columns:
            raise GBMExperimentError("TARGET cannot be present in X_train.")
        if X_test is not None and list(X_train.columns) != list(X_test.columns):
            raise GBMExperimentError("X_train and X_test columns must be aligned.")

    def _build_model(self, seed: int) -> Any:
        raise NotImplementedError

    def _fit_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        raise NotImplementedError

    def _predict_proba(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(model.predict_proba(X)[:, 1], dtype="float64")

    def _feature_importance(self, model: Any, columns: pd.Index) -> np.ndarray:
        values = getattr(model, "feature_importances_", np.zeros(len(columns)))
        return np.asarray(values, dtype="float64")

    def _resolved_params(self) -> dict[str, Any]:
        raise NotImplementedError


class LightGBMOOFRunner(_BaseOOFRunner):
    """OOF runner for LightGBM on relational features."""

    model_name = "lightgbm_relational"

    def _resolved_params(self) -> dict[str, Any]:
        return self.config.resolved_lightgbm_params()

    def _build_model(self, seed: int) -> Any:
        try:
            import lightgbm as lgb
        except ModuleNotFoundError as exc:
            raise GBMExperimentError(
                "LightGBM is not installed. Install requirements.txt to run this extra."
            ) from exc
        params = self._resolved_params()
        params["random_state"] = seed
        return lgb.LGBMClassifier(**params)

    def _fit_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        import lightgbm as lgb

        callbacks = [lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)]
        if self.config.log_period > 0:
            callbacks.append(lgb.log_evaluation(self.config.log_period))
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=callbacks,
        )


class XGBoostOOFRunner(_BaseOOFRunner):
    """OOF runner for XGBoost on relational features."""

    model_name = "xgboost_relational"

    def _resolved_params(self) -> dict[str, Any]:
        return self.config.resolved_xgboost_params()

    def _build_model(self, seed: int) -> Any:
        try:
            import xgboost as xgb
        except ModuleNotFoundError as exc:
            raise GBMExperimentError(
                "XGBoost is not installed. Install requirements.txt to run this extra."
            ) from exc
        params = self._resolved_params()
        params["random_state"] = seed
        return xgb.XGBClassifier(
            **params,
            early_stopping_rounds=self.config.early_stopping_rounds,
        )

    def _fit_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        verbose: bool | int = self.config.log_period if self.config.log_period > 0 else False
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=verbose,
        )


def write_extras_manifest(
    paths: GBMArtifactPaths,
    *,
    rows: list[Mapping[str, Any]],
) -> Path:
    """Write a compact manifest for all extras models run in one notebook."""

    paths.create()
    frame = pd.DataFrame(rows)
    frame.insert(0, "run_id", paths.run_id)
    frame.to_csv(paths.manifest_path, index=False)
    return paths.manifest_path


def serializable_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Convert model params to JSON-friendly scalars."""

    result: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        else:
            result[key] = str(value)
    return result


def safe_slug(value: str) -> str:
    """Return a filesystem-safe slug."""

    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    cleaned = "".join(char if char in allowed else "-" for char in value.strip())
    cleaned = cleaned.strip("-_").lower()
    return cleaned or "model"


def feature_source(feature_name: str) -> str:
    """Map a feature name to the table family used in the extras notebook."""

    if feature_name.startswith("BUR"):
        return "bureau"
    if feature_name.startswith("PREV"):
        return "previous_application"
    if feature_name.startswith("INST"):
        return "installments_payments"
    if feature_name.startswith("POS"):
        return "pos_cash_balance"
    if feature_name.startswith("CC"):
        return "credit_card_balance"
    return "application"


__all__ = [
    "FeatureImportanceReporter",
    "GBMArtifactPaths",
    "GBMExperimentConfig",
    "GBMExperimentError",
    "LightGBMOOFRunner",
    "OOFResult",
    "OOFResultWriter",
    "XGBoostOOFRunner",
    "extras_timestamp_run_id",
    "feature_source",
    "safe_slug",
    "write_extras_manifest",
]
