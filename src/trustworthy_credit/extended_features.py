"""Leakage-safe extended feature pipeline for advanced Home Credit experiments.

The canonical MVP uses a compact, validated feature set. This module turns the
42-feature exploratory path into a reusable object-oriented pipeline so the
team can reproduce the information-enrichment experiment without keeping the
logic hidden inside a notebook.

The pipeline is intentionally optional: it does not replace the MVP
preprocessing. Its main design constraint is leakage control: split first, fit
target encoders/imputers/scalers only on train, then transform validation and
test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ExtendedFeatureError(ValueError):
    """Raised when the extended feature pipeline cannot run safely."""


@dataclass(frozen=True, slots=True)
class ExtendedFeatureSelectionConfig:
    """Configuration for the extended application-train feature pipeline."""

    id_col: str = "SK_ID_CURR"
    target_col: str = "TARGET"
    sensitive_col: str = "CODE_GENDER"
    test_size: float = 0.15
    validation_size: float = 0.15
    random_state: int = 42
    target_encoding_smoothing: float = 20.0
    include_sensitive_as_feature: bool = False
    shuffle: bool = True

    original_numeric_cols: tuple[str, ...] = (
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
    )
    additional_numeric_cols: tuple[str, ...] = (
        "REGION_RATING_CLIENT_W_CITY",
        "DAYS_LAST_PHONE_CHANGE",
        "OWN_CAR_AGE",
        "DAYS_ID_PUBLISH",
        "REG_CITY_NOT_WORK_CITY",
        "FLAG_EMP_PHONE",
        "DAYS_REGISTRATION",
        "AMT_GOODS_PRICE",
        "DAYS_EMPLOYED",
    )
    one_hot_cols: tuple[str, ...] = (
        "NAME_FAMILY_STATUS",
        "NAME_EDUCATION_TYPE",
        "NAME_HOUSING_TYPE",
    )
    binary_categorical_cols: tuple[str, ...] = ("NAME_CONTRACT_TYPE",)
    target_encoded_cols: tuple[str, ...] = (
        "NAME_INCOME_TYPE",
        "OCCUPATION_TYPE",
        "ORGANIZATION_TYPE",
    )
    days_cols: tuple[str, ...] = (
        "DAYS_BIRTH",
        "DAYS_LAST_PHONE_CHANGE",
        "DAYS_ID_PUBLISH",
        "DAYS_REGISTRATION",
        "DAYS_EMPLOYED",
    )
    log_cols: tuple[str, ...] = (
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
    )
    missing_mask_cols: tuple[str, ...] = (
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "OWN_CAR_AGE",
        "DAYS_EMPLOYED",
        "OCCUPATION_TYPE",
    )
    protected_binary_cols: tuple[str, ...] = (
        "NAME_CONTRACT_TYPE",
        "REG_CITY_NOT_WORK_CITY",
        "FLAG_EMP_PHONE",
        "DEBT_RATIO",
        "DAYS_EMPLOYED_ANOMALY",
    )

    def __post_init__(self) -> None:
        """Validate split and smoothing settings."""

        if self.test_size <= 0.0 or self.validation_size <= 0.0:
            raise ExtendedFeatureError("test_size and validation_size must be positive.")
        if self.test_size + self.validation_size >= 1.0:
            raise ExtendedFeatureError(
                "test_size + validation_size must leave positive train data."
            )
        if self.target_encoding_smoothing < 0.0:
            raise ExtendedFeatureError("target_encoding_smoothing must be non-negative.")

    @property
    def validation_size_relative_to_trainval(self) -> float:
        """Return validation fraction after the test split is removed."""

        return self.validation_size / (1.0 - self.test_size)


@dataclass(frozen=True, slots=True)
class ExtendedFeatureSet:
    """Processed train/validation/test matrices and audit metadata."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    s_train: np.ndarray
    s_val: np.ndarray
    s_test: np.ndarray
    train_ids: tuple[Any, ...]
    val_ids: tuple[Any, ...]
    test_ids: tuple[Any, ...]
    feature_names: tuple[str, ...]
    X_train_frame: pd.DataFrame
    X_val_frame: pd.DataFrame
    X_test_frame: pd.DataFrame
    selected_numeric_features: tuple[str, ...]
    selected_one_hot_features: tuple[str, ...]
    selected_target_encoded_features: tuple[str, ...]
    missing_mask_features: tuple[str, ...]
    scale_features: tuple[str, ...]
    audit_table: pd.DataFrame
    split_report: pd.DataFrame
    target_encoding_maps: dict[str, dict[Any, float]]
    target_encoding_global_mean: float


@dataclass(slots=True)
class SmoothedTargetEncoder:
    """Train-only target encoder with additive smoothing."""

    smoothing: float = 20.0
    global_mean_: float | None = None
    mapping_: dict[Any, float] = field(default_factory=dict)

    def fit(self, categories: pd.Series, target: pd.Series) -> "SmoothedTargetEncoder":
        """Fit category means using only the provided training target."""

        if len(categories) != len(target):
            raise ExtendedFeatureError("categories and target must have the same length.")
        target_numeric = pd.Series(target, index=categories.index).astype(float)
        self.global_mean_ = float(target_numeric.mean())
        stats = (
            pd.DataFrame({"category": categories, "target": target_numeric})
            .groupby("category", dropna=False)["target"]
            .agg(["mean", "count"])
        )
        smooth = (
            stats["count"] * stats["mean"] + self.smoothing * self.global_mean_
        ) / (stats["count"] + self.smoothing)
        self.mapping_ = smooth.to_dict()
        return self

    def transform(self, categories: pd.Series) -> pd.Series:
        """Encode categories, mapping unseen values to the train global mean."""

        if self.global_mean_ is None:
            raise ExtendedFeatureError("SmoothedTargetEncoder must be fitted first.")
        return categories.map(self.mapping_).fillna(self.global_mean_).astype(float)


class ExtendedFeaturePreprocessingPipeline:
    """Build leakage-safe extended features from ``application_train.csv`` rows."""

    def __init__(self, config: ExtendedFeatureSelectionConfig | None = None) -> None:
        """Initialize the pipeline with immutable configuration."""

        self.config = config or ExtendedFeatureSelectionConfig()
        self.target_encoders_: dict[str, SmoothedTargetEncoder] = {}
        self.one_hot_categories_: dict[str, list[Any]] = {}
        self.medians_: pd.Series | None = None
        self.scaler_: StandardScaler | None = None
        self.feature_names_: tuple[str, ...] = ()
        self.scale_features_: tuple[str, ...] = ()

    def fit_transform(self, frame: pd.DataFrame) -> ExtendedFeatureSet:
        """Split, fit train-only transformers, and return processed matrices."""

        prepared = self._prepare_frame(frame)
        X, y, sensitive, ids = self._build_model_frame(prepared)
        X_train, X_val, X_test, y_train, y_val, y_test, s_train, s_val, s_test, ids_train, ids_val, ids_test = self._split(
            X, y, sensitive, ids
        )

        target_encoded_features = self._fit_transform_target_encoding(
            X_train, X_val, X_test, y_train
        )
        one_hot_features = self._fit_transform_one_hot(X_train, X_val, X_test)
        self._fit_transform_impute_log_scale(X_train, X_val, X_test)

        self.feature_names_ = tuple(X_train.columns)
        audit_table = ExtendedFeatureAuditReporter().build(
            X_train=X_train,
            y_train=y_train,
            feature_roles=self._feature_roles(
                feature_names=self.feature_names_,
                target_encoded_features=target_encoded_features,
                one_hot_features=one_hot_features,
            ),
        )
        split_report = self._split_report(
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            s_train=s_train,
            s_val=s_val,
            s_test=s_test,
        )

        return ExtendedFeatureSet(
            X_train=X_train.to_numpy(dtype=np.float32),
            X_val=X_val.to_numpy(dtype=np.float32),
            X_test=X_test.to_numpy(dtype=np.float32),
            y_train=y_train.to_numpy(dtype=np.float32),
            y_val=y_val.to_numpy(dtype=np.float32),
            y_test=y_test.to_numpy(dtype=np.float32),
            s_train=s_train.to_numpy(dtype=np.float32),
            s_val=s_val.to_numpy(dtype=np.float32),
            s_test=s_test.to_numpy(dtype=np.float32),
            train_ids=tuple(ids_train.tolist()),
            val_ids=tuple(ids_val.tolist()),
            test_ids=tuple(ids_test.tolist()),
            feature_names=self.feature_names_,
            X_train_frame=X_train.copy(),
            X_val_frame=X_val.copy(),
            X_test_frame=X_test.copy(),
            selected_numeric_features=tuple(
                column
                for column in self._candidate_numeric_columns(prepared)
                if column in self.feature_names_
            ),
            selected_one_hot_features=tuple(one_hot_features),
            selected_target_encoded_features=tuple(target_encoded_features),
            missing_mask_features=tuple(
                column for column in self.feature_names_ if column.endswith("_MISSING")
            ),
            scale_features=self.scale_features_,
            audit_table=audit_table,
            split_report=split_report,
            target_encoding_maps={
                column: dict(encoder.mapping_)
                for column, encoder in self.target_encoders_.items()
            },
            target_encoding_global_mean=float(y_train.mean()),
        )

    def _prepare_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply deterministic row-wise transformations before splitting."""

        self._require_columns(frame, (self.config.target_col, self.config.sensitive_col))
        data = frame.copy()
        data = data[data[self.config.sensitive_col] != "XNA"].copy()
        data[self.config.sensitive_col] = data[self.config.sensitive_col].map(
            {"M": 0.0, "F": 1.0}
        )
        if data[self.config.sensitive_col].isna().any():
            raise ExtendedFeatureError("CODE_GENDER must contain only M/F after filtering.")

        if "DAYS_EMPLOYED" in data.columns:
            data["DAYS_EMPLOYED_ANOMALY"] = (
                data["DAYS_EMPLOYED"] == 365243
            ).astype(float)
            data["DAYS_EMPLOYED"] = data["DAYS_EMPLOYED"].replace(365243, np.nan)

        if "DAYS_BIRTH" in data.columns:
            data["AGE_YEARS"] = data["DAYS_BIRTH"].abs() / 365.0

        for column in self.config.days_cols:
            if column in data.columns:
                data[column] = data[column].abs()

        if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(data.columns):
            income = data["AMT_INCOME_TOTAL"].replace(0, np.nan)
            data["DEBT_RATIO"] = (data["AMT_ANNUITY"] / income).clip(upper=3.0)

        for column in self.config.missing_mask_cols:
            if column in data.columns:
                data[f"{column}_MISSING"] = data[column].isna().astype(float)

        if "NAME_CONTRACT_TYPE" in data.columns:
            data["NAME_CONTRACT_TYPE"] = (
                data["NAME_CONTRACT_TYPE"] == "Cash loans"
            ).astype(float)
        return data

    def _build_model_frame(
        self,
        prepared: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Build raw X/y/s/id objects from the prepared frame."""

        feature_columns = self._feature_columns(prepared)
        if not feature_columns:
            raise ExtendedFeatureError("No extended feature columns are available.")
        y = prepared[self.config.target_col].astype(float)
        sensitive = prepared[self.config.sensitive_col].astype(float)
        if self.config.id_col in prepared.columns:
            ids = prepared[self.config.id_col]
        else:
            ids = pd.Series(prepared.index, index=prepared.index, name=self.config.id_col)
        return prepared[feature_columns].copy(), y, sensitive, ids

    def _feature_columns(self, prepared: pd.DataFrame) -> list[str]:
        """Return ordered candidate feature columns available in the DataFrame."""

        columns: list[str] = []
        if self.config.include_sensitive_as_feature:
            columns.append(self.config.sensitive_col)
        columns.extend(self._candidate_numeric_columns(prepared))
        columns.extend(column for column in self.config.binary_categorical_cols)
        columns.extend(column for column in self.config.one_hot_cols)
        columns.extend(column for column in self.config.target_encoded_cols)
        columns.extend(
            f"{column}_MISSING"
            for column in self.config.missing_mask_cols
            if f"{column}_MISSING" in prepared.columns
        )
        seen: set[str] = set()
        return [
            column
            for column in columns
            if column in prepared.columns
            and column not in {self.config.target_col, self.config.id_col}
            and not (column == self.config.sensitive_col and not self.config.include_sensitive_as_feature)
            and not (column in seen or seen.add(column))
        ]

    def _candidate_numeric_columns(self, prepared: pd.DataFrame) -> list[str]:
        """Return numeric/engineered candidates in stable order."""

        candidates = (
            self.config.original_numeric_cols
            + ("AGE_YEARS", "DEBT_RATIO")
            + self.config.additional_numeric_cols
            + ("DAYS_EMPLOYED_ANOMALY",)
        )
        return [column for column in candidates if column in prepared.columns]

    def _split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive: pd.Series,
        ids: pd.Series,
    ):
        """Create a reproducible train/validation/test split."""

        stratify = self._joint_stratification(y, sensitive)
        X_trainval, X_test, y_trainval, y_test, s_trainval, s_test, ids_trainval, ids_test = train_test_split(
            X,
            y,
            sensitive,
            ids,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            shuffle=self.config.shuffle,
            stratify=stratify if self.config.shuffle else None,
        )
        stratify_trainval = self._joint_stratification(y_trainval, s_trainval)
        X_train, X_val, y_train, y_val, s_train, s_val, ids_train, ids_val = train_test_split(
            X_trainval,
            y_trainval,
            s_trainval,
            ids_trainval,
            test_size=self.config.validation_size_relative_to_trainval,
            random_state=self.config.random_state,
            shuffle=self.config.shuffle,
            stratify=stratify_trainval if self.config.shuffle else None,
        )
        return (
            X_train.copy(),
            X_val.copy(),
            X_test.copy(),
            y_train.copy(),
            y_val.copy(),
            y_test.copy(),
            s_train.copy(),
            s_val.copy(),
            s_test.copy(),
            ids_train.copy(),
            ids_val.copy(),
            ids_test.copy(),
        )

    @staticmethod
    def _joint_stratification(y: pd.Series, sensitive: pd.Series) -> pd.Series:
        """Build a joint target-sensitive stratification label."""

        return y.astype(int).astype(str) + "_" + sensitive.astype(int).astype(str)

    def _fit_transform_target_encoding(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
    ) -> list[str]:
        """Fit target encoders on train and transform all splits."""

        encoded_features: list[str] = []
        self.target_encoders_ = {}
        for column in self.config.target_encoded_cols:
            if column not in X_train.columns:
                continue
            encoder = SmoothedTargetEncoder(
                smoothing=self.config.target_encoding_smoothing
            ).fit(X_train[column], y_train)
            self.target_encoders_[column] = encoder
            for split in (X_train, X_val, X_test):
                split[column] = encoder.transform(split[column])
            encoded_features.append(column)
        return encoded_features

    def _fit_transform_one_hot(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> list[str]:
        """Fit one-hot categories on train and transform all splits."""

        created_features: list[str] = []
        self.one_hot_categories_ = {}
        for column in self.config.one_hot_cols:
            if column not in X_train.columns:
                continue
            categories = sorted(X_train[column].dropna().astype(str).unique().tolist())
            categories_to_encode = categories[:-1]
            self.one_hot_categories_[column] = categories_to_encode
            for category in categories_to_encode:
                feature_name = self._safe_one_hot_name(column, category)
                for split in (X_train, X_val, X_test):
                    split[feature_name] = (split[column].astype(str) == category).astype(
                        float
                    )
                created_features.append(feature_name)
            for split in (X_train, X_val, X_test):
                split.drop(columns=[column], inplace=True)
        return created_features

    def _fit_transform_impute_log_scale(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> None:
        """Fit median imputation and StandardScaler on train only."""

        for split in (X_train, X_val, X_test):
            for column in split.columns:
                split[column] = pd.to_numeric(split[column], errors="coerce")

        self.medians_ = X_train.median(numeric_only=True)
        for split in (X_train, X_val, X_test):
            split.fillna(self.medians_, inplace=True)
            split.fillna(0.0, inplace=True)

        log_columns = [column for column in self.config.log_cols if column in X_train]
        for split in (X_train, X_val, X_test):
            for column in log_columns:
                split[column] = np.log1p(split[column].clip(lower=0.0))

        binary_like = set(self.config.protected_binary_cols)
        binary_like.update(
            column for column in X_train.columns if column.endswith("_MISSING")
        )
        binary_like.update(
            column
            for column in X_train.columns
            if any(column.startswith(f"{base}_") for base in self.config.one_hot_cols)
        )
        binary_like.add(self.config.sensitive_col)

        self.scale_features_ = tuple(
            column for column in X_train.columns if column not in binary_like
        )
        if self.scale_features_:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X_train.loc[:, self.scale_features_])
            for split in (X_train, X_val, X_test):
                split.loc[:, self.scale_features_] = self.scaler_.transform(
                    split.loc[:, self.scale_features_]
                )

        for split in (X_train, X_val, X_test):
            split.sort_index(axis=1, inplace=True)

    def _feature_roles(
        self,
        feature_names: tuple[str, ...],
        target_encoded_features: list[str],
        one_hot_features: list[str],
    ) -> dict[str, str]:
        """Assign human-readable feature roles for audit reporting."""

        roles: dict[str, str] = {}
        for feature in feature_names:
            if feature in target_encoded_features:
                roles[feature] = "target_encoded_categorical"
            elif feature in one_hot_features:
                roles[feature] = "one_hot_categorical"
            elif feature.endswith("_MISSING"):
                roles[feature] = "missingness_mask"
            elif feature in {"AGE_YEARS", "DEBT_RATIO", "DAYS_EMPLOYED_ANOMALY"}:
                roles[feature] = "engineered"
            elif feature in self.scale_features_:
                roles[feature] = "scaled_numeric"
            else:
                roles[feature] = "unscaled_numeric_or_binary"
        return roles

    def _split_report(
        self,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        s_train: pd.Series,
        s_val: pd.Series,
        s_test: pd.Series,
    ) -> pd.DataFrame:
        """Return split composition diagnostics."""

        records = []
        for split_name, y_values, s_values in (
            ("train", y_train, s_train),
            ("val", y_val, s_val),
            ("test", y_test, s_test),
        ):
            records.append(
                {
                    "split": split_name,
                    "n": int(len(y_values)),
                    "target_rate": float(y_values.mean()),
                    "sensitive_rate": float(s_values.mean()),
                }
            )
        return pd.DataFrame.from_records(records)

    @staticmethod
    def _safe_one_hot_name(column: str, category: Any) -> str:
        """Create stable one-hot column names from raw category values."""

        safe_category = str(category).strip().replace(" ", "_").replace("/", "_")
        return f"{column}_{safe_category}"

    @staticmethod
    def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
        """Validate required columns."""

        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ExtendedFeatureError(
                "Missing required columns: " + ", ".join(missing)
            )


@dataclass(slots=True)
class ExtendedFeatureAuditReporter:
    """Build feature-level audit tables for the extended pipeline."""

    def build(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_roles: dict[str, str],
    ) -> pd.DataFrame:
        """Return missingness, Spearman, and Cohen's d proxies per feature."""

        records = []
        y_numeric = y_train.astype(float)
        for feature in X_train.columns:
            series = pd.to_numeric(X_train[feature], errors="coerce")
            spearman = series.corr(y_numeric, method="spearman")
            cohen_d = self._cohens_d(series, y_numeric)
            records.append(
                {
                    "feature": feature,
                    "role": feature_roles.get(feature, "unknown"),
                    "missing_rate_train": float(series.isna().mean()),
                    "spearman_abs_train": float(abs(spearman))
                    if pd.notna(spearman)
                    else 0.0,
                    "cohen_d_train": cohen_d,
                }
            )
        return (
            pd.DataFrame.from_records(records)
            .sort_values(
                ["spearman_abs_train", "cohen_d_train"],
                ascending=False,
            )
            .reset_index(drop=True)
        )

    @staticmethod
    def _cohens_d(series: pd.Series, y: pd.Series) -> float:
        """Return absolute Cohen's d between TARGET=0 and TARGET=1 values."""

        values_0 = series[y == 0].dropna()
        values_1 = series[y == 1].dropna()
        if values_0.empty or values_1.empty:
            return 0.0
        pooled_std = pd.concat([values_0, values_1]).std()
        if pooled_std == 0 or pd.isna(pooled_std):
            return 0.0
        return float(abs(values_1.mean() - values_0.mean()) / pooled_std)


__all__ = [
    "ExtendedFeatureAuditReporter",
    "ExtendedFeatureError",
    "ExtendedFeaturePreprocessingPipeline",
    "ExtendedFeatureSelectionConfig",
    "ExtendedFeatureSet",
    "SmoothedTargetEncoder",
]
