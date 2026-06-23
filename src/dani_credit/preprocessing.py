"""Leakage-safe preprocessing for the Home Credit MVP.

This module implements Block 2 of the project. It converts raw
`application_train.csv` rows into deterministic features and provides a
statistical preprocessing class that must be fitted only on the train split.

The module intentionally does not create train/validation/test splits. Splitting
is Block 3's responsibility. This design prevents hidden leakage by forcing the
caller to split before fitting imputers, scalers and encoders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from .data_contract import (
    DataContractError,
    DataContractValidator,
    HomeCreditMVPDataContract,
    build_default_home_credit_contract,
)


class PreprocessingError(ValueError):
    """Raised when preprocessing cannot be completed safely.

    Args:
        message: Human-readable explanation of the preprocessing failure.

    Returns:
        None.

    Raises:
        This exception is raised by preprocessing classes when the input data
        violates assumptions that are stricter than the raw data contract.
    """


@dataclass(frozen=True)
class DeterministicDataset:
    """Container returned after deterministic transformations.

    Args:
        features: Feature DataFrame after deterministic transformations and
            after removing target and sensitive columns from the model inputs.
        target: Binary target series aligned with `features`.
        sensitive: Numeric sensitive series aligned with `features`.

    Returns:
        Immutable container with aligned `X`, `y` and `s`.

    Raises:
        None.
    """

    # Features remain a DataFrame at this stage because we still need column
    # names and indices for splitting, tracing and later ColumnTransformer logic.
    features: pd.DataFrame

    # Target remains a Series so the index stays aligned with `features`.
    target: pd.Series

    # Sensitive remains separate from features by design; it is never included
    # in X as a normal predictor.
    sensitive: pd.Series


@dataclass(frozen=True)
class RawSplitDataset:
    """Raw train/validation/test split before statistical preprocessing.

    Args:
        X_train: Raw train features after deterministic transformations.
        X_val: Raw validation features after deterministic transformations.
        X_test: Raw test features after deterministic transformations.
        y_train: Train target aligned with `X_train`.
        y_val: Validation target aligned with `X_val`.
        y_test: Test target aligned with `X_test`.
        s_train: Train sensitive values aligned with `X_train`.
        s_val: Validation sensitive values aligned with `X_val`.
        s_test: Test sensitive values aligned with `X_test`.

    Returns:
        Immutable split container used by `HomeCreditFeaturePreprocessor`.

    Raises:
        None.
    """

    # The X objects intentionally stay as DataFrames until the statistical
    # preprocessor transforms them into numeric arrays.
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame

    # y and s stay separate to keep the modelling API clean in later blocks.
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    s_train: pd.Series
    s_val: pd.Series
    s_test: pd.Series


@dataclass(frozen=True)
class PreprocessingColumnSpec:
    """Column groups used by the statistical preprocessing stage.

    Args:
        financial_cols: Monetary columns imputed but not scaled.
        continuous_scaled_cols: Numeric columns imputed and robust-scaled.
        binary_cols: Binary/flag columns imputed but not scaled.
        categorical_cols: Low-cardinality categorical columns one-hot encoded.

    Returns:
        Immutable grouping used to build the sklearn `ColumnTransformer`.

    Raises:
        None.
    """

    # Monetary columns are intentionally left in original scale after median
    # imputation so Block 5 can compute interpretable ratios.
    financial_cols: tuple[str, ...]

    # Continuous non-financial columns are scaled because dense neural networks
    # train more stably when most numeric inputs have comparable ranges.
    continuous_scaled_cols: tuple[str, ...]

    # Binary flags are left as 0/1 to preserve their meaning.
    binary_cols: tuple[str, ...]

    # Categorical columns are handled with one-hot encoding because the selected
    # MVP categoricals have manageable cardinality.
    categorical_cols: tuple[str, ...]

    def all_columns(self) -> tuple[str, ...]:
        """Return all columns expected by the statistical preprocessor.

        Args:
            None.

        Returns:
            Ordered tuple with all preprocessing columns.

        Raises:
            None.
        """

        # The order mirrors the ColumnTransformer order and later feature_names.
        return (
            self.financial_cols
            + self.continuous_scaled_cols
            + self.binary_cols
            + self.categorical_cols
        )


@dataclass(frozen=True)
class ProcessedSplitDataset:
    """Numeric train/validation/test data ready for Keras.

    Args:
        X_train: Processed train matrix as float32.
        X_val: Processed validation matrix as float32.
        X_test: Processed test matrix as float32.
        y_train: Train target as float32.
        y_val: Validation target as float32.
        y_test: Test target as float32.
        s_train: Train sensitive values as float32.
        s_val: Validation sensitive values as float32.
        s_test: Test sensitive values as float32.
        train_ids: Original `SK_ID_CURR` values for train rows.
        val_ids: Original `SK_ID_CURR` values for validation rows.
        test_ids: Original `SK_ID_CURR` values for test rows.
        feature_names: Names of the columns in the processed matrices.
        preprocessor: Fitted sklearn `ColumnTransformer`.

    Returns:
        Immutable container with model-ready arrays and metadata.

    Raises:
        None.
    """

    # Keras will consume these float32 matrices in later blocks.
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray

    # Targets are float32 because Keras losses expect numeric tensors.
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

    # Sensitive arrays are also float32 because the FAIR penalty later uses
    # TensorFlow arithmetic with them.
    s_train: np.ndarray
    s_val: np.ndarray
    s_test: np.ndarray

    # IDs are preserved for reproducibility, saved predictions and split audits.
    train_ids: tuple[Any, ...]
    val_ids: tuple[Any, ...]
    test_ids: tuple[Any, ...]

    # Feature names are essential for Block 5 ratio indices.
    feature_names: tuple[str, ...]

    # The fitted preprocessor is kept so validation/test and future inference
    # use exactly the train-fitted transformations.
    preprocessor: ColumnTransformer


class HomeCreditRawDataLoader:
    """Load raw Home Credit MVP columns from `application_train.csv`.

    Args:
        contract: Data contract defining the expected file and columns.
        validator: Optional validator. If omitted, one is created from the
            contract.

    Returns:
        Loader object able to read the MVP raw DataFrame.

    Raises:
        None during initialization.
    """

    def __init__(
        self,
        contract: HomeCreditMVPDataContract,
        validator: DataContractValidator | None = None,
    ) -> None:
        """Initialize the raw data loader.

        Args:
            contract: Contract with the raw path and required columns.
            validator: Optional validator bound to the same contract.

        Returns:
            None.

        Raises:
            None.
        """

        # The contract is injected so the loader never hardcodes column names.
        self._contract = contract

        # A validator can be injected in tests; otherwise we build the default
        # validator for this contract.
        self._validator = validator or DataContractValidator(contract)

    def load(self, path: str | Path | None = None) -> pd.DataFrame:
        """Read the MVP raw columns from CSV.

        Args:
            path: Optional explicit CSV path. When omitted, the contract's
                `training_file_path()` is used.

        Returns:
            DataFrame containing only the MVP required raw columns.

        Raises:
            DataContractError: If the expected file does not exist.
            DataContractError: If required columns are missing after reading.
        """

        # Use the contract path by default so notebooks do not duplicate it.
        csv_path = Path(path) if path is not None else self._contract.training_file_path()

        # Fail early with a project-specific error if the CSV has not been
        # downloaded yet.
        if not csv_path.exists():
            raise DataContractError(f"Missing MVP training file at {csv_path}.")

        # Read only required columns to reduce memory and enforce the MVP scope.
        df = pd.read_csv(csv_path, usecols=self._contract.required_raw_columns())

        # Validate the DataFrame columns even after using `usecols`, because this
        # gives a clearer error if a Kaggle file or path is wrong.
        self._validator.assert_columns(df.columns)

        return df


class HomeCreditDeterministicTransformer:
    """Apply deterministic, leakage-free transformations before splitting.

    Args:
        contract: Data contract that defines target, sensitive and feature
            columns.

    Returns:
        Transformer object that converts a raw DataFrame into aligned `X`, `y`
        and `s`.

    Raises:
        None during initialization.
    """

    # Home Credit uses this sentinel for anomalous employment duration.
    DAYS_EMPLOYED_SENTINEL: int = 365243

    def __init__(self, contract: HomeCreditMVPDataContract) -> None:
        """Initialize the deterministic transformer.

        Args:
            contract: MVP data contract.

        Returns:
            None.

        Raises:
            None.
        """

        # The contract gives us the names of target, sensitive, external scores
        # and financial columns without duplicating strings.
        self._contract = contract

        # This validator checks that raw inputs satisfy Block 1 before we mutate
        # them in Block 2.
        self._validator = DataContractValidator(contract)

    def transform(self, raw_df: pd.DataFrame) -> DeterministicDataset:
        """Transform raw MVP rows into deterministic features, target and sensitive.

        Args:
            raw_df: Raw DataFrame containing the required MVP columns.

        Returns:
            DeterministicDataset with `features`, `target` and `sensitive`.

        Raises:
            DataContractError: If required raw columns are missing.
            PreprocessingError: If binary mappings or financial assumptions fail.
        """

        # Validate first so later column access errors become clear contract
        # errors instead of generic KeyErrors.
        self._validator.assert_columns(raw_df.columns)

        # Work on a copy to avoid mutating a DataFrame that may be reused by a
        # notebook cell or another pipeline component.
        df = raw_df.copy()

        # Keep only binary gender values defined by the assignment. XNA rows are
        # very rare in Home Credit and would complicate a binary sensitive setup.
        df = self._filter_supported_gender_values(df)

        # Create the numeric sensitive column before removing CODE_GENDER from X.
        df = self._add_sensitive_column(df)

        # Preserve SK_ID_CURR as the index so splits and predictions can later be
        # traced back to original clients.
        df = self._set_identifier_index(df)

        # Missingness flags must be created before any imputation; otherwise the
        # information that a score was missing would be lost.
        df = self._add_external_source_missing_flags(df)

        # DAYS_EMPLOYED has a known sentinel that must become NaN before years
        # are computed.
        df = self._handle_days_employed_anomaly(df)

        # Convert negative day counts into interpretable positive year values.
        df = self._add_positive_year_columns(df)

        # Map simple Y/N car ownership to a numeric binary flag.
        df = self._map_binary_columns(df)

        # Ensure financial columns are non-negative before they later feed ratio
        # calculations in Block 5.
        self._validate_financial_non_negative(df)

        # Remove raw temporal columns after creating AGE_YEARS and EMPLOYED_YEARS.
        df = df.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED"])

        # Separate X, y and s so sensitive information does not become a normal
        # model feature.
        return self._split_features_target_sensitive(df)

    def _filter_supported_gender_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter rows to supported binary gender values.

        Args:
            df: Raw DataFrame with `CODE_GENDER`.

        Returns:
            Filtered DataFrame containing only F and M in `CODE_GENDER`.

        Raises:
            PreprocessingError: If no rows remain after filtering.
        """

        # The assignment frames fairness around CODE_GENDER as a binary
        # sensitive variable; unsupported values are removed before mapping.
        filtered = df[df[self._contract.sensitive_column].isin(["F", "M"])].copy()

        # A completely empty frame would indicate the wrong file or corrupted
        # data, so we fail loudly.
        if filtered.empty:
            raise PreprocessingError("No rows remain after filtering CODE_GENDER to F/M.")

        return filtered

    def _add_sensitive_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create numeric `SENSITIVE` from raw `CODE_GENDER`.

        Args:
            df: DataFrame filtered to F/M gender values.

        Returns:
            DataFrame with the engineered sensitive column added.

        Raises:
            PreprocessingError: If an unmapped gender value remains.
        """

        # The map is deliberately explicit so the encoding is transparent in the
        # defence and consistent across all later blocks.
        mapping = {"F": 0.0, "M": 1.0}

        # Create a copy because pandas chained assignment is easy to trigger
        # after filtering operations.
        transformed = df.copy()

        # Convert to float32 now because TensorFlow later uses the sensitive
        # tensor in arithmetic operations.
        transformed[self._contract.engineered_sensitive_column] = (
            transformed[self._contract.sensitive_column].map(mapping).astype("float32")
        )

        # If any value could not be mapped, pandas would create NaN; that must
        # never silently continue.
        if transformed[self._contract.engineered_sensitive_column].isna().any():
            raise PreprocessingError("Unmapped CODE_GENDER value found.")

        return transformed

    def _set_identifier_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set `SK_ID_CURR` as index and remove it from columns.

        Args:
            df: DataFrame containing the identifier column.

        Returns:
            DataFrame indexed by `SK_ID_CURR`.

        Raises:
            PreprocessingError: If duplicate identifiers are detected.
        """

        # Duplicate loan IDs would break traceability of splits and predictions.
        if df[self._contract.identifier_column].duplicated().any():
            raise PreprocessingError("Duplicate SK_ID_CURR values found.")

        # The index keeps the ID attached to each row without exposing it as a
        # model feature.
        return df.set_index(self._contract.identifier_column, drop=True)

    def _add_external_source_missing_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create missing indicators for EXT_SOURCE columns.

        Args:
            df: DataFrame containing EXT_SOURCE columns.

        Returns:
            DataFrame with EXT_SOURCE missing flags and EXT_NULL_COUNT.

        Raises:
            None.
        """

        # Work on a copy to preserve method purity for callers.
        transformed = df.copy()

        # The contract centralizes which EXT_SOURCE columns are part of the MVP.
        ext_cols = self._contract.external_score_columns()

        # Each missing flag records whether that individual external score was
        # absent before imputation.
        for col in ext_cols:
            transformed[f"{col}_WAS_MISSING"] = transformed[col].isna().astype("int8")

        # EXT_NULL_COUNT is a compact summary used later to test whether missing
        # external data is associated with higher uncertainty.
        transformed["EXT_NULL_COUNT"] = transformed[list(ext_cols)].isna().sum(axis=1).astype(
            "int8"
        )

        return transformed

    def _handle_days_employed_anomaly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace anomalous DAYS_EMPLOYED sentinel with missing value.

        Args:
            df: DataFrame containing `DAYS_EMPLOYED`.

        Returns:
            DataFrame with `DAYS_EMPLOYED_ANOM` and sentinel replaced by NaN.

        Raises:
            None.
        """

        # Copy so the sentinel replacement does not mutate external references.
        transformed = df.copy()

        # The anomaly flag can carry information, so we keep it instead of only
        # deleting the sentinel.
        transformed["DAYS_EMPLOYED_ANOM"] = (
            transformed["DAYS_EMPLOYED"] == self.DAYS_EMPLOYED_SENTINEL
        ).astype("int8")

        # Replace the impossible employment duration with NaN so the train-only
        # imputer can handle it later.
        transformed.loc[
            transformed["DAYS_EMPLOYED"] == self.DAYS_EMPLOYED_SENTINEL,
            "DAYS_EMPLOYED",
        ] = np.nan

        return transformed

    def _add_positive_year_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert negative day columns into positive years.

        Args:
            df: DataFrame containing `DAYS_BIRTH` and `DAYS_EMPLOYED`.

        Returns:
            DataFrame with `AGE_YEARS` and `EMPLOYED_YEARS` columns.

        Raises:
            None.
        """

        # Copy for method isolation.
        transformed = df.copy()

        # DAYS_BIRTH is negative in the Kaggle data; multiplying by -1 makes age
        # positive and interpretable.
        transformed["AGE_YEARS"] = (-transformed["DAYS_BIRTH"] / 365.25).astype(
            "float32"
        )

        # DAYS_EMPLOYED may now contain NaN after sentinel replacement; float
        # conversion preserves that missingness for later train-only imputation.
        transformed["EMPLOYED_YEARS"] = (
            -transformed["DAYS_EMPLOYED"].astype("float32") / 365.25
        ).astype("float32")

        return transformed

    def _map_binary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map simple binary categorical columns to numeric flags.

        Args:
            df: DataFrame containing `FLAG_OWN_CAR`.

        Returns:
            DataFrame with `FLAG_OWN_CAR` mapped to 0/1 float32.

        Raises:
            PreprocessingError: If values other than Y/N are present.
        """

        # The MVP only maps FLAG_OWN_CAR here; other categoricals go through
        # one-hot encoding in the statistical preprocessor.
        mapping = {"N": 0.0, "Y": 1.0}

        # Copy before changing dtype.
        transformed = df.copy()

        # Apply explicit mapping for transparency and reproducibility.
        transformed["FLAG_OWN_CAR"] = transformed["FLAG_OWN_CAR"].map(mapping)

        # Any NaN after mapping means the raw data had an unexpected category.
        if transformed["FLAG_OWN_CAR"].isna().any():
            raise PreprocessingError("Unexpected FLAG_OWN_CAR value; expected Y/N.")

        # Cast to float32 for Keras compatibility.
        transformed["FLAG_OWN_CAR"] = transformed["FLAG_OWN_CAR"].astype("float32")

        return transformed

    def _validate_financial_non_negative(self, df: pd.DataFrame) -> None:
        """Validate financial columns are non-negative when present.

        Args:
            df: DataFrame containing MVP financial columns.

        Returns:
            None.

        Raises:
            PreprocessingError: If any non-null financial value is negative.
        """

        # Financial amounts should be non-negative; negative values would make
        # direct debt ratios difficult to interpret.
        for col in self._contract.financial_columns():
            if (df[col].dropna() < 0).any():
                raise PreprocessingError(f"{col} contains negative values.")

    def _split_features_target_sensitive(self, df: pd.DataFrame) -> DeterministicDataset:
        """Separate transformed data into X, y and sensitive series.

        Args:
            df: Deterministically transformed DataFrame.

        Returns:
            DeterministicDataset with features, target and sensitive values.

        Raises:
            PreprocessingError: If target or sensitive contain invalid values.
        """

        # y is the binary target used by classification losses.
        target = df[self._contract.target_column].astype("int8")

        # s is the numeric sensitive value used for fairness penalties and
        # metrics, but never included in the dense feature branch.
        sensitive = df[self._contract.engineered_sensitive_column].astype("float32")

        # Confirm y is binary before later splitting/training.
        if not set(target.dropna().unique()).issubset({0, 1}):
            raise PreprocessingError("TARGET must contain only 0/1 values.")

        # Confirm sensitive is binary after mapping.
        if not set(sensitive.dropna().unique()).issubset({0.0, 1.0}):
            raise PreprocessingError("SENSITIVE must contain only 0.0/1.0 values.")

        # Drop target and both sensitive representations from X.
        features = df.drop(
            columns=[
                self._contract.target_column,
                self._contract.sensitive_column,
                self._contract.engineered_sensitive_column,
            ]
        )

        return DeterministicDataset(
            features=features,
            target=target,
            sensitive=sensitive,
        )


class HomeCreditPreprocessingColumnSpecFactory:
    """Build the Block 2 preprocessing column specification.

    Args:
        contract: Data contract used to derive financial and categorical groups.

    Returns:
        Factory instance able to create `PreprocessingColumnSpec`.

    Raises:
        None during initialization.
    """

    def __init__(self, contract: HomeCreditMVPDataContract) -> None:
        """Initialize the column spec factory.

        Args:
            contract: MVP data contract.

        Returns:
            None.

        Raises:
            None.
        """

        # The contract remains the single source for raw feature groups.
        self._contract = contract

    def build(self) -> PreprocessingColumnSpec:
        """Build the preprocessing groups used by the ColumnTransformer.

        Args:
            None.

        Returns:
            PreprocessingColumnSpec with financial, scaled, binary and
            categorical column groups.

        Raises:
            None.
        """

        # Financial columns are deliberately not scaled.
        financial_cols = self._contract.financial_columns()

        # Continuous non-financial columns include derived temporal features,
        # external scores, missing count and simple numeric/discrete features.
        continuous_scaled_cols = (
            "AGE_YEARS",
            "EMPLOYED_YEARS",
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
            "EXT_NULL_COUNT",
            "REGION_RATING_CLIENT_W_CITY",
            "CNT_CHILDREN",
        )

        # Binary columns remain 0/1, so they are imputed but not scaled.
        binary_cols = (
            "EXT_SOURCE_1_WAS_MISSING",
            "EXT_SOURCE_2_WAS_MISSING",
            "EXT_SOURCE_3_WAS_MISSING",
            "DAYS_EMPLOYED_ANOM",
            "FLAG_OWN_CAR",
        )

        # Selected low-cardinality categoricals are one-hot encoded.
        categorical_cols = self._contract.categorical_columns()

        return PreprocessingColumnSpec(
            financial_cols=financial_cols,
            continuous_scaled_cols=continuous_scaled_cols,
            binary_cols=binary_cols,
            categorical_cols=categorical_cols,
        )


class HomeCreditFeaturePreprocessor:
    """Fit and apply leakage-safe statistical preprocessing.

    Args:
        column_spec: Column grouping used to build the sklearn transformer.

    Returns:
        Preprocessor object wrapping a train-fitted `ColumnTransformer`.

    Raises:
        None during initialization.
    """

    def __init__(self, column_spec: PreprocessingColumnSpec) -> None:
        """Initialize the feature preprocessor.

        Args:
            column_spec: Column groups for financial, continuous, binary and
                categorical features.

        Returns:
            None.

        Raises:
            None.
        """

        # Store the spec so feature name generation and validation use the exact
        # same column groups as the transformer.
        self._column_spec = column_spec

        # The ColumnTransformer is built lazily during initialization and fitted
        # later on train only.
        self._preprocessor = self._build_column_transformer()

        # Feature names are unknown until the categorical encoder has been fit.
        self._feature_names: tuple[str, ...] | None = None

    @property
    def column_spec(self) -> PreprocessingColumnSpec:
        """Return the preprocessing column specification.

        Args:
            None.

        Returns:
            PreprocessingColumnSpec used by this object.

        Raises:
            None.
        """

        # Expose the immutable spec for later model-building code.
        return self._column_spec

    @property
    def preprocessor(self) -> ColumnTransformer:
        """Return the sklearn ColumnTransformer.

        Args:
            None.

        Returns:
            ColumnTransformer, fitted after `fit` or `fit_transform_splits`.

        Raises:
            None.
        """

        # Returning the object lets callers persist it with joblib in later work.
        return self._preprocessor

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return processed feature names after fitting.

        Args:
            None.

        Returns:
            Tuple of processed feature names in matrix order.

        Raises:
            PreprocessingError: If the preprocessor has not been fit yet.
        """

        # Feature names depend on one-hot categories learned during fit.
        if self._feature_names is None:
            raise PreprocessingError("Feature names are unavailable before fitting.")

        return self._feature_names

    def fit(self, X_train: pd.DataFrame) -> "HomeCreditFeaturePreprocessor":
        """Fit imputers, scaler and encoder using train features only.

        Args:
            X_train: Raw train features after deterministic transformations.

        Returns:
            Self, fitted.

        Raises:
            PreprocessingError: If expected columns are missing.
        """

        # Validate train columns before sklearn emits less clear messages.
        self._assert_expected_columns(X_train)

        # This is the leakage-critical line: fit happens only on train.
        self._preprocessor.fit(X_train)

        # Once fit, the categorical encoder knows its categories and feature
        # names can be generated reliably.
        self._feature_names = self._build_feature_names()

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using the train-fitted preprocessor.

        Args:
            X: Raw features after deterministic transformations.

        Returns:
            Numeric float32 matrix ready for Keras.

        Raises:
            PreprocessingError: If expected columns are missing.
        """

        # Every split must have the same expected columns before transformation.
        self._assert_expected_columns(X)

        # Transform uses medians, scaler parameters and categories learned from
        # train only.
        transformed = self._preprocessor.transform(X)

        # ColumnTransformer can return sparse or dense arrays depending on
        # configuration; force a dense numpy array for TensorFlow.
        dense = np.asarray(transformed)

        # Keras should receive float32 arrays for memory and speed.
        return dense.astype("float32")

    def fit_transform_splits(self, raw_splits: RawSplitDataset) -> ProcessedSplitDataset:
        """Fit on train and transform train/validation/test splits.

        Args:
            raw_splits: Raw split data produced after Block 3 splitting.

        Returns:
            ProcessedSplitDataset containing numeric arrays and metadata.

        Raises:
            PreprocessingError: If any split has invalid columns or NaNs remain.
        """

        # Fit only on X_train. This is what prevents statistical leakage.
        self.fit(raw_splits.X_train)

        # Transform each split with the same train-fitted transformer.
        X_train = self.transform(raw_splits.X_train)
        X_val = self.transform(raw_splits.X_val)
        X_test = self.transform(raw_splits.X_test)

        # Convert y and s to float32 arrays for TensorFlow compatibility.
        y_train = self._series_to_float32_array(raw_splits.y_train)
        y_val = self._series_to_float32_array(raw_splits.y_val)
        y_test = self._series_to_float32_array(raw_splits.y_test)
        s_train = self._series_to_float32_array(raw_splits.s_train)
        s_val = self._series_to_float32_array(raw_splits.s_val)
        s_test = self._series_to_float32_array(raw_splits.s_test)

        # Validate final arrays so downstream Keras code does not receive hidden
        # NaNs from an unexpected preprocessing issue.
        self._assert_no_nan("X_train", X_train)
        self._assert_no_nan("X_val", X_val)
        self._assert_no_nan("X_test", X_test)

        return ProcessedSplitDataset(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            s_train=s_train,
            s_val=s_val,
            s_test=s_test,
            train_ids=tuple(raw_splits.X_train.index.tolist()),
            val_ids=tuple(raw_splits.X_val.index.tolist()),
            test_ids=tuple(raw_splits.X_test.index.tolist()),
            feature_names=self.feature_names,
            preprocessor=self._preprocessor,
        )

    def financial_feature_indices(self) -> dict[str, int]:
        """Return processed indices of financial columns.

        Args:
            None.

        Returns:
            Mapping from financial column name to processed matrix index.

        Raises:
            PreprocessingError: If the preprocessor has not been fit.
        """

        # Feature names must exist before indices can be calculated.
        names = self.feature_names

        # Build a reverse lookup for feature name to index.
        name_to_idx = {name: index for index, name in enumerate(names)}

        # Keep only the financial columns used by the Block 5 ratio layer.
        return {
            column: name_to_idx[column]
            for column in self._column_spec.financial_cols
        }

    def _build_column_transformer(self) -> ColumnTransformer:
        """Create the sklearn ColumnTransformer for Block 2.

        Args:
            None.

        Returns:
            Unfitted ColumnTransformer.

        Raises:
            None.
        """

        # Financial amounts: median imputation only, no scaling.
        financial_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

        # Continuous non-financial values: median imputation plus RobustScaler.
        continuous_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
            ]
        )

        # Binary flags: most frequent imputation only, preserving 0/1 semantics.
        binary_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ]
        )

        # Categoricals: explicit MISSING token plus one-hot encoding.
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )

        # The transformer order is the processed feature order.
        return ColumnTransformer(
            transformers=[
                ("financial", financial_pipeline, list(self._column_spec.financial_cols)),
                (
                    "continuous_scaled",
                    continuous_pipeline,
                    list(self._column_spec.continuous_scaled_cols),
                ),
                ("binary", binary_pipeline, list(self._column_spec.binary_cols)),
                (
                    "categorical",
                    categorical_pipeline,
                    list(self._column_spec.categorical_cols),
                ),
            ],
            remainder="drop",
        )

    def _build_feature_names(self) -> tuple[str, ...]:
        """Build feature names in exact transformed matrix order.

        Args:
            None.

        Returns:
            Tuple of processed feature names.

        Raises:
            PreprocessingError: If categorical encoder is not fitted.
        """

        # Financial, continuous and binary pipelines preserve one output per
        # input column in the same order.
        base_names = (
            self._column_spec.financial_cols
            + self._column_spec.continuous_scaled_cols
            + self._column_spec.binary_cols
        )

        # Access the fitted categorical pipeline by its ColumnTransformer name.
        categorical_pipeline = self._preprocessor.named_transformers_["categorical"]

        # Access the fitted one-hot encoder inside the pipeline.
        encoder = categorical_pipeline.named_steps["onehot"]

        # One-hot feature names must be generated by sklearn because categories
        # are learned from train.
        categorical_names = tuple(
            encoder.get_feature_names_out(self._column_spec.categorical_cols).tolist()
        )

        return base_names + categorical_names

    def _assert_expected_columns(self, X: pd.DataFrame) -> None:
        """Validate that a feature frame contains all preprocessing columns.

        Args:
            X: Feature DataFrame to validate.

        Returns:
            None.

        Raises:
            PreprocessingError: If any expected column is missing.
        """

        # The preprocessor expects deterministic features created before split.
        expected = self._column_spec.all_columns()

        # Preserve expected order in the error message.
        missing = tuple(column for column in expected if column not in X.columns)

        if missing:
            raise PreprocessingError(
                "Missing columns for preprocessing: " + ", ".join(missing)
            )

    def _series_to_float32_array(self, series: pd.Series) -> np.ndarray:
        """Convert a pandas Series to a flat float32 numpy array.

        Args:
            series: Series containing target or sensitive values.

        Returns:
            One-dimensional float32 numpy array.

        Raises:
            None.
        """

        # Keras accepts 1D arrays for binary targets and sensitive side inputs.
        return series.to_numpy(dtype="float32", copy=True).reshape(-1)

    def _assert_no_nan(self, name: str, array: np.ndarray) -> None:
        """Raise if a processed matrix still contains NaN.

        Args:
            name: Human-readable array name for error messages.
            array: Numeric matrix to validate.

        Returns:
            None.

        Raises:
            PreprocessingError: If the matrix contains NaN.
        """

        # NaNs after preprocessing indicate an imputation or column routing bug.
        if np.isnan(array).any():
            raise PreprocessingError(f"{name} contains NaN after preprocessing.")


class HomeCreditMVPPreprocessingPipeline:
    """Facade that wires Block 1 contract and Block 2 preprocessing classes.

    Args:
        contract: Optional MVP data contract. If omitted, the default contract
            is created.

    Returns:
        Pipeline object with loader, deterministic transformer and feature
        preprocessor helpers.

    Raises:
        None during initialization.
    """

    def __init__(self, contract: HomeCreditMVPDataContract | None = None) -> None:
        """Initialize the preprocessing facade.

        Args:
            contract: Optional MVP data contract.

        Returns:
            None.

        Raises:
            None.
        """

        # Default contract keeps notebooks short while still allowing tests to
        # inject a temporary raw_data_dir.
        self._contract = contract or build_default_home_credit_contract()

        # The loader handles CSV reading and raw column validation.
        self._loader = HomeCreditRawDataLoader(self._contract)

        # The deterministic transformer handles all transformations that are
        # safe before splitting.
        self._deterministic_transformer = HomeCreditDeterministicTransformer(
            self._contract
        )

        # The column spec factory knows which deterministic columns feed each
        # statistical preprocessing branch.
        self._column_spec_factory = HomeCreditPreprocessingColumnSpecFactory(
            self._contract
        )

    @property
    def contract(self) -> HomeCreditMVPDataContract:
        """Return the pipeline contract.

        Args:
            None.

        Returns:
            HomeCreditMVPDataContract used by this pipeline.

        Raises:
            None.
        """

        return self._contract

    def load_raw(self, path: str | Path | None = None) -> pd.DataFrame:
        """Load raw MVP data using the contract columns.

        Args:
            path: Optional explicit path to `application_train.csv`.

        Returns:
            Raw MVP DataFrame.

        Raises:
            DataContractError: If the file is missing or required columns fail.
        """

        # Delegate to the loader to keep responsibilities isolated.
        return self._loader.load(path=path)

    def apply_deterministic_transforms(
        self,
        raw_df: pd.DataFrame,
    ) -> DeterministicDataset:
        """Apply leakage-free deterministic transformations.

        Args:
            raw_df: Raw MVP DataFrame.

        Returns:
            DeterministicDataset with features, target and sensitive.

        Raises:
            DataContractError: If required raw columns are missing.
            PreprocessingError: If deterministic assumptions fail.
        """

        # Deterministic transformations can run before splitting because they do
        # not learn medians, scales, categories or other dataset statistics.
        return self._deterministic_transformer.transform(raw_df)

    def build_feature_preprocessor(self) -> HomeCreditFeaturePreprocessor:
        """Create a new unfitted statistical feature preprocessor.

        Args:
            None.

        Returns:
            HomeCreditFeaturePreprocessor ready to fit on train split only.

        Raises:
            None.
        """

        # Each experiment should get a fresh unfitted preprocessor to avoid
        # accidental reuse of fitted state across different splits.
        column_spec = self._column_spec_factory.build()

        return HomeCreditFeaturePreprocessor(column_spec=column_spec)

    def fit_transform_splits(
        self,
        raw_splits: RawSplitDataset,
    ) -> ProcessedSplitDataset:
        """Fit preprocessing on train and transform all splits.

        Args:
            raw_splits: Raw split data produced by Block 3.

        Returns:
            ProcessedSplitDataset ready for Keras.

        Raises:
            PreprocessingError: If preprocessing fails.
        """

        # Build a fresh preprocessor so fitting is explicit and isolated.
        preprocessor = self.build_feature_preprocessor()

        # Delegate leakage-sensitive train-only fitting to the preprocessor.
        return preprocessor.fit_transform_splits(raw_splits)


__all__ = [
    "DeterministicDataset",
    "HomeCreditDeterministicTransformer",
    "HomeCreditFeaturePreprocessor",
    "HomeCreditMVPPreprocessingPipeline",
    "HomeCreditPreprocessingColumnSpecFactory",
    "HomeCreditRawDataLoader",
    "PreprocessingColumnSpec",
    "PreprocessingError",
    "ProcessedSplitDataset",
    "RawSplitDataset",
]
