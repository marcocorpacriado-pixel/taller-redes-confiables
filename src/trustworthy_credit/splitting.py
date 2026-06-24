"""Honest train/validation/test splitting for the Home Credit MVP.

This module implements Block 3. It receives the deterministic output from Block
2 and creates a single reproducible 70/15/15 split stratified by TARGET and
SENSITIVE together.

The module deliberately does not preprocess statistically, train models, choose
thresholds or inspect test performance. Its job is only to create and audit the
split that all later blocks must reuse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .preprocessing import DeterministicDataset, RawSplitDataset


class SplitError(ValueError):
    """Raised when an honest split cannot be created safely.

    Args:
        message: Human-readable explanation of the split failure.

    Returns:
        None.

    Raises:
        This exception is raised by Block 3 classes when sizes, alignment or
        stratification assumptions are violated.
    """


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for the MVP train/validation/test split.

    Args:
        test_size: Fraction of the full dataset reserved for final test.
        validation_size: Fraction of the full dataset reserved for validation.
        random_state: Seed used by sklearn for reproducibility.
        shuffle: Whether to shuffle before splitting.

    Returns:
        Immutable split configuration.

    Raises:
        None.
    """

    # Test is isolated first and used only for final reporting.
    test_size: float = 0.15

    # Validation is used for tuning, lambda selection, thresholds and early
    # stopping decisions.
    validation_size: float = 0.15

    # Fixed seed makes the split reproducible across teammates and reruns.
    random_state: int = 42

    # Stratified splitting requires shuffling in sklearn.
    shuffle: bool = True

    @property
    def train_size(self) -> float:
        """Return the implied train fraction.

        Args:
            None.

        Returns:
            Fraction of the full dataset left for train.

        Raises:
            None.
        """

        # Train receives the remaining mass after validation and test.
        return 1.0 - self.test_size - self.validation_size

    @property
    def validation_size_relative_to_trainval(self) -> float:
        """Return validation fraction inside the train+validation subset.

        Args:
            None.

        Returns:
            Relative validation size used in the second `train_test_split`.

        Raises:
            SplitError: If test size leaves no train+validation data.
        """

        # After test is removed, validation must be expressed relative to the
        # remaining trainval subset.
        trainval_size = 1.0 - self.test_size

        # This should never be zero with valid config, but the guard makes the
        # property safe and explicit.
        if trainval_size <= 0.0:
            raise SplitError("test_size leaves no data for train/validation.")

        return self.validation_size / trainval_size


@dataclass(frozen=True)
class SplitReportRow:
    """Summary statistics for one split.

    Args:
        split: Split name: train, validation or test.
        n: Number of rows in the split.
        target_rate: Mean TARGET value.
        sensitive_rate: Mean SENSITIVE value.
        target_0_sensitive_0: Count of rows with TARGET=0 and SENSITIVE=0.
        target_0_sensitive_1: Count of rows with TARGET=0 and SENSITIVE=1.
        target_1_sensitive_0: Count of rows with TARGET=1 and SENSITIVE=0.
        target_1_sensitive_1: Count of rows with TARGET=1 and SENSITIVE=1.

    Returns:
        Immutable report row.

    Raises:
        None.
    """

    # The split name makes the row self-describing once converted to a DataFrame.
    split: str

    # Number of rows is the first sanity check for 70/15/15 proportions.
    n: int

    # TARGET rate checks class balance across splits.
    target_rate: float

    # SENSITIVE rate checks gender composition across splits.
    sensitive_rate: float

    # The four joint counts verify that stratification preserved all groups.
    target_0_sensitive_0: int
    target_0_sensitive_1: int
    target_1_sensitive_0: int
    target_1_sensitive_1: int

    def to_dict(self) -> dict[str, int | float | str]:
        """Convert the report row to a plain dictionary.

        Args:
            None.

        Returns:
            Dictionary suitable for pandas, JSON or logging.

        Raises:
            None.
        """

        # Avoid dataclasses.asdict to keep the return type explicit and simple.
        return {
            "split": self.split,
            "n": self.n,
            "target_rate": self.target_rate,
            "sensitive_rate": self.sensitive_rate,
            "target_0_sensitive_0": self.target_0_sensitive_0,
            "target_0_sensitive_1": self.target_0_sensitive_1,
            "target_1_sensitive_0": self.target_1_sensitive_0,
            "target_1_sensitive_1": self.target_1_sensitive_1,
        }


@dataclass(frozen=True)
class SplitArtifacts:
    """Complete output of Block 3.

    Args:
        raw_splits: RawSplitDataset consumed later by Block 2 statistical
            preprocessing.
        report: DataFrame with split composition diagnostics.
        config: SplitConfig used to create the split.

    Returns:
        Immutable container with split data and audit information.

    Raises:
        None.
    """

    # RawSplitDataset is the handoff object expected by Block 2's feature
    # preprocessor.
    raw_splits: RawSplitDataset

    # Report is kept next to the data because every downstream experiment should
    # know which split proportions it is using.
    report: pd.DataFrame

    # Config is stored for reproducibility.
    config: SplitConfig


class SplitConfigValidator:
    """Validate split configuration values before splitting.

    Args:
        None.

    Returns:
        Validator instance.

    Raises:
        None.
    """

    def validate(self, config: SplitConfig) -> None:
        """Raise if a split configuration is invalid.

        Args:
            config: Split configuration to validate.

        Returns:
            None.

        Raises:
            SplitError: If sizes are non-positive or do not leave train data.
        """

        # Both held-out fractions must be strictly positive for this project.
        if config.test_size <= 0.0:
            raise SplitError("test_size must be greater than 0.")

        # Validation must exist because tuner, lambda and threshold depend on it.
        if config.validation_size <= 0.0:
            raise SplitError("validation_size must be greater than 0.")

        # The sum must leave positive train data.
        if config.test_size + config.validation_size >= 1.0:
            raise SplitError("test_size + validation_size must be less than 1.")

        # This property also checks that trainval size is positive.
        _ = config.validation_size_relative_to_trainval


class DatasetAlignmentValidator:
    """Validate that X, y and sensitive values are aligned.

    Args:
        None.

    Returns:
        Validator instance.

    Raises:
        None.
    """

    def validate(self, dataset: DeterministicDataset) -> None:
        """Raise if deterministic dataset components are misaligned.

        Args:
            dataset: DeterministicDataset from Block 2.

        Returns:
            None.

        Raises:
            SplitError: If lengths differ, indices differ or labels are invalid.
        """

        # All components must have exactly the same number of rows.
        if not (
            len(dataset.features)
            == len(dataset.target)
            == len(dataset.sensitive)
        ):
            raise SplitError("features, target and sensitive lengths differ.")

        # The DataFrame and Series indices must match, otherwise rows could be
        # split with the wrong target or sensitive value.
        if not dataset.features.index.equals(dataset.target.index):
            raise SplitError("features and target indices are not aligned.")

        # Sensitive alignment is equally important because fairness metrics use
        # row-level gender membership.
        if not dataset.features.index.equals(dataset.sensitive.index):
            raise SplitError("features and sensitive indices are not aligned.")

        # TARGET should be binary after Block 2 deterministic transformations.
        target_values = set(dataset.target.dropna().astype(int).unique().tolist())
        if not target_values.issubset({0, 1}):
            raise SplitError("TARGET must be binary before splitting.")

        # SENSITIVE should also be binary after Block 2 mapping.
        sensitive_values = set(dataset.sensitive.dropna().astype(int).unique().tolist())
        if not sensitive_values.issubset({0, 1}):
            raise SplitError("SENSITIVE must be binary before splitting.")


class StratificationKeyBuilder:
    """Build joint stratification keys from TARGET and SENSITIVE.

    Args:
        None.

    Returns:
        Builder instance.

    Raises:
        None.
    """

    def build(self, target: pd.Series, sensitive: pd.Series) -> pd.Series:
        """Create `TARGET_SENSITIVE` strata labels.

        Args:
            target: Binary target series.
            sensitive: Binary sensitive series.

        Returns:
            Series of string labels such as `0_0`, `0_1`, `1_0`, `1_1`.

        Raises:
            SplitError: If indices are not aligned.
        """

        # The target and sensitive series must describe the same rows.
        if not target.index.equals(sensitive.index):
            raise SplitError("target and sensitive indices are not aligned.")

        # Cast through int so 0.0/1.0 sensitive values become clean 0/1 labels.
        target_part = target.astype(int).astype(str)

        # Same int casting for sensitive values.
        sensitive_part = sensitive.astype(int).astype(str)

        # Combine both variables to preserve the four joint groups.
        return target_part + "_" + sensitive_part

    def value_counts(self, target: pd.Series, sensitive: pd.Series) -> pd.Series:
        """Return counts of joint stratification groups.

        Args:
            target: Binary target series.
            sensitive: Binary sensitive series.

        Returns:
            Series with counts per joint stratum.

        Raises:
            SplitError: If indices are not aligned.
        """

        # Reuse build so all key formatting stays in one place.
        strata = self.build(target=target, sensitive=sensitive)

        # Sort by index for deterministic reports.
        return strata.value_counts().sort_index()


class SplitReportBuilder:
    """Build post-split composition reports.

    Args:
        None.

    Returns:
        Reporter instance.

    Raises:
        None.
    """

    def build(self, raw_splits: RawSplitDataset) -> pd.DataFrame:
        """Build a DataFrame report for train, validation and test.

        Args:
            raw_splits: Raw split dataset produced by the splitter.

        Returns:
            DataFrame with one row per split and composition statistics.

        Raises:
            None.
        """

        # Build each row independently for clarity and testability.
        rows = [
            self._build_row("train", raw_splits.y_train, raw_splits.s_train),
            self._build_row("validation", raw_splits.y_val, raw_splits.s_val),
            self._build_row("test", raw_splits.y_test, raw_splits.s_test),
        ]

        # Convert row objects to dictionaries before handing them to pandas.
        return pd.DataFrame([row.to_dict() for row in rows])

    def _build_row(
        self,
        split_name: str,
        target: pd.Series,
        sensitive: pd.Series,
    ) -> SplitReportRow:
        """Build one split report row.

        Args:
            split_name: Name of the split.
            target: Target series for the split.
            sensitive: Sensitive series for the split.

        Returns:
            SplitReportRow with rates and joint counts.

        Raises:
            None.
        """

        # Cast to int for stable comparisons even if series dtype is float.
        y = target.astype(int)

        # Sensitive is also cast to int for 0/1 comparisons.
        s = sensitive.astype(int)

        # Joint group counts are the core fairness-related split diagnostic.
        return SplitReportRow(
            split=split_name,
            n=int(len(target)),
            target_rate=float(y.mean()) if len(y) else 0.0,
            sensitive_rate=float(s.mean()) if len(s) else 0.0,
            target_0_sensitive_0=int(((y == 0) & (s == 0)).sum()),
            target_0_sensitive_1=int(((y == 0) & (s == 1)).sum()),
            target_1_sensitive_0=int(((y == 1) & (s == 0)).sum()),
            target_1_sensitive_1=int(((y == 1) & (s == 1)).sum()),
        )


class SplitIndexExporter:
    """Export split indices for reproducibility.

    Args:
        None.

    Returns:
        Exporter instance.

    Raises:
        None.
    """

    def to_dict(self, raw_splits: RawSplitDataset) -> dict[str, list[Any]]:
        """Convert split indices to a serializable dictionary.

        Args:
            raw_splits: Raw split dataset.

        Returns:
            Dictionary with train, validation and test IDs.

        Raises:
            None.
        """

        # Indices are SK_ID_CURR because Block 2 set them before splitting.
        return {
            "train_idx": raw_splits.X_train.index.tolist(),
            "val_idx": raw_splits.X_val.index.tolist(),
            "test_idx": raw_splits.X_test.index.tolist(),
        }

    def save_json(self, raw_splits: RawSplitDataset, path: str | Path) -> Path:
        """Save split indices to a JSON file.

        Args:
            raw_splits: Raw split dataset.
            path: Destination JSON path.

        Returns:
            Path object pointing to the written file.

        Raises:
            OSError: If the destination cannot be created or written.
        """

        # Normalize to Path so caller can pass strings or Path objects.
        output_path = Path(path)

        # Ensure parent directories exist, matching the project structure.
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert indices to plain Python objects before JSON serialization.
        payload = self.to_dict(raw_splits)

        # Write stable, indented JSON to make diffs and manual inspection easy.
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

        return output_path


class HomeCreditTrainValTestSplitter:
    """Create the official internal MVP split.

    Args:
        config: SplitConfig controlling proportions and random seed.
        config_validator: Optional validator for split configuration.
        alignment_validator: Optional validator for dataset alignment.
        strata_builder: Optional builder for joint stratification keys.
        report_builder: Optional builder for split diagnostics.

    Returns:
        Splitter object able to create `RawSplitDataset` and reports.

    Raises:
        SplitError: If the provided config is invalid.
    """

    def __init__(
        self,
        config: SplitConfig | None = None,
        config_validator: SplitConfigValidator | None = None,
        alignment_validator: DatasetAlignmentValidator | None = None,
        strata_builder: StratificationKeyBuilder | None = None,
        report_builder: SplitReportBuilder | None = None,
    ) -> None:
        """Initialize the train/validation/test splitter.

        Args:
            config: Optional split configuration. Defaults to 70/15/15 with
                `random_state=42`.
            config_validator: Optional config validator.
            alignment_validator: Optional dataset alignment validator.
            strata_builder: Optional stratification key builder.
            report_builder: Optional report builder.

        Returns:
            None.

        Raises:
            SplitError: If split configuration is invalid.
        """

        # Use the project default config unless a test or experiment injects one.
        self._config = config or SplitConfig()

        # Dependency injection keeps the splitter modular and testable.
        self._config_validator = config_validator or SplitConfigValidator()
        self._alignment_validator = alignment_validator or DatasetAlignmentValidator()
        self._strata_builder = strata_builder or StratificationKeyBuilder()
        self._report_builder = report_builder or SplitReportBuilder()

        # Validate configuration immediately so errors appear before data work.
        self._config_validator.validate(self._config)

    @property
    def config(self) -> SplitConfig:
        """Return the split configuration.

        Args:
            None.

        Returns:
            SplitConfig used by this splitter.

        Raises:
            None.
        """

        # The config is immutable, so exposing it is safe.
        return self._config

    def split(self, dataset: DeterministicDataset) -> SplitArtifacts:
        """Split deterministic data into train, validation and test.

        Args:
            dataset: DeterministicDataset produced by Block 2.

        Returns:
            SplitArtifacts containing RawSplitDataset, report and config.

        Raises:
            SplitError: If alignment or stratified splitting fails.
        """

        # Validate X/y/s alignment before any random split.
        self._alignment_validator.validate(dataset)

        # First stage: isolate final test from the full dataset.
        trainval = self._split_trainval_test(dataset)

        # Second stage: split train and validation from trainval.
        raw_splits = self._split_train_validation(trainval)

        # Build diagnostics immediately so issues with proportions are visible.
        report = self._report_builder.build(raw_splits)

        return SplitArtifacts(
            raw_splits=raw_splits,
            report=report,
            config=self._config,
        )

    def _split_trainval_test(
        self,
        dataset: DeterministicDataset,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
        """Create trainval/test split from the full deterministic dataset.

        Args:
            dataset: DeterministicDataset from Block 2.

        Returns:
            Tuple containing X_trainval, X_test, y_trainval, y_test, s_trainval,
            s_test.

        Raises:
            SplitError: If sklearn cannot create the stratified split.
        """

        # Joint TARGET+SENSITIVE strata preserve both class and gender
        # composition in the final holdout.
        strata = self._strata_builder.build(
            target=dataset.target,
            sensitive=dataset.sensitive,
        )

        try:
            # We split all aligned objects in one call so sklearn applies the
            # same row selection to X, y and s.
            return train_test_split(
                dataset.features,
                dataset.target,
                dataset.sensitive,
                test_size=self._config.test_size,
                random_state=self._config.random_state,
                shuffle=self._config.shuffle,
                stratify=strata,
            )
        except ValueError as exc:
            # Convert sklearn's generic error into a project-specific one.
            raise SplitError(f"Unable to create stratified test split: {exc}") from exc

    def _split_train_validation(
        self,
        trainval: tuple[
            pd.DataFrame,
            pd.DataFrame,
            pd.Series,
            pd.Series,
            pd.Series,
            pd.Series,
        ],
    ) -> RawSplitDataset:
        """Split trainval into train and validation.

        Args:
            trainval: Tuple returned by `_split_trainval_test`.

        Returns:
            RawSplitDataset with train, validation and test pieces.

        Raises:
            SplitError: If sklearn cannot create the stratified validation split.
        """

        # Unpack the first-stage split using explicit names for readability.
        X_trainval, X_test, y_trainval, y_test, s_trainval, s_test = trainval

        # Rebuild strata only on the trainval subset because validation is carved
        # out of that subset, not the original full dataset.
        strata_trainval = self._strata_builder.build(
            target=y_trainval,
            sensitive=s_trainval,
        )

        try:
            # Validation size is relative to trainval, not to the full dataset.
            X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(
                X_trainval,
                y_trainval,
                s_trainval,
                test_size=self._config.validation_size_relative_to_trainval,
                random_state=self._config.random_state,
                shuffle=self._config.shuffle,
                stratify=strata_trainval,
            )
        except ValueError as exc:
            # Project-specific error helps the team inspect joint group counts.
            raise SplitError(
                f"Unable to create stratified validation split: {exc}"
            ) from exc

        return RawSplitDataset(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            s_train=s_train,
            s_val=s_val,
            s_test=s_test,
        )


__all__ = [
    "DatasetAlignmentValidator",
    "HomeCreditTrainValTestSplitter",
    "SplitArtifacts",
    "SplitConfig",
    "SplitConfigValidator",
    "SplitError",
    "SplitIndexExporter",
    "SplitReportBuilder",
    "SplitReportRow",
    "StratificationKeyBuilder",
]
