"""Data contract for the Home Credit MVP.

This module implements Block 1 of the project: the formal contract that defines
which raw file, columns, roles and exclusions are valid for the MVP.

The module deliberately does not preprocess, split, impute or train anything.
Those responsibilities belong to later blocks. Here we only encode the rules
that protect the rest of the pipeline from ambiguous input data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class DataContractError(ValueError):
    """Raised when the input data does not satisfy the MVP data contract.

    Args:
        message: Human-readable explanation of the contract violation.

    Returns:
        None.

    Raises:
        This exception is itself raised by validator methods when mandatory
        files or columns are missing.
    """


@dataclass(frozen=True)
class DatasetFileSpec:
    """Immutable description of one raw Kaggle file.

    Args:
        file_name: Name of the CSV file expected inside the raw data folder.
        role: Short explanation of how the file is used by the project.
        required_for_mvp: Whether the MVP needs this file to run.
        has_target: Whether the file contains the `TARGET` column.
        notes: Extra explanation useful for the team and documentation.

    Returns:
        A read-only value object describing a dataset file.

    Raises:
        None.
    """

    # The physical CSV name is kept separate from paths so the same contract can
    # be used on any machine where `data/raw/` points somewhere different.
    file_name: str

    # The role tells future code whether the file is training data, official
    # Kaggle test data, relational enrichment data, or metadata.
    role: str

    # Only `application_train.csv` is required in the MVP; relational tables are
    # intentionally postponed until the advanced phase.
    required_for_mvp: bool

    # This flag prevents the common mistake of treating `application_test.csv`
    # as an evaluable test set even though it has no ground-truth target.
    has_target: bool

    # Notes are documentation carried with the code, useful in notebooks and
    # generated summaries.
    notes: str = ""


@dataclass(frozen=True)
class ColumnGroup:
    """Named immutable group of columns with one modelling responsibility.

    Args:
        name: Stable internal name for the group.
        columns: Ordered column names belonging to the group.
        description: Explanation of why the group exists.

    Returns:
        A read-only value object grouping related columns.

    Raises:
        None.
    """

    # The group name is used by documentation and downstream code to refer to
    # logical groups instead of repeating raw lists everywhere.
    name: str

    # Tuples make the object immutable and safe to share across modules.
    columns: tuple[str, ...]

    # Description keeps the "why" close to the actual declaration.
    description: str


@dataclass(frozen=True)
class DataContractValidationResult:
    """Result object returned by non-throwing validation methods.

    Args:
        is_valid: True when the checked input satisfies the contract.
        missing_required_columns: Required raw columns absent from the input.
        unexpected_columns: Columns present in strict mode but outside the MVP
            contract.
        checked_columns: Columns that were actually checked.
        message: Human-readable summary of the validation result.

    Returns:
        A read-only validation summary.

    Raises:
        None.
    """

    # This Boolean lets callers decide whether to continue without parsing the
    # lists manually.
    is_valid: bool

    # Missing required columns are always serious because later blocks depend on
    # these names exactly.
    missing_required_columns: tuple[str, ...] = field(default_factory=tuple)

    # Unexpected columns are not necessarily fatal unless strict mode is used.
    unexpected_columns: tuple[str, ...] = field(default_factory=tuple)

    # Keeping the checked columns helps debug typos and wrong CSV files.
    checked_columns: tuple[str, ...] = field(default_factory=tuple)

    # The message is intentionally friendly because it will often surface in
    # notebooks used by teammates.
    message: str = ""


@dataclass(frozen=True)
class HomeCreditMVPDataContract:
    """Complete MVP data contract for the Home Credit Default Risk practice.

    Args:
        raw_data_dir: Directory where the Kaggle CSV files are expected.
        training_file: Specification of the file used for MVP training and
            internal evaluation.
        official_test_file: Specification of Kaggle's official test file.
        relational_files: Specifications of advanced relational files excluded
            from the MVP.
        identifier_column: Column used only for traceability.
        target_column: Binary target column.
        sensitive_column: Raw sensitive column used to create `SENSITIVE`.
        engineered_sensitive_column: Name of the numeric sensitive column that
            later blocks will create.
        feature_groups: Ordered column groups that define MVP raw features.
        derived_columns: Deterministic columns that later preprocessing will
            create from the raw MVP columns.

    Returns:
        A read-only contract object that can validate files, columns and schema.

    Raises:
        None.
    """

    # The default follows the repository structure created in Block 0.
    raw_data_dir: Path = Path("data/raw")

    # The MVP training file is the only raw CSV with TARGET available.
    training_file: DatasetFileSpec = DatasetFileSpec(
        file_name="application_train.csv",
        role="mvp_train_and_internal_evaluation",
        required_for_mvp=True,
        has_target=True,
        notes="Main table with TARGET. Used to create train/validation/test.",
    )

    # The Kaggle official test file is useful later for inference, but it cannot
    # produce test metrics because TARGET is missing.
    official_test_file: DatasetFileSpec = DatasetFileSpec(
        file_name="application_test.csv",
        role="official_kaggle_inference_only",
        required_for_mvp=False,
        has_target=False,
        notes="No TARGET column; not valid as an evaluable test set.",
    )

    # Relational tables are registered so the team knows they are intentionally
    # out of MVP scope, not forgotten.
    relational_files: tuple[DatasetFileSpec, ...] = (
        DatasetFileSpec(
            file_name="bureau.csv",
            role="advanced_relational_enrichment",
            required_for_mvp=False,
            has_target=False,
            notes="Previous credits reported to Credit Bureau.",
        ),
        DatasetFileSpec(
            file_name="bureau_balance.csv",
            role="advanced_relational_enrichment",
            required_for_mvp=False,
            has_target=False,
            notes="Monthly balances for previous bureau credits.",
        ),
        DatasetFileSpec(
            file_name="previous_application.csv",
            role="advanced_relational_enrichment",
            required_for_mvp=False,
            has_target=False,
            notes="Previous Home Credit applications.",
        ),
        DatasetFileSpec(
            file_name="installments_payments.csv",
            role="advanced_relational_enrichment",
            required_for_mvp=False,
            has_target=False,
            notes="Repayment history for previous credits.",
        ),
        DatasetFileSpec(
            file_name="POS_CASH_balance.csv",
            role="advanced_relational_enrichment",
            required_for_mvp=False,
            has_target=False,
            notes="Monthly POS and cash loan balances.",
        ),
        DatasetFileSpec(
            file_name="credit_card_balance.csv",
            role="advanced_relational_enrichment",
            required_for_mvp=False,
            has_target=False,
            notes="Monthly credit card balance snapshots.",
        ),
    )

    # This ID is preserved for traceability and split saving, never as a model
    # input.
    identifier_column: str = "SK_ID_CURR"

    # TARGET is the label: 1 means payment difficulties, 0 means paid on time.
    target_column: str = "TARGET"

    # CODE_GENDER is the raw sensitive column from Kaggle.
    sensitive_column: str = "CODE_GENDER"

    # SENSITIVE is the numeric 0/1 column created later from CODE_GENDER.
    engineered_sensitive_column: str = "SENSITIVE"

    # Feature groups express the MVP feature contract without mixing it with
    # preprocessing details from Block 2.
    feature_groups: tuple[ColumnGroup, ...] = (
        ColumnGroup(
            name="financial",
            columns=(
                "AMT_INCOME_TOTAL",
                "AMT_CREDIT",
                "AMT_ANNUITY",
                "AMT_GOODS_PRICE",
            ),
            description=(
                "Original monetary amounts kept for interpretable custom "
                "financial ratios."
            ),
        ),
        ColumnGroup(
            name="temporal",
            columns=("DAYS_BIRTH", "DAYS_EMPLOYED"),
            description="Raw temporal columns later converted to positive years.",
        ),
        ColumnGroup(
            name="external_scores",
            columns=("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"),
            description=(
                "External scores with missingness used later for uncertainty "
                "analysis."
            ),
        ),
        ColumnGroup(
            name="categorical_low_cardinality",
            columns=(
                "NAME_EDUCATION_TYPE",
                "NAME_FAMILY_STATUS",
                "NAME_INCOME_TYPE",
            ),
            description=(
                "Selected low-cardinality categorical features with useful "
                "predictive signal and possible proxy behavior."
            ),
        ),
        ColumnGroup(
            name="simple_numeric_and_binary",
            columns=(
                "REGION_RATING_CLIENT_W_CITY",
                "FLAG_OWN_CAR",
                "CNT_CHILDREN",
            ),
            description="Small extra feature set that strengthens the MVP signal.",
        ),
    )

    # These columns are not required in the raw CSV; they are documented here so
    # Block 2 can create them consistently.
    derived_columns: tuple[str, ...] = (
        "AGE_YEARS",
        "EMPLOYED_YEARS",
        "DAYS_EMPLOYED_ANOM",
        "EXT_SOURCE_1_WAS_MISSING",
        "EXT_SOURCE_2_WAS_MISSING",
        "EXT_SOURCE_3_WAS_MISSING",
        "EXT_NULL_COUNT",
    )

    def training_file_path(self) -> Path:
        """Return the expected path to `application_train.csv`.

        Args:
            None.

        Returns:
            Path to the MVP training CSV inside `raw_data_dir`.

        Raises:
            None.
        """

        # Keeping path construction in one method avoids duplicated string
        # concatenation across notebooks and modules.
        return self.raw_data_dir / self.training_file.file_name

    def official_test_file_path(self) -> Path:
        """Return the expected path to Kaggle's official test CSV.

        Args:
            None.

        Returns:
            Path to `application_test.csv` inside `raw_data_dir`.

        Raises:
            None.
        """

        # This path is exposed for optional inference, not for final metrics.
        return self.raw_data_dir / self.official_test_file.file_name

    def required_raw_columns(self) -> tuple[str, ...]:
        """Return the complete ordered list of raw columns required for the MVP.

        Args:
            None.

        Returns:
            Tuple containing identifier, target, sensitive column and all MVP
            raw feature columns.

        Raises:
            None.
        """

        # The identifier is first because it anchors traceability.
        leading_columns: tuple[str, ...] = (
            self.identifier_column,
            self.target_column,
            self.sensitive_column,
        )

        # Feature columns are flattened from their logical groups, preserving
        # the declared order for reproducible reading.
        return leading_columns + self.raw_feature_columns()

    def raw_feature_columns(self) -> tuple[str, ...]:
        """Return the raw feature columns allowed in the MVP model pipeline.

        Args:
            None.

        Returns:
            Ordered tuple of raw feature columns before deterministic
            transformations.

        Raises:
            None.
        """

        # We build the tuple explicitly instead of storing a second duplicate
        # list, reducing the risk of drift between groups and total features.
        columns: list[str] = []

        # Each group contributes its columns to the final raw feature list.
        for group in self.feature_groups:
            columns.extend(group.columns)

        # Returning a tuple keeps the public API immutable.
        return tuple(columns)

    def excluded_from_model_columns(self) -> tuple[str, ...]:
        """Return columns that must not be used as normal model features.

        Args:
            None.

        Returns:
            Tuple with identifier, target, raw sensitive and engineered
            sensitive columns.

        Raises:
            None.
        """

        # TARGET is the label, not an input.
        # CODE_GENDER/SENSITIVE are used for fairness auditing and penalty, not
        # concatenated to X as ordinary predictors.
        # SK_ID_CURR is an identifier, not behavioral information.
        return (
            self.identifier_column,
            self.target_column,
            self.sensitive_column,
            self.engineered_sensitive_column,
        )

    def financial_columns(self) -> tuple[str, ...]:
        """Return monetary columns used later by the custom ratio layer.

        Args:
            None.

        Returns:
            Tuple of raw financial amount columns.

        Raises:
            DataContractError: If the financial group is missing from the
                contract definition.
        """

        # Financial columns are looked up by group name so downstream code does
        # not depend on magic positions in `feature_groups`.
        return self._columns_for_group("financial")

    def external_score_columns(self) -> tuple[str, ...]:
        """Return EXT_SOURCE columns used for prediction and uncertainty checks.

        Args:
            None.

        Returns:
            Tuple containing `EXT_SOURCE_1`, `EXT_SOURCE_2` and `EXT_SOURCE_3`.

        Raises:
            DataContractError: If the external score group is missing.
        """

        # These columns are central because their missingness is studied later
        # by the uncertainty block.
        return self._columns_for_group("external_scores")

    def categorical_columns(self) -> tuple[str, ...]:
        """Return low-cardinality categorical columns selected for the MVP.

        Args:
            None.

        Returns:
            Tuple of categorical columns that Block 2 will one-hot encode.

        Raises:
            DataContractError: If the categorical group is missing.
        """

        # Keeping this accessor prevents preprocessing code from hardcoding the
        # categorical list in multiple places.
        return self._columns_for_group("categorical_low_cardinality")

    def all_registered_files(self) -> tuple[DatasetFileSpec, ...]:
        """Return all Kaggle files known by the project contract.

        Args:
            None.

        Returns:
            Tuple containing MVP training file, official test file and advanced
            relational files.

        Raises:
            None.
        """

        # The order communicates priority: MVP first, optional inference second,
        # advanced enrichment after that.
        return (self.training_file, self.official_test_file) + self.relational_files

    def to_dict(self) -> dict[str, Any]:
        """Serialize the contract into plain Python objects.

        Args:
            None.

        Returns:
            Dictionary suitable for JSON serialization, logging or notebooks.

        Raises:
            None.
        """

        # The dictionary intentionally contains both raw and derived columns so
        # teammates can inspect the whole Block 1 decision in one object.
        return {
            "raw_data_dir": str(self.raw_data_dir),
            "training_file": self.training_file.file_name,
            "official_test_file": self.official_test_file.file_name,
            "relational_files": [item.file_name for item in self.relational_files],
            "identifier_column": self.identifier_column,
            "target_column": self.target_column,
            "sensitive_column": self.sensitive_column,
            "engineered_sensitive_column": self.engineered_sensitive_column,
            "required_raw_columns": list(self.required_raw_columns()),
            "raw_feature_columns": list(self.raw_feature_columns()),
            "excluded_from_model_columns": list(self.excluded_from_model_columns()),
            "derived_columns": list(self.derived_columns),
            "feature_groups": {
                group.name: {
                    "columns": list(group.columns),
                    "description": group.description,
                }
                for group in self.feature_groups
            },
        }

    def _columns_for_group(self, group_name: str) -> tuple[str, ...]:
        """Return columns for one registered feature group.

        Args:
            group_name: Name of the group to retrieve.

        Returns:
            Tuple of columns in the requested group.

        Raises:
            DataContractError: If no group with `group_name` exists.
        """

        # We scan the small immutable tuple because clarity matters more than
        # micro-optimization in a contract object.
        for group in self.feature_groups:
            if group.name == group_name:
                return group.columns

        # Failing loudly here protects later blocks from silently running with
        # missing schema groups.
        raise DataContractError(f"Unknown feature group in contract: {group_name}")


class DataContractValidator:
    """Validate raw files and column collections against a data contract.

    Args:
        contract: Contract object defining the expected MVP file and columns.

    Returns:
        Validator instance bound to one contract.

    Raises:
        None.
    """

    def __init__(self, contract: HomeCreditMVPDataContract) -> None:
        """Initialize the validator with a specific contract.

        Args:
            contract: Immutable Home Credit MVP contract to enforce.

        Returns:
            None.

        Raises:
            None.
        """

        # The validator receives the contract through dependency injection,
        # which keeps validation logic separate from schema declaration.
        self._contract = contract

    @property
    def contract(self) -> HomeCreditMVPDataContract:
        """Return the contract used by this validator.

        Args:
            None.

        Returns:
            HomeCreditMVPDataContract associated with the validator.

        Raises:
            None.
        """

        # Exposing the contract read-only helps notebooks inspect the active
        # schema without mutating it.
        return self._contract

    def validate_training_file_exists(self) -> DataContractValidationResult:
        """Validate that the MVP training CSV exists on disk.

        Args:
            None.

        Returns:
            Validation result with `is_valid=True` when the file exists.

        Raises:
            None. Use `assert_training_file_exists` for throwing behavior.
        """

        # The contract knows where the file should be.
        path = self._contract.training_file_path()

        # Existence is a Block 1 concern because all later blocks depend on the
        # correct source file being present.
        exists = path.exists()

        # The message is precise so a teammate knows exactly where to place the
        # downloaded Kaggle CSV.
        message = (
            f"Found MVP training file at {path}."
            if exists
            else f"Missing MVP training file at {path}."
        )

        return DataContractValidationResult(is_valid=exists, message=message)

    def assert_training_file_exists(self) -> None:
        """Raise if `application_train.csv` is not present.

        Args:
            None.

        Returns:
            None.

        Raises:
            DataContractError: If the expected MVP training file is missing.
        """

        # We reuse the non-throwing validator so both code paths stay aligned.
        result = self.validate_training_file_exists()

        # Raising early is safer than letting pandas fail later with a less
        # project-specific message.
        if not result.is_valid:
            raise DataContractError(result.message)

    def validate_columns(
        self,
        available_columns: Iterable[str],
        *,
        strict: bool = False,
    ) -> DataContractValidationResult:
        """Validate a collection of column names against the MVP contract.

        Args:
            available_columns: Column names found in a CSV or DataFrame.
            strict: When True, unexpected columns are reported as invalid. When
                False, only missing required columns invalidate the result.

        Returns:
            Validation result describing missing and unexpected columns.

        Raises:
            None. Use `assert_columns` for throwing behavior.
        """

        # Convert to tuple first so the result can report the exact checked
        # order, which helps debug wrong file reads.
        checked_columns = tuple(str(column) for column in available_columns)

        # Sets make membership checks simple and robust to ordering differences.
        checked_set = set(checked_columns)

        # Required columns are declared by the contract and must all be present.
        required_columns = self._contract.required_raw_columns()

        # Preserve contract order in the missing list so error messages are
        # deterministic.
        missing = tuple(
            column for column in required_columns if column not in checked_set
        )

        # Unexpected columns are only a problem in strict mode; they can appear
        # when a teammate reads the whole Kaggle CSV instead of `usecols`.
        expected_set = set(required_columns)
        unexpected = tuple(
            column for column in checked_columns if column not in expected_set
        )

        # Non-strict mode is valid as long as all required columns exist.
        is_valid = not missing and (not strict or not unexpected)

        # Messages are explicit because schema failures are common early in a
        # data project.
        if is_valid:
            message = "Column contract validation passed."
        elif missing:
            message = "Missing required MVP columns: " + ", ".join(missing)
        else:
            message = "Unexpected columns in strict mode: " + ", ".join(unexpected)

        return DataContractValidationResult(
            is_valid=is_valid,
            missing_required_columns=missing,
            unexpected_columns=unexpected if strict else tuple(),
            checked_columns=checked_columns,
            message=message,
        )

    def assert_columns(
        self,
        available_columns: Iterable[str],
        *,
        strict: bool = False,
    ) -> None:
        """Raise if available columns violate the MVP contract.

        Args:
            available_columns: Column names found in a CSV or DataFrame.
            strict: Whether columns outside the MVP contract should fail.

        Returns:
            None.

        Raises:
            DataContractError: If required columns are missing or strict mode
                detects unexpected columns.
        """

        # Use the result-producing method to keep all validation rules in one
        # place.
        result = self.validate_columns(available_columns, strict=strict)

        # A contract violation should stop later preprocessing before leakage or
        # wrong feature selection can happen.
        if not result.is_valid:
            raise DataContractError(result.message)

    def validate_mapping_keys(
        self,
        mapping: Mapping[str, Any],
        *,
        strict: bool = False,
    ) -> DataContractValidationResult:
        """Validate keys of a mapping as if they were data columns.

        Args:
            mapping: Mapping whose keys represent column names.
            strict: Whether keys outside the MVP contract should fail.

        Returns:
            Validation result for the mapping keys.

        Raises:
            None.
        """

        # This helper lets unit tests validate simple dictionaries without
        # importing pandas.
        return self.validate_columns(mapping.keys(), strict=strict)


class DataContractReporter:
    """Create human-readable summaries from the MVP data contract.

    Args:
        contract: Contract object to summarize.

    Returns:
        Reporter instance bound to the given contract.

    Raises:
        None.
    """

    def __init__(self, contract: HomeCreditMVPDataContract) -> None:
        """Initialize the reporter.

        Args:
            contract: Contract object whose decisions will be summarized.

        Returns:
            None.

        Raises:
            None.
        """

        # Keeping reporting outside the contract follows single responsibility:
        # the contract stores rules, the reporter formats them.
        self._contract = contract

    def summary_lines(self) -> list[str]:
        """Build a concise text summary of the MVP contract.

        Args:
            None.

        Returns:
            List of text lines suitable for printing in notebooks.

        Raises:
            None.
        """

        # A list of lines is easier to test than a pre-joined string.
        lines: list[str] = []

        # File decisions appear first because they are the highest-level scope
        # boundary of the MVP.
        lines.append("MVP data contract")
        lines.append(f"- training file: {self._contract.training_file.file_name}")
        lines.append(
            f"- official test file: {self._contract.official_test_file.file_name} "
            "(inference only, no TARGET)"
        )

        # Column roles are summarized next so the team can quickly verify that
        # CODE_GENDER and TARGET are not model features.
        lines.append(f"- identifier: {self._contract.identifier_column}")
        lines.append(f"- target: {self._contract.target_column}")
        lines.append(f"- sensitive raw column: {self._contract.sensitive_column}")
        lines.append(
            "- engineered sensitive column: "
            f"{self._contract.engineered_sensitive_column}"
        )

        # Feature groups are listed group-by-group to mirror the contract
        # structure.
        for group in self._contract.feature_groups:
            columns = ", ".join(group.columns)
            lines.append(f"- {group.name}: {columns}")

        # Derived columns remind the reader that these are created later, not
        # expected in the raw CSV.
        derived = ", ".join(self._contract.derived_columns)
        lines.append(f"- derived later: {derived}")

        return lines

    def as_text(self) -> str:
        """Return the contract summary as a printable text block.

        Args:
            None.

        Returns:
            Newline-joined summary string.

        Raises:
            None.
        """

        # Joining happens here so callers can choose either list or text form.
        return "\n".join(self.summary_lines())


def build_default_home_credit_contract(
    raw_data_dir: str | Path = Path("data/raw"),
) -> HomeCreditMVPDataContract:
    """Build the default Home Credit MVP contract.

    Args:
        raw_data_dir: Directory where raw Kaggle CSV files are expected.

    Returns:
        HomeCreditMVPDataContract configured for the project MVP.

    Raises:
        None.
    """

    # The factory keeps object creation simple for notebooks and later modules.
    return HomeCreditMVPDataContract(raw_data_dir=Path(raw_data_dir))


def build_default_contract_validator(
    raw_data_dir: str | Path = Path("data/raw"),
) -> DataContractValidator:
    """Build a validator for the default Home Credit MVP contract.

    Args:
        raw_data_dir: Directory where raw Kaggle CSV files are expected.

    Returns:
        DataContractValidator bound to the default contract.

    Raises:
        None.
    """

    # This helper is intentionally small but convenient for first notebook cells.
    contract = build_default_home_credit_contract(raw_data_dir=raw_data_dir)

    # The validator is returned separately so downstream code can validate files
    # without knowing construction details.
    return DataContractValidator(contract=contract)


__all__ = [
    "ColumnGroup",
    "DataContractError",
    "DataContractReporter",
    "DataContractValidationResult",
    "DataContractValidator",
    "DatasetFileSpec",
    "HomeCreditMVPDataContract",
    "build_default_contract_validator",
    "build_default_home_credit_contract",
]
