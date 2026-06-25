"""Relational feature engineering for the Home Credit grade-10 extras.

The MVP deliberately uses a small, auditable feature contract. This module is
the advanced path: it lifts the strongest ideas from the full-dataset
XGBoost/LightGBM notebooks into reusable classes while keeping the same safety
rules that matter for the final project:

- one row per ``SK_ID_CURR`` after every relational aggregation;
- ``TARGET`` never appears in the final feature matrix;
- train and official Kaggle test frames finish with aligned feature columns;
- generated artifacts can be explained source by source in the extras notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class RelationalFeatureError(ValueError):
    """Raised when relational feature construction violates its contract."""


@dataclass(frozen=True)
class RelationalFeatureConfig:
    """Configuration for the full relational feature pipeline."""

    raw_data_dir: Path = Path("data/raw")
    id_column: str = "SK_ID_CURR"
    target_column: str = "TARGET"
    sensitive_column: str = "CODE_GENDER"
    include_sensitive_feature: bool = True
    recent_installment_days: int = 365
    recent_months: int = 12


@dataclass(frozen=True)
class ApplicationFeatureResult:
    """Application table features plus supervised metadata."""

    train: pd.DataFrame
    test: pd.DataFrame
    target: pd.Series
    train_ids: pd.Series
    test_ids: pd.Series


@dataclass(frozen=True)
class RelationalFeatureDataset:
    """Final aligned train/test dataset produced by relational features."""

    train: pd.DataFrame
    test: pd.DataFrame
    target: pd.Series
    train_ids: pd.Series
    test_ids: pd.Series
    feature_columns: tuple[str, ...]
    source_summary: pd.DataFrame

    @property
    def X_train(self) -> pd.DataFrame:
        """Return the training feature matrix."""

        return self.train.loc[:, list(self.feature_columns)]

    @property
    def X_test(self) -> pd.DataFrame:
        """Return the official Kaggle test feature matrix."""

        return self.test.loc[:, list(self.feature_columns)]


class RelationalDataLoader:
    """Load Home Credit CSVs with conservative memory downcasting."""

    expected_files: tuple[str, ...] = (
        "application_train.csv",
        "application_test.csv",
        "bureau.csv",
        "bureau_balance.csv",
        "previous_application.csv",
        "installments_payments.csv",
        "POS_CASH_balance.csv",
        "credit_card_balance.csv",
    )

    def __init__(self, raw_data_dir: str | Path = Path("data/raw")) -> None:
        self.raw_data_dir = Path(raw_data_dir)

    def assert_expected_files_exist(self) -> None:
        """Raise if any CSV needed by the extras pipeline is missing."""

        missing = [
            file_name
            for file_name in self.expected_files
            if not (self.raw_data_dir / file_name).exists()
        ]
        if missing:
            raise RelationalFeatureError(
                "Missing relational CSV files: " + ", ".join(missing)
            )

    def read_csv(self, file_name: str, usecols: Sequence[str] | None = None) -> pd.DataFrame:
        """Read one CSV from ``raw_data_dir`` and reduce numeric memory usage."""

        path = self.raw_data_dir / file_name
        if not path.exists():
            raise RelationalFeatureError(f"Missing CSV file: {path}")
        return reduce_memory(pd.read_csv(path, usecols=usecols))


class ApplicationFeatureBuilder:
    """Build derived features from application_train/application_test."""

    days_employed_sentinel: int = 365243

    def __init__(self, config: RelationalFeatureConfig | None = None) -> None:
        self.config = config or RelationalFeatureConfig()

    def build(self, app_train: pd.DataFrame, app_test: pd.DataFrame) -> ApplicationFeatureResult:
        """Return application features, encoded categoricals and target."""

        train = app_train.copy()
        test = app_test.copy()

        if self.config.sensitive_column in train.columns:
            train = train[train[self.config.sensitive_column] != "XNA"].reset_index(drop=True)

        train = self.transform(train)
        test = self.transform(test)
        train, test = self._encode_categoricals(train, test)

        target = train[self.config.target_column].copy()
        train_ids = train[self.config.id_column].copy()
        test_ids = test[self.config.id_column].copy()

        return ApplicationFeatureResult(
            train=reduce_memory(train),
            test=reduce_memory(test),
            target=target,
            train_ids=train_ids,
            test_ids=test_ids,
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add application-level ratios, EXT_SOURCE interactions and flags."""

        out = frame.copy()
        if "DAYS_EMPLOYED" in out.columns:
            out["DAYS_EMPLOYED_ANOM"] = (
                out["DAYS_EMPLOYED"] == self.days_employed_sentinel
            ).astype("int8")
            out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].replace(
                self.days_employed_sentinel,
                np.nan,
            )

        ext = [col for col in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if col in out]
        if ext:
            out["EXT_MEAN"] = out[ext].mean(axis=1)
            out["EXT_STD"] = out[ext].std(axis=1)
            out["EXT_MIN"] = out[ext].min(axis=1)
            out["EXT_MAX"] = out[ext].max(axis=1)
            out["EXT_NANCOUNT"] = out[ext].isna().sum(axis=1)
            out["EXT_RANGE"] = out["EXT_MAX"] - out["EXT_MIN"]
        self._add_product(out, "EXT_PROD", ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"])
        self._add_product(out, "EXT_S1xS2", ["EXT_SOURCE_1", "EXT_SOURCE_2"])
        self._add_product(out, "EXT_S1xS3", ["EXT_SOURCE_1", "EXT_SOURCE_3"])
        self._add_product(out, "EXT_S2xS3", ["EXT_SOURCE_2", "EXT_SOURCE_3"])
        self._add_product(out, "EXT_S2xBIRTH", ["EXT_SOURCE_2", "DAYS_BIRTH"])
        self._add_product(out, "EXT_S3xBIRTH", ["EXT_SOURCE_3", "DAYS_BIRTH"])
        self._add_product(out, "EXT_S2xEMPL", ["EXT_SOURCE_2", "DAYS_EMPLOYED"])
        self._add_product(out, "EXT_S3xEMPL", ["EXT_SOURCE_3", "DAYS_EMPLOYED"])
        if {"EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"}.issubset(out.columns):
            out["EXT_WEIGHTED"] = (
                2 * out["EXT_SOURCE_2"] + out["EXT_SOURCE_3"] + 0.5 * out["EXT_SOURCE_1"]
            )

        self._add_ratio(out, "CREDIT_INCOME_RATIO", "AMT_CREDIT", "AMT_INCOME_TOTAL")
        self._add_ratio(out, "ANNUITY_INCOME_RATIO", "AMT_ANNUITY", "AMT_INCOME_TOTAL")
        self._add_ratio(out, "CREDIT_ANNUITY_RATIO", "AMT_CREDIT", "AMT_ANNUITY")
        self._add_ratio(out, "CREDIT_GOODS_RATIO", "AMT_CREDIT", "AMT_GOODS_PRICE")
        self._add_ratio(out, "GOODS_INCOME_RATIO", "AMT_GOODS_PRICE", "AMT_INCOME_TOTAL")
        self._add_ratio(out, "PAYMENT_LENGTH", "AMT_CREDIT", "AMT_ANNUITY")
        if {"AMT_GOODS_PRICE", "AMT_CREDIT"}.issubset(out.columns):
            out["DOWN_PAYMENT"] = out["AMT_GOODS_PRICE"] - out["AMT_CREDIT"]
            out["DOWN_PAYMENT_RATIO"] = out["DOWN_PAYMENT"] / (out["AMT_GOODS_PRICE"] + 1)
        self._add_ratio(out, "INCOME_PER_CHILD", "AMT_INCOME_TOTAL", "CNT_CHILDREN")
        self._add_ratio(out, "INCOME_PER_FAM", "AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS")
        self._add_ratio(out, "CREDIT_PER_PERSON", "AMT_CREDIT", "CNT_FAM_MEMBERS")
        self._add_ratio(out, "ANNUITY_PER_PERSON", "AMT_ANNUITY", "CNT_FAM_MEMBERS")
        self._add_ratio(out, "CREDIT_TERM", "AMT_ANNUITY", "AMT_CREDIT")

        if "DAYS_BIRTH" in out.columns:
            out["DAYS_BIRTH_YRS"] = out["DAYS_BIRTH"] / -365.25
            out["AGE_RANGE"] = pd.cut(
                out["DAYS_BIRTH_YRS"],
                bins=[0, 25, 30, 35, 40, 45, 50, 55, 60, 65, 100],
                labels=False,
            )
        if "DAYS_EMPLOYED" in out.columns:
            out["DAYS_EMPLOYED_YRS"] = out["DAYS_EMPLOYED"] / -365.25
        self._add_ratio(out, "EMPLOYED_TO_BIRTH", "DAYS_EMPLOYED", "DAYS_BIRTH")
        self._add_ratio(out, "ID_TO_BIRTH", "DAYS_ID_PUBLISH", "DAYS_BIRTH")
        self._add_ratio(out, "PHONE_TO_BIRTH", "DAYS_LAST_PHONE_CHANGE", "DAYS_BIRTH")
        self._add_ratio(out, "PHONE_TO_EMPLOYED", "DAYS_LAST_PHONE_CHANGE", "DAYS_EMPLOYED")
        self._add_ratio(out, "REG_TO_BIRTH", "DAYS_REGISTRATION", "DAYS_BIRTH")
        self._add_ratio(out, "EMPLOYED_TO_ID", "DAYS_EMPLOYED", "DAYS_ID_PUBLISH")
        self._add_ratio(out, "DAYS_EMPLOYED_PERC", "DAYS_EMPLOYED", "DAYS_BIRTH")

        doc_cols = [col for col in out.columns if "FLAG_DOCUMENT" in col]
        if doc_cols:
            out["DOCUMENT_COUNT"] = out[doc_cols].sum(axis=1).astype("float32")
        contact_cols = [
            col
            for col in (
                "FLAG_MOBIL",
                "FLAG_EMP_PHONE",
                "FLAG_WORK_PHONE",
                "FLAG_CONT_MOBILE",
                "FLAG_PHONE",
                "FLAG_EMAIL",
            )
            if col in out.columns
        ]
        if contact_cols:
            out["FLAG_CONTACTS_SUM"] = out[contact_cols].sum(axis=1).astype("float32")

        self._add_ratio(out, "DEF_30_RATIO", "DEF_30_CNT_SOCIAL_CIRCLE", "OBS_30_CNT_SOCIAL_CIRCLE")
        self._add_ratio(out, "DEF_60_RATIO", "DEF_60_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE")
        bureau_request_cols = [
            col for col in out.columns if col.startswith("AMT_REQ_CREDIT_BUREAU_")
        ]
        if bureau_request_cols:
            out["AMT_REQ_SUM"] = out[bureau_request_cols].sum(axis=1).astype("float32")

        out["APP_NULLS"] = out.isna().sum(axis=1).astype("int16")
        return out

    def _encode_categoricals(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Label-encode object columns using the train+official-test vocabulary."""

        train_out = train.copy()
        test_out = test.copy()
        categorical_cols = sorted(
            set(train_out.select_dtypes("object").columns).union(
                set(test_out.select_dtypes("object").columns)
            )
        )
        for col in categorical_cols:
            values = []
            if col in train_out.columns:
                values.append(train_out[col].astype(str))
            if col in test_out.columns:
                values.append(test_out[col].astype(str))
            encoder = LabelEncoder().fit(pd.concat(values, ignore_index=True))
            if col in train_out.columns:
                train_out[col] = encoder.transform(train_out[col].astype(str)).astype("int16")
            if col in test_out.columns:
                test_out[col] = encoder.transform(test_out[col].astype(str)).astype("int16")
        return train_out, test_out

    def _add_ratio(self, frame: pd.DataFrame, name: str, numerator: str, denominator: str) -> None:
        if {numerator, denominator}.issubset(frame.columns):
            frame[name] = frame[numerator] / (frame[denominator] + 1)

    def _add_product(self, frame: pd.DataFrame, name: str, columns: Sequence[str]) -> None:
        if set(columns).issubset(frame.columns):
            value = frame[columns[0]]
            for col in columns[1:]:
                value = value * frame[col]
            frame[name] = value


class BureauBalanceAggregator:
    """Aggregate bureau_balance from monthly status rows to bureau credits."""

    status_mapping: dict[str, float] = {
        "0": 0.0,
        "1": 1.0,
        "2": 2.0,
        "3": 3.0,
        "4": 4.0,
        "5": 5.0,
        "C": 0.0,
        "X": np.nan,
    }

    def transform(self, bureau_balance: pd.DataFrame) -> pd.DataFrame:
        frame = bureau_balance.copy()
        frame["STATUS_NUM"] = frame["STATUS"].map(self.status_mapping).astype("float32")
        aggregated = frame.groupby("SK_ID_BUREAU").agg(
            BB_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
            BB_STATUS_MAX=("STATUS_NUM", "max"),
            BB_STATUS_MEAN=("STATUS_NUM", "mean"),
            BB_STATUS_STD=("STATUS_NUM", "std"),
            BB_MONTHS_MIN=("MONTHS_BALANCE", "min"),
            BB_DPD_MONTHS=("STATUS_NUM", lambda values: (values > 0).sum()),
        )
        return reduce_memory(aggregated.reset_index())


class BureauFeatureBuilder:
    """Aggregate bureau and bureau_balance to one row per applicant."""

    def __init__(self, balance_aggregator: BureauBalanceAggregator | None = None) -> None:
        self.balance_aggregator = balance_aggregator or BureauBalanceAggregator()

    def build(self, bureau: pd.DataFrame, bureau_balance: pd.DataFrame) -> pd.DataFrame:
        frame = bureau.copy()
        balance = self.balance_aggregator.transform(bureau_balance)
        frame = frame.merge(balance, on="SK_ID_BUREAU", how="left")

        if "CREDIT_ACTIVE" in frame.columns:
            frame["CREDIT_ACTIVE_BIN"] = (frame["CREDIT_ACTIVE"] == "Active").astype("int8")
        encode_object_columns_inplace(frame)

        active = frame[frame["CREDIT_ACTIVE_BIN"] == 1] if "CREDIT_ACTIVE_BIN" in frame else frame.iloc[0:0]
        closed = frame[frame["CREDIT_ACTIVE_BIN"] == 0] if "CREDIT_ACTIVE_BIN" in frame else frame.iloc[0:0]

        blocks = [
            aggregate_numeric_by_curr(
                frame,
                "BUR",
                exclude=("SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE_BIN"),
            ),
            aggregate_numeric_by_curr(
                active,
                "BUR_ACT",
                exclude=("SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE_BIN"),
            ),
            aggregate_numeric_by_curr(
                closed,
                "BUR_CLS",
                exclude=("SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE_BIN"),
            ),
            self._extra_features(frame),
        ]
        return merge_feature_blocks(blocks)

    def _extra_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = {
            "SK_ID_BUREAU",
            "CREDIT_ACTIVE_BIN",
            "AMT_CREDIT_SUM_DEBT",
            "AMT_CREDIT_SUM",
            "AMT_CREDIT_SUM_OVERDUE",
            "CNT_CREDIT_PROLONG",
            "AMT_CREDIT_MAX_OVERDUE",
        }
        if not required.issubset(frame.columns):
            return frame[["SK_ID_CURR"]].drop_duplicates().reset_index(drop=True)

        extra = frame.groupby("SK_ID_CURR").agg(
            BUR_N_CREDITS=("SK_ID_BUREAU", "count"),
            BUR_N_ACTIVE=("CREDIT_ACTIVE_BIN", "sum"),
            BUR_DEBT_TOTAL=("AMT_CREDIT_SUM_DEBT", "sum"),
            BUR_CREDIT_TOTAL=("AMT_CREDIT_SUM", "sum"),
            BUR_OVERDUE_TOTAL=("AMT_CREDIT_SUM_OVERDUE", "sum"),
            BUR_PROLONG_TOTAL=("CNT_CREDIT_PROLONG", "sum"),
            BUR_DEBT_RATIO=("AMT_CREDIT_SUM_DEBT", "mean"),
            BUR_MAX_OVERDUE=("AMT_CREDIT_MAX_OVERDUE", "max"),
        )
        extra = extra.reset_index()
        extra["BUR_DEBT_CREDIT_RATIO"] = (
            extra["BUR_DEBT_TOTAL"] / (extra["BUR_CREDIT_TOTAL"] + 1)
        ).astype("float32")
        return reduce_memory(extra)


class PreviousApplicationAggregator:
    """Aggregate previous_application to applicant-level features."""

    days_sentinel: int = 365243

    def build(self, previous_application: pd.DataFrame) -> pd.DataFrame:
        frame = previous_application.copy()
        status_raw = (
            frame["NAME_CONTRACT_STATUS"].astype(str)
            if "NAME_CONTRACT_STATUS" in frame
            else pd.Series("", index=frame.index)
        )

        for col in [col for col in frame.columns if "DAYS" in col]:
            frame[col] = frame[col].replace(self.days_sentinel, np.nan)

        add_ratio_columns(
            frame,
            specs=(
                ("PREV_CREDIT_DIFF", "AMT_APPLICATION", "AMT_CREDIT", "diff"),
                ("PREV_CREDIT_RATIO", "AMT_CREDIT", "AMT_APPLICATION", "ratio"),
                ("PREV_DOWN_PAYMENT_RATIO", "AMT_DOWN_PAYMENT", "AMT_APPLICATION", "ratio"),
                ("PREV_ANNUITY_RATIO", "AMT_ANNUITY", "AMT_APPLICATION", "ratio"),
                ("PREV_PAYMENT_LENGTH", "AMT_CREDIT", "AMT_ANNUITY", "ratio"),
            ),
        )
        encode_object_columns_inplace(frame)

        approved = frame[status_raw == "Approved"]
        refused = frame[status_raw != "Approved"]
        blocks = [
            aggregate_numeric_by_curr(frame, "PREV", exclude=("SK_ID_CURR", "SK_ID_PREV")),
            aggregate_numeric_by_curr(
                approved,
                "PREV_APPR",
                exclude=("SK_ID_CURR", "SK_ID_PREV"),
            ),
            aggregate_numeric_by_curr(
                refused,
                "PREV_REF",
                exclude=("SK_ID_CURR", "SK_ID_PREV"),
            ),
            self._extra_features(frame, status_raw),
        ]
        return merge_feature_blocks(blocks)

    def _extra_features(self, frame: pd.DataFrame, status_raw: pd.Series) -> pd.DataFrame:
        if "SK_ID_PREV" not in frame.columns:
            return frame[["SK_ID_CURR"]].drop_duplicates().reset_index(drop=True)

        temp = frame.copy()
        temp["_APPROVED"] = (status_raw == "Approved").astype("int8")
        named_aggs = {
            "PREV_N_APPS": ("SK_ID_PREV", "count"),
            "PREV_N_APPROVED": ("_APPROVED", "sum"),
            "PREV_APPROVAL_RATE": ("_APPROVED", "mean"),
        }
        optional = {
            "PREV_MAX_CREDIT": ("AMT_CREDIT", "max"),
            "PREV_MEAN_CREDIT": ("AMT_CREDIT", "mean"),
            "PREV_LAST_DAYS_DEC": ("DAYS_DECISION", "max"),
            "PREV_CREDIT_DIFF_MEAN": ("PREV_CREDIT_DIFF", "mean"),
            "PREV_CREDIT_RATIO_MEAN": ("PREV_CREDIT_RATIO", "mean"),
            "PREV_ANNUITY_RATIO_MEAN": ("PREV_ANNUITY_RATIO", "mean"),
            "PREV_PAYMENT_LEN_MEAN": ("PREV_PAYMENT_LENGTH", "mean"),
        }
        named_aggs.update({name: spec for name, spec in optional.items() if spec[0] in temp})
        return reduce_memory(temp.groupby("SK_ID_CURR").agg(**named_aggs).reset_index())


class InstallmentsAggregator:
    """Aggregate repayment behavior from installments_payments."""

    def __init__(self, recent_days: int = 365) -> None:
        self.recent_days = int(recent_days)

    def build(self, installments: pd.DataFrame) -> pd.DataFrame:
        frame = installments.copy()
        if {"AMT_INSTALMENT", "AMT_PAYMENT"}.issubset(frame.columns):
            frame["PAYMENT_DIFF"] = frame["AMT_INSTALMENT"] - frame["AMT_PAYMENT"]
            frame["PAYMENT_RATIO"] = frame["AMT_PAYMENT"] / (frame["AMT_INSTALMENT"] + 1)
            frame["UNDERPAYMENT"] = (frame["PAYMENT_DIFF"] > 0).astype("int8")
        if {"DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT"}.issubset(frame.columns):
            frame["DPD"] = (frame["DAYS_ENTRY_PAYMENT"] - frame["DAYS_INSTALMENT"]).clip(lower=0)
            frame["DBD"] = (frame["DAYS_INSTALMENT"] - frame["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
            frame["LATE_PAYMENT"] = (frame["DPD"] > 0).astype("int8")
            frame["EARLY_PAYMENT"] = (frame["DBD"] > 0).astype("int8")

        recent = frame[frame["DAYS_INSTALMENT"] >= -self.recent_days] if "DAYS_INSTALMENT" in frame else frame.iloc[0:0]
        return merge_feature_blocks(
            [
                self._aggregate(frame, "INST"),
                self._aggregate(recent, "INST_REC"),
            ]
        )

    def _aggregate(self, frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        feature_cols = [
            col
            for col in ("AMT_INSTALMENT", "AMT_PAYMENT", "PAYMENT_DIFF", "PAYMENT_RATIO", "DPD", "DBD")
            if col in frame.columns
        ]
        if frame.empty or not feature_cols:
            return frame[["SK_ID_CURR"]].drop_duplicates().reset_index(drop=True)
        stats = frame.groupby("SK_ID_CURR")[feature_cols].agg(["mean", "max", "sum", "std"])
        stats.columns = [f"{prefix}_{col}_{stat.upper()}" for col, stat in stats.columns]
        extra_specs = {}
        if "LATE_PAYMENT" in frame:
            extra_specs[f"{prefix}_LATE_RATE"] = ("LATE_PAYMENT", "mean")
        if "EARLY_PAYMENT" in frame:
            extra_specs[f"{prefix}_EARLY_RATE"] = ("EARLY_PAYMENT", "mean")
        if "UNDERPAYMENT" in frame:
            extra_specs[f"{prefix}_UNDER_RATE"] = ("UNDERPAYMENT", "mean")
        if "AMT_PAYMENT" in frame:
            extra_specs[f"{prefix}_N_PAYMENTS"] = ("AMT_PAYMENT", "count")
        extra = frame.groupby("SK_ID_CURR").agg(**extra_specs).reset_index()
        return reduce_memory(stats.reset_index().merge(extra, on="SK_ID_CURR", how="left"))


class POSCashAggregator:
    """Aggregate POS_CASH_balance to applicant-level features."""

    def __init__(self, recent_months: int = 12) -> None:
        self.recent_months = int(recent_months)

    def build(self, pos_cash: pd.DataFrame) -> pd.DataFrame:
        frame = pos_cash.copy()
        encode_object_columns_inplace(frame)
        if "SK_DPD" in frame:
            frame["POS_DPD_FLAG"] = (frame["SK_DPD"] > 0).astype("int8")
        if "SK_DPD_DEF" in frame:
            frame["POS_DPD_DEF_FLAG"] = (frame["SK_DPD_DEF"] > 0).astype("int8")
        recent = frame[frame["MONTHS_BALANCE"] >= -self.recent_months] if "MONTHS_BALANCE" in frame else frame.iloc[0:0]
        return merge_feature_blocks(
            [
                self._aggregate(frame, "POS"),
                self._aggregate(recent, "POS_REC"),
            ]
        )

    def _aggregate(self, frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if frame.empty:
            return frame[["SK_ID_CURR"]].drop_duplicates().reset_index(drop=True)
        numeric_cols = [
            col
            for col in frame.select_dtypes(include=[np.number]).columns
            if col not in {"SK_ID_CURR", "SK_ID_PREV"}
        ]
        if not numeric_cols:
            return frame[["SK_ID_CURR"]].drop_duplicates().reset_index(drop=True)
        stats = frame.groupby("SK_ID_CURR")[numeric_cols].agg(["mean", "max", "min", "sum"])
        stats.columns = [f"{prefix}_{col}_{stat.upper()}" for col, stat in stats.columns]
        extra_specs = {}
        if "MONTHS_BALANCE" in frame:
            extra_specs[f"{prefix}_N_MONTHS"] = ("MONTHS_BALANCE", "count")
        if "POS_DPD_FLAG" in frame:
            extra_specs[f"{prefix}_DPD_RATE"] = ("POS_DPD_FLAG", "mean")
        if "NAME_CONTRACT_STATUS" in frame:
            extra_specs[f"{prefix}_COMPLETED"] = (
                "NAME_CONTRACT_STATUS",
                lambda values: values.value_counts(normalize=True).iloc[0],
            )
        output = stats.reset_index()
        if extra_specs:
            extra = frame.groupby("SK_ID_CURR").agg(**extra_specs).reset_index()
            output = output.merge(extra, on="SK_ID_CURR", how="left")
        return reduce_memory(output)


class CreditCardAggregator:
    """Aggregate credit_card_balance to applicant-level features."""

    def __init__(self, recent_months: int = 12) -> None:
        self.recent_months = int(recent_months)

    def build(self, credit_card: pd.DataFrame) -> pd.DataFrame:
        frame = credit_card.copy()
        encode_object_columns_inplace(frame)
        add_ratio_columns(
            frame,
            specs=(
                ("CC_UTIL_RATIO", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "ratio"),
                ("CC_PAYMENT_RATIO", "AMT_PAYMENT_CURRENT", "AMT_INST_MIN_REGULARITY", "ratio"),
                ("CC_DRAWING_RATIO", "AMT_DRAWINGS_CURRENT", "AMT_CREDIT_LIMIT_ACTUAL", "ratio"),
            ),
        )
        if "SK_DPD" in frame:
            frame["CC_DPD_FLAG"] = (frame["SK_DPD"] > 0).astype("int8")
        recent = frame[frame["MONTHS_BALANCE"] >= -self.recent_months] if "MONTHS_BALANCE" in frame else frame.iloc[0:0]
        return merge_feature_blocks(
            [
                self._aggregate(frame, "CC"),
                self._aggregate(recent, "CC_REC"),
            ]
        )

    def _aggregate(self, frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if frame.empty:
            return frame[["SK_ID_CURR"]].drop_duplicates().reset_index(drop=True)
        numeric_cols = [
            col
            for col in frame.select_dtypes(include=[np.number]).columns
            if col not in {"SK_ID_CURR", "SK_ID_PREV"}
        ]
        if not numeric_cols:
            return frame[["SK_ID_CURR"]].drop_duplicates().reset_index(drop=True)
        stats = frame.groupby("SK_ID_CURR")[numeric_cols].agg(["mean", "max", "min", "sum"])
        stats.columns = [f"{prefix}_{col}_{stat.upper()}" for col, stat in stats.columns]
        extra_specs = {}
        if "MONTHS_BALANCE" in frame:
            extra_specs[f"{prefix}_N_MONTHS"] = ("MONTHS_BALANCE", "count")
        if "CC_DPD_FLAG" in frame:
            extra_specs[f"{prefix}_DPD_RATE"] = ("CC_DPD_FLAG", "mean")
        if "CC_UTIL_RATIO" in frame:
            extra_specs[f"{prefix}_UTIL_MEAN"] = ("CC_UTIL_RATIO", "mean")
            extra_specs[f"{prefix}_UTIL_MAX"] = ("CC_UTIL_RATIO", "max")
        if "CC_PAYMENT_RATIO" in frame:
            extra_specs[f"{prefix}_PAYMENT_MEAN"] = ("CC_PAYMENT_RATIO", "mean")
        output = stats.reset_index()
        if extra_specs:
            extra = frame.groupby("SK_ID_CURR").agg(**extra_specs).reset_index()
            output = output.merge(extra, on="SK_ID_CURR", how="left")
        return reduce_memory(output)


class RelationalFeaturePipeline:
    """End-to-end relational feature builder used by the extras notebook."""

    def __init__(
        self,
        config: RelationalFeatureConfig | None = None,
        loader: RelationalDataLoader | None = None,
    ) -> None:
        self.config = config or RelationalFeatureConfig()
        self.loader = loader or RelationalDataLoader(self.config.raw_data_dir)

    def build(self) -> RelationalFeatureDataset:
        """Load every CSV, build relational features and align train/test."""

        self.loader.assert_expected_files_exist()
        app = ApplicationFeatureBuilder(self.config).build(
            self.loader.read_csv("application_train.csv"),
            self.loader.read_csv("application_test.csv"),
        )

        train = app.train
        test = app.test
        blocks = [
            BureauFeatureBuilder().build(
                self.loader.read_csv("bureau.csv"),
                self.loader.read_csv("bureau_balance.csv"),
            ),
            PreviousApplicationAggregator().build(
                self.loader.read_csv("previous_application.csv")
            ),
            InstallmentsAggregator(self.config.recent_installment_days).build(
                self.loader.read_csv("installments_payments.csv")
            ),
            POSCashAggregator(self.config.recent_months).build(
                self.loader.read_csv("POS_CASH_balance.csv")
            ),
            CreditCardAggregator(self.config.recent_months).build(
                self.loader.read_csv("credit_card_balance.csv")
            ),
        ]

        for block in blocks:
            assert_unique_id(block, self.config.id_column)
            train = train.merge(block, on=self.config.id_column, how="left")
            test = test.merge(block, on=self.config.id_column, how="left")

        feature_columns = self._feature_columns(train)
        train, test, feature_columns = align_train_test_features(
            train=train,
            test=test,
            feature_columns=feature_columns,
            id_column=self.config.id_column,
            target_column=self.config.target_column,
        )
        source_summary = build_source_summary(feature_columns)

        return RelationalFeatureDataset(
            train=train,
            test=test,
            target=app.target,
            train_ids=app.train_ids,
            test_ids=app.test_ids,
            feature_columns=tuple(feature_columns),
            source_summary=source_summary,
        )

    def _feature_columns(self, train: pd.DataFrame) -> list[str]:
        excluded = {self.config.id_column, self.config.target_column}
        if not self.config.include_sensitive_feature:
            excluded.add(self.config.sensitive_column)
        columns = [col for col in train.columns if col not in excluded]
        if self.config.target_column in columns:
            raise RelationalFeatureError("TARGET leaked into feature columns.")
        return columns


def reduce_memory(frame: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns in-place and return the same frame."""

    for col in frame.columns:
        dtype = frame[col].dtype
        if pd.api.types.is_float_dtype(dtype):
            frame[col] = pd.to_numeric(frame[col], downcast="float")
        elif pd.api.types.is_integer_dtype(dtype):
            frame[col] = pd.to_numeric(frame[col], downcast="integer")
    return frame


def encode_object_columns_inplace(frame: pd.DataFrame) -> None:
    """Label-encode all object columns inside one relational table."""

    for col in frame.select_dtypes("object").columns:
        frame[col] = LabelEncoder().fit_transform(frame[col].astype(str)).astype("int32")


def aggregate_numeric_by_curr(
    frame: pd.DataFrame,
    prefix: str,
    *,
    exclude: Iterable[str],
    stats: Sequence[str] = ("mean", "max", "min", "sum", "count"),
) -> pd.DataFrame:
    """Aggregate numeric columns to one row per ``SK_ID_CURR``."""

    if frame.empty:
        return pd.DataFrame(columns=["SK_ID_CURR"])
    excluded = set(exclude)
    numeric_cols = [
        col
        for col in frame.select_dtypes(include=[np.number]).columns
        if col not in excluded
    ]
    if not numeric_cols:
        return frame[["SK_ID_CURR"]].drop_duplicates().reset_index(drop=True)
    aggregated = frame.groupby("SK_ID_CURR")[numeric_cols].agg(list(stats))
    aggregated.columns = [
        f"{prefix}_{col}_{stat.upper()}" for col, stat in aggregated.columns
    ]
    return reduce_memory(aggregated.reset_index())


def merge_feature_blocks(blocks: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Outer-merge applicant-level feature blocks and validate uniqueness."""

    non_empty = [block for block in blocks if not block.empty]
    if not non_empty:
        return pd.DataFrame(columns=["SK_ID_CURR"])

    merged = non_empty[0]
    assert_unique_id(merged, "SK_ID_CURR")
    for block in non_empty[1:]:
        assert_unique_id(block, "SK_ID_CURR")
        merged = merged.merge(block, on="SK_ID_CURR", how="outer")
    assert_unique_id(merged, "SK_ID_CURR")
    return reduce_memory(merged)


def assert_unique_id(frame: pd.DataFrame, id_column: str = "SK_ID_CURR") -> None:
    """Raise if an applicant-level frame has duplicated applicant IDs."""

    if id_column not in frame.columns:
        raise RelationalFeatureError(f"Missing id column: {id_column}")
    if frame[id_column].duplicated().any():
        raise RelationalFeatureError(f"Duplicated {id_column} values after aggregation.")


def add_ratio_columns(
    frame: pd.DataFrame,
    *,
    specs: Sequence[tuple[str, str, str, str]],
) -> None:
    """Add simple ratio or difference columns when source columns exist."""

    for output, left, right, kind in specs:
        if {left, right}.issubset(frame.columns):
            if kind == "ratio":
                frame[output] = frame[left] / (frame[right] + 1)
            elif kind == "diff":
                frame[output] = frame[left] - frame[right]
            else:
                raise RelationalFeatureError(f"Unknown feature operation: {kind}")


def align_train_test_features(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: Sequence[str],
    id_column: str,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Replace infinities, align feature columns and validate no TARGET leak."""

    if target_column in feature_columns:
        raise RelationalFeatureError("TARGET cannot be part of feature_columns.")

    train_out = train.copy().replace([np.inf, -np.inf], np.nan)
    test_out = test.copy().replace([np.inf, -np.inf], np.nan)

    for col in feature_columns:
        if col not in test_out.columns:
            test_out[col] = np.nan
    aligned_features = [col for col in feature_columns if col in train_out.columns]
    ordered_train_cols = [id_column, target_column] + aligned_features
    ordered_test_cols = [id_column] + aligned_features
    return (
        reduce_memory(train_out.loc[:, ordered_train_cols]),
        reduce_memory(test_out.loc[:, ordered_test_cols]),
        list(aligned_features),
    )


def build_source_summary(feature_columns: Sequence[str]) -> pd.DataFrame:
    """Count final features by their source table prefix."""

    rows: dict[str, int] = {
        "application": 0,
        "bureau": 0,
        "previous_application": 0,
        "installments_payments": 0,
        "pos_cash_balance": 0,
        "credit_card_balance": 0,
    }
    for col in feature_columns:
        if col.startswith("BUR"):
            rows["bureau"] += 1
        elif col.startswith("PREV"):
            rows["previous_application"] += 1
        elif col.startswith("INST"):
            rows["installments_payments"] += 1
        elif col.startswith("POS"):
            rows["pos_cash_balance"] += 1
        elif col.startswith("CC"):
            rows["credit_card_balance"] += 1
        else:
            rows["application"] += 1
    return pd.DataFrame(
        [{"source": source, "n_features": n_features} for source, n_features in rows.items()]
    )


__all__ = [
    "ApplicationFeatureBuilder",
    "ApplicationFeatureResult",
    "BureauBalanceAggregator",
    "BureauFeatureBuilder",
    "CreditCardAggregator",
    "InstallmentsAggregator",
    "POSCashAggregator",
    "PreviousApplicationAggregator",
    "RelationalDataLoader",
    "RelationalFeatureConfig",
    "RelationalFeatureDataset",
    "RelationalFeatureError",
    "RelationalFeaturePipeline",
    "aggregate_numeric_by_curr",
    "align_train_test_features",
    "assert_unique_id",
    "build_source_summary",
    "merge_feature_blocks",
    "reduce_memory",
]
