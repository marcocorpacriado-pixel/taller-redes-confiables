"""Tests for executable extended-feature training experiments."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.trustworthy_credit.extended_features import ExtendedFeatureSelectionConfig
from src.trustworthy_credit.extended_training import (
    CompactVsExtendedComparator,
    ExtendedFeatureExperimentRunner,
    ExtendedTrainingConfig,
    ExtendedTrainingDatasetBuilder,
    ExtendedTrainingError,
)


def _synthetic_home_credit_frame(n_rows: int = 120) -> pd.DataFrame:
    """Build a small Home Credit-like frame with all extended columns."""

    rows = []
    genders = ["M", "F"]
    family = ["Single / not married", "Married", "Civil marriage", "Widow"]
    education = ["Secondary / secondary special", "Higher education", "Incomplete higher"]
    housing = ["House / apartment", "With parents", "Municipal apartment"]
    income_type = ["Working", "Commercial associate", "Pensioner"]
    occupation = ["Laborers", "Core staff", "Managers", None]
    organization = ["Business Entity Type 3", "Self-employed", "Government"]

    for i in range(n_rows):
        target = i % 2
        gender = genders[(i // 2) % 2]
        rows.append(
            {
                "SK_ID_CURR": 200000 + i,
                "TARGET": target,
                "CODE_GENDER": gender,
                "AMT_INCOME_TOTAL": 100000.0 + 700.0 * i,
                "AMT_CREDIT": 250000.0 + 1200.0 * i,
                "AMT_ANNUITY": np.nan if i % 19 == 0 else 12000.0 + 110.0 * i,
                "EXT_SOURCE_1": np.nan if i % 5 == 0 else 0.2 + (i % 10) / 20.0,
                "EXT_SOURCE_2": np.nan if i % 7 == 0 else 0.3 + (i % 9) / 20.0,
                "EXT_SOURCE_3": np.nan if i % 11 == 0 else 0.4 + (i % 8) / 20.0,
                "REGION_RATING_CLIENT_W_CITY": 1 + i % 3,
                "DAYS_LAST_PHONE_CHANGE": -float(20 + i),
                "OWN_CAR_AGE": np.nan if i % 4 == 0 else float(1 + i % 12),
                "DAYS_ID_PUBLISH": -float(100 + i),
                "REG_CITY_NOT_WORK_CITY": float(i % 2),
                "FLAG_EMP_PHONE": float((i + 1) % 2),
                "DAYS_REGISTRATION": -float(300 + i),
                "AMT_GOODS_PRICE": 220000.0 + 950.0 * i,
                "DAYS_EMPLOYED": 365243 if i % 13 == 0 else -float(200 + i),
                "DAYS_BIRTH": -float(9000 + i),
                "NAME_CONTRACT_TYPE": "Cash loans" if i % 3 else "Revolving loans",
                "NAME_FAMILY_STATUS": family[i % len(family)],
                "NAME_EDUCATION_TYPE": education[i % len(education)],
                "NAME_HOUSING_TYPE": housing[i % len(housing)],
                "NAME_INCOME_TYPE": income_type[i % len(income_type)],
                "OCCUPATION_TYPE": occupation[i % len(occupation)],
                "ORGANIZATION_TYPE": organization[i % len(organization)],
            }
        )
    return pd.DataFrame(rows)


def test_extended_training_config_rejects_invalid_values() -> None:
    """Invalid extended training settings should fail early."""

    with pytest.raises(ExtendedTrainingError, match="model_id"):
        ExtendedTrainingConfig(model_ids=("BAD",))
    with pytest.raises(ExtendedTrainingError, match="epochs"):
        ExtendedTrainingConfig(epochs=0)
    with pytest.raises(ExtendedTrainingError, match="dropout"):
        ExtendedTrainingConfig(dropout_rates=(1.0, 0.2))


def test_extended_training_dataset_builder_returns_progression_data() -> None:
    """Builder should connect extended preprocessing to progression inputs."""

    builder = ExtendedTrainingDatasetBuilder(
        ExtendedFeatureSelectionConfig(random_state=21)
    )
    data = builder.build(_synthetic_home_credit_frame())

    assert data.progression_dataset.X_train.shape[1] == len(
        data.feature_set.feature_names
    )
    assert data.progression_dataset.X_val.shape[1] == len(data.feature_set.feature_names)
    assert data.progression_dataset.X_test.shape[1] == len(data.feature_set.feature_names)
    assert data.feature_indices.require_debt_ratio() >= 0
    assert len(data.feature_indices.require_ext_sources()) == 3
    assert not data.feature_set.audit_table.empty


def test_extended_feature_experiment_runner_trains_and_writes_outputs(tmp_path) -> None:
    """Runner should train selected models and persist tables/histories."""

    runner = ExtendedFeatureExperimentRunner(
        training_config=ExtendedTrainingConfig(
            model_ids=("M0", "M3"),
            epochs=2,
            batch_size=16,
            learning_rate=1e-2,
            hidden_units=(12, 6),
            m1_units=8,
            dropout_rates=(0.0, 0.0),
            early_stopping_patience=2,
            seed=17,
        ),
        dataset_builder=ExtendedTrainingDatasetBuilder(
            ExtendedFeatureSelectionConfig(random_state=17)
        ),
    )

    results = runner.run(
        _synthetic_home_credit_frame(),
        output_dir=tmp_path,
        verbose=0,
    )

    assert results["model_id"].tolist() == ["M0", "M3"]
    assert set(results.columns) >= {
        "feature_setup",
        "n_features",
        "test_auc",
        "test_pr_auc",
        "test_abs_rho",
        "test_dpd",
        "test_eod",
    }
    assert results["feature_setup"].eq("extended").all()
    assert results["test_auc"].between(0.0, 1.0).all()
    assert results["test_pr_auc"].between(0.0, 1.0).all()

    assert (tmp_path / "extended_feature_training_results.csv").exists()
    assert (tmp_path / "extended_feature_audit.csv").exists()
    assert (tmp_path / "extended_split_report.csv").exists()
    for model_id in ("M0", "M3"):
        history_path = tmp_path / f"extended_{model_id}_history.json"
        assert history_path.exists()
        history = json.loads(history_path.read_text(encoding="utf-8"))
        assert "val_auc" in history


def test_compact_vs_extended_comparator_builds_gain_table_and_figure(tmp_path) -> None:
    """Comparator should quantify and plot AUC lift from compact to extended."""

    compact = pd.DataFrame(
        {
            "model_id": ["M0", "M3"],
            "reported_test_auc": [0.7335, 0.7457],
        }
    )
    extended = pd.DataFrame(
        {
            "model_id": ["M0", "M3"],
            "test_auc": [0.7501, 0.7555],
        }
    )
    comparator = CompactVsExtendedComparator()

    comparison = comparator.compare(compact, extended)
    figure_path = comparator.save_auc_figure(
        comparison,
        tmp_path / "compact_vs_extended_auc.png",
    )

    assert comparison["model_id"].tolist() == ["M0", "M3"]
    assert comparison.loc[0, "auc_gain_extended_vs_compact"] == pytest.approx(
        0.0166
    )
    assert comparison["relative_auc_gain_pct"].gt(0).all()
    assert figure_path.exists()


def test_compact_vs_extended_comparator_rejects_no_shared_models() -> None:
    """Comparison should fail clearly when model identifiers do not overlap."""

    comparator = CompactVsExtendedComparator()
    compact = pd.DataFrame({"model_id": ["M0"], "test_auc": [0.73]})
    extended = pd.DataFrame({"model_id": ["M3"], "test_auc": [0.75]})

    with pytest.raises(ExtendedTrainingError, match="No shared"):
        comparator.compare(compact, extended)
