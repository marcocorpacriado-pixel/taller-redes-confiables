"""Tests for the optional extended 42-feature preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.trustworthy_credit.extended_features import (
    ExtendedFeatureError,
    ExtendedFeaturePreprocessingPipeline,
    ExtendedFeatureSelectionConfig,
    SmoothedTargetEncoder,
)


def _synthetic_home_credit_frame(n_rows: int = 80) -> pd.DataFrame:
    """Build a small Home Credit-like frame with balanced target/gender groups."""

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
                "SK_ID_CURR": 100000 + i,
                "TARGET": target,
                "CODE_GENDER": gender,
                "AMT_INCOME_TOTAL": 100000.0 + 500.0 * i,
                "AMT_CREDIT": 250000.0 + 1000.0 * i,
                "AMT_ANNUITY": np.nan if i % 17 == 0 else 12000.0 + 100.0 * i,
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
                "AMT_GOODS_PRICE": 220000.0 + 900.0 * i,
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


def test_extended_config_rejects_invalid_split_settings() -> None:
    """Invalid split or target encoding settings should fail early."""

    with pytest.raises(ExtendedFeatureError, match="positive"):
        ExtendedFeatureSelectionConfig(test_size=0.0)
    with pytest.raises(ExtendedFeatureError, match="leave positive train"):
        ExtendedFeatureSelectionConfig(test_size=0.7, validation_size=0.4)
    with pytest.raises(ExtendedFeatureError, match="smoothing"):
        ExtendedFeatureSelectionConfig(target_encoding_smoothing=-1.0)


def test_smoothed_target_encoder_maps_unseen_categories_to_train_global_mean() -> None:
    """Target encoding must not learn from validation/test categories."""

    categories = pd.Series(["A", "A", "B", "B"])
    target = pd.Series([0.0, 1.0, 1.0, 1.0])
    encoder = SmoothedTargetEncoder(smoothing=2.0).fit(categories, target)

    encoded = encoder.transform(pd.Series(["A", "B", "UNSEEN"]))

    assert encoder.global_mean_ == pytest.approx(0.75)
    assert encoded.iloc[2] == pytest.approx(0.75)
    assert encoded.iloc[0] != encoded.iloc[1]


def test_extended_pipeline_outputs_leakage_safe_processed_splits() -> None:
    """The extended pipeline should split first and return numeric clean matrices."""

    frame = _synthetic_home_credit_frame()
    pipeline = ExtendedFeaturePreprocessingPipeline(
        ExtendedFeatureSelectionConfig(random_state=7)
    )

    result = pipeline.fit_transform(frame)

    assert result.X_train.shape[0] == len(result.y_train) == len(result.s_train)
    assert result.X_val.shape[0] == len(result.y_val) == len(result.s_val)
    assert result.X_test.shape[0] == len(result.y_test) == len(result.s_test)
    assert result.X_train.shape[1] == len(result.feature_names)
    assert result.X_val.shape[1] == result.X_train.shape[1]
    assert result.X_test.shape[1] == result.X_train.shape[1]

    assert "TARGET" not in result.feature_names
    assert "SK_ID_CURR" not in result.feature_names
    assert "CODE_GENDER" not in result.feature_names
    assert "AGE_YEARS" in result.feature_names
    assert "DEBT_RATIO" in result.feature_names
    assert "DAYS_EMPLOYED_ANOMALY" in result.feature_names
    assert "NAME_INCOME_TYPE" in result.feature_names
    assert any(feature.startswith("NAME_FAMILY_STATUS_") for feature in result.feature_names)

    assert not np.isnan(result.X_train).any()
    assert not np.isnan(result.X_val).any()
    assert not np.isnan(result.X_test).any()
    assert set(np.unique(result.s_train)).issubset({0.0, 1.0})
    assert set(result.split_report["split"]) == {"train", "val", "test"}
    assert not result.audit_table.empty


def test_extended_pipeline_can_include_sensitive_as_explicit_feature() -> None:
    """Sensitive-as-feature should be opt-in and traceable."""

    frame = _synthetic_home_credit_frame()
    pipeline = ExtendedFeaturePreprocessingPipeline(
        ExtendedFeatureSelectionConfig(
            include_sensitive_as_feature=True,
            random_state=11,
        )
    )

    result = pipeline.fit_transform(frame)

    assert "CODE_GENDER" in result.feature_names
    assert result.X_train_frame["CODE_GENDER"].isin([0.0, 1.0]).all()


def test_extended_pipeline_scales_only_train_fit_columns() -> None:
    """Scaled train columns should be centered using train-only statistics."""

    frame = _synthetic_home_credit_frame()
    result = ExtendedFeaturePreprocessingPipeline(
        ExtendedFeatureSelectionConfig(random_state=13)
    ).fit_transform(frame)

    assert result.scale_features
    train_scaled_means = result.X_train_frame.loc[:, result.scale_features].mean()
    assert np.allclose(train_scaled_means.to_numpy(), 0.0, atol=1e-6)
