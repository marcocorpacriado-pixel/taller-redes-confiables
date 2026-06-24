"""Regression tests for the uncertainty and audit metadata pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.trustworthy_credit.preprocessing import (
    HomeCreditFeaturePreprocessor,
    PreprocessingColumnSpec,
    RawSplitDataset,
)


def test_m2_softplus_head_produces_non_constant_synthetic_uncertainty() -> None:
    """M2 should learn a non-constant error signal on a simple synthetic case."""

    tf = pytest.importorskip("tensorflow")
    uncertainty_module = pytest.importorskip("src.trustworthy_credit.uncertainty")

    tf.keras.utils.set_random_seed(123)

    z_train = np.column_stack(
        [
            np.linspace(-2.0, 2.0, 64),
            np.linspace(1000.0, 64000.0, 64),
            np.linspace(0.05, 0.95, 64),
        ]
    ).astype("float32")
    error_train = (
        0.05
        + 0.20 * (z_train[:, 0] > 0.0).astype("float32")
        + 0.10 * z_train[:, 2]
    ).astype("float32")

    config = uncertainty_module.UncertaintyModelConfig(
        hidden_units=(16, 8),
        dropout=0.0,
        output_activation="softplus",
        learning_rate=0.01,
        batch_size=16,
        epochs=40,
        early_stopping_patience=5,
    )
    model = uncertainty_module.UncertaintyM2ModelBuilder(config).build(
        input_dim=z_train.shape[1],
        normalization_data=z_train,
    )

    model.fit(
        z_train,
        error_train,
        validation_split=0.2,
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=0,
    )

    raw_uncertainty = model.predict(z_train[:12], verbose=0).reshape(-1)
    uncertainty = np.clip(raw_uncertainty, 0.0, 1.0)

    assert np.isfinite(uncertainty).all()
    assert uncertainty.min() >= 0.0
    assert uncertainty.max() <= 1.0
    assert np.unique(np.round(uncertainty, decimals=6)).size > 1


def test_prediction_builder_rejects_constant_uncertainty() -> None:
    """A constant uncertainty vector should fail before artifacts are saved."""

    uncertainty_module = pytest.importorskip("src.trustworthy_credit.uncertainty")

    builder = uncertainty_module.UncertaintyPredictionBuilder()

    with pytest.raises(uncertainty_module.UncertaintyError, match="constant uncertainty"):
        builder._clip_and_validate_uncertainty(np.zeros(8, dtype="float32"))


def test_processed_dataset_preserves_raw_ext_null_count_values() -> None:
    """ProcessedSplitDataset should keep EXT_NULL_COUNT as raw 0-3 metadata."""

    column_spec = PreprocessingColumnSpec(
        financial_cols=("AMT_INCOME_TOTAL",),
        continuous_scaled_cols=("EXT_NULL_COUNT",),
        binary_cols=("FLAG_OWN_CAR",),
        categorical_cols=("NAME_CONTRACT_TYPE",),
    )

    raw_splits = RawSplitDataset(
        X_train=_synthetic_frame(index_start=100, ext_null_count=(0, 1, 2, 3, 0, 1)),
        X_val=_synthetic_frame(index_start=200, ext_null_count=(2, 3, 0, 1)),
        X_test=_synthetic_frame(index_start=300, ext_null_count=(0, 1, 2, 3)),
        y_train=_synthetic_series(index_start=100, values=(0, 1, 0, 1, 0, 1)),
        y_val=_synthetic_series(index_start=200, values=(0, 1, 0, 1)),
        y_test=_synthetic_series(index_start=300, values=(0, 1, 0, 1)),
        s_train=_synthetic_series(index_start=100, values=(0, 1, 0, 1, 0, 1)),
        s_val=_synthetic_series(index_start=200, values=(0, 1, 0, 1)),
        s_test=_synthetic_series(index_start=300, values=(0, 1, 0, 1)),
    )

    processed = HomeCreditFeaturePreprocessor(column_spec).fit_transform_splits(
        raw_splits
    )

    assert processed.ext_null_count_train.tolist() == [0, 1, 2, 3, 0, 1]
    assert processed.ext_null_count_val.tolist() == [2, 3, 0, 1]
    assert processed.ext_null_count_test.tolist() == [0, 1, 2, 3]
    assert set(processed.ext_null_count_test.tolist()).issubset({0, 1, 2, 3})


def test_prediction_builder_rejects_scaled_ext_null_count_values() -> None:
    """Scaled EXT_NULL_COUNT values such as -1.0 are invalid for reporting."""

    uncertainty_module = pytest.importorskip("src.trustworthy_credit.uncertainty")

    builder = uncertainty_module.UncertaintyPredictionBuilder()

    with pytest.raises(uncertainty_module.UncertaintyError, match="raw integer values"):
        builder._validate_ext_null_count(
            np.array([-1.0, 0.0, 1.0], dtype="float32"),
            expected_rows=3,
        )


def _synthetic_frame(
    *,
    index_start: int,
    ext_null_count: tuple[int, ...],
) -> pd.DataFrame:
    """Build a minimal deterministic feature frame for preprocessing tests."""

    index = list(range(index_start, index_start + len(ext_null_count)))
    rows = len(ext_null_count)

    return pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": np.linspace(100000.0, 250000.0, rows),
            "EXT_NULL_COUNT": np.array(ext_null_count, dtype="int8"),
            "FLAG_OWN_CAR": np.arange(rows) % 2,
            "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"]
            * ((rows + 1) // 2),
        },
        index=index,
    ).iloc[:rows]


def _synthetic_series(*, index_start: int, values: tuple[int, ...]) -> pd.Series:
    """Build an indexed numeric series aligned with a synthetic frame."""

    index = list(range(index_start, index_start + len(values)))
    return pd.Series(values, index=index)
