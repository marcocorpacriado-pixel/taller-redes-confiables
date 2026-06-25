import json

import numpy as np
import pandas as pd

from src.trustworthy_credit.gbm_experiments import (
    GBMArtifactPaths,
    OOFResult,
    OOFResultWriter,
)
from src.trustworthy_credit.relational_features import (
    BureauFeatureBuilder,
    RelationalFeatureConfig,
    RelationalFeaturePipeline,
)


def test_bureau_aggregation_keeps_one_row_per_customer():
    bureau = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2],
            "SK_ID_BUREAU": [10, 11, 20],
            "CREDIT_ACTIVE": ["Active", "Closed", "Active"],
            "AMT_CREDIT_SUM_DEBT": [100.0, 0.0, 50.0],
            "AMT_CREDIT_SUM": [200.0, 150.0, 100.0],
            "AMT_CREDIT_SUM_OVERDUE": [0.0, 5.0, 0.0],
            "CNT_CREDIT_PROLONG": [0, 1, 0],
            "AMT_CREDIT_MAX_OVERDUE": [0.0, 5.0, 0.0],
        }
    )
    bureau_balance = pd.DataFrame(
        {
            "SK_ID_BUREAU": [10, 10, 11, 20],
            "MONTHS_BALANCE": [-1, -2, -1, -1],
            "STATUS": ["0", "1", "C", "0"],
        }
    )

    features = BureauFeatureBuilder().build(bureau, bureau_balance)

    assert features["SK_ID_CURR"].is_unique
    assert set(features["SK_ID_CURR"]) == {1, 2}
    assert "BUR_N_CREDITS" in features.columns


def test_relational_pipeline_aligns_train_test_and_excludes_target(tmp_path):
    _write_csv(
        tmp_path,
        "application_train.csv",
        pd.DataFrame(
            {
                "SK_ID_CURR": [1, 2],
                "TARGET": [0, 1],
                "CODE_GENDER": ["F", "M"],
            }
        ),
    )
    _write_csv(
        tmp_path,
        "application_test.csv",
        pd.DataFrame({"SK_ID_CURR": [3], "CODE_GENDER": ["F"]}),
    )
    _write_csv(
        tmp_path,
        "bureau.csv",
        pd.DataFrame({"SK_ID_CURR": [1, 2], "SK_ID_BUREAU": [10, 20]}),
    )
    _write_csv(
        tmp_path,
        "bureau_balance.csv",
        pd.DataFrame(
            {
                "SK_ID_BUREAU": [10, 20],
                "MONTHS_BALANCE": [-1, -1],
                "STATUS": ["0", "1"],
            }
        ),
    )
    _write_csv(
        tmp_path,
        "previous_application.csv",
        pd.DataFrame({"SK_ID_CURR": [1, 2], "SK_ID_PREV": [100, 200]}),
    )
    _write_csv(
        tmp_path,
        "installments_payments.csv",
        pd.DataFrame({"SK_ID_CURR": [1, 2], "DAYS_INSTALMENT": [-10, -20]}),
    )
    _write_csv(
        tmp_path,
        "POS_CASH_balance.csv",
        pd.DataFrame({"SK_ID_CURR": [1, 2], "SK_ID_PREV": [100, 200]}),
    )
    _write_csv(
        tmp_path,
        "credit_card_balance.csv",
        pd.DataFrame({"SK_ID_CURR": [1, 2], "SK_ID_PREV": [100, 200]}),
    )

    dataset = RelationalFeaturePipeline(
        RelationalFeatureConfig(raw_data_dir=tmp_path)
    ).build()

    assert "TARGET" not in dataset.feature_columns
    assert list(dataset.X_train.columns) == list(dataset.X_test.columns)
    assert len(dataset.target) == 2
    assert dataset.train["SK_ID_CURR"].is_unique
    assert dataset.test["SK_ID_CURR"].is_unique


def test_oof_result_writer_uses_results_extras_contract(tmp_path):
    paths = GBMArtifactPaths(project_root=tmp_path, run_id=" smoke run ")
    result = OOFResult(
        model_name="LightGBM Relational",
        oof_auc=0.75,
        oof_accuracy=0.9,
        fold_aucs=(0.7, 0.8),
        oof_predictions=np.array([0.1, 0.8]),
        test_predictions=np.array([0.2]),
        feature_importance=pd.DataFrame(
            {"feature": ["EXT_SOURCE_2", "BUR_N_CREDITS"], "importance": [5.0, 3.0]}
        ),
        params={"learning_rate": 0.01},
    )

    written = OOFResultWriter().write(result, paths)

    assert paths.run_dir == tmp_path / "results" / "extras" / "smoke-run"
    for path in written.values():
        assert path.exists()
        assert path.resolve().is_relative_to(paths.run_dir.resolve())

    metrics = json.loads(written["metrics"].read_text(encoding="utf-8"))
    assert metrics["model_name"] == "LightGBM Relational"
    assert metrics["oof_auc"] == 0.75


def _write_csv(root, file_name, frame):
    frame.to_csv(root / file_name, index=False)
