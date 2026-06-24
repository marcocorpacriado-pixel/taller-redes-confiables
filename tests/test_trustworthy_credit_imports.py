"""Regression tests for the unified trustworthy credit bridge package."""

import src.trustworthy_credit as trustworthy_credit
from src.trustworthy_credit.artifacts import (
    TrainingArtifactWriter,
    TuningArtifactPaths,
    UncertaintyArtifactPaths,
    UncertaintyArtifactWriter,
)
from src.trustworthy_credit.config import (
    BaseModelConfig,
    CustomMLPConfig,
    FairModelConfig,
    ReproducibilityConfig,
    TuningConfig,
    UncertaintyModelConfig,
)
from src.trustworthy_credit.data_contract import HomeCreditMVPDataContract
from src.trustworthy_credit.extended_features import ExtendedFeaturePreprocessingPipeline
from src.trustworthy_credit.layers import FairnessPenalty, FinancialRatiosLayer
from src.trustworthy_credit.metrics import FairnessMetricCalculator
from src.trustworthy_credit.models import FairCustomModelBuilder
from src.trustworthy_credit.preprocessing import (
    HomeCreditMVPPreprocessingPipeline,
    ProcessedSplitDataset,
)
from src.trustworthy_credit.splitting import HomeCreditTrainValTestSplitter
from src.trustworthy_credit.tuning import FairLambdaSweepTrainer
from src.trustworthy_credit.uncertainty import UncertaintyMVPTrainer


def test_unified_package_has_version() -> None:
    """The unified package should expose a lightweight root import."""

    assert trustworthy_credit.__version__ == "0.1.0"


def test_unified_bridges_point_to_validated_dani_modules() -> None:
    """Bridge imports should keep using the validated Dani MVP implementation."""

    assert HomeCreditMVPDataContract.__module__ == "src.dani_credit.data_contract"
    assert ProcessedSplitDataset.__module__ == "src.dani_credit.preprocessing"
    assert HomeCreditMVPPreprocessingPipeline.__module__ == "src.dani_credit.preprocessing"
    assert HomeCreditTrainValTestSplitter.__module__ == "src.dani_credit.splitting"
    assert FinancialRatiosLayer.__module__ == "src.dani_credit.layers"
    assert FairnessPenalty.__module__ == "src.dani_credit.layers"
    assert FairCustomModelBuilder.__module__ == "src.dani_credit.models"
    assert FairnessMetricCalculator.__module__ == "src.dani_credit.metrics"
    assert FairLambdaSweepTrainer.__module__ == "src.dani_credit.tuning"
    assert UncertaintyMVPTrainer.__module__ == "src.dani_credit.uncertainty"
    assert (
        ExtendedFeaturePreprocessingPipeline.__module__
        == "src.trustworthy_credit.extended_features"
    )


def test_unified_config_and_artifact_bridges_are_available() -> None:
    """Common configuration and artifact classes should be importable."""

    assert ReproducibilityConfig.__module__ == "src.dani_credit.base_model"
    assert BaseModelConfig.__module__ == "src.dani_credit.base_model"
    assert CustomMLPConfig.__module__ == "src.dani_credit.models"
    assert FairModelConfig.__module__ == "src.dani_credit.models"
    assert TuningConfig.__module__ == "src.dani_credit.tuning"
    assert TuningArtifactPaths.__module__ == "src.dani_credit.tuning"
    assert UncertaintyModelConfig.__module__ == "src.dani_credit.uncertainty"
    assert UncertaintyArtifactPaths.__module__ == "src.dani_credit.uncertainty"
    assert TrainingArtifactWriter.__module__ == "src.dani_credit.tuning"
    assert UncertaintyArtifactWriter.__module__ == "src.dani_credit.uncertainty"
