"""M2 uncertainty bridge for the unified trustworthy credit package.

The professor-aligned MVP uncertainty approach is M1 -> M2: M2 predicts the
absolute error expected from the classifier. This module exposes the validated
Dani implementation through the future common package path.
"""

from src.dani_credit.uncertainty import (
    DualInputModelPredictor,
    FairModelLoader,
    UncertaintyArtifactPaths,
    UncertaintyArtifactWriter,
    UncertaintyCallbackFactory,
    UncertaintyError,
    UncertaintyFeatureBuilder,
    UncertaintyInternalSplitter,
    UncertaintyM2ModelBuilder,
    UncertaintyMVPResult,
    UncertaintyMVPTrainer,
    UncertaintyModelConfig,
    UncertaintyPredictionBuilder,
    UncertaintyPredictionResult,
    UncertaintySummaryBuilder,
    UncertaintyTrainingData,
    UncertaintyTrainingDataBuilder,
)

__all__ = [
    "DualInputModelPredictor",
    "FairModelLoader",
    "UncertaintyArtifactPaths",
    "UncertaintyArtifactWriter",
    "UncertaintyCallbackFactory",
    "UncertaintyError",
    "UncertaintyFeatureBuilder",
    "UncertaintyInternalSplitter",
    "UncertaintyM2ModelBuilder",
    "UncertaintyMVPResult",
    "UncertaintyMVPTrainer",
    "UncertaintyModelConfig",
    "UncertaintyPredictionBuilder",
    "UncertaintyPredictionResult",
    "UncertaintySummaryBuilder",
    "UncertaintyTrainingData",
    "UncertaintyTrainingDataBuilder",
]
