"""Model builder bridge for the unified trustworthy credit package."""

from src.dani_credit.models import (
    CustomMLPConfig,
    CustomMLPModelBuilder,
    CustomModelBuildResult,
    CustomModelError,
    CustomProbabilityGraph,
    FairCustomModelBuilder,
    FairModelBuildResult,
    FairModelConfig,
    FairModelError,
    build_fair_custom_model,
    custom_model_objects,
    lambda_slug,
)

__all__ = [
    "CustomMLPConfig",
    "CustomMLPModelBuilder",
    "CustomModelBuildResult",
    "CustomModelError",
    "CustomProbabilityGraph",
    "FairCustomModelBuilder",
    "FairModelBuildResult",
    "FairModelConfig",
    "FairModelError",
    "build_fair_custom_model",
    "custom_model_objects",
    "lambda_slug",
]
