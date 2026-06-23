"""Configuration bridge for the unified trustworthy credit package.

The validated configuration dataclasses still live in ``src.dani_credit``.
This module provides their future canonical import location without changing
runtime behavior.
"""

from src.dani_credit.base_model import BaseModelConfig, ReproducibilityConfig
from src.dani_credit.models import CustomMLPConfig, FairModelConfig
from src.dani_credit.tuning import TuningArtifactPaths, TuningConfig
from src.dani_credit.uncertainty import (
    UncertaintyArtifactPaths,
    UncertaintyModelConfig,
)

__all__ = [
    "BaseModelConfig",
    "CustomMLPConfig",
    "FairModelConfig",
    "ReproducibilityConfig",
    "TuningArtifactPaths",
    "TuningConfig",
    "UncertaintyArtifactPaths",
    "UncertaintyModelConfig",
]
