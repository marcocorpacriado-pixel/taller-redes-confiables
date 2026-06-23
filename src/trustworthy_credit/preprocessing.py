"""Preprocessing bridge for the unified trustworthy credit package.

The Dani preprocessing pipeline is the canonical MVP implementation because it
preserves raw audit features such as ``EXT_NULL_COUNT`` while keeping the model
matrix processed and scaled.
"""

from src.dani_credit.preprocessing import (
    DeterministicDataset,
    HomeCreditDeterministicTransformer,
    HomeCreditFeaturePreprocessor,
    HomeCreditMVPPreprocessingPipeline,
    HomeCreditPreprocessingColumnSpecFactory,
    HomeCreditRawDataLoader,
    PreprocessingColumnSpec,
    PreprocessingError,
    ProcessedSplitDataset,
    RawSplitDataset,
)

__all__ = [
    "DeterministicDataset",
    "HomeCreditDeterministicTransformer",
    "HomeCreditFeaturePreprocessor",
    "HomeCreditMVPPreprocessingPipeline",
    "HomeCreditPreprocessingColumnSpecFactory",
    "HomeCreditRawDataLoader",
    "PreprocessingColumnSpec",
    "PreprocessingError",
    "ProcessedSplitDataset",
    "RawSplitDataset",
]
