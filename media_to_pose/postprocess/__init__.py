"""Post-processing: clean-up modules, their settings schema and the pipeline."""

from __future__ import annotations

from .base import CorrectionModule, PostProcessingSession, SequenceProcessingContext
from .config import (
    compile_postprocess_config,
    ensure_module_configs,
    load_module_configs,
    module_config,
    ordered_module_configs,
    reset_module_defaults,
    serialize_module_configs,
)
from .pipeline import (
    iter_pipeline,
    iter_sequence_pipeline,
    run_pipeline,
    validate_modules,
)
from .registry import CORRECTION_REGISTRY
from .settings_schema import (
    MODULE_SETTING_FIELDS,
    POST_PROCESSING_DEFAULT_ENABLED,
    POST_PROCESSING_MODULES,
)

__all__ = [
    "CORRECTION_REGISTRY",
    "CorrectionModule",
    "MODULE_SETTING_FIELDS",
    "POST_PROCESSING_DEFAULT_ENABLED",
    "POST_PROCESSING_MODULES",
    "PostProcessingSession",
    "SequenceProcessingContext",
    "compile_postprocess_config",
    "ensure_module_configs",
    "iter_pipeline",
    "iter_sequence_pipeline",
    "load_module_configs",
    "module_config",
    "ordered_module_configs",
    "reset_module_defaults",
    "run_pipeline",
    "serialize_module_configs",
    "validate_modules",
]
