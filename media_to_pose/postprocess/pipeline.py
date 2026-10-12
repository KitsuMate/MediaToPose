"""Run clean-up modules over one pose or a whole baked sequence."""

from __future__ import annotations

from collections.abc import Iterator
from typing import (
    Any,
)

from ..retarget.target_pose import TargetPose
from .base import PostProcessingSession, SequenceProcessingContext
from .config import compile_postprocess_config, ensure_module_configs
from .registry import CORRECTION_REGISTRY


def validate_modules(armature) -> list[str]:
    config = armature.mtp_rig_config
    ensure_module_configs(config)
    warnings = []
    for item in config.post_processing_modules:
        module = CORRECTION_REGISTRY[item.module_id]
        profile = config.detected_profile
        if profile not in module.supported_profiles:
            issue = (
                f"{item.display_name or item.module_id} does not support "
                f"the {profile} rig profile"
            )
            item.status = "WARNING"
            item.last_warning = issue
            if item.enabled:
                warnings.append(issue)
            continue
        issues = module.validate(armature, item)
        item.status = "WARNING" if issues else "VALID"
        item.last_warning = "; ".join(issues)
        if item.enabled:
            warnings.extend(issues)
    return warnings


def iter_pipeline(
    pose: TargetPose,
    config,
    *,
    session: PostProcessingSession | None = None,
    module_filter: set[str] | None = None,
) -> Iterator[tuple[str, int, int]]:
    """Apply enabled modules cooperatively, yielding before every module."""
    compiled = compile_postprocess_config(config)
    pose.diagnostic_mode = compiled.diagnostic_mode
    session = session or PostProcessingSession()
    session.prepare_frame(int(pose.frame.timestamp_ms))
    warnings = []
    modules = [
        item
        for item in compiled.modules
        if CORRECTION_REGISTRY[item.module_id].scope == "FRAME"
        if (
            item.module_id in module_filter
            if module_filter is not None
            else item.enabled
        )
    ]
    total = len(modules)
    for index, item in enumerate(modules, 1):
        module_id = item.module_id
        yield item.display_name or module_id, index, total
        module = CORRECTION_REGISTRY[module_id]
        profile = compiled.detected_profile
        if profile not in module.supported_profiles:
            warning = (
                f"{item.display_name or item.module_id} does not support "
                f"the {profile} rig profile"
            )
            warnings.append(warning)
            continue
        before = (
            pose.semantic_endpoints()
            if compiled.diagnostic_mode and not module.allows_endpoint_motion
            else None
        )
        module_warnings = module.process_frame(pose, item, session)
        pose.applied_modules.append(item.module_id)
        if before is not None:
            after = pose.semantic_endpoints()
            displacement, semantic_id, endpoint = pose.maximum_endpoint_delta(
                before, after
            )
            if displacement > 1.0e-3:
                module_warnings.append(
                    f"{item.display_name or item.module_id} remained enabled: "
                    f"{semantic_id} {endpoint} moved "
                    f"{displacement * 100.0:.2f}% of its bone length"
                )
        if module_warnings:
            warnings.extend(module_warnings)
    pose.warnings.extend(warnings)
    return warnings


def iter_sequence_pipeline(
    context: SequenceProcessingContext,
    config,
) -> Iterator[Any]:
    """Run enabled complete-sequence processors in registry order."""
    compiled = compile_postprocess_config(config)
    modules = [
        item
        for item in compiled.modules
        if item.enabled and CORRECTION_REGISTRY[item.module_id].scope == "SEQUENCE"
    ]
    for item in modules:
        yield from CORRECTION_REGISTRY[item.module_id].process_sequence(context, item)


def run_pipeline(
    pose: TargetPose,
    config,
    *,
    session: PostProcessingSession | None = None,
    module_filter: set[str] | None = None,
) -> list[str]:
    """Apply the complete pipeline synchronously."""
    iterator = iter_pipeline(
        pose,
        config,
        session=session,
        module_filter=module_filter,
    )
    while True:
        try:
            next(iterator)
        except StopIteration as stopped:
            return stopped.value
