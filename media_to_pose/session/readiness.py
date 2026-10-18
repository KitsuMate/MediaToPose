"""Side-effect-free capture capability and apply-readiness evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.constants import FACE_OUTPUTS


def _descriptor(provider: str):
    # Provider descriptors pull in the inference backends; import on use so
    # the readiness rules stay importable (and testable) without Blender.
    from ..providers import descriptor

    return descriptor(provider)


def _fk_switch_issues(armature, *, include_arms: bool, include_legs: bool):
    from ..rig.profiles.fk_switches import fk_switch_issues

    return fk_switch_issues(
        armature, include_arms=include_arms, include_legs=include_legs
    )


@dataclass(frozen=True)
class CaptureReadiness:
    ready: bool
    issue: str
    provider_ready: bool
    provider_issue: str
    enabled_scope: bool
    mapping_issues: tuple[str, ...]
    media_source: Any


def enabled_capture_scope(settings, capabilities) -> bool:
    return bool(
        (settings.use_body and "BODY" in capabilities)
        or (settings.use_hands and "HANDS" in capabilities)
        or (settings.use_face and "FACE" in capabilities)
    )


def mapping_scope_issues(armature, settings) -> tuple[str, ...]:
    """Return current mapping blockers without mutating mapping status."""
    config = armature.mtp_rig_config
    if not config.mappings:
        return ("Rig setup has not been initialized",)
    issues = []
    if settings.use_body:
        missing = [
            entry.label
            for entry in config.mappings
            if (
                entry.section == "BODY"
                and entry.required
                and entry.status in {"MISSING", "ERROR"}
            )
        ]
        if missing:
            issues.append("Missing required body mappings: " + ", ".join(missing))
    if settings.use_hands and not any(
        entry.section == "HANDS" and entry.enabled and bool(entry.target_name)
        for entry in config.mappings
    ):
        issues.append("No finger mappings are available")
    if settings.use_body or settings.use_hands:
        issues.extend(
            _fk_switch_issues(
                armature,
                include_arms=settings.use_body or settings.use_hands,
                include_legs=settings.use_body,
            )
        )
    provider = _descriptor(settings.provider)
    face_enabled = settings.use_face and "FACE" in provider.capabilities
    if face_enabled:
        completeness = _output_completeness(config, settings.face_output)
        if completeness == 0.0 and not (settings.use_body or settings.use_hands):
            issues.append(
                f"The selected {settings.face_output} face output is fully missing"
            )
    return tuple(issues)


def _output_completeness(config, face_output: str) -> float:
    """How completely the rig maps the selected face output (0 to 1)."""
    return float(getattr(config, FACE_OUTPUTS[face_output][0]))


def face_output_warning(armature, settings) -> str:
    provider = _descriptor(settings.provider)
    if not settings.use_face or "FACE" not in provider.capabilities:
        return ""
    config = armature.mtp_rig_config
    completeness = _output_completeness(config, settings.face_output)
    label = FACE_OUTPUTS[settings.face_output][2]
    if completeness == 0.0:
        return f"{label} are fully missing; face output will be skipped"
    if completeness < 1.0:
        return f"{label} are partially mapped ({completeness:.0%})"
    return ""


def evaluate_apply_readiness(
    settings,
    armature,
    *,
    provider=None,
    media_source=None,
) -> CaptureReadiness:
    """Evaluate the exact side-effect-free state shared by UI and Apply."""
    from ..backends.sam3d import backend as sam3d_backend
    from . import capture_options

    provider = provider or _descriptor(settings.provider)
    source = (
        media_source
        if media_source is not None
        else capture_options.resolve_media_source(settings)
    )
    enabled_scope = enabled_capture_scope(settings, provider.capabilities)
    options = (
        {
            "variant": str(settings.sam3d_model_variant),
            "device": str(settings.sam3d_device),
        }
        if settings.provider == "SAM3D_BODY"
        else {}
    )
    provider_ready, provider_issue = provider.dependency_status(options)
    mapping_issues = (
        mapping_scope_issues(armature, settings) if armature is not None else ()
    )

    issue = ""
    if armature is None:
        issue = "Select a target armature"
    elif source.state == "EMPTY":
        issue = "Choose an image or video"
    elif not source.usable:
        issue = source.message
    elif not enabled_scope:
        issue = f"Enable a capture type supported by {provider.label}"
    elif sam3d_backend.model_load_running():
        issue = "SAM 3D Body model is loading"
    elif not provider_ready:
        issue = provider_issue
    elif mapping_issues:
        issue = mapping_issues[0]

    return CaptureReadiness(
        ready=not issue,
        issue=issue,
        provider_ready=provider_ready,
        provider_issue=provider_issue,
        enabled_scope=enabled_scope,
        mapping_issues=mapping_issues,
        media_source=source,
    )
