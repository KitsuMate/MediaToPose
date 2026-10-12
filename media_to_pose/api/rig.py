"""Describe how an armature is mapped for capture."""

from __future__ import annotations

from typing import Any

from ..core.constants import SECTIONS
from ..rig import mapping
from ..rig.face_schemes import associated_meshes, shape_scheme
from ..rig.profiles.common import PROFILE_LABELS, detect_profile
from .common import resolve_armature, resolve_scene


def prepare_rig(armature, settings) -> None:
    """Map a rig that was never set up, as choosing it in the panel does.

    An Automatic mapping whose rig changed is refreshed; manual and locked
    mappings are kept. The mapping is then validated for ``settings``.
    """
    mapping.ensure_mapping_entries(armature)
    mapping.refresh_auto_mapping(armature, settings=settings)
    mapping.validate_mapping(armature, settings=settings)


def auto_map(armature, *, force_profile: bool = False, scene=None) -> dict[str, Any]:
    """Run automatic mapping (keeping locked entries) and describe the rig."""
    target = resolve_armature(armature)
    settings = getattr(resolve_scene(scene), "mtp_settings", None)
    mapping.auto_map(target, force_profile=force_profile, settings=settings)
    return describe_rig(target, scene=scene)


def describe_rig(armature, *, scene=None) -> dict[str, Any]:
    """Mapping coverage, unmapped targets, profile, shape keys and face bones.

    Validates the mapping first (as the Rig Setup panel does), so statuses
    reflect the scene's current capture settings. The result is JSON-ready.
    """
    target = resolve_armature(armature)
    settings = getattr(resolve_scene(scene), "mtp_settings", None)
    prepare_rig(target, settings)
    config = target.mtp_rig_config
    sections: dict[str, Any] = {}
    for section, label, _description in SECTIONS:
        entries = [entry for entry in config.mappings if entry.section == section]
        enabled = [entry for entry in entries if entry.enabled]
        mapped = [
            entry
            for entry in enabled
            if entry.target_name and entry.status in {"VALID", "INFO", "WARNING"}
        ]
        sections[section] = {
            "label": label,
            "entries": len(entries),
            "enabled": len(enabled),
            "mapped": len(mapped),
            "coverage": round(len(mapped) / len(enabled), 4) if enabled else 0.0,
            "unmapped": [
                {
                    "semantic_id": entry.semantic_id,
                    "label": entry.label,
                    "required": bool(entry.required),
                    "status": entry.status,
                    "message": entry.message,
                }
                for entry in enabled
                if entry not in mapped
            ],
        }
    shape_keys = {}
    for mesh in associated_meshes(target):
        keys = getattr(mesh.data, "shape_keys", None)
        names = [block.name for block in keys.key_blocks[1:]] if keys else []
        if names:
            shape_keys[mesh.name] = {
                "count": len(names),
                "schemes": sorted(
                    {scheme for name in names if (scheme := shape_scheme(name))}
                ),
            }
    face_bones = sorted(
        entry.target_name
        for entry in config.mappings
        if entry.section == "FACE_RIG"
        and entry.enabled
        and entry.target_kind == "BONE"
        and entry.target_name
    )
    detected = detect_profile(target)
    face_requested = bool(getattr(settings, "use_face", False))
    return {
        "armature": target.name,
        "profile": config.profile,
        "detected_profile": detected,
        "profile_label": PROFILE_LABELS.get(detected, detected),
        "validation": config.validation_summary,
        # Face sections are only validated when face capture is on;
        # otherwise their completeness is not meaningful and is null.
        "completeness": {
            "body": round(config.body_complete, 4),
            "hands": round(config.hands_complete, 4),
            **{
                name: round(value, 4) if face_requested else None
                for name, value in (
                    ("shapes", config.shapes_complete),
                    ("arkit", config.arkit_complete),
                    ("facs", config.facs_complete),
                    ("vrm", config.vrm_complete),
                    ("face", config.face_complete),
                )
            },
        },
        "sections": sections,
        "shape_keys": shape_keys,
        "face_bones": face_bones,
        "bones": len(target.data.bones),
    }


__all__ = ["auto_map", "describe_rig", "prepare_rig"]
