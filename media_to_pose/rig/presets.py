"""Mapping presets: JSON export, import and copying between armatures."""

from __future__ import annotations

import json
from pathlib import Path

import bpy

from ..core.constants import (
    PROFILE_ARP,
    PROFILE_GENERIC,
    PROFILE_RIGIFY,
    SCHEMA_VERSION,
)
from ..core.errors import InvalidRequest
from ..core.fs import atomic_json
from .locks import suppress_manual_locks
from .mapping import ensure_mapping_entries, mapping_by_id, validate_mapping
from .profiles.fk_switches import FK_SWITCH_FIELDS


def serialize_config(armature: bpy.types.Object) -> dict:
    config = armature.mtp_rig_config
    from ..postprocess import serialize_module_configs

    return {
        "schema_version": SCHEMA_VERSION,
        "profile": config.profile,
        "detected_profile": config.detected_profile,
        "armature_signature": config.armature_signature,
        "fk_switches": {field: getattr(config, field) for field in FK_SWITCH_FIELDS}
        | {"manual": config.fk_switches_manual},
        "post_processing": serialize_module_configs(config),
        "mappings": [
            {
                key: getattr(entry, key)
                for key in (
                    "semantic_id",
                    "label",
                    "section",
                    "side",
                    "required",
                    "enabled",
                    "target_kind",
                    "shape_scheme",
                    "target_name",
                    "target_object",
                    "target_data_name",
                    "property_name",
                    "channel_mode",
                    "axis",
                    "gain",
                    "offset",
                    "clamp_min",
                    "clamp_max",
                    "invert",
                    "influence",
                    "confidence",
                    "source",
                    "manual_lock",
                )
            }
            | {
                "correction_euler": list(entry.correction_euler),
                "shape_overrides": [
                    {
                        "object_name": override.object_name,
                        "target_name": override.target_name,
                        "enabled": override.enabled,
                    }
                    for override in entry.shape_overrides
                ],
            }
            for entry in config.mappings
        ],
    }


def load_config(armature: bpy.types.Object, payload: dict, *, settings=None) -> None:
    schema_version = int(payload.get("schema_version", 0))
    if schema_version != SCHEMA_VERSION:
        raise InvalidRequest(
            f"Unsupported mapping schema {schema_version}; expected {SCHEMA_VERSION}"
        )
    ensure_mapping_entries(armature)
    entries = mapping_by_id(armature.mtp_rig_config)
    with suppress_manual_locks():
        profile = payload.get("profile")
        if profile in {
            "AUTO",
            PROFILE_ARP,
            PROFILE_RIGIFY,
            PROFILE_GENERIC,
        }:
            armature.mtp_rig_config.profile = profile
        switch_values = payload.get("fk_switches", {})
        for field in FK_SWITCH_FIELDS:
            if field in switch_values:
                setattr(
                    armature.mtp_rig_config,
                    field,
                    switch_values[field],
                )
        if "manual" in switch_values:
            armature.mtp_rig_config.fk_switches_manual = bool(switch_values["manual"])
        from ..postprocess import load_module_configs

        load_module_configs(
            armature.mtp_rig_config,
            payload.get("post_processing", ()),
        )
        for item in payload.get("mappings", []):
            entry = entries.get(item.get("semantic_id", ""))
            if not entry:
                continue
            for key, value in item.items():
                if key in {"semantic_id", "shape_overrides"} or not hasattr(entry, key):
                    continue
                setattr(entry, key, value)
            if "shape_overrides" in item:
                entry.shape_overrides.clear()
                for override_data in item.get("shape_overrides", []):
                    override = entry.shape_overrides.add()
                    override.semantic_id = entry.semantic_id
                    override.object_name = str(override_data.get("object_name", ""))
                    override.target_name = str(override_data.get("target_name", ""))
                    override.enabled = bool(override_data.get("enabled", True))
    validate_mapping(armature, settings=settings)


def copy_section(
    source: bpy.types.Object,
    target: bpy.types.Object,
    section: str | None = None,
    *,
    settings=None,
) -> None:
    ensure_mapping_entries(source)
    ensure_mapping_entries(target)
    source_entries = mapping_by_id(source.mtp_rig_config)
    target_entries = mapping_by_id(target.mtp_rig_config)
    with suppress_manual_locks():
        if section in {None, "POST_PROCESSING"}:
            from ..postprocess import load_module_configs, serialize_module_configs

            load_module_configs(
                target.mtp_rig_config,
                serialize_module_configs(source.mtp_rig_config),
            )
        if section in {None, "BODY"}:
            for field in FK_SWITCH_FIELDS:
                setattr(
                    target.mtp_rig_config,
                    field,
                    getattr(source.mtp_rig_config, field),
                )
            target.mtp_rig_config.fk_switches_manual = True
        for semantic_id, source_entry in source_entries.items():
            if section and source_entry.section != section:
                continue
            target_entry = target_entries.get(semantic_id)
            if not target_entry:
                continue
            for key in (
                "enabled",
                "target_kind",
                "target_name",
                "target_object",
                "target_data_name",
                "property_name",
                "channel_mode",
                "axis",
                "gain",
                "offset",
                "clamp_min",
                "clamp_max",
                "invert",
                "influence",
                "correction_euler",
            ):
                setattr(target_entry, key, getattr(source_entry, key))
            target_entry.shape_overrides.clear()
            for source_override in source_entry.shape_overrides:
                target_override = target_entry.shape_overrides.add()
                target_override.semantic_id = semantic_id
                target_override.object_name = source_override.object_name
                target_override.target_name = source_override.target_name
                target_override.enabled = source_override.enabled
            target_entry.source = "MANUAL"
            target_entry.manual_lock = True
            target_entry.confidence = 1.0
    validate_mapping(target, settings=settings)


def save_json(filepath: str, armature: bpy.types.Object) -> None:
    atomic_json(
        Path(filepath),
        serialize_config(armature),
        sort_keys=False,
        ensure_ascii=False,
    )


def load_json(filepath: str, armature: bpy.types.Object, *, settings=None) -> None:
    with Path(filepath).open(encoding="utf8") as handle:
        load_config(armature, json.load(handle), settings=settings)
