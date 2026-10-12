"""Per-armature role mapping: automatic matching and validation."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import bpy

from ..core.constants import (
    ALL_ROLES,
    FACE_BLENDSHAPES,
    FACS_ACTION_UNITS,
    MAPPING_TABS,
    PROFILE_ARP,
    PROFILE_AUTO,
    PROFILE_GENERIC,
    PROFILE_RIGIFY,
    SHAPE_SCHEMES,
    VRM_EXPRESSIONS,
)
from .face_schemes import (
    SIGNED_FACE_CONTROLS,
    _facs_aliases,
    associated_meshes,
    find_shape_targets,
    matching_shape_blocks,
    shape_priority,
    shape_scheme,
)
from .locks import suppress_manual_locks
from .names import normalize_name, side_from_role
from .profiles.arp import ARP_BODY, ARP_FACE
from .profiles.common import detect_profile
from .profiles.fk_switches import configure_fk_switches
from .profiles.generic import (
    _best_generic_bone,
    _find_first_bone,
    _finger_candidates,
    _generic_structural_body_mapping,
    _generic_torso_mapping,
)
from .profiles.rigify import RIGIFY_BODY, RIGIFY_FACE
from .profiles.vrm import vrm_humanoid_mapping


def armature_signature(armature: bpy.types.Object) -> str:
    parts = []
    for bone in sorted(armature.data.bones, key=lambda item: item.name):
        parent = bone.parent.name if bone.parent else ""
        head = ",".join(f"{v:.5f}" for v in bone.head_local)
        tail = ",".join(f"{v:.5f}" for v in bone.tail_local)
        parts.append(f"{bone.name}|{parent}|{head}|{tail}")
    return hashlib.sha256("\n".join(parts).encode("utf8")).hexdigest()


def _add_entry(config, section: str, role: Sequence) -> None:
    semantic_id, label, required = role
    with suppress_manual_locks():
        entry = config.mappings.add()
        entry.semantic_id = semantic_id
        entry.label = label
        entry.section = section
        entry.side = side_from_role(semantic_id)
        entry.required = bool(required)
        entry.target_kind = "SHAPE_KEY" if section == "SHAPES" else "BONE"
        if section == "SHAPES":
            entry.shape_scheme = semantic_id.split("_", 1)[0]
        elif section == "FACE_RIG" and semantic_id in SIGNED_FACE_CONTROLS:
            entry.clamp_min = -1.0
        entry.channel_mode = "VALUE" if section == "SHAPES" else "AUTO"
        entry.status = "MISSING" if required else "OPTIONAL"
        entry.source = "NONE"
        entry.manual_lock = False


def ensure_mapping_entries(armature: bpy.types.Object) -> None:
    config = armature.mtp_rig_config
    if config.subtab not in {item[0] for item in MAPPING_TABS}:
        config.subtab = "BODY"

    canonical_roles = {
        role[0]: (section, role[1], bool(role[2]))
        for section, roles in ALL_ROLES.items()
        for role in roles
    }
    valid_ids = set(canonical_roles)
    for index in reversed(range(len(config.mappings))):
        if config.mappings[index].semantic_id not in valid_ids:
            config.mappings.remove(index)
    existing = {entry.semantic_id for entry in config.mappings}
    if not existing:
        for section, roles in ALL_ROLES.items():
            for role in roles:
                _add_entry(config, section, role)
    else:
        for section, roles in ALL_ROLES.items():
            for role in roles:
                if role[0] not in existing:
                    _add_entry(config, section, role)
    # Labels and structural metadata are defined by the current role catalog,
    # not by mutable data stored in a .blend. This keeps the Mapping tab exact
    # even after entries have been edited or copied between armatures.
    with suppress_manual_locks():
        for entry in config.mappings:
            section, label, required = canonical_roles[entry.semantic_id]
            entry.label = label
            entry.section = section
            entry.side = side_from_role(entry.semantic_id)
            entry.required = required
    from ..postprocess import ensure_module_configs

    ensure_module_configs(config)


def mapping_by_id(config) -> dict[str, object]:
    return {entry.semantic_id: entry for entry in config.mappings}


def clear_unlocked_automatic_mappings(armature: bpy.types.Object) -> None:
    """Discard generated assignments before applying another mapping method."""
    ensure_mapping_entries(armature)
    with suppress_manual_locks():
        for entry in armature.mtp_rig_config.mappings:
            if entry.manual_lock:
                continue
            entry.target_name = ""
            entry.target_object = ""
            entry.target_data_name = ""
            entry.property_name = ""
            entry.shape_overrides.clear()
            entry.source = "NONE"
            entry.confidence = 0.0
            entry.status = "MISSING" if entry.required else "OPTIONAL"
            entry.message = ""


AUTO_MATCH_CONFIDENCE_EXPLANATION = (
    "Auto-match % estimates mapping confidence from the target name, side, "
    "and bone hierarchy. It is not pose-detection accuracy; exact rig-profile "
    "matches are 100%."
)


def heuristic_mapping_message(entry) -> str:
    target_type = "shape key" if entry.target_kind == "SHAPE_KEY" else "bone"
    return (
        f'Auto-matched to {target_type} "{entry.target_name}" '
        f"({entry.confidence:.0%}). This percentage estimates the target name, "
        "side, and hierarchy match, not pose-detection accuracy."
    )


def _set_mapping(
    entry,
    target: str,
    *,
    source: str,
    confidence: float,
    target_kind: str = "BONE",
    target_object: str = "",
    channel: str = "AUTO",
    axis: str = "X",
    force: bool = False,
) -> bool:
    if entry.manual_lock and entry.source == "MANUAL" and not force:
        return False
    with suppress_manual_locks():
        entry.target_kind = target_kind
        entry.target_name = target
        entry.target_object = target_object
        entry.channel_mode = channel
        entry.axis = axis
        entry.source = source
        entry.confidence = confidence
        informational = source == "HEURISTIC" and confidence < 1.0
        entry.status = "INFO" if informational else "VALID"
        entry.message = heuristic_mapping_message(entry) if informational else ""
        if force:
            entry.manual_lock = False
            if target_kind == "SHAPE_KEY":
                entry.shape_overrides.clear()
    return True


def auto_map(
    armature: bpy.types.Object,
    *,
    force_profile: bool = False,
    settings=None,
) -> str:
    ensure_mapping_entries(armature)
    config = armature.mtp_rig_config
    detected = detect_profile(armature)
    profile = config.profile if config.profile != "AUTO" else detected
    config.detected_profile = profile
    config.armature_signature = armature_signature(armature)
    configure_fk_switches(armature, profile, force=force_profile)
    entries = mapping_by_id(config)
    bones = set(armature.pose.bones.keys())
    vrm_map = vrm_humanoid_mapping(armature) if profile == PROFILE_GENERIC else {}

    profile_body = (
        ARP_BODY
        if profile == PROFILE_ARP
        else RIGIFY_BODY
        if profile == PROFILE_RIGIFY
        else vrm_map
        if profile == PROFILE_GENERIC
        else {}
    )
    generic_torso = (
        _generic_torso_mapping(armature) if profile == PROFILE_GENERIC else {}
    )
    reserved_torso_bones = {
        bone_name: role for role, bone_name in generic_torso.items()
    }
    resolved_body = {
        entry.semantic_id: entry.target_name
        for entry in config.mappings
        if entry.section == "BODY" and entry.manual_lock and entry.target_name in bones
    }
    used_body_bones = set(resolved_body.values())
    for semantic_id, _label, _required in ALL_ROLES["BODY"]:
        entry = entries[semantic_id]
        exact = profile_body.get(semantic_id, "")
        if exact and exact in bones:
            channel = (
                "LOCATION"
                if profile == PROFILE_ARP and semantic_id in {"EYE_L", "EYE_R"}
                else "ROTATION"
                if semantic_id in {"EYE_L", "EYE_R"}
                else "AUTO"
            )
            _set_mapping(
                entry,
                exact,
                source="PROFILE",
                confidence=1.0,
                channel=channel,
                axis="X",
                force=force_profile,
            )
        elif (
            not entry.manual_lock
            and generic_torso.get(semantic_id) in bones
            and generic_torso[semantic_id] not in used_body_bones
        ):
            _set_mapping(
                entry,
                generic_torso[semantic_id],
                source="HEURISTIC",
                confidence=0.9,
                force=force_profile,
            )
        elif not entry.manual_lock:
            candidate, confidence = _best_generic_bone(
                armature,
                semantic_id,
                mapped=resolved_body,
                excluded=used_body_bones
                | {
                    bone_name
                    for bone_name, role in reserved_torso_bones.items()
                    if role != semantic_id
                },
            )
            if candidate:
                _set_mapping(
                    entry, candidate, source="HEURISTIC", confidence=confidence
                )
        if entry.target_name in bones:
            resolved_body[semantic_id] = entry.target_name
            used_body_bones.add(entry.target_name)

    if profile == PROFILE_GENERIC:
        structural = _generic_structural_body_mapping(
            armature,
            resolved_body,
            used_body_bones,
        )
        for semantic_id, (bone_name, confidence) in structural.items():
            entry = entries[semantic_id]
            if entry.manual_lock or entry.target_name:
                continue
            _set_mapping(
                entry,
                bone_name,
                source="HEURISTIC",
                confidence=confidence,
                force=force_profile,
            )

    for semantic_id, _label, _required in ALL_ROLES["HANDS"]:
        entry = entries[semantic_id]
        vrm_target = vrm_map.get(semantic_id, "")
        if vrm_target and vrm_target in bones:
            _set_mapping(
                entry,
                vrm_target,
                source="PROFILE",
                confidence=1.0,
                force=force_profile,
            )
            continue
        match = re.match(r"([A-Z]+)_(\d+)_(L|R)$", semantic_id)
        if not match:
            continue
        finger, segment, side = match.group(1), int(match.group(2)), match.group(3)
        target = _find_first_bone(
            armature, _finger_candidates(profile, finger, segment, side)
        )
        if target:
            source = (
                "PROFILE" if profile in {PROFILE_ARP, PROFILE_RIGIFY} else "HEURISTIC"
            )
            confidence = 1.0 if source == "PROFILE" else 0.65
            _set_mapping(
                entry, target, source=source, confidence=confidence, force=force_profile
            )

    shape_targets = find_shape_targets(armature)
    for blendshape in FACE_BLENDSHAPES:
        if blendshape == "_neutral":
            continue
        entry = entries.get(f"ARKIT_{blendshape}")
        if not entry or (entry.manual_lock and not force_profile):
            continue
        hits = sorted(
            (
                hit
                for hit in shape_targets.get(normalize_name(blendshape, shape=True), [])
                if shape_scheme(hit[1]) == "ARKIT"
            ),
            key=lambda hit: shape_priority(hit[1]),
        )
        if hits:
            _set_mapping(
                entry,
                hits[0][1],
                target_kind="SHAPE_KEY",
                target_object="*",
                channel="VALUE",
                source="PROFILE",
                confidence=1.0,
                force=force_profile,
            )

    for name, _label, keys in VRM_EXPRESSIONS:
        entry = entries.get(f"VRM_{name}")
        if not entry or (entry.manual_lock and not force_profile):
            continue
        hits = [
            hit
            for key in keys
            for hit in shape_targets.get(normalize_name(key, shape=True), [])
            if shape_scheme(hit[1]) == "VRM"
        ]
        if hits:
            _set_mapping(
                entry,
                hits[0][1],
                target_kind="SHAPE_KEY",
                target_object="*",
                channel="VALUE",
                source="PROFILE",
                confidence=1.0,
                force=force_profile,
            )

    for action_unit, _label in FACS_ACTION_UNITS:
        entry = entries.get(f"FACS_{action_unit}")
        if not entry or (entry.manual_lock and not force_profile):
            continue
        hits = []
        for alias in _facs_aliases(action_unit):
            hits.extend(
                hit
                for hit in shape_targets.get(normalize_name(alias, shape=True), [])
                if shape_scheme(hit[1]) == "FACS"
            )
        hits.sort(key=lambda hit: shape_priority(hit[1]))
        if hits:
            _set_mapping(
                entry,
                hits[0][1],
                target_kind="SHAPE_KEY",
                target_object="*",
                channel="VALUE",
                source="PROFILE",
                confidence=1.0,
                force=force_profile,
            )

    face_map = (
        ARP_FACE
        if profile == PROFILE_ARP
        else RIGIFY_FACE
        if profile == PROFILE_RIGIFY
        else {}
    )
    for semantic_id, (target, channel, axis) in face_map.items():
        entry = entries.get(semantic_id)
        if entry and target in bones:
            _set_mapping(
                entry,
                target,
                channel=channel,
                axis=axis,
                source="PROFILE",
                confidence=1.0,
                force=force_profile,
            )

    if profile == PROFILE_GENERIC:
        # Generic rigs can still expose the small set of facial bone controls
        # that Media to Pose understands, including VRoid jaw conventions
        # when VRM metadata is missing or unavailable. Head and eye controls
        # are resolved as body mappings above.
        generic_face = {
            "FACE_JAW_OPEN": vrm_map.get("FACE_JAW_OPEN")
            or _find_first_bone(
                armature,
                ("J_Bip_C_Jaw", "Jaw", "jaw", "Mandible", "mandible"),
            ),
        }
        for semantic_id, target in generic_face.items():
            entry = entries.get(semantic_id)
            if not entry or not target or target not in bones:
                continue
            _set_mapping(
                entry,
                target,
                channel="ROTATION",
                axis="X",
                source="HEURISTIC",
                confidence=0.8,
                force=force_profile,
            )

    validate_mapping(armature, settings=settings)
    from ..postprocess import validate_modules

    validate_modules(armature)
    return profile


def refresh_auto_mapping(armature: bpy.types.Object, *, settings=None) -> bool:
    """Refresh an Automatic mapping when its rig identity or profile changed.

    VRM importers can populate humanoid metadata after the armature object is
    first visible. In that interval an Automatic mapping may legitimately be
    detected as Generic. Refreshing only stale Automatic configurations fills
    unlocked entries from the now-available metadata while preserving every
    manual lock.
    """
    if armature is None or getattr(armature, "type", None) != "ARMATURE":
        return False
    config = getattr(armature, "mtp_rig_config", None)
    if config is None:
        return False
    # Correction definitions can change without the armature geometry or
    # mapping profile changing. Synchronize them before the mapping staleness
    # checks so newly registered processors appear on existing target rigs.
    from ..postprocess import ensure_module_configs

    ensure_module_configs(config)
    if config.profile != PROFILE_AUTO:
        return False
    detected = detect_profile(armature)
    signature = armature_signature(armature)
    metadata = vrm_humanoid_mapping(armature)
    entries = mapping_by_id(config)
    metadata_changed = any(
        target
        and semantic_id in entries
        and not entries[semantic_id].manual_lock
        and entries[semantic_id].target_name != target
        for semantic_id, target in metadata.items()
    )
    if (
        config.mappings
        and config.detected_profile == detected
        and config.armature_signature == signature
        and not metadata_changed
    ):
        return False
    auto_map(armature, settings=settings)
    return True


def validate_mapping(
    armature: bpy.types.Object,
    *,
    settings=None,
) -> tuple[int, int, int]:
    config = armature.mtp_rig_config
    ensure_mapping_entries(armature)
    if settings is None:
        face_enabled = True
    else:
        from ..providers import descriptor

        face_enabled = bool(
            settings.use_face and "FACE" in descriptor(settings.provider).capabilities
        )
    bone_names = set(armature.pose.bones.keys())
    associated = {obj.name: obj for obj in associated_meshes(armature)}
    used: dict[tuple[str, ...], list[object]] = {}
    errors = warnings = information = valid = 0

    section_counts = {
        section: [0, 0] for section in ("BODY", "HANDS", "SHAPES", "FACE_RIG")
    }
    scheme_counts = {scheme: [0, 0] for scheme in SHAPE_SCHEMES}
    for entry in config.mappings:
        if entry.section in {"SHAPES", "FACE_RIG"} and not face_enabled:
            entry.status = "OPTIONAL"
            entry.message = "Facial posing is disabled"
            for override in entry.shape_overrides:
                override.status = "OPTIONAL"
                override.message = "Facial posing is disabled"
            continue
        if not entry.enabled:
            entry.status = "OPTIONAL"
            entry.message = "Disabled"
            continue
        section_counts[entry.section][1] += 1
        if entry.section == "SHAPES" and entry.shape_scheme in scheme_counts:
            scheme_counts[entry.shape_scheme][1] += 1
        exists = False
        access_error = ""
        control_warning = ""
        if entry.target_kind in {"BONE", "CUSTOM_PROPERTY"}:
            exists = bool(entry.target_name and entry.target_name in bone_names)
            if exists and (
                entry.target_kind == "CUSTOM_PROPERTY"
                or entry.channel_mode == "CUSTOM_PROPERTY"
            ):
                bone = armature.pose.bones[entry.target_name]
                if not entry.property_name or entry.property_name not in bone:
                    access_error = "Custom property no longer exists"
            if exists and not getattr(armature, "is_editable", True):
                access_error = "Armature is linked or read-only"
            if exists and entry.source != "PROFILE":
                bone = armature.pose.bones[entry.target_name]
                active_constraints = [
                    constraint
                    for constraint in bone.constraints
                    if not constraint.mute
                    and float(getattr(constraint, "influence", 1.0)) > 0.0
                ]
                if active_constraints:
                    control_warning = (
                        "Target has active constraints; test before capture"
                    )
                elif entry.channel_mode == "ROTATION" and all(bone.lock_rotation):
                    control_warning = "All rotation channels are locked"
                elif entry.channel_mode == "LOCATION" and all(bone.lock_location):
                    control_warning = "All location channels are locked"
        elif entry.target_kind == "SHAPE_KEY":
            override_error = ""
            seen_override_objects = set()
            for override in entry.shape_overrides:
                if override.object_name in seen_override_objects:
                    override.status = "ERROR"
                    override.message = "Duplicate override for this mesh"
                    override_error = override.message
                    continue
                seen_override_objects.add(override.object_name)
                obj = associated.get(override.object_name)
                if not obj:
                    override.status = "ERROR"
                    override.message = "Mesh is missing or not associated"
                    override_error = override.message
                    continue
                if entry.target_object not in {"", "*"}:
                    override.status = "ERROR"
                    override.message = "Overrides require an all-mesh group"
                    override_error = override.message
                    continue
                if not override.enabled:
                    override.status = "OPTIONAL"
                    override.message = "Intentionally skipped"
                    continue
                keys = obj.data.shape_keys
                block = (
                    next(
                        (
                            candidate
                            for candidate in keys.key_blocks
                            if normalize_name(candidate.name, shape=True)
                            == normalize_name(override.target_name, shape=True)
                            and (
                                entry.shape_scheme == "NONE"
                                or shape_scheme(candidate.name) == entry.shape_scheme
                            )
                        ),
                        None,
                    )
                    if keys and override.target_name
                    else None
                )
                if not block:
                    override.status = "ERROR"
                    override.message = "Shape key no longer exists"
                    override_error = (
                        f"{override.object_name}: shape key no longer exists"
                    )
                else:
                    if not getattr(block.id_data, "is_editable", True):
                        override.status = "ERROR"
                        override.message = "Shape-key datablock is linked or read-only"
                        override_error = (
                            f"{override.object_name}: shape-key datablock is read-only"
                        )
                    else:
                        override.status = "VALID"
                        override.message = ""
            if override_error:
                entry.status = "ERROR"
                entry.message = override_error
                errors += 1
                continue
            resolved_blocks = (
                list(matching_shape_blocks(armature, entry))
                if entry.target_name
                else []
            )
            exists = bool(resolved_blocks)
            read_only = [
                obj.name
                for obj, block in resolved_blocks
                if not getattr(block.id_data, "is_editable", True)
            ]
            if read_only:
                access_error = (
                    "Shape-key datablock is linked or read-only: "
                    + ", ".join(read_only[:3])
                )
        if not entry.target_name:
            entry.status = "MISSING" if entry.required else "OPTIONAL"
            entry.message = (
                "Required target is missing"
                if entry.required
                else "Optional target is not mapped"
            )
            errors += int(entry.required)
            continue
        if not exists:
            entry.status = "ERROR"
            entry.message = "Target no longer exists"
            errors += 1
            continue
        if access_error:
            entry.status = "ERROR"
            entry.message = access_error
            errors += 1
            continue
        # Facial rigs often use one control for multiple compatible motions,
        # such as jaw opening and sliding on separate axes. Only the exact same
        # destination channel is a collision.
        key = (
            entry.target_kind,
            f"{entry.target_object}:{entry.target_name}",
            entry.channel_mode,
            entry.property_name,
            entry.axis,
        )
        used.setdefault(key, []).append(entry)
        section_counts[entry.section][0] += 1
        if entry.section == "SHAPES" and entry.shape_scheme in scheme_counts:
            scheme_counts[entry.shape_scheme][0] += 1
        if control_warning:
            entry.status = "WARNING"
            entry.message = control_warning
            warnings += 1
        elif (
            entry.source == "HEURISTIC"
            and entry.confidence < 1.0
            and not entry.manual_lock
        ):
            entry.status = "INFO"
            entry.message = heuristic_mapping_message(entry)
            information += 1
            valid += 1
        else:
            entry.status = "VALID"
            entry.message = ""
            valid += 1

    for entries in used.values():
        if len(entries) < 2:
            continue
        # Shape keys may intentionally share a name across multiple meshes.
        if entries[0].target_kind == "SHAPE_KEY":
            continue
        for entry in entries:
            entry.status = "WARNING"
            entry.message = "Target is used by multiple semantic roles"
            warnings += 1

    # A symmetric AU target and its left/right variants should not all be
    # driven together unless the user explicitly disables one representation.
    facs_entries = {
        entry.semantic_id.removeprefix("FACS_"): entry
        for entry in config.mappings
        if entry.section == "SHAPES"
        and entry.shape_scheme == "FACS"
        and entry.enabled
        and entry.target_name
        and entry.status not in {"ERROR", "MISSING"}
    }
    for action_unit, _label in FACS_ACTION_UNITS:
        if action_unit.endswith(("_L", "_R")):
            continue
        center = facs_entries.get(action_unit)
        sides = [
            facs_entries.get(f"{action_unit}_L"),
            facs_entries.get(f"{action_unit}_R"),
        ]
        if center and any(sides):
            for entry in (center, *sides):
                if not entry:
                    continue
                entry.status = "WARNING"
                entry.message = (
                    "Symmetric and sided variants are both mapped; "
                    "disable one representation"
                )
                warnings += 1

    # A section with nothing in scope (face capture off) is not complete:
    # it reports 0 and readers treat it as not requested.
    def completeness(section: str) -> float:
        mapped, total = section_counts[section]
        return mapped / total if total else 0.0

    config.body_complete = completeness("BODY")
    config.hands_complete = completeness("HANDS")
    config.shapes_complete = completeness("SHAPES")
    config.arkit_complete = (
        scheme_counts["ARKIT"][0] / scheme_counts["ARKIT"][1]
        if scheme_counts["ARKIT"][1]
        else 0.0
    )
    config.facs_complete = (
        scheme_counts["FACS"][0] / scheme_counts["FACS"][1]
        if scheme_counts["FACS"][1]
        else 0.0
    )
    config.vrm_complete = (
        scheme_counts["VRM"][0] / scheme_counts["VRM"][1]
        if scheme_counts["VRM"][1]
        else 0.0
    )
    config.face_complete = completeness("FACE_RIG")
    config.validation_summary = (
        f"{valid} valid ({information} auto-detected), "
        f"{warnings} warnings, {errors} errors"
    )
    return valid, warnings, errors
