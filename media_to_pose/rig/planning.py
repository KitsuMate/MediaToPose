"""Compiled, armature-static data used by the per-frame pose solver."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class ConstraintBinding:
    bone_name: str
    constraint_name: str
    constraint_type: str
    target_name: str
    subtarget: str


@dataclass(frozen=True)
class ShapeOverrideSnapshot:
    object_name: str
    target_name: str
    enabled: bool


@dataclass(frozen=True)
class MappingEntrySnapshot:
    """RNA-free mapping data safe to retain across modal/undo boundaries."""

    semantic_id: str
    section: str
    required: bool
    enabled: bool
    target_kind: str
    shape_scheme: str
    target_name: str
    target_object: str
    target_data_name: str
    property_name: str
    channel_mode: str
    axis: str
    gain: float
    offset: float
    clamp_min: float
    clamp_max: float
    invert: bool
    influence: float
    correction_euler: tuple[float, float, float]
    shape_overrides: tuple[ShapeOverrideSnapshot, ...]


@dataclass(frozen=True)
class RigSolvePlan:
    """Immutable topology and rest transforms for one mapping revision."""

    armature_pointer: int
    mapping_signature: tuple
    entries: Mapping[str, object]
    semantic_bones: Mapping[str, object]
    hierarchy_order: tuple[str, ...]
    hierarchy_depth: Mapping[str, int]
    semantic_endpoints: Mapping[str, tuple]
    constraints: tuple[ConstraintBinding, ...]
    driver_paths: tuple[str, ...]
    ik_chains: Mapping[tuple[str, str], tuple[str, ...]]
    rest_matrices: Mapping[str, object]
    rest_basis: object
    constraint_free: bool


_PLAN_CACHE: dict[int, RigSolvePlan] = {}


def snapshot_mapping_entries(
    entries: Mapping[str, object],
) -> Mapping[str, MappingEntrySnapshot]:
    """Copy Blender mapping PropertyGroups into plain immutable Python data."""
    snapshots = {}
    for semantic_id, entry in entries.items():
        snapshots[str(semantic_id)] = MappingEntrySnapshot(
            semantic_id=str(semantic_id),
            section=str(getattr(entry, "section", "BODY")),
            required=bool(getattr(entry, "required", False)),
            enabled=bool(getattr(entry, "enabled", True)),
            target_kind=str(getattr(entry, "target_kind", "BONE")),
            shape_scheme=str(getattr(entry, "shape_scheme", "NONE")),
            target_name=str(getattr(entry, "target_name", "")),
            target_object=str(getattr(entry, "target_object", "")),
            target_data_name=str(getattr(entry, "target_data_name", "")),
            property_name=str(getattr(entry, "property_name", "")),
            channel_mode=str(getattr(entry, "channel_mode", "AUTO")),
            axis=str(getattr(entry, "axis", "X")),
            gain=float(getattr(entry, "gain", 1.0)),
            offset=float(getattr(entry, "offset", 0.0)),
            clamp_min=float(getattr(entry, "clamp_min", 0.0)),
            clamp_max=float(getattr(entry, "clamp_max", 1.0)),
            invert=bool(getattr(entry, "invert", False)),
            influence=float(getattr(entry, "influence", 1.0)),
            correction_euler=tuple(
                float(value)
                for value in getattr(entry, "correction_euler", (0.0, 0.0, 0.0))
            ),
            shape_overrides=tuple(
                ShapeOverrideSnapshot(
                    object_name=str(getattr(override, "object_name", "")),
                    target_name=str(getattr(override, "target_name", "")),
                    enabled=bool(getattr(override, "enabled", True)),
                )
                for override in getattr(entry, "shape_overrides", ())
            ),
        )
    return MappingProxyType(snapshots)


def _mapping_signature(entries: Mapping[str, object]) -> tuple:
    return tuple(
        (
            semantic_id,
            bool(entry.enabled),
            str(entry.target_kind),
            str(entry.target_name),
            str(entry.target_object),
            str(entry.target_data_name),
            str(entry.property_name),
            str(entry.channel_mode),
            str(entry.axis),
            round(float(entry.gain), 8),
            round(float(entry.offset), 8),
            bool(entry.invert),
            round(float(entry.influence), 8),
            tuple(round(float(value), 8) for value in entry.correction_euler),
        )
        for semantic_id, entry in sorted(entries.items())
    )


def _driver_paths(armature) -> tuple[str, ...]:
    paths = []
    for owner in (armature, armature.data):
        animation = owner.animation_data
        if animation is None:
            continue
        paths.extend(str(driver.data_path) for driver in animation.drivers)
    return tuple(sorted(set(paths)))


def _ik_chains(semantic_bones: Mapping[str, str]):
    result = {}
    for kind, roles in (
        ("ARM", ("UPPER_ARM", "LOWER_ARM", "HAND")),
        ("LEG", ("UPPER_LEG", "LOWER_LEG", "FOOT", "TOE")),
    ):
        for side in ("L", "R"):
            names = tuple(
                semantic_bones[f"{role}_{side}"]
                for role in roles
                if f"{role}_{side}" in semantic_bones
            )
            if names:
                result[(kind, side)] = names
    return MappingProxyType(result)


def compile_rig_solve_plan(
    armature,
    entries: Mapping[str, object],
    rest_basis_factory: Callable[[], object],
) -> RigSolvePlan:
    """Return a cached plan, rebuilding only after mapping/topology changes."""
    pointer = int(armature.as_pointer())
    immutable_entries = snapshot_mapping_entries(entries)
    hierarchy = tuple(
        sorted(
            armature.pose.bones,
            key=lambda bone: (len(bone.parent_recursive), bone.name.casefold()),
        )
    )
    topology_signature = tuple(
        (
            bone.name,
            bone.parent.name if bone.parent is not None else "",
            tuple(
                round(float(value), 8)
                for row in bone.bone.matrix_local
                for value in row
            ),
        )
        for bone in hierarchy
    )
    signature = (
        str(armature.name),
        str(armature.data.name),
        topology_signature,
        _mapping_signature(immutable_entries),
    )
    cached = _PLAN_CACHE.get(pointer)
    if cached is not None and cached.mapping_signature == signature:
        return cached

    semantic_pose_bones = {
        semantic_id: bone
        for semantic_id, entry in immutable_entries.items()
        if entry.enabled
        and entry.target_name
        and (bone := armature.pose.bones.get(entry.target_name)) is not None
    }
    semantic_bones = {
        semantic_id: bone.name for semantic_id, bone in semantic_pose_bones.items()
    }
    constraints = tuple(
        ConstraintBinding(
            bone.name,
            constraint.name,
            constraint.type,
            str(getattr(getattr(constraint, "target", None), "name", "")),
            str(getattr(constraint, "subtarget", "")),
        )
        for bone in hierarchy
        for constraint in bone.constraints
    )
    driver_paths = _driver_paths(armature)
    plan = RigSolvePlan(
        armature_pointer=pointer,
        mapping_signature=signature,
        entries=immutable_entries,
        semantic_bones=MappingProxyType(semantic_bones),
        hierarchy_order=tuple(bone.name for bone in hierarchy),
        hierarchy_depth=MappingProxyType(
            {bone.name: len(bone.parent_recursive) for bone in hierarchy}
        ),
        semantic_endpoints=MappingProxyType(
            {
                semantic_id: (
                    bone.bone.head_local.copy(),
                    bone.bone.tail_local.copy(),
                )
                for semantic_id, bone in semantic_pose_bones.items()
            }
        ),
        constraints=constraints,
        driver_paths=driver_paths,
        ik_chains=_ik_chains(semantic_bones),
        rest_matrices=MappingProxyType(
            {bone.name: bone.bone.matrix_local.copy() for bone in hierarchy}
        ),
        rest_basis=rest_basis_factory().copy(),
        constraint_free=not constraints and not driver_paths,
    )
    _PLAN_CACHE[pointer] = plan
    return plan


def clear_rig_solve_plans() -> None:
    _PLAN_CACHE.clear()
