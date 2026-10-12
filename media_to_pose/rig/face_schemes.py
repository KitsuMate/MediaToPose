"""Face shape-key naming schemes and shape-key lookup on rig meshes."""

from __future__ import annotations

import re

import bpy

from ..core.constants import FACE_BLENDSHAPES, SHAPE_SCHEMES
from .names import normalize_name

_ARKIT_NORMALIZED = {
    normalize
    for normalize in (
        re.sub(r"[^a-z0-9]", "", name.lower())
        for name in FACE_BLENDSHAPES
        if name != "_neutral"
    )
}


_FACS_NAME_RE = re.compile(
    r"^au0*(\d{1,2})(?:(left|right|l|r))?$",
    re.IGNORECASE,
)

# Shape-key name prefixes, as rig exporters write them. A corrective
# (``cbs``) is combined automatically by the rig and is never driven
# directly; a control (``ctrl``) is a fallback when no base shape exists.
_PREFIX_RE = re.compile(
    r"^(?P<scheme>facs|arkit|ar)?[_-]?(?P<kind>bs|cbs|ctrl)?[_-]",
    re.IGNORECASE,
)


def shape_kind(name: str) -> str:
    """``CORRECTIVE``, ``CONTROL`` or ``BASE`` from the name's prefix."""
    match = _PREFIX_RE.match(name)
    kind = (match.group("kind") or "").casefold() if match else ""
    return {"cbs": "CORRECTIVE", "ctrl": "CONTROL"}.get(kind, "BASE")


def shape_priority(name: str) -> int:
    """Lower is preferred when several keys share a normalised name."""
    lowered = name.casefold()
    if lowered.startswith(("arkit_", "arkit-", "ar_")):
        return 0
    return {"BASE": 1, "CONTROL": 2, "CORRECTIVE": 3}[shape_kind(name)]


SIGNED_FACE_CONTROLS = {
    "FACE_JAW_SIDE",
    "FACE_BROW_OUTER_L",
    "FACE_BROW_OUTER_R",
    "FACE_MOUTH_CORNER_L",
    "FACE_MOUTH_CORNER_R",
}


def associated_meshes(armature: bpy.types.Object) -> list[bpy.types.Object]:
    meshes = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if obj.parent == armature:
            meshes.append(obj)
            continue
        for modifier in obj.modifiers:
            if modifier.type == "ARMATURE" and modifier.object == armature:
                meshes.append(obj)
                break
    return meshes


def find_shape_targets(armature: bpy.types.Object) -> dict[str, list[tuple[str, str]]]:
    targets: dict[str, list[tuple[str, str]]] = {}
    for obj in associated_meshes(armature):
        keys = obj.data.shape_keys
        if not keys:
            continue
        for block in keys.key_blocks:
            if block.name == "Basis":
                continue
            normalized = normalize_name(block.name, shape=True)
            targets.setdefault(normalized, []).append((obj.name, block.name))
    return targets


def shape_override_for_object(entry, object_name: str):
    return next(
        (
            override
            for override in entry.shape_overrides
            if override.object_name == object_name
        ),
        None,
    )


def resolved_shape_name(entry, object_name: str) -> str | None:
    """Resolve a grouped shape mapping for one mesh.

    An enabled override replaces the shared target name. A disabled override
    intentionally excludes that mesh from the group.
    """
    override = shape_override_for_object(entry, object_name)
    if override is not None:
        return override.target_name if override.enabled else None
    return entry.target_name


def matching_shape_blocks(armature: bpy.types.Object, entry):
    for obj in associated_meshes(armature):
        if entry.target_object not in {"", "*", obj.name}:
            continue
        target_name = resolved_shape_name(entry, obj.name)
        if not target_name:
            continue
        keys = obj.data.shape_keys
        if not keys:
            continue
        normalized = normalize_name(target_name, shape=True)
        candidates = [
            block
            for block in keys.key_blocks
            if normalize_name(block.name, shape=True) == normalized
            and not (
                entry.shape_scheme in SHAPE_SCHEMES
                and shape_scheme(block.name) != entry.shape_scheme
            )
        ]
        if not candidates:
            continue
        # One key per mesh: driving both ``arkit_EyeBlinkLeft`` and
        # ``facs_bs_EyeBlinkLeft`` would apply the expression twice.
        exact = [block for block in candidates if block.name == target_name]
        yield (
            obj,
            (exact or sorted(candidates, key=lambda b: shape_priority(b.name)))[0],
        )


def shape_scheme(name: str) -> str:
    """``ARKIT``, ``FACS``, ``VRM`` or ``NONE`` from what the name says.

    ``facs_bs_EyeBlinkLeft`` is the ARKit coefficient ``eyeBlinkLeft``
    behind an exporter prefix; correctives are never driven directly;
    VRoid expression keys start with ``Fcl_``.
    """
    if shape_kind(name) == "CORRECTIVE":
        return "NONE"
    if name.casefold().startswith(("fcl_", "vrm_")):
        return "VRM"
    normalized = normalize_name(name, shape=True)
    if normalized in _ARKIT_NORMALIZED:
        return "ARKIT"
    if _FACS_NAME_RE.match(normalized):
        return "FACS"
    if name.casefold().startswith(("arkit_", "arkit-", "ar_")):
        return "ARKIT"
    return "NONE"


def _facs_aliases(action_unit: str) -> tuple[str, ...]:
    """Return normalized-name-compatible aliases for common AU conventions."""
    match = re.match(r"AU0*(\d+)(?:_(L|R))?$", action_unit)
    if not match:
        return (action_unit,)
    number = int(match.group(1))
    side = match.group(2)
    bases = (f"AU{number:02d}", f"AU{number}")
    if not side:
        return bases
    word = "Left" if side == "L" else "Right"
    lower_word = word.lower()
    return tuple(
        dict.fromkeys(
            alias
            for base in bases
            for alias in (
                f"{base}_{side}",
                f"{base}.{side}",
                f"{base}-{side}",
                f"{base}{side}",
                f"{base}_{word}",
                f"{base}{word}",
                f"{word}_{base}",
                f"{lower_word}_{base}",
            )
        )
    )
