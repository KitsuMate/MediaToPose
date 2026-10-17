"""Fit face bone controls to the rig's own ARKit shape keys.

A rig that has both an ARKit shape key and a bone control for the same
expression (an Auto-Rig Pro face with ARKit morphs, for example) can tune
the bone's output range from the key: for each control, the output that
moves the meshes closest to the key at full strength (least squares over
every vertex of every associated mesh) becomes ``output_full``, and the key
of the opposite expression gives ``output_negative`` for signed controls.

The search samples the control along its mapped axis and refines the best
sample by golden-section search; each sample evaluates the meshes with the
bone posed and every shape key at zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from mathutils import Matrix, Quaternion, Vector

from .face_defaults import apply_default_ranges, default_ranges, is_signed
from .face_schemes import associated_meshes, matching_shape_blocks
from .face_space import face_axis, face_frame, to_bone
from .locks import suppress_manual_locks

# Control -> (ARKit keys of the positive side, keys of the negative side).
FIT_SHAPES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "FACE_JAW_OPEN": (("jawOpen",), ()),
    "FACE_JAW_SIDE": (("jawRight",), ("jawLeft",)),
    "FACE_BROW_INNER": (("browInnerUp",), ()),
    "FACE_BROW_OUTER_L": (("browOuterUpLeft",), ("browDownLeft",)),
    "FACE_BROW_OUTER_R": (("browOuterUpRight",), ("browDownRight",)),
    "FACE_LID_L": (("eyeBlinkLeft",), ("eyeWideLeft",)),
    "FACE_LID_R": (("eyeBlinkRight",), ("eyeWideRight",)),
    "FACE_CHEEK_L": (("cheekSquintLeft",), ()),
    "FACE_CHEEK_R": (("cheekSquintRight",), ()),
    "FACE_MOUTH_CORNER_L": (("mouthSmileLeft",), ("mouthFrownLeft",)),
    "FACE_MOUTH_CORNER_R": (("mouthSmileRight",), ("mouthFrownRight",)),
    "FACE_LIP_TOP": (("mouthUpperUpLeft", "mouthUpperUpRight"), ()),
    "FACE_LIP_BOTTOM": (("mouthLowerDownLeft", "mouthLowerDownRight"), ()),
}
# Samples as multiples of the default output; the best is refined.
SAMPLES = (-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
REFINE_STEPS = 14
# A fit is used only when the bone explains at least this share of the
# key's motion (squared vertex error removed, relative to the bone at
# rest); below it the bone moves differently from the key (an Auto-Rig Pro
# lip centre against ARKit keys that lift the lip sides) and the default
# stays.
MIN_EXPLAINED = 0.25
MAX_EXPANSIONS = 8
MOVED = 1.0e-5


@dataclass(frozen=True)
class SideFit:
    output: float
    default_output: float
    # Root mean square vertex distance to the key, over the vertices the key
    # moves: with the bone at rest, at the default output and at the fit.
    rms_rest: float
    rms_default: float
    rms_fit: float
    moved_vertices: int
    # Share of the key's squared vertex error the fitted bone removes.
    explained: float
    accepted: bool

    def to_dict(self) -> dict:
        return {
            name: value if isinstance(value, bool) else round(value, 6)
            for name, value in self.__dict__.items()
        }


class _Meshes:
    """Evaluated vertex positions of the rig's meshes, concatenated."""

    def __init__(self, armature, view_layer):
        self.view_layer = view_layer
        self.objects = [
            obj
            for obj in associated_meshes(armature)
            if obj.data.shape_keys is not None or obj.find_armature() == armature
        ]

    def positions(self) -> np.ndarray:
        self.view_layer.update()
        depsgraph = self.view_layer.depsgraph
        chunks = []
        for obj in self.objects:
            data = obj.evaluated_get(depsgraph).data
            coordinates = np.empty(len(data.vertices) * 3, dtype=np.float64)
            data.vertices.foreach_get("co", coordinates)
            chunks.append(coordinates.reshape(-1, 3))
        return np.concatenate(chunks) if chunks else np.zeros((0, 3))


def _blocks(meshes: _Meshes):
    for obj in meshes.objects:
        keys = obj.data.shape_keys
        yield from keys.key_blocks if keys else ()


def _reset(armature, meshes: _Meshes) -> None:
    for bone in armature.pose.bones:
        bone.matrix_basis = Matrix.Identity(4)
    for block in _blocks(meshes):
        block.value = 0.0


def _pose(bone, entry, space, value: float) -> None:
    if entry.channel_mode == "LOCATION":
        bone.matrix_basis = Matrix.Translation(
            to_bone(bone, face_axis(space, entry.axis) * value)
        )
    else:
        axis = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[entry.axis[-1]]
        sign = -1.0 if entry.axis.startswith("-") else 1.0
        rotation = Quaternion(Vector(axis) * sign, math.radians(value))
        bone.matrix_basis = rotation.to_matrix().to_4x4()


def _key_target(armature, entries, meshes: _Meshes, names) -> np.ndarray | None:
    blocks = []
    for name in names:
        entry = entries.get(f"ARKIT_{name}")
        if entry is None or not entry.enabled or not entry.target_name:
            return None
        blocks.extend(block for _obj, block in matching_shape_blocks(armature, entry))
    if not blocks:
        return None
    for block in blocks:
        block.value = 1.0
    target = meshes.positions()
    for block in blocks:
        block.value = 0.0
    return target


def _fit_side(armature, meshes, bone, entry, space, target, rest, default) -> SideFit:
    moved = np.linalg.norm(target - rest, axis=1) > MOVED
    count = max(int(moved.sum()), 1)

    def error(value: float) -> float:
        _pose(bone, entry, space, value)
        difference = meshes.positions() - target
        return float(np.einsum("ij,ij->", difference, difference))

    scale = default if abs(default) > 1.0e-9 else 1.0
    samples = sorted({scale * factor for factor in SAMPLES})
    errors = [error(value) for value in samples]
    best = min(range(len(samples)), key=errors.__getitem__)
    # The best output may lie beyond the samples: widen until it is inside.
    for _expansion in range(MAX_EXPANSIONS):
        if 0 < best < len(samples) - 1:
            break
        edge = samples[best]
        step = (
            abs(samples[1] - samples[0])
            if best == 0
            else abs(samples[-1] - samples[-2])
        )
        outward = edge - step * 2.0 if best == 0 else edge + step * 2.0
        if best == 0:
            samples.insert(0, outward)
            errors.insert(0, error(outward))
        else:
            samples.append(outward)
            errors.append(error(outward))
        best = min(range(len(samples)), key=errors.__getitem__)
    low = samples[max(best - 1, 0)]
    high = samples[min(best + 1, len(samples) - 1)]
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = high - ratio * (high - low), low + ratio * (high - low)
    error_a, error_b = error(a), error(b)
    for _step in range(REFINE_STEPS):
        if error_a < error_b:
            high, b, error_b = b, a, error_a
            a = high - ratio * (high - low)
            error_a = error(a)
        else:
            low, a, error_a = a, b, error_b
            b = low + ratio * (high - low)
            error_b = error(b)
    candidates = [(errors[best], samples[best]), (error_a, a), (error_b, b)]
    fit_error, fit_value = min(candidates)
    default_error = error(default)
    rest_error = float(np.einsum("ij,ij->", rest - target, rest - target))
    _pose(bone, entry, space, 0.0)
    explained = 1.0 - fit_error / rest_error if rest_error > 0.0 else 0.0
    return SideFit(
        output=fit_value,
        default_output=default,
        rms_rest=math.sqrt(rest_error / count),
        rms_default=math.sqrt(default_error / count),
        rms_fit=math.sqrt(fit_error / count),
        moved_vertices=int(moved.sum()),
        explained=explained,
        accepted=explained >= MIN_EXPLAINED,
    )


def fit_face_controls(
    armature, entries, view_layer, *, controls=None, apply: bool = False
) -> dict[str, dict]:
    """Fit each mapped bone control to its ARKit key; see the module doc.

    Returns, per control, a ``SideFit`` dict for ``full`` and, for signed
    controls with an opposite key, ``negative``. With ``apply`` the accepted
    fits are written to the mapping (other ranges are reset to defaults). The pose and shape key values are
    restored afterwards.
    """
    meshes = _Meshes(armature, view_layer)
    space = face_frame(armature, entries)
    report: dict[str, dict] = {}
    saved_pose = {bone.name: bone.matrix_basis.copy() for bone in armature.pose.bones}
    saved_keys = [(block, block.value) for block in _blocks(meshes)]
    saved_position = armature.data.pose_position
    try:
        armature.data.pose_position = "POSE"
        _reset(armature, meshes)
        rest = meshes.positions()
        for semantic_id, (positive, negative) in FIT_SHAPES.items():
            if controls is not None and semantic_id not in controls:
                continue
            entry = entries.get(semantic_id)
            if entry is None or not entry.enabled or not entry.target_name:
                continue
            if entry.channel_mode not in {"LOCATION", "ROTATION"}:
                continue
            bone = armature.pose.bones.get(entry.target_name)
            if bone is None:
                continue
            defaults = default_ranges(armature, entries, entry)
            sides = {"full": (positive, defaults["output_full"])}
            if is_signed(semantic_id) and negative:
                sides["negative"] = (negative, defaults["output_negative"])
            result = {}
            for side, (names, default) in sides.items():
                target = _key_target(armature, entries, meshes, names)
                if target is None:
                    continue
                result[side] = _fit_side(
                    armature, meshes, bone, entry, space, target, rest, default
                )
            if not result:
                continue
            report[semantic_id] = {side: fit.to_dict() for side, fit in result.items()}
            if apply:
                with suppress_manual_locks():
                    apply_default_ranges(armature, entries, entry)
                    full, opposite = result.get("full"), result.get("negative")
                    if full is not None and full.accepted:
                        entry.output_full = full.output
                    if opposite is not None and opposite.accepted:
                        entry.output_negative = opposite.output
    finally:
        armature.data.pose_position = saved_position
        for bone in armature.pose.bones:
            bone.matrix_basis = saved_pose[bone.name]
        for block, value in saved_keys:
            block.value = value
        view_layer.update()
    return report


__all__ = ["FIT_SHAPES", "SideFit", "fit_face_controls"]
