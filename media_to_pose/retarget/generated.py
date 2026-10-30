"""Apply generated SOMA motion to a rig: bake, warp to the guides, key.

Every scene frame becomes a ``_BakedFrameSample`` through ``soma_pose``.
The guide warp then makes each keyed guide frame match the user's keys
exactly, spreading the correction to its neighbours (``guide_warp``), and
the samples are keyed with the capture's key writer.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from mathutils import Matrix, Quaternion, Vector

from ..core.motion import guide_warp
from ..core.motion.kimodo import rep
from ..core.scene import SceneContext
from .guides import SceneGuide
from .keyframes import _iter_commit_baked_samples
from .samples import (
    SequenceApplyProgress,
    _apply_baked_sample_pose,
    _BakedFrameSample,
    _restore_sequence_state,
    _scene_set_float,
    _sequence_state,
)
from .soma_pose import SomaRigBinding, apply_soma_frame

DEFAULT_BLEND_FRAMES = 8.0


@dataclass(frozen=True)
class WarpReport:
    frame: float
    rotation_deg: float
    location_m: float
    bone: str = ""


def iter_bake(
    binding: SomaRigBinding,
    frames: rep.KimodoFrames,
    scene_frames: Sequence[float],
    warnings: list[str],
    *,
    scene_context: SceneContext,
) -> Iterator[SequenceApplyProgress]:
    """Pose the rig at every scene frame; returns the samples."""
    armature = binding.armature
    state = _sequence_state(armature, scene_context=scene_context)
    samples: list[_BakedFrameSample] = []
    try:
        for index, frame in enumerate(scene_frames):
            _scene_set_float(scene_context.scene, frame)
            sample = _BakedFrameSample(destination=float(frame), warning_sink=warnings)
            apply_soma_frame(
                binding,
                frames.global_rotations[index],
                frames.root_positions[index],
                sample,
                scene_context=scene_context,
                smooth_root=frames.smooth_root[index],
                heading=float(frames.heading[index]),
            )
            samples.append(sample)
            yield SequenceApplyProgress("Posing the rig", index + 1, len(scene_frames))
    finally:
        _restore_sequence_state(armature, state, scene_context=scene_context)
    return samples


def _quaternions(samples: Sequence[_BakedFrameSample], name: str) -> np.ndarray:
    return np.array([tuple(sample.bones[name].rotation) for sample in samples])


def _local_targets(
    binding: SomaRigBinding,
    samples: list[_BakedFrameSample],
    keyed: Sequence[SceneGuide],
    rows: dict[float, int],
    *,
    scene_context: SceneContext,
) -> dict[float, dict[str, tuple[Vector, Quaternion]]]:
    """Each guide's keyed armature-space poses as local transforms under the
    generated parents at that frame.

    Restoring local transforms directly would be wrong when a parent the
    user did not key (such as a ROOT that follows the heading) moved.
    Unkeyed locations keep the generated value.
    """
    armature = binding.armature
    state = _sequence_state(armature, scene_context=scene_context)
    result: dict[float, dict[str, tuple[Vector, Quaternion]]] = {}
    try:
        for guide in keyed:
            row = rows.get(guide.frame)
            if row is None:
                continue
            _apply_baked_sample_pose(
                armature, samples[row], scene_context=scene_context
            )
            targets = {}
            bones = [armature.pose.bones.get(name) for name in guide.poses]
            for bone in sorted(
                (bone for bone in bones if bone is not None),
                key=lambda item: len(item.parent_recursive),
            ):
                values = guide.poses[bone.name]
                matrix = Matrix([values[0:4], values[4:8], values[8:12], values[12:16]])
                if bone.name not in guide.located:
                    matrix.translation = bone.matrix.translation
                bone.matrix = matrix
                scene_context.update()
                location, rotation, _scale = bone.matrix_basis.decompose()
                targets[bone.name] = (location.copy(), rotation.normalized())
            result[guide.frame] = targets
    finally:
        _restore_sequence_state(armature, state, scene_context=scene_context)
    return result


def warp_to_guides(
    binding: SomaRigBinding,
    samples: list[_BakedFrameSample],
    guides: Sequence[SceneGuide],
    blend_frames: float = DEFAULT_BLEND_FRAMES,
    *,
    scene_context: SceneContext,
) -> list[WarpReport]:
    """Make every keyed guide frame match its keys; returns the size of the
    correction at each guide."""
    keyed = [guide for guide in guides if guide.poses]
    if not keyed or not samples:
        return []
    times = np.array([sample.destination for sample in samples])
    rows = {}
    for guide in keyed:
        matches = np.flatnonzero(np.isclose(times, guide.frame, atol=1.0e-3))
        if len(matches):
            rows[guide.frame] = int(matches[0])
    local = _local_targets(binding, samples, keyed, rows, scene_context=scene_context)
    reports: dict[float, list] = {frame: [0.0, 0.0, ""] for frame in rows}
    names = {name for targets in local.values() for name in targets}
    for name in sorted(names):
        if any(name not in sample.bones for sample in samples):
            continue
        targets = {
            frame: values[name] for frame, values in local.items() if name in values
        }
        rotations = {
            frame: np.array(tuple(value[1])) for frame, value in targets.items()
        }
        warped, angles = guide_warp.warp_quaternions(
            times, _quaternions(samples, name), rotations, blend_frames
        )
        for frame, angle in zip(sorted(rotations), angles, strict=True):
            if math.degrees(float(angle)) > reports[frame][0]:
                reports[frame][0] = math.degrees(float(angle))
                reports[frame][2] = name
        for sample, value in zip(samples, warped, strict=True):
            sample.bones[name].rotation = Quaternion(value).normalized()
        if not samples[0].bones[name].key_location:
            continue
        locations = {
            frame: np.array(tuple(value[0])) for frame, value in targets.items()
        }
        values = np.array([tuple(sample.bones[name].location) for sample in samples])
        moved, deltas = guide_warp.warp_vectors(times, values, locations, blend_frames)
        for frame, delta in zip(sorted(locations), deltas, strict=True):
            reports[frame][1] = max(reports[frame][1], float(np.linalg.norm(delta)))
        for sample, value in zip(samples, moved, strict=True):
            sample.bones[name].location = Vector(value)
    return [
        WarpReport(frame, round(values[0], 3), round(values[1], 4), values[2])
        for frame, values in sorted(reports.items())
    ]


def iter_write(
    binding: SomaRigBinding,
    samples: list[_BakedFrameSample],
    *,
    scene_context: SceneContext,
) -> Iterator[SequenceApplyProgress]:
    """Key the samples; returns (bone keys, shape keys) written."""
    return (
        yield from _iter_commit_baked_samples(
            binding.armature,
            samples,
            "BONE",
            binding.pelvis,
            False,
            scene_context=scene_context,
        )
    )


__all__ = [
    "DEFAULT_BLEND_FRAMES",
    "WarpReport",
    "iter_bake",
    "iter_write",
    "warp_to_guides",
]
