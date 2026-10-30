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
from mathutils import Quaternion, Vector

from ..core.motion import guide_warp
from ..core.motion.kimodo import rep
from ..core.scene import SceneContext
from .guides import SceneGuide
from .keyframes import _iter_commit_baked_samples
from .samples import (
    SequenceApplyProgress,
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


def warp_to_guides(
    samples: list[_BakedFrameSample],
    guides: Sequence[SceneGuide],
    blend_frames: float = DEFAULT_BLEND_FRAMES,
) -> list[WarpReport]:
    """Make every keyed guide frame match its keys; returns the size of the
    correction at each guide."""
    keyed = [guide for guide in guides if guide.basis]
    if not keyed or not samples:
        return []
    times = np.array([sample.destination for sample in samples])
    rows = {}
    for guide in keyed:
        matches = np.flatnonzero(np.isclose(times, guide.frame, atol=1.0e-3))
        if len(matches):
            rows[guide.frame] = int(matches[0])
    reports = {frame: [0.0, 0.0] for frame in rows}
    names = {name for guide in keyed for name in guide.basis}
    for name in sorted(names):
        if any(name not in sample.bones for sample in samples):
            continue
        targets = {
            guide.frame: guide.basis[name]
            for guide in keyed
            if name in guide.basis and guide.frame in rows
        }
        if not targets:
            continue
        rotations = {frame: np.array(value[1]) for frame, value in targets.items()}
        warped, angles = guide_warp.warp_quaternions(
            times, _quaternions(samples, name), rotations, blend_frames
        )
        for frame, angle in zip(sorted(rotations), angles, strict=True):
            reports[frame][0] = max(reports[frame][0], math.degrees(float(angle)))
        for sample, value in zip(samples, warped, strict=True):
            sample.bones[name].rotation = Quaternion(value).normalized()
        if not samples[0].bones[name].key_location:
            continue
        locations = {frame: np.array(value[0]) for frame, value in targets.items()}
        values = np.array([tuple(sample.bones[name].location) for sample in samples])
        moved, deltas = guide_warp.warp_vectors(times, values, locations, blend_frames)
        for frame, delta in zip(sorted(locations), deltas, strict=True):
            reports[frame][1] = max(reports[frame][1], float(np.linalg.norm(delta)))
        for sample, value in zip(samples, moved, strict=True):
            sample.bones[name].location = Vector(value)
    return [
        WarpReport(frame, round(values[0], 3), round(values[1], 4))
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
