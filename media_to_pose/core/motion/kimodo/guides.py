"""Guides (Kimodo constraints) and their compilation into model inputs.

A guide fixes part of the motion at one window frame. The channels each type
writes match upstream ``kimodo/constraints.py``:

- ``FullBodyGuide``: positions of all 30 joints, plus the smooth root x/z,
  the pelvis height and the heading at that frame. Rotations are not used.
- ``EffectorGuide``: the positions of each chosen chain (hand and middle
  finger end, foot and toe, or the hips) and the global rotation of the
  chain's first joint, plus smooth root x/z, pelvis height and heading.
- ``RootGuide``: the smooth root x/z and, optionally, the heading.

When two guides fix the same channel at the same frame, the first one wins.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ...errors import InvalidRequest
from .. import soma
from . import rep

EFFECTORS = frozenset(soma.EFFECTOR_CHAINS)
_POSE_SHAPE = (soma.JOINT_COUNT, 3)


def _frozen(values, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.array(values, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise InvalidRequest(f"Guide {name} must be finite values of shape {shape}")
    array.setflags(write=False)
    return array


def _optional_xz(values) -> np.ndarray | None:
    return None if values is None else _frozen(values, (2,), "root x/z")


@dataclass(frozen=True)
class FullBodyGuide:
    frame: int
    positions: np.ndarray
    smooth_root_xz: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "positions", _frozen(self.positions, _POSE_SHAPE, "positions")
        )
        object.__setattr__(self, "smooth_root_xz", _optional_xz(self.smooth_root_xz))


@dataclass(frozen=True)
class EffectorGuide:
    frame: int
    effectors: frozenset[str]
    positions: np.ndarray
    rotations: np.ndarray
    smooth_root_xz: np.ndarray | None = None

    def __post_init__(self) -> None:
        effectors = frozenset(self.effectors)
        unknown = effectors - EFFECTORS
        if not effectors or unknown:
            raise InvalidRequest(
                f"Effector guides take a non-empty subset of {sorted(EFFECTORS)}"
            )
        object.__setattr__(self, "effectors", effectors)
        object.__setattr__(
            self, "positions", _frozen(self.positions, _POSE_SHAPE, "positions")
        )
        object.__setattr__(
            self,
            "rotations",
            _frozen(self.rotations, (soma.JOINT_COUNT, 3, 3), "rotations"),
        )
        object.__setattr__(self, "smooth_root_xz", _optional_xz(self.smooth_root_xz))


@dataclass(frozen=True)
class RootGuide:
    frame: int
    xz: np.ndarray
    heading: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "xz", _frozen(self.xz, (2,), "root x/z"))
        if self.heading is not None and not np.isfinite(self.heading):
            raise InvalidRequest("Guide heading must be finite")


Guide = FullBodyGuide | EffectorGuide | RootGuide


@dataclass
class _Writes:
    """Per-channel values in upstream's compile order, first write kept."""

    root_xz: dict[int, np.ndarray] = field(default_factory=dict)
    root_y: dict[int, float] = field(default_factory=dict)
    heading: dict[int, float] = field(default_factory=dict)
    rotations: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    positions: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)


def _pose_writes(writes: _Writes, guide: FullBodyGuide | EffectorGuide) -> None:
    frame, positions = guide.frame, guide.positions
    root_xz = (
        guide.smooth_root_xz
        if guide.smooth_root_xz is not None
        else positions[soma.ROOT, [0, 2]]
    )
    writes.root_xz.setdefault(frame, root_xz)
    writes.root_y.setdefault(frame, float(positions[soma.ROOT, 1]))
    writes.heading.setdefault(frame, float(soma.heading_angle(positions)))
    if isinstance(guide, FullBodyGuide):
        joints = range(soma.JOINT_COUNT)
    else:
        joints = sorted(
            {joint for name in guide.effectors for joint in soma.EFFECTOR_CHAINS[name]}
        )
        for name in guide.effectors:
            base = soma.EFFECTOR_CHAINS[name][0]
            writes.rotations.setdefault((frame, base), guide.rotations[base])
    for joint in joints:
        writes.positions.setdefault((frame, joint), positions[joint])


def compile_guides(
    guides: Sequence[Guide], frame_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Unnormalised observed frames [T, 369] and the constraint mask [T, 369]."""
    if not 1 <= frame_count <= rep.MAX_WINDOW_FRAMES:
        raise InvalidRequest(
            f"A Kimodo window has 1 to {rep.MAX_WINDOW_FRAMES} frames, not {frame_count}"
        )
    writes = _Writes()
    for guide in guides:
        if not 0 <= int(guide.frame) < frame_count:
            raise InvalidRequest(
                f"Guide frame {guide.frame} is outside the window of {frame_count} frames"
            )
        if isinstance(guide, RootGuide):
            writes.root_xz.setdefault(guide.frame, guide.xz)
            if guide.heading is not None:
                writes.heading.setdefault(guide.frame, float(guide.heading))
        else:
            _pose_writes(writes, guide)

    observed = np.zeros((frame_count, rep.FEATURE_COUNT), dtype=np.float64)
    mask = np.zeros((frame_count, rep.FEATURE_COUNT), dtype=bool)
    root = rep.SMOOTH_ROOT.start
    for frame, xz in writes.root_xz.items():
        observed[frame, [root, root + 2]] = xz
        mask[frame, [root, root + 2]] = True
    for frame, height in writes.root_y.items():
        observed[frame, root + 1] = height
        mask[frame, root + 1] = True
    for frame, angle in writes.heading.items():
        observed[frame, rep.HEADING] = (np.cos(angle), np.sin(angle))
        mask[frame, rep.HEADING] = True
    for (frame, joint), rotation in writes.rotations.items():
        start = rep.ROTATIONS.start + 6 * joint
        observed[frame, start : start + 6] = rep.matrix_to_rot6d(rotation)
        mask[frame, start : start + 6] = True
    for (frame, joint), position in writes.positions.items():
        # Positions are stored relative to the smooth root on the ground,
        # which every pose guide fixes at the same frame.
        reference = observed[frame, [root, root + 2]]
        start = rep.POSITIONS.start + 3 * joint
        observed[frame, start : start + 3] = (
            position[0] - reference[0],
            position[1],
            position[2] - reference[1],
        )
        mask[frame, start : start + 3] = True
    return observed, mask


def translate_guide(guide: Guide, offset_xz: np.ndarray, frame_shift: int = 0) -> Guide:
    """The guide moved along the ground by ``(dx, dz)`` and in time."""
    shift = np.array((offset_xz[0], 0.0, offset_xz[1]), dtype=np.float64)
    offset = np.asarray(offset_xz, dtype=np.float64)
    frame = int(guide.frame) + int(frame_shift)
    if isinstance(guide, RootGuide):
        return RootGuide(frame, guide.xz + offset, guide.heading)
    smooth = None if guide.smooth_root_xz is None else guide.smooth_root_xz + offset
    if isinstance(guide, FullBodyGuide):
        return FullBodyGuide(frame, guide.positions + shift, smooth)
    return EffectorGuide(
        frame, guide.effectors, guide.positions + shift, guide.rotations, smooth
    )


def guides_from_frames(
    frames: rep.KimodoFrames, window_frames: Sequence[int]
) -> list[Guide]:
    """Full-body and four-effector guides that pin decoded frames, as upstream
    does for the transition frames between windows."""
    result: list[Guide] = []
    for index, frame in enumerate(window_frames):
        root_xz = frames.smooth_root[index, [0, 2]]
        result.append(FullBodyGuide(frame, frames.positions[index], root_xz))
    limbs = frozenset(("LEFT_HAND", "RIGHT_HAND", "LEFT_FOOT", "RIGHT_FOOT"))
    for index, frame in enumerate(window_frames):
        root_xz = frames.smooth_root[index, [0, 2]]
        result.append(
            EffectorGuide(
                frame,
                limbs,
                frames.positions[index],
                frames.global_rotations[index],
                root_xz,
            )
        )
    return result


__all__ = [
    "EFFECTORS",
    "EffectorGuide",
    "FullBodyGuide",
    "Guide",
    "RootGuide",
    "compile_guides",
    "guides_from_frames",
    "translate_guide",
]
