"""Plant the feet of generated motion during the model's own contacts.

Kimodo predicts heel and toe contacts per frame but its feet still drift
while planted (raw foot skate). Upstream's post-processing pins each contact
interval with leg IK; this is the NumPy equivalent on SOMA joint positions:

- the ankle is pinned over each run of heel contact (score above 0.5) and
  the toe over each run of toe contact, the toe only while the heel is off,
  so the heel can lift at toe-off;
- a pinned point's target is its position at the run's middle frame, or at
  a guide frame inside the run, never lower than its rest height above the
  floor;
- the correction is held over the interval and faded in and out over a few
  frames on either side, so nothing pops;
- a two-bone IK from the fixed hip places the knee in its original bend
  plane, keeping both bone lengths; the foot moves rigidly with the ankle.

Only positions change. ``limb_solve`` then rebuilds the leg rotations from
them. Only the standard library and NumPy.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from . import soma

_J = soma.JOINT_INDEX
_REST_LOW = soma.REST_POSITIONS[:, 1].min()
# Rest heights above the floor of the ankle and toe joints.
_REST_HEIGHT = {
    joint: float(soma.REST_POSITIONS[_J[joint], 1] - _REST_LOW)
    for side in ("Left", "Right")
    for joint in (f"{side}Foot", f"{side}ToeBase")
}
# Contact columns (heel, toe) per side, in feature order.
_COLUMNS = {"Left": (0, 1), "Right": (2, 3)}
MIN_FRAMES = 3
FADE_FRAMES = 4


@dataclass(frozen=True)
class PlantReport:
    intervals: int
    planted_frames: int
    sliding_before: float
    sliding_after: float


def contact_intervals(flags: np.ndarray, minimum: int) -> list[tuple[int, int]]:
    """``[start, stop)`` runs of True at least ``minimum`` frames long."""
    runs, start = [], None
    for index, value in enumerate(list(flags) + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= minimum:
                runs.append((start, index))
            start = None
    return runs


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def pin_offsets(
    pinned: np.ndarray,
    runs: list[tuple[int, int]],
    anchors: list[int],
    floor: float,
    fade: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame translations [T, 3] that pin ``pinned`` on each run, held
    into the fade frames, and their weights [T] (1 inside, easing out)."""
    count = len(pinned)
    offsets = np.zeros((count, 3))
    weights = np.zeros(count)
    for (start, stop), anchor in zip(runs, anchors, strict=True):
        target = pinned[anchor].copy()
        target[1] = max(target[1], floor)
        inside = target[None, :] - pinned[start:stop]
        offsets[start:stop] = inside
        weights[start:stop] = 1.0
        for step in range(1, fade + 1):
            weight = float(_smoothstep(np.array(1.0 - step / (fade + 1))))
            for frame, edge in ((start - step, start), (stop - 1 + step, stop - 1)):
                if 0 <= frame < count and weight > weights[frame]:
                    offsets[frame] = inside[edge - start]
                    weights[frame] = weight
    return offsets, weights


def _anchors(runs: list[tuple[int, int]], guides: list[int]) -> list[int]:
    anchors = []
    for start, stop in runs:
        inside = [frame for frame in guides if start <= frame < stop]
        anchors.append(inside[0] if inside else (start + stop - 1) // 2)
    return anchors


def foot_translation(
    ankle: np.ndarray,
    toe: np.ndarray,
    heel_flags: np.ndarray,
    toe_flags: np.ndarray,
    *,
    guide_frames: Iterable[int] = (),
    floors: tuple[float, float] = (-np.inf, -np.inf),
    min_frames: int = MIN_FRAMES,
    fade_frames: int = FADE_FRAMES,
) -> tuple[np.ndarray, int, int]:
    """Per-frame translations [T, 3] of one foot that plant it.

    The ankle is pinned while the heel touches; the toe is pinned while only
    the toe touches, so the heel can lift at toe-off. Both blend with their
    fades. Returns the translations, the intervals and the planted frames.
    """
    guides = sorted({int(frame) for frame in guide_frames})
    heel_runs = contact_intervals(np.asarray(heel_flags, bool), min_frames)
    toe_runs = contact_intervals(np.asarray(toe_flags, bool), min_frames)
    heel_move, heel_weight = pin_offsets(
        ankle, heel_runs, _anchors(heel_runs, guides), floors[0], fade_frames
    )
    toe_move, toe_weight = pin_offsets(
        toe, toe_runs, _anchors(toe_runs, guides), floors[1], fade_frames
    )
    move = (
        heel_weight[:, None] * heel_move
        + ((1.0 - heel_weight) * toe_weight)[:, None] * toe_move
    )
    planted = int(np.count_nonzero((heel_weight >= 1.0) | (toe_weight >= 1.0)))
    return move, len(heel_runs) + len(toe_runs), planted


def two_bone_ik(
    hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Knee and ankle [T, 3] reaching ``target`` from ``hip``, keeping bone
    lengths and the original bend plane."""
    upper = np.linalg.norm(knee - hip, axis=-1, keepdims=True)
    lower = np.linalg.norm(ankle - knee, axis=-1, keepdims=True)
    reach = target - hip
    distance = np.linalg.norm(reach, axis=-1, keepdims=True)
    distance = np.clip(distance, np.abs(upper - lower) + 1.0e-6, upper + lower - 1.0e-6)
    along = reach / np.maximum(np.linalg.norm(reach, axis=-1, keepdims=True), 1.0e-9)
    bend = knee - hip
    bend = bend - along * np.sum(bend * along, axis=-1, keepdims=True)
    fallback = np.cross(along, np.array((1.0, 0.0, 0.0)))
    side = np.linalg.norm(bend, axis=-1, keepdims=True)
    bend = np.where(side > 1.0e-6, bend / np.maximum(side, 1.0e-9), fallback)
    cosine = np.clip(
        (upper**2 + distance**2 - lower**2) / (2.0 * upper * distance), -1.0, 1.0
    )
    sine = np.sqrt(1.0 - cosine**2)
    new_knee = hip + upper * (cosine * along + sine * bend)
    new_ankle = hip + along * distance
    return new_knee, new_ankle


def _sliding(points: np.ndarray, flags: np.ndarray) -> float:
    """Mean horizontal speed (m per frame) of ``points`` while ``flags``."""
    moving = np.linalg.norm(np.diff(points[:, [0, 2]], axis=0), axis=-1)
    both = flags[1:] & flags[:-1]
    return float(moving[both].mean()) if both.any() else 0.0


def plant_feet(
    positions: np.ndarray,
    contacts: np.ndarray,
    *,
    guide_frames: Iterable[int] = (),
    min_frames: int = MIN_FRAMES,
    fade_frames: int = FADE_FRAMES,
) -> tuple[np.ndarray, PlantReport]:
    """Positions [T, 30, 3] with planted feet, and what was done.

    ``contacts`` are the decoded contact flags [T, 4] (left heel, left toe,
    right heel, right toe). ``guide_frames`` anchor an interval that holds
    one, so a keyed pose keeps its foot where the key put it.
    """
    result = np.array(positions, dtype=np.float64, copy=True)
    flags = np.asarray(contacts, dtype=bool)
    intervals = planted = 0
    before = after = 0.0
    for side, (heel, toe) in _COLUMNS.items():
        hip, knee = _J[f"{side}Leg"], _J[f"{side}Shin"]
        ankle, tip = _J[f"{side}Foot"], _J[f"{side}ToeBase"]
        touching = flags[:, heel] | flags[:, toe]
        if not touching.any():
            continue
        before += _sliding(result[:, ankle], flags[:, heel])
        move, runs, count = foot_translation(
            result[:, ankle],
            result[:, tip],
            flags[:, heel],
            flags[:, toe],
            guide_frames=guide_frames,
            floors=(
                _REST_HEIGHT[f"{side}Foot"],
                _REST_HEIGHT[f"{side}ToeBase"],
            ),
            min_frames=min_frames,
            fade_frames=fade_frames,
        )
        intervals += runs
        planted += count
        foot = result[:, tip] - result[:, ankle]
        new_knee, new_ankle = two_bone_ik(
            result[:, hip], result[:, knee], result[:, ankle], result[:, ankle] + move
        )
        result[:, knee] = new_knee
        result[:, ankle] = new_ankle
        result[:, tip] = new_ankle + foot
        after += _sliding(result[:, ankle], flags[:, heel])
    return result, PlantReport(intervals, planted, before, after)


__all__ = [
    "FADE_FRAMES",
    "MIN_FRAMES",
    "PlantReport",
    "contact_intervals",
    "foot_translation",
    "pin_offsets",
    "plant_feet",
    "two_bone_ik",
]
