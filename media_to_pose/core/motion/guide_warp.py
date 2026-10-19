"""Make generated channels pass exactly through keyed values.

Diffusion only approximates its guides. For each channel the difference
between the key and the generated value is measured at every guide frame and
spread to the neighbouring frames:

- between two guides closer than ``2 * blend_frames`` (or always, when
  ``blend_frames`` is ``None``) the differences are interpolated;
- otherwise each difference fades out over ``blend_frames`` on either side.

Both use a smoothstep weight, whose slope is zero at the guide, so the result
keeps a continuous first derivative wherever the generated motion has one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ..errors import InvalidRequest
from . import quaternions as quat


def _smoothstep(value: np.ndarray) -> np.ndarray:
    x = np.clip(value, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _plan(
    frames: np.ndarray, guide_frames: np.ndarray, blend_frames: float | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every frame: left guide index, right guide index, right weight.

    A missing side is -1. The delta at a frame is
    ``slerp(identity or left, right or identity, weight)``.
    """
    count = len(frames)
    left = np.full(count, -1, dtype=np.int64)
    right = np.full(count, -1, dtype=np.int64)
    weight = np.zeros(count)
    position = np.searchsorted(guide_frames, frames, side="right")
    for index, frame in enumerate(frames):
        after = int(position[index])
        before = after - 1
        if before >= 0 and np.isclose(guide_frames[before], frame):
            left[index], weight[index] = before, 0.0
            continue
        has_before = before >= 0
        has_after = after < len(guide_frames)
        if has_before and has_after:
            gap = guide_frames[after] - guide_frames[before]
            if blend_frames is None or gap <= 2.0 * blend_frames:
                left[index], right[index] = before, after
                weight[index] = _smoothstep(
                    np.asarray((frame - guide_frames[before]) / gap)
                )
                continue
        if blend_frames is None:
            # Outside the guides, hold the nearest difference.
            if has_before:
                left[index] = before
            else:
                right[index], weight[index] = after, 1.0
            continue
        # Fade the nearest guide's difference to nothing.
        distance_before = frame - guide_frames[before] if has_before else np.inf
        distance_after = guide_frames[after] - frame if has_after else np.inf
        if distance_before <= distance_after:
            left[index] = before
            weight[index] = _smoothstep(np.asarray(distance_before / blend_frames))
        else:
            right[index] = after
            weight[index] = 1.0 - _smoothstep(np.asarray(distance_after / blend_frames))
    return left, right, weight


def _check(frames: np.ndarray, guides: Mapping[float, object]) -> np.ndarray:
    if frames.ndim != 1 or np.any(np.diff(frames) <= 0.0):
        raise InvalidRequest("Warp frames must be strictly increasing")
    guide_frames = np.asarray(sorted(guides), dtype=np.float64)
    missing = [frame for frame in guide_frames if not np.any(np.isclose(frames, frame))]
    if missing:
        raise InvalidRequest(f"Guide frames {missing} are not among the warped frames")
    return guide_frames


def _guide_rows(frames: np.ndarray, guide_frames: np.ndarray) -> np.ndarray:
    return np.asarray(
        [int(np.argmin(np.abs(frames - frame))) for frame in guide_frames]
    )


def warp_vectors(
    frames: Sequence[float],
    values: np.ndarray,
    guides: Mapping[float, np.ndarray],
    blend_frames: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Warped values [F, n] and the difference applied at each guide [G, n]."""
    times = np.asarray(frames, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    if not guides:
        return data.copy(), np.zeros((0, *data.shape[1:]))
    guide_frames = _check(times, guides)
    rows = _guide_rows(times, guide_frames)
    targets = np.stack(
        [np.asarray(guides[frame], dtype=np.float64) for frame in sorted(guides)]
    )
    deltas = targets - data[rows]
    left, right, weight = _plan(times, guide_frames, blend_frames)
    zero = np.zeros_like(deltas[0])
    start = np.stack([deltas[i] if i >= 0 else zero for i in left])
    end = np.stack([deltas[i] if i >= 0 else zero for i in right])
    shape = (len(times),) + (1,) * (data.ndim - 1)
    offset = start * (1.0 - weight.reshape(shape)) + end * weight.reshape(shape)
    return data + offset, deltas


def warp_quaternions(
    frames: Sequence[float],
    values: np.ndarray,
    guides: Mapping[float, np.ndarray],
    blend_frames: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Warped quaternions [F, 4] (delta applied on the left) and the angle in
    radians of the correction at each guide."""
    times = np.asarray(frames, dtype=np.float64)
    data = quat.continuous(quat.normalize(values))
    if not guides:
        return data.copy(), np.zeros(0)
    guide_frames = _check(times, guides)
    rows = _guide_rows(times, guide_frames)
    targets = quat.normalize(np.stack([guides[frame] for frame in sorted(guides)]))
    deltas = quat.canonical(quat.multiply(targets, quat.inverse(data[rows])))
    left, right, weight = _plan(times, guide_frames, blend_frames)
    identity = quat.IDENTITY
    start = np.stack([deltas[i] if i >= 0 else identity for i in left])
    end = np.stack([deltas[i] if i >= 0 else identity for i in right])
    offset = quat.slerp(start, end, weight)
    warped = quat.multiply(offset, data)
    # Hit the guides exactly, whatever rounding the blend introduced.
    warped[rows] = targets
    return quat.continuous(warped), quat.angle(deltas)


__all__ = ["warp_quaternions", "warp_vectors"]
