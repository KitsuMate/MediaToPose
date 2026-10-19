"""Mapping between scene frames and Kimodo's 30 fps frames, and resampling."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..errors import InvalidRequest
from . import quaternions as quat
from .kimodo import rep


def model_index(scene_frame: float, start_frame: float, scene_fps: float) -> int:
    """The nearest 30 fps frame for a scene frame, counted from ``start_frame``."""
    _check_fps(scene_fps)
    return round((float(scene_frame) - float(start_frame)) * rep.FPS / scene_fps)


def model_frame_count(first: float, last: float, scene_fps: float) -> int:
    """How many 30 fps frames cover the scene frames ``first..last``."""
    if last < first:
        raise InvalidRequest(f"Frame range {first}..{last} is empty")
    return model_index(last, first, scene_fps) + 1


def _check_fps(scene_fps: float) -> None:
    if not np.isfinite(scene_fps) or scene_fps <= 0.0:
        raise InvalidRequest(f"Scene frame rate must be positive, not {scene_fps}")


def resample(
    frames: rep.KimodoFrames,
    scene_frames: Sequence[float],
    start_frame: float,
    scene_fps: float,
) -> rep.KimodoFrames:
    """Motion sampled at scene frames: lerp for positions, slerp for rotations.

    Times outside the generated range hold the first or last frame.
    """
    _check_fps(scene_fps)
    count = frames.frame_count
    if count == 0:
        raise InvalidRequest("There are no generated frames to resample")
    times = (
        (np.asarray(scene_frames, dtype=np.float64) - start_frame) * rep.FPS / scene_fps
    )
    times = np.clip(times, 0.0, count - 1)
    lower = np.floor(times).astype(np.int64)
    upper = np.minimum(lower + 1, count - 1)
    factor = times - lower

    def lerp(values: np.ndarray) -> np.ndarray:
        shape = (len(factor),) + (1,) * (values.ndim - 1)
        weight = factor.reshape(shape)
        return values[lower] * (1.0 - weight) + values[upper] * weight

    rotations = quat.continuous(quat.from_matrix(frames.global_rotations))
    blended = quat.slerp(rotations[lower], rotations[upper], factor[:, None])
    heading = np.unwrap(frames.heading)
    return rep.KimodoFrames(
        global_rotations=quat.to_matrix(blended),
        positions=lerp(frames.positions),
        root_positions=lerp(frames.root_positions),
        smooth_root=lerp(frames.smooth_root),
        heading=np.angle(np.exp(1j * lerp(heading))),
        contact_scores=lerp(frames.contact_scores),
    )


__all__ = ["model_frame_count", "model_index", "resample"]
