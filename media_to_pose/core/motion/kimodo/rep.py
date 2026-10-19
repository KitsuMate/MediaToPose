"""Kimodo's 369-feature motion frames: layout, normalisation and decoding.

Each frame at 30 fps holds, in order:

- ``0:3`` smooth root x and z, with the pelvis height as y;
- ``3:5`` heading as (cos, sin);
- ``5:95`` the 30 joint positions minus the smooth root's x and z;
- ``95:275`` the 30 global rotations as 6D (the first two matrix columns);
- ``275:365`` the 30 joint velocities;
- ``365:369`` foot contacts: left heel, left toe, right heel, right toe.

All positions are Kimodo space (see ``core.motion.soma``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...errors import BackendError
from .. import soma

FPS = 30.0
FEATURE_COUNT = 369
MAX_WINDOW_FRAMES = 300
TEXT_TOKENS = 50
TEXT_WIDTH = 4096

SMOOTH_ROOT = slice(0, 3)
HEADING = slice(3, 5)
POSITIONS = slice(5, 95)
ROTATIONS = slice(95, 275)
VELOCITIES = slice(275, 365)
CONTACTS = slice(365, 369)
ROOT_FEATURES = 5

CONTACT_THRESHOLD = 0.5
STATS_EPSILON = 1.0e-5


@dataclass(frozen=True)
class FeatureStats:
    """Per-feature mean and standard deviation used to normalise frames."""

    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        for name in ("mean", "std"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != (FEATURE_COUNT,) or not np.all(np.isfinite(values)):
                raise BackendError(
                    f"Kimodo {name} statistics must be {FEATURE_COUNT} finite values"
                )
            values = values.copy()
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    @classmethod
    def from_parts(
        cls,
        global_root_mean: np.ndarray,
        global_root_std: np.ndarray,
        body_mean: np.ndarray,
        body_std: np.ndarray,
    ) -> FeatureStats:
        """Join the upstream ``global_root`` (5) and ``body`` (364) statistics."""
        parts = (global_root_mean, global_root_std, body_mean, body_std)
        root_mean, root_std, mean, std = (
            np.asarray(part, dtype=np.float64).reshape(-1) for part in parts
        )
        if root_mean.shape != (ROOT_FEATURES,) or root_std.shape != (ROOT_FEATURES,):
            raise BackendError("Kimodo global_root statistics must have 5 values")
        return cls(np.concatenate((root_mean, mean)), np.concatenate((root_std, std)))

    @property
    def scale(self) -> np.ndarray:
        return np.sqrt(self.std**2 + STATS_EPSILON)

    def normalize(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features)
        dtype = values.dtype if np.issubdtype(values.dtype, np.floating) else np.float64
        return ((values - self.mean) / self.scale).astype(dtype, copy=False)

    def denormalize(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features)
        dtype = values.dtype if np.issubdtype(values.dtype, np.floating) else np.float64
        return (values * self.scale + self.mean).astype(dtype, copy=False)


def matrix_to_rot6d(matrices: np.ndarray) -> np.ndarray:
    """6D rotations [..., 6]: the first and second matrix columns."""
    values = np.asarray(matrices)
    return np.concatenate((values[..., :, 0], values[..., :, 1]), axis=-1)


def rot6d_to_matrix(rotations: np.ndarray) -> np.ndarray:
    """Rotation matrices [..., 3, 3] from 6D values by Gram-Schmidt."""
    values = np.asarray(rotations, dtype=np.float64)
    if values.shape[-1] != 6:
        raise BackendError("6D rotations need 6 values in the last axis")
    x = _unit(values[..., 0:3])
    z = _unit(np.cross(x, values[..., 3:6]))
    y = np.cross(z, x)
    return np.stack((x, y, z), axis=-1)


def _unit(vectors: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(length, 1.0e-12)


@dataclass(frozen=True)
class KimodoFrames:
    """Decoded motion in Kimodo space at 30 fps.

    ``global_rotations`` [T, 30, 3, 3] are identity at the neutral T-pose;
    ``positions`` [T, 30, 3] come from forward kinematics of those rotations
    on the SOMA skeleton; ``root_positions`` [T, 3] are the Hips;
    ``smooth_root`` [T, 3] is the model's smoothed root path (y is the pelvis
    height); ``heading`` [T] is in radians about +Y; ``contact_scores``
    [T, 4] are the raw contact features.
    """

    global_rotations: np.ndarray
    positions: np.ndarray
    root_positions: np.ndarray
    smooth_root: np.ndarray
    heading: np.ndarray
    contact_scores: np.ndarray

    @property
    def frame_count(self) -> int:
        return len(self.root_positions)

    @property
    def contacts(self) -> np.ndarray:
        return self.contact_scores > CONTACT_THRESHOLD

    @property
    def local_rotations(self) -> np.ndarray:
        return soma.global_to_local(self.global_rotations)

    def slice(self, start: int, stop: int) -> KimodoFrames:
        return KimodoFrames(
            self.global_rotations[start:stop],
            self.positions[start:stop],
            self.root_positions[start:stop],
            self.smooth_root[start:stop],
            self.heading[start:stop],
            self.contact_scores[start:stop],
        )

    def translated(self, offset_xz: np.ndarray) -> KimodoFrames:
        """The same motion moved along the ground by ``(dx, dz)``."""
        shift = np.array((offset_xz[0], 0.0, offset_xz[1]), dtype=np.float64)
        return KimodoFrames(
            self.global_rotations,
            self.positions + shift,
            self.root_positions + shift,
            self.smooth_root + shift,
            self.heading,
            self.contact_scores,
        )


def decode(features: np.ndarray, stats: FeatureStats | None = None) -> KimodoFrames:
    """Decode frames [T, 369]; pass ``stats`` when the frames are normalised."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != FEATURE_COUNT:
        raise BackendError(f"Kimodo frames must have shape [T, {FEATURE_COUNT}]")
    if not np.all(np.isfinite(values)):
        raise BackendError("Kimodo produced non-finite motion features")
    if stats is not None:
        values = stats.denormalize(values)
    count = len(values)
    smooth_root = values[:, SMOOTH_ROOT].copy()
    heading = np.arctan2(values[:, HEADING][:, 1], values[:, HEADING][:, 0])
    local_positions = values[:, POSITIONS].reshape(count, soma.JOINT_COUNT, 3)
    global_rotations = rot6d_to_matrix(
        values[:, ROTATIONS].reshape(count, soma.JOINT_COUNT, 6)
    )
    root_positions = local_positions[:, soma.ROOT, :].copy()
    root_positions[:, 0] += smooth_root[:, 0]
    root_positions[:, 2] += smooth_root[:, 2]
    # Rebuild positions from the rotations, as upstream does by default, so
    # bone lengths are always the skeleton's.
    global_rotations, positions = soma.forward_kinematics(
        soma.global_to_local(global_rotations), root_positions
    )
    return KimodoFrames(
        global_rotations=global_rotations,
        positions=positions,
        root_positions=root_positions,
        smooth_root=smooth_root,
        heading=heading,
        contact_scores=values[:, CONTACTS].copy(),
    )


__all__ = [
    "CONTACTS",
    "CONTACT_THRESHOLD",
    "FEATURE_COUNT",
    "FPS",
    "HEADING",
    "MAX_WINDOW_FRAMES",
    "POSITIONS",
    "ROOT_FEATURES",
    "ROTATIONS",
    "SMOOTH_ROOT",
    "STATS_EPSILON",
    "TEXT_TOKENS",
    "TEXT_WIDTH",
    "VELOCITIES",
    "FeatureStats",
    "KimodoFrames",
    "decode",
    "matrix_to_rot6d",
    "rot6d_to_matrix",
]
