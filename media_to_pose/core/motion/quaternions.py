"""Vectorised unit quaternions (w, x, y, z) for NumPy motion arrays."""

from __future__ import annotations

import numpy as np


def from_matrix(matrices: np.ndarray) -> np.ndarray:
    """Unit quaternions [..., 4] from rotation matrices [..., 3, 3], w >= 0."""
    m = np.asarray(matrices, dtype=np.float64)
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    candidates = np.stack(
        (
            1.0 + trace,
            1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
            1.0 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
            1.0 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2],
        ),
        axis=-1,
    )
    choice = np.argmax(candidates, axis=-1)
    result = np.empty((*m.shape[:-2], 4), dtype=np.float64)
    # Pick the numerically largest component for each matrix.
    rows = {
        0: (
            1.0 + trace,
            m[..., 2, 1] - m[..., 1, 2],
            m[..., 0, 2] - m[..., 2, 0],
            m[..., 1, 0] - m[..., 0, 1],
        ),
        1: (
            m[..., 2, 1] - m[..., 1, 2],
            1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
            m[..., 0, 1] + m[..., 1, 0],
            m[..., 0, 2] + m[..., 2, 0],
        ),
        2: (
            m[..., 0, 2] - m[..., 2, 0],
            m[..., 0, 1] + m[..., 1, 0],
            1.0 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
            m[..., 1, 2] + m[..., 2, 1],
        ),
        3: (
            m[..., 1, 0] - m[..., 0, 1],
            m[..., 0, 2] + m[..., 2, 0],
            m[..., 1, 2] + m[..., 2, 1],
            1.0 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2],
        ),
    }
    for index, components in rows.items():
        selected = choice == index
        if np.any(selected):
            result[selected] = np.stack(components, axis=-1)[selected]
    return canonical(normalize(result))


def to_matrix(quaternions: np.ndarray) -> np.ndarray:
    """Rotation matrices [..., 3, 3] from quaternions [..., 4]."""
    w, x, y, z = np.moveaxis(normalize(quaternions), -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape((*w.shape, 3, 3))


def normalize(quaternions: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float64)
    length = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(length, 1.0e-12)


def canonical(quaternions: np.ndarray) -> np.ndarray:
    """The same rotations with w >= 0."""
    values = np.asarray(quaternions, dtype=np.float64)
    return np.where(values[..., :1] < 0.0, -values, values)


def continuous(quaternions: np.ndarray, axis: int = 0) -> np.ndarray:
    """Flip signs along ``axis`` so neighbours stay in the same hemisphere."""
    values = np.moveaxis(np.array(quaternions, dtype=np.float64), axis, 0)
    for index in range(1, len(values)):
        flip = np.sum(values[index] * values[index - 1], axis=-1) < 0.0
        values[index][flip] *= -1.0
    return np.moveaxis(values, 0, axis)


def multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(np.asarray(left, dtype=np.float64), -1, 0)
    rw, rx, ry, rz = np.moveaxis(np.asarray(right, dtype=np.float64), -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def inverse(quaternions: np.ndarray) -> np.ndarray:
    values = normalize(quaternions)
    return values * np.array((1.0, -1.0, -1.0, -1.0))


def slerp(start: np.ndarray, end: np.ndarray, factor) -> np.ndarray:
    """Shortest-path interpolation; ``factor`` broadcasts over [...]."""
    a = normalize(start)
    b = normalize(end)
    t = np.asarray(factor, dtype=np.float64)[..., None]
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0.0, -b, b)
    dot = np.abs(dot)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    sin = np.sin(angle)
    near = sin < 1.0e-6
    safe = np.where(near, 1.0, sin)
    weight_a = np.where(near, 1.0 - t, np.sin((1.0 - t) * angle) / safe)
    weight_b = np.where(near, t, np.sin(t * angle) / safe)
    return normalize(weight_a * a + weight_b * b)


def angle(quaternions: np.ndarray) -> np.ndarray:
    """Rotation angle in radians, 0 to pi."""
    w = np.abs(normalize(quaternions)[..., 0])
    return 2.0 * np.arccos(np.clip(w, 0.0, 1.0))


IDENTITY = np.array((1.0, 0.0, 0.0, 0.0))
IDENTITY.setflags(write=False)

__all__ = [
    "IDENTITY",
    "angle",
    "canonical",
    "continuous",
    "from_matrix",
    "inverse",
    "multiply",
    "normalize",
    "slerp",
    "to_matrix",
]
