"""Rotation conversions used by MHR decoding (NumPy)."""

from __future__ import annotations

import numpy as np

from .common import VithRuntimeError


def _normalize(value: np.ndarray, *, epsilon: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    return value / np.maximum(norm, np.float32(epsilon))


def rotation_6d_to_matrix(value: object) -> np.ndarray:
    """Match the upstream first-two-columns Gram-Schmidt conversion."""

    data = np.asarray(value, dtype=np.float32)
    if data.shape[-1:] != (6,):
        raise VithRuntimeError(f"6-D rotations must end in 6, got {data.shape}")
    first = _normalize(data[..., :3])
    third = _normalize(np.cross(first, data[..., 3:]))
    second = np.cross(third, first)
    return np.stack((first, second, third), axis=-1).astype(np.float32, copy=False)


def matrix_to_euler_zyx(matrix: object) -> np.ndarray:
    """Intrinsic ZYX angles ``[z, y, x]`` used for the global MHR rotation."""

    value = np.asarray(matrix, dtype=np.float32)
    sy = np.sqrt(value[..., 0, 0] ** 2 + value[..., 1, 0] ** 2)
    singular = sy < np.float32(1e-6)
    x = np.arctan2(value[..., 2, 1], value[..., 2, 2])
    y = np.arctan2(-value[..., 2, 0], sy)
    z = np.arctan2(value[..., 1, 0], value[..., 0, 0])
    x_singular = np.arctan2(-value[..., 1, 2], value[..., 1, 1])
    x = np.where(singular, x_singular, x)
    z = np.where(singular, np.float32(0.0), z)
    return np.stack((z, y, x), axis=-1).astype(np.float32, copy=False)


def euler_xyz_to_matrix(angles: object) -> np.ndarray:
    """Extrinsic xyz / equivalent intrinsic ZYX rotation matrix."""

    value = np.asarray(angles, dtype=np.float32)
    x, y, z = np.moveaxis(value, -1, 0)
    cx, cy, cz = np.cos(x), np.cos(y), np.cos(z)
    sx, sy, sz = np.sin(x), np.sin(y), np.sin(z)
    return (
        np.stack(
            (
                cy * cz,
                -cx * sz + sx * sy * cz,
                sx * sz + cx * sy * cz,
                cy * sz,
                cx * cz + sx * sy * sz,
                -sx * cz + cx * sy * sz,
                -sy,
                sx * cy,
                cx * cy,
            ),
            axis=-1,
        )
        .reshape(*value.shape[:-1], 3, 3)
        .astype(np.float32, copy=False)
    )


def euler_xzy_to_matrix(angles: object) -> np.ndarray:
    """Intrinsic XZY matrix for the two wrist joints."""

    value = np.asarray(angles, dtype=np.float32)
    x, z, y = np.moveaxis(value, -1, 0)
    cx, cz, cy = np.cos(x), np.cos(z), np.cos(y)
    sx, sz, sy = np.sin(x), np.sin(z), np.sin(y)
    return (
        np.stack(
            (
                cz * cy,
                -sz,
                cz * sy,
                cx * sz * cy + sx * sy,
                cx * cz,
                cx * sz * sy - sx * cy,
                sx * sz * cy - cx * sy,
                sx * cz,
                sx * sz * sy + cx * cy,
            ),
            axis=-1,
        )
        .reshape(*value.shape[:-1], 3, 3)
        .astype(np.float32, copy=False)
    )


def matrix_to_euler_xzy(matrix: object) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float32)
    z = np.arcsin(np.clip(-value[..., 0, 1], -1.0, 1.0))
    cos_z = np.cos(z)
    regular = np.abs(cos_z) > np.float32(1e-7)
    x = np.arctan2(value[..., 2, 1], value[..., 1, 1])
    y = np.arctan2(value[..., 0, 2], value[..., 0, 0])
    # Roma fixes the third angle to zero at gimbal lock.  Solve the remaining
    # first angle from the stable lower-right block.
    x_singular = np.arctan2(-value[..., 1, 2], value[..., 2, 2])
    return np.stack(
        (np.where(regular, x, x_singular), z, np.where(regular, y, 0.0)), axis=-1
    ).astype(np.float32, copy=False)


def rotation_angle_difference(first: object, second: object) -> np.ndarray:
    a = np.asarray(first, dtype=np.float32)
    b = np.asarray(second, dtype=np.float32)
    relative = a @ np.swapaxes(b, -1, -2)
    trace = np.trace(relative, axis1=-2, axis2=-1)
    return np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))


def fix_wrist_euler(value: object) -> np.ndarray:
    wrist = np.asarray(value, dtype=np.float32)
    x, z, y = np.moveaxis(wrist, -1, 0)
    alternate = np.stack(
        (
            np.arctan2(np.sin(x + np.pi), np.cos(x + np.pi)),
            np.arctan2(np.sin(-(z + np.pi)), np.cos(-(z + np.pi))),
            np.arctan2(np.sin(y + np.pi), np.cos(y + np.pi)),
        ),
        axis=-1,
    )
    limits = ((-2.2, 1.0), (-2.2, 1.5), (-1.2, 1.5))

    def violation(candidate):
        result = np.zeros(candidate.shape[:-1], dtype=np.float32)
        for index, (low, high) in enumerate(limits):
            item = candidate[..., index]
            result += np.maximum(low - item, 0.0) ** 2
            result += np.maximum(item - high, 0.0) ** 2
        return result

    use_alternate = violation(alternate) < violation(wrist)
    return np.where(use_alternate[..., None], alternate, wrist).astype(np.float32)
