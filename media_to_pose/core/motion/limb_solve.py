"""Bone rotations from SOMA joint positions, with anatomical roll.

Kimodo predicts a global rotation per joint, but nothing pins how a nearly
straight bone is rolled about its own length: no child joint moves when it
twists. With straight legs (lying, standing still) the model may twist a
thigh by 150° and the shin back again, or flip a clavicle by 166° in one
frame. Every joint position stays right while a skinned rig is wrung.

This module keeps the hips, chest and head, whose children pin their roll,
and rebuilds every other bone from the joint positions:

- the direction of a bone comes from its joint to its child;
- spine and neck segments roll between the hips, chest and head;
- a clavicle rolls with the chest, carried along its swing;
- the upper and lower limb share the knee or elbow hinge axis, from the bend
  plane where it agrees with the previous frame's axis; a straight or
  over-straightened limb keeps the previous axis, carried along its swing
  and pulled gently towards the parent's;
- the foot rolls with the knee hinge, and the hand takes its roll from the
  thumb, with half of the hand's roll about the forearm moved into the
  forearm, as pronation spreads along it.

Positions are unchanged: every rebuilt rotation sends the rest bone to the
same child joint. Only the standard library and NumPy.
"""

from __future__ import annotations

import numpy as np

from . import soma

_REST = soma.REST_POSITIONS
_J = soma.JOINT_INDEX
# Hinge bend below this sine counts as straight; above the upper one the
# bend plane alone sets the roll.
_STRAIGHT = 0.05
_BENT = 0.25


def _unit(vectors: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(length, 1.0e-9)


def _frames(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """Rotation matrices [..., 3, 3] with columns primary, secondary made
    orthogonal to it, and their cross product."""
    x = _unit(primary)
    y = secondary - x * np.sum(secondary * x, axis=-1, keepdims=True)
    y = _unit(y)
    z = np.cross(x, y)
    return np.stack((x, y, z), axis=-1)


def _rotation(
    direction: np.ndarray, roll: np.ndarray, rest_direction, rest_roll
) -> np.ndarray:
    """The global rotation taking the rest bone frame to the posed one."""
    rest = _frames(np.asarray(rest_direction, float), np.asarray(rest_roll, float))
    return _frames(direction, roll) @ rest.T


def _swing(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Shortest-arc rotations [T, 3, 3] taking directions ``start`` to ``end``."""
    u, v = _unit(start), _unit(end)
    axis = np.cross(u, v)
    sine = np.linalg.norm(axis, axis=-1)
    cosine = np.sum(u * v, axis=-1)
    # Opposite directions: turn about any axis perpendicular to ``u``.
    other = np.where(np.abs(u[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    fallback = _unit(np.cross(u, other))
    axis = np.where(
        sine[:, None] > 1.0e-8, axis / np.maximum(sine, 1.0e-12)[:, None], fallback
    )
    return _about(axis, np.arctan2(sine, cosine))


def _carried(parent: np.ndarray, rest_direction, rest_axis, direction) -> np.ndarray:
    """The parent's rest hinge axis carried along the limb's swing: the roll
    with the least twist against the parent."""
    start = parent @ np.asarray(rest_direction, float)
    axis = parent @ np.asarray(rest_axis, float)
    return np.einsum("tij,tj->ti", _swing(start, direction), axis)


def _smoothstep(value: np.ndarray) -> np.ndarray:
    x = np.clip((value - _STRAIGHT) / (_BENT - _STRAIGHT), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _hinge_track(
    upper: np.ndarray,
    lower: np.ndarray,
    parent: np.ndarray,
    rest_direction,
    rest_axis,
) -> np.ndarray:
    """Hinge axes [T, 3] of a knee or elbow, continuous in time.

    Each frame starts from the previous axis carried along the upper bone's
    small swing, pulled gently towards the parent's rest axis carried along
    the swing from rest (where that is well defined: it flips when the limb
    points opposite its rest direction). The bend plane takes over as the
    limb bends, but only where its normal agrees with that axis: knees and
    elbows bend one way, so a reversed normal is an over-straightened limb.
    """
    reference = _carried(parent, rest_direction, rest_axis, upper)
    start = _unit(parent @ np.asarray(rest_direction, float))
    alignment = np.sum(start * _unit(upper), axis=-1)
    pull = 0.2 * np.clip((alignment + 0.5) / 0.5, 0.0, 1.0)
    normal = np.cross(_unit(upper), _unit(lower))
    bend = _smoothstep(np.linalg.norm(normal, axis=-1))
    normal = _unit(normal)
    axes = np.empty_like(reference)
    previous = reference[0]
    for frame in range(len(upper)):
        if frame:
            swing = _swing(upper[frame - 1 : frame], upper[frame : frame + 1])[0]
            previous = swing @ previous
        carried = (1.0 - pull[frame]) * previous + pull[frame] * reference[frame]
        carried /= max(float(np.linalg.norm(carried)), 1.0e-9)
        agree = np.clip((float(normal[frame] @ carried) + 0.3) / 0.6, 0.0, 1.0)
        weight = float(bend[frame]) * agree
        axis = weight * normal[frame] + (1.0 - weight) * carried
        previous = axis / max(float(np.linalg.norm(axis)), 1.0e-9)
        axes[frame] = previous
    return axes


def _leg(globals_: np.ndarray, positions: np.ndarray, side: str) -> None:
    hip, knee, ankle, toe = (
        _J[f"{side}Leg"],
        _J[f"{side}Shin"],
        _J[f"{side}Foot"],
        _J[f"{side}ToeBase"],
    )
    # Knees bend backwards: the rest hinge axis points to the character's
    # left (+X) for both legs.
    rest_axis = np.array((1.0, 0.0, 0.0))
    thigh = positions[:, knee] - positions[:, hip]
    shin = positions[:, ankle] - positions[:, knee]
    foot = positions[:, toe] - positions[:, ankle]
    # A normal knee bend (shin back from the knee) has its bend normal
    # along +X.
    axis = _hinge_track(
        thigh, shin, globals_[:, soma.ROOT], _REST[knee] - _REST[hip], rest_axis
    )
    globals_[:, hip] = _rotation(thigh, axis, _REST[knee] - _REST[hip], rest_axis)
    globals_[:, knee] = _rotation(shin, axis, _REST[ankle] - _REST[knee], rest_axis)
    globals_[:, ankle] = _rotation(foot, axis, _REST[toe] - _REST[ankle], rest_axis)


def _twist(rotation: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Angles [T] of rotations [T, 3, 3] about unit axes [T, 3]."""
    trace = np.trace(rotation, axis1=-2, axis2=-1)
    w = np.sqrt(np.clip(1.0 + trace, 0.0, None)) / 2.0
    v = (
        np.stack(
            (
                rotation[:, 2, 1] - rotation[:, 1, 2],
                rotation[:, 0, 2] - rotation[:, 2, 0],
                rotation[:, 1, 0] - rotation[:, 0, 1],
            ),
            axis=-1,
        )
        / 4.0
    )
    return 2.0 * np.arctan2(np.sum(v * axis, axis=-1), np.maximum(w, 1.0e-9))


def _about(axes: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Rotation matrices [T, 3, 3] about unit axes by angles (Rodrigues)."""
    x, y, z = axes[:, 0], axes[:, 1], axes[:, 2]
    zero = np.zeros_like(x)
    k = np.stack((zero, -z, y, z, zero, -x, -y, x, zero), axis=-1).reshape(-1, 3, 3)
    s, c = np.sin(angles)[:, None, None], np.cos(angles)[:, None, None]
    return np.eye(3) + s * k + (1.0 - c) * (k @ k)


def _arm(globals_: np.ndarray, positions: np.ndarray, side: str) -> None:
    shoulder, elbow, wrist = _J[f"{side}Arm"], _J[f"{side}ForeArm"], _J[f"{side}Hand"]
    middle, thumb = _J[f"{side}HandMiddleEnd"], _J[f"{side}HandThumbEnd"]
    # Elbows bend forwards (+Z) from the T-pose.
    rest_axis = np.array((0.0, -1.0 if side == "Left" else 1.0, 0.0))
    upper = positions[:, elbow] - positions[:, shoulder]
    lower = positions[:, wrist] - positions[:, elbow]
    axis = _hinge_track(
        upper,
        lower,
        globals_[:, _J[f"{side}Shoulder"]],
        _REST[elbow] - _REST[shoulder],
        rest_axis,
    )
    globals_[:, shoulder] = _rotation(
        upper, axis, _REST[elbow] - _REST[shoulder], rest_axis
    )
    forearm = _rotation(lower, axis, _REST[wrist] - _REST[elbow], rest_axis)
    hand = _rotation(
        positions[:, middle] - positions[:, wrist],
        positions[:, thumb] - positions[:, wrist],
        _REST[middle] - _REST[wrist],
        _REST[thumb] - _REST[wrist],
    )
    # Half of the hand's roll about the forearm goes into the forearm.
    along = _unit(lower)
    local = hand @ np.swapaxes(forearm, -1, -2)
    half = _about(along, 0.5 * _twist(local, along))
    globals_[:, elbow] = half @ forearm
    globals_[:, wrist] = hand


def _segment(
    globals_: np.ndarray, positions: np.ndarray, joint: str, child: str, roll
) -> None:
    j, c = _J[joint], _J[child]
    right = np.array((1.0, 0.0, 0.0))
    globals_[:, j] = _rotation(
        positions[:, c] - positions[:, j], roll, _REST[c] - _REST[j], right
    )


def _torso(globals_: np.ndarray, positions: np.ndarray) -> None:
    """Spine, neck and clavicles; the hips, chest and head are kept."""
    right = np.array((1.0, 0.0, 0.0))
    hips = globals_[:, soma.ROOT] @ right
    chest = globals_[:, _J["Chest"]] @ right
    head = globals_[:, _J["Head"]] @ right
    for joint, child, low, high, share in (
        ("Spine1", "Spine2", hips, chest, 1.0 / 3.0),
        ("Spine2", "Chest", hips, chest, 2.0 / 3.0),
        ("Neck1", "Neck2", chest, head, 1.0 / 3.0),
        ("Neck2", "Head", chest, head, 2.0 / 3.0),
    ):
        _segment(globals_, positions, joint, child, (1.0 - share) * low + share * high)
    # A clavicle takes the chest's forward axis carried along its swing: the
    # least twist, and never degenerate as a fixed axis is on a steep shrug.
    forward = np.array((0.0, 0.0, 1.0))
    for side in ("Left", "Right"):
        j, c = _J[f"{side}Shoulder"], _J[f"{side}Arm"]
        direction = positions[:, c] - positions[:, j]
        roll = _carried(
            globals_[:, _J["Chest"]], _REST[c] - _REST[j], forward, direction
        )
        globals_[:, j] = _rotation(direction, roll, _REST[c] - _REST[j], forward)


def solve_limbs(global_rotations: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Global rotations [T, 30, 3, 3] rebuilt from ``positions`` [T, 30, 3]
    for every bone whose roll no child pins; the hips, chest and head are
    kept."""
    result = np.array(global_rotations, dtype=np.float64, copy=True)
    points = np.asarray(positions, dtype=np.float64)
    _torso(result, points)
    for side in ("Left", "Right"):
        _leg(result, points, side)
        _arm(result, points, side)
    # Finger end joints follow the hand, as they have no rotation of their own.
    for side in ("Left", "Right"):
        for end in ("HandMiddleEnd", "HandThumbEnd"):
            result[:, _J[f"{side}{end}"]] = result[:, _J[f"{side}Hand"]]
        result[:, _J[f"{side}ToeBase"]] = result[:, _J[f"{side}Foot"]]
    return result


__all__ = ["solve_limbs"]
