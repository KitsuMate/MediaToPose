"""Rest alignment between the SOMA skeleton and a rig, in the anatomical frame.

The anatomical frame is (right, forward, up), Z up. Kimodo's global joint
rotations are relative to SOMA's T-pose rest; a rig's are relative to its
own rest, which may be an A-pose or have differently oriented bones.

For every mapped joint a rest frame is built on both skeletons from the bone
direction (joint to child) and a reference axis: the character's right for
the torso, head and legs, its forward for the shoulders, arms and hands. The
fixed rotation ``O = F_rig @ F_somaᵀ`` takes one rest frame to the other, so

- SOMA to rig: ``D = R @ Oᵀ`` with ``R = C @ G @ Cᵀ`` (the rig bone's
  anatomical rotation from its rest);
- rig to SOMA: ``G = Cᵀ @ (D @ O) @ C``.

``C`` is ``soma.KIMODO_TO_ANATOMICAL``. Only the standard library and NumPy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..errors import MappingIncomplete
from . import soma

C = soma.KIMODO_TO_ANATOMICAL
RIGHT = np.array((1.0, 0.0, 0.0))
FORWARD = np.array((0.0, 1.0, 0.0))

# SOMA joint -> rig semantic id. Joints without one follow their parent.
JOINT_SEMANTICS: dict[str, str] = {
    "Hips": "PELVIS",
    "Spine1": "SPINE_01",
    "Spine2": "SPINE_02",
    "Chest": "CHEST",
    "Neck1": "NECK",
    "Head": "HEAD",
}
for _side, _suffix in (("Left", "L"), ("Right", "R")):
    JOINT_SEMANTICS.update(
        {
            f"{_side}Shoulder": f"SHOULDER_{_suffix}",
            f"{_side}Arm": f"UPPER_ARM_{_suffix}",
            f"{_side}ForeArm": f"LOWER_ARM_{_suffix}",
            f"{_side}Hand": f"HAND_{_suffix}",
            f"{_side}Leg": f"UPPER_LEG_{_suffix}",
            f"{_side}Shin": f"LOWER_LEG_{_suffix}",
            f"{_side}Foot": f"FOOT_{_suffix}",
            f"{_side}ToeBase": f"TOE_{_suffix}",
        }
    )

# Roles generation cannot do without (the capture's required body roles).
REQUIRED_SEMANTICS: tuple[str, ...] = (
    "PELVIS",
    "CHEST",
    "HEAD",
    *(
        f"{role}_{side}"
        for side in ("L", "R")
        for role in ("UPPER_ARM", "LOWER_ARM", "HAND", "UPPER_LEG", "LOWER_LEG", "FOOT")
    ),
)

# The child whose rest position sets each joint's bone direction on SOMA.
_SOMA_CHILD: dict[str, str] = {
    "Hips": "Spine1",
    "Spine1": "Spine2",
    "Spine2": "Chest",
    "Chest": "Neck1",
    "Neck1": "Neck2",
    "Head": "",
}
for _side in ("Left", "Right"):
    _SOMA_CHILD.update(
        {
            f"{_side}Shoulder": f"{_side}Arm",
            f"{_side}Arm": f"{_side}ForeArm",
            f"{_side}ForeArm": f"{_side}Hand",
            f"{_side}Hand": f"{_side}HandMiddleEnd",
            f"{_side}Leg": f"{_side}Shin",
            f"{_side}Shin": f"{_side}Foot",
            f"{_side}Foot": f"{_side}ToeBase",
            f"{_side}ToeBase": "",
        }
    )

# Rig semantic -> semantics whose head gives the bone direction, in order of
# preference; the bone's own tail is the fallback.
RIG_CHILDREN: dict[str, tuple[str, ...]] = {
    "PELVIS": ("SPINE_01", "SPINE_02", "CHEST"),
    "SPINE_01": ("SPINE_02", "CHEST"),
    "SPINE_02": ("CHEST",),
    "CHEST": ("NECK", "HEAD"),
    "NECK": ("HEAD",),
    "HEAD": (),
}
for _suffix in ("L", "R"):
    RIG_CHILDREN.update(
        {
            f"SHOULDER_{_suffix}": (f"UPPER_ARM_{_suffix}",),
            f"UPPER_ARM_{_suffix}": (f"LOWER_ARM_{_suffix}",),
            f"LOWER_ARM_{_suffix}": (f"HAND_{_suffix}",),
            f"HAND_{_suffix}": (f"MIDDLE_01_{_suffix}",),
            f"UPPER_LEG_{_suffix}": (f"LOWER_LEG_{_suffix}",),
            f"LOWER_LEG_{_suffix}": (f"FOOT_{_suffix}",),
            f"FOOT_{_suffix}": (f"TOE_{_suffix}",),
            f"TOE_{_suffix}": (),
        }
    )

_FORWARD_REFERENCE = ("SHOULDER", "UPPER_ARM", "LOWER_ARM", "HAND")


def reference_axis(semantic_id: str) -> np.ndarray:
    """The second axis of a joint's rest frame: forward for arms, else right."""
    return FORWARD if semantic_id.startswith(_FORWARD_REFERENCE) else RIGHT


def rest_frame(direction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """An orthonormal frame [3, 3] whose columns are the bone direction, the
    reference made orthogonal to it, and their cross product."""
    primary = np.asarray(direction, dtype=np.float64)
    length = np.linalg.norm(primary)
    primary = np.array((0.0, 0.0, 1.0)) if length < 1.0e-9 else primary / length
    secondary = np.asarray(reference, dtype=np.float64)
    secondary = secondary - primary * float(secondary @ primary)
    if np.linalg.norm(secondary) < 1.0e-6:
        fallback = FORWARD if abs(primary @ FORWARD) < 0.9 else RIGHT
        secondary = fallback - primary * float(fallback @ primary)
    secondary = secondary / np.linalg.norm(secondary)
    return np.stack((primary, secondary, np.cross(primary, secondary)), axis=1)


def soma_rest_direction(joint: str) -> np.ndarray:
    """SOMA's rest bone direction of ``joint`` in the anatomical frame."""
    index = soma.JOINT_INDEX[joint]
    child = _SOMA_CHILD.get(joint, "")
    rest = soma.REST_POSITIONS
    if child:
        direction = rest[soma.JOINT_INDEX[child]] - rest[index]
    else:
        direction = rest[index] - rest[soma.PARENTS[index]]
    return C @ direction


@dataclass(frozen=True)
class SomaAlignment:
    """Fixed rest rotations ``O`` per SOMA joint index, for the mapped joints."""

    rotations: Mapping[int, np.ndarray]

    @property
    def joints(self) -> tuple[int, ...]:
        return tuple(sorted(self.rotations))


def align(rig_directions: Mapping[str, np.ndarray]) -> SomaAlignment:
    """Alignment from the rig's rest bone directions per semantic id.

    ``rig_directions`` holds the anatomical rest direction of every mapped
    semantic in ``JOINT_SEMANTICS``. Raises ``MappingIncomplete`` when a
    required role is missing.
    """
    missing = [role for role in REQUIRED_SEMANTICS if role not in rig_directions]
    if missing:
        raise MappingIncomplete(
            "Generation needs these body roles mapped: " + ", ".join(missing),
            missing=missing,
        )
    rotations: dict[int, np.ndarray] = {}
    for joint, semantic_id in JOINT_SEMANTICS.items():
        direction = rig_directions.get(semantic_id)
        if direction is None:
            continue
        reference = reference_axis(semantic_id)
        soma_frame = rest_frame(soma_rest_direction(joint), reference)
        rig_frame = rest_frame(direction, reference)
        rotations[soma.JOINT_INDEX[joint]] = rig_frame @ soma_frame.T
    return SomaAlignment(rotations)


def soma_to_rig(
    global_rotations: np.ndarray, alignment: SomaAlignment
) -> dict[int, np.ndarray]:
    """Anatomical rig rotations from rest [..., 3, 3] per mapped joint index,
    from Kimodo global rotations [..., 30, 3, 3]."""
    rotations = np.asarray(global_rotations, dtype=np.float64)
    return {
        joint: C @ rotations[..., joint, :, :] @ C.T @ offset.T
        for joint, offset in alignment.rotations.items()
    }


def rig_to_soma(
    rig_rotations: Mapping[int, np.ndarray], alignment: SomaAlignment
) -> np.ndarray:
    """Kimodo global rotations [30, 3, 3] from anatomical rig rotations.

    Joints without a rig bone take their parent's global rotation, so the
    next mapped joint absorbs the missing bend.
    """
    result = np.empty((soma.JOINT_COUNT, 3, 3), dtype=np.float64)
    for joint in range(soma.JOINT_COUNT):
        rotation = rig_rotations.get(joint)
        offset = alignment.rotations.get(joint)
        if rotation is not None and offset is not None:
            result[joint] = C.T @ (np.asarray(rotation) @ offset) @ C
        elif soma.PARENTS[joint] >= 0:
            result[joint] = result[soma.PARENTS[joint]]
        else:
            result[joint] = np.eye(3)
    return result


def pose_positions(global_rotations: np.ndarray, hips: np.ndarray) -> np.ndarray:
    """SOMA joint positions [30, 3] for global rotations, on SOMA's own
    proportions, with the Hips at ``hips`` (Kimodo space)."""
    local = soma.global_to_local(np.asarray(global_rotations, dtype=np.float64))
    _globals, positions = soma.forward_kinematics(local, np.asarray(hips))
    return positions


__all__ = [
    "JOINT_SEMANTICS",
    "REQUIRED_SEMANTICS",
    "RIG_CHILDREN",
    "SomaAlignment",
    "align",
    "pose_positions",
    "reference_axis",
    "rest_frame",
    "rig_to_soma",
    "soma_rest_direction",
    "soma_to_rig",
]
