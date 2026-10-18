"""The face's own directions and size, from the head and eye bones.

Face controls move in face space (X to the subject's left, Y forward out of
the face, Z up along the head) by amounts sized by the distance between the
eyes, so the same settings suit faces of any size and orientation.
"""

from __future__ import annotations

from dataclasses import dataclass

from mathutils import Vector


def mapped_bone(armature, entries, semantic_id: str):
    """The pose bone mapped to ``semantic_id``, if it is enabled and set."""
    entry = entries.get(semantic_id)
    if not entry or not entry.enabled or not entry.target_name:
        return None
    return armature.pose.bones.get(entry.target_name)


@dataclass(frozen=True)
class FaceFrame:
    """The face's directions in armature space and its size.

    ``lateral`` points to the subject's left, ``up`` along the head and
    ``forward`` out of the face; ``scale`` is the distance between the eyes.
    """

    lateral: Vector
    up: Vector
    forward: Vector
    scale: float


def eye_position(armature, bone) -> Vector:
    """Where the eye is: the bone tracking a look-at target, else the bone."""
    tracker = gaze_tracker(armature, bone)
    return (tracker or bone).bone.head_local


def face_frame(armature, entries) -> FaceFrame:
    head = mapped_bone(armature, entries, "HEAD")
    eye_left = mapped_bone(armature, entries, "EYE_L")
    eye_right = mapped_bone(armature, entries, "EYE_R")
    up = (
        Vector(head.bone.matrix_local.col[1][:3]).normalized()
        if head is not None
        else Vector((0.0, 0.0, 1.0))
    )
    if eye_left is not None and eye_right is not None:
        between = eye_position(armature, eye_left) - eye_position(armature, eye_right)
        scale = between.length
        lateral = between
    else:
        scale = 0.0
        lateral = Vector((1.0, 0.0, 0.0))
    lateral = lateral - up * up.dot(lateral)
    if lateral.length < 1.0e-8:
        lateral = Vector((1.0, 0.0, 0.0))
    lateral.normalize()
    if scale < 1.0e-6:
        # No eyes: about a third of the head bone, or a typical adult 6 cm.
        scale = max(head.bone.length * 0.35, 0.01) if head is not None else 0.06
    return FaceFrame(lateral, up, lateral.cross(up).normalized(), scale)


def face_axis(frame: FaceFrame, axis_name: str) -> Vector:
    """A face-space axis: X sideways (subject's left), Y forward, Z up."""
    sign = -1.0 if axis_name.startswith("-") else 1.0
    return {"X": frame.lateral, "Y": frame.forward, "Z": frame.up}[axis_name[-1]] * sign


def bone_axis(axis_name: str) -> Vector:
    """A bone-local axis such as ``X`` or ``-Z`` as a unit vector."""
    sign = -1.0 if axis_name.startswith("-") else 1.0
    return {
        "X": Vector((sign, 0.0, 0.0)),
        "Y": Vector((0.0, sign, 0.0)),
        "Z": Vector((0.0, 0.0, sign)),
    }[axis_name[-1]]


def to_bone(bone, vector: Vector) -> Vector:
    """An armature-space direction in the bone's rest (pose channel) space."""
    return bone.bone.matrix_local.to_3x3().inverted() @ vector


def gaze_tracker(armature, target):
    """The bone that aims at ``target`` (an eye following a look-at bone)."""
    for bone in armature.pose.bones:
        for constraint in bone.constraints:
            if (
                constraint.type in {"DAMPED_TRACK", "TRACK_TO", "LOCKED_TRACK"}
                and getattr(constraint, "target", None) == armature
                and getattr(constraint, "subtarget", "") == target.name
            ):
                return bone
    return None


def gaze_pivot_distance(armature, target, frame: FaceFrame) -> float:
    """Distance from the eye to a look-at target bone."""
    tracker = gaze_tracker(armature, target)
    if tracker is None:
        # No bone tracks it: assume the target sits two eye distances ahead.
        return frame.scale * 2.0
    return (target.bone.head_local - tracker.bone.head_local).length


__all__ = [
    "FaceFrame",
    "bone_axis",
    "eye_position",
    "face_axis",
    "face_frame",
    "gaze_pivot_distance",
    "gaze_tracker",
    "mapped_bone",
    "to_bone",
]
