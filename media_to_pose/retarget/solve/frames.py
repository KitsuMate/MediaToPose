"""Source and target spaces, reference directions and rotation limits."""

from __future__ import annotations

import math

from mathutils import Matrix, Quaternion, Vector

from ...core.types import CanonicalFrame
from .common import _canonical_vector, _mapped_bone, body_points


def _root_frame(right: Vector, up: Vector) -> Matrix:
    """Build a proper right/forward/up rotation frame."""
    up = up.normalized() if up.length > 1.0e-8 else Vector((0.0, 0.0, 1.0))
    right = right - up * right.dot(up)
    if right.length < 1.0e-8:
        right = Vector((-1.0, 0.0, 0.0))
        right = right - up * right.dot(up)
    right.normalize()
    forward = up.cross(right)
    if forward.length < 1.0e-8:
        forward = Vector((0.0, -1.0, 0.0))
    else:
        forward.normalize()
    up = right.cross(forward).normalized()
    return Matrix((right, forward, up)).transposed()


def source_root_basis(frame: CanonicalFrame) -> Matrix:
    """Return the detected person's yaw frame in canonical source space."""
    points = body_points(frame)
    lateral_vectors = []
    for left_name, right_name in (
        ("LEFT_HIP", "RIGHT_HIP"),
        ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ):
        left = points.get(left_name)
        right = points.get(right_name)
        if not left or not right or not left.valid or not right.valid:
            continue
        lateral = _canonical_vector(Vector(right.position) - Vector(left.position))
        # Use detected depth for subject yaw, while image vertical remains the
        # stable up axis. Body lean stays in the individual pose vectors.
        lateral.z = 0.0
        if lateral.length > 1.0e-8:
            lateral_vectors.append(lateral.normalized())

    if lateral_vectors:
        # Hips and shoulders estimate the same anatomical axis. Averaging them
        # damps monocular depth jitter without involving any Blender camera.
        right = sum(lateral_vectors, Vector())
        if right.length < 1.0e-8:
            right = lateral_vectors[0]
    else:
        right = Vector((-1.0, 0.0, 0.0))
    return _root_frame(right, Vector((0.0, 0.0, 1.0)))


def target_rest_basis(armature, entries) -> Matrix:
    """Return the immutable character rest frame in armature-local space."""
    pelvis = _mapped_bone(armature, entries, "PELVIS")
    chest = _mapped_bone(armature, entries, "CHEST")
    left_leg = _mapped_bone(armature, entries, "UPPER_LEG_L")
    right_leg = _mapped_bone(armature, entries, "UPPER_LEG_R")
    left_arm = _mapped_bone(armature, entries, "UPPER_ARM_L")
    right_arm = _mapped_bone(armature, entries, "UPPER_ARM_R")

    def head(bone):
        return bone.bone.head_local.copy() if bone else None

    left_hip = head(left_leg)
    right_hip = head(right_leg)
    left_shoulder = head(left_arm)
    right_shoulder = head(right_arm)
    right = (
        right_hip - left_hip
        if left_hip is not None and right_hip is not None
        else Vector((-1.0, 0.0, 0.0))
    )
    if (
        left_hip is not None
        and right_hip is not None
        and left_shoulder is not None
        and right_shoulder is not None
    ):
        hip_center = (left_hip + right_hip) * 0.5
        shoulder_center = (left_shoulder + right_shoulder) * 0.5
        up = shoulder_center - hip_center
    else:
        center = head(pelvis)
        upper = head(chest)
        up = (
            upper - center
            if center is not None and upper is not None
            else Vector((0.0, 0.0, 1.0))
        )
    return _root_frame(right, up)


def target_basis(
    armature,
    entries,
    *,
    rest_basis: Matrix | None = None,
) -> Matrix:
    """Return the character root frame including its current root control."""
    rest_frame = (
        rest_basis.copy()
        if rest_basis is not None
        else target_rest_basis(armature, entries)
    )

    # The armature object transform already stays outside pose space. Include
    # the mapped root control's current rotation so the capture is relative to
    # the character placement, without ever consulting a Blender camera.
    root = _mapped_bone(armature, entries, "ROOT")
    if root:
        # Extract rotation only. Pose/control scale must not shear the
        # camera-independent retarget frame.
        root_delta = (
            root.matrix.to_quaternion().to_matrix()
            @ root.bone.matrix_local.to_quaternion().inverted().to_matrix()
        )
        return root_delta @ rest_frame
    return rest_frame


def _to_target_direction(
    raw_direction: Vector,
    source_basis: Matrix,
    target_basis: Matrix,
) -> Vector:
    canonical = _canonical_vector(raw_direction)
    root_relative = source_basis.inverted_safe() @ canonical
    result = target_basis @ root_relative
    return result.normalized() if result.length > 1.0e-8 else result


def _joint_frame_target_delta(
    frame: CanonicalFrame,
    name: str,
    source_basis: Matrix,
    target_basis: Matrix,
) -> Quaternion | None:
    """Express a neutral-relative provider joint rotation in target space."""
    observation = frame.joint_frames.get(name)
    if observation is None or not observation.valid:
        return None
    values = observation.rotation_matrix
    if len(values) != 9 or not all(math.isfinite(float(value)) for value in values):
        return None
    raw_delta = Matrix(
        tuple(
            tuple(float(values[row * 3 + column]) for column in range(3))
            for row in range(3)
        )
    )
    # Provider positions use image-right, image-down, camera-depth. Convert
    # the rotation itself through the same basis change as point directions,
    # then express its axis relative to the detected and target root frames.
    canonical = Matrix(
        (
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, -1.0, 0.0),
        )
    )
    canonical_delta = canonical @ raw_delta @ canonical.inverted_safe()
    root_relative = source_basis.inverted_safe() @ canonical_delta @ source_basis
    target_delta = target_basis @ root_relative @ target_basis.inverted_safe()
    result = target_delta.to_quaternion().normalized()
    if result.w < 0.0:
        result.negate()
    return result


_BODY_CHILD_ROLE = {
    "SPINE_01": "SPINE_02",
    "SPINE_02": "CHEST",
    "CHEST": "NECK",
    "NECK": "HEAD",
    "SHOULDER_L": "UPPER_ARM_L",
    "UPPER_ARM_L": "LOWER_ARM_L",
    "LOWER_ARM_L": "HAND_L",
    "SHOULDER_R": "UPPER_ARM_R",
    "UPPER_ARM_R": "LOWER_ARM_R",
    "LOWER_ARM_R": "HAND_R",
    "UPPER_LEG_L": "LOWER_LEG_L",
    "LOWER_LEG_L": "FOOT_L",
    "FOOT_L": "TOE_L",
    "UPPER_LEG_R": "LOWER_LEG_R",
    "LOWER_LEG_R": "FOOT_R",
    "FOOT_R": "TOE_R",
}


def _rest_head(armature, entries, semantic_id: str) -> Vector | None:
    bone = _mapped_bone(armature, entries, semantic_id)
    return bone.bone.head_local.copy() if bone else None


def _finger_role(finger: str, segment: int, side: str) -> str:
    return f"{finger}_{segment:02d}_{side}"


def _target_palm_vectors(
    armature, entries, side: str
) -> tuple[Vector | None, Vector | None]:
    hand = _rest_head(armature, entries, f"HAND_{side}")
    finger_heads = [
        _rest_head(armature, entries, _finger_role(finger, 1, side))
        for finger in ("INDEX", "MIDDLE", "RING", "PINKY")
    ]
    valid_heads = [point for point in finger_heads if point is not None]
    primary = (
        sum(valid_heads, Vector()) / len(valid_heads) - hand
        if hand is not None and valid_heads
        else None
    )
    index = _rest_head(armature, entries, _finger_role("INDEX", 1, side))
    pinky = _rest_head(armature, entries, _finger_role("PINKY", 1, side))
    lateral = index - pinky if index is not None and pinky is not None else None
    return primary, lateral


def _orthonormal_palm_basis(
    primary: Vector | None,
    lateral: Vector | None,
) -> Matrix | None:
    """Return a right-handed palm-local frame with axes stored as columns."""
    if (
        primary is None
        or lateral is None
        or primary.length <= 1.0e-8
        or lateral.length <= 1.0e-8
    ):
        return None
    forward = primary.normalized()
    across = lateral - forward * lateral.dot(forward)
    if across.length <= 1.0e-8:
        return None
    across.normalize()
    normal = forward.cross(across)
    if normal.length <= 1.0e-8:
        return None
    normal.normalize()
    return Matrix((forward, across, normal)).transposed()


def _detected_palm_basis(hand) -> Matrix | None:
    wrist = hand.get("WRIST")
    palm_points = [
        hand.get("INDEX_FINGER_MCP"),
        hand.get("MIDDLE_FINGER_MCP"),
        hand.get("RING_FINGER_MCP"),
        hand.get("PINKY_MCP"),
    ]
    valid_palm = [point for point in palm_points if point and point.valid]
    index = hand.get("INDEX_FINGER_MCP")
    pinky = hand.get("PINKY_MCP")
    if (
        wrist is None
        or not wrist.valid
        or not valid_palm
        or index is None
        or not index.valid
        or pinky is None
        or not pinky.valid
    ):
        return None
    palm_center = sum(
        (Vector(point.position) for point in valid_palm),
        Vector(),
    ) / len(valid_palm)
    return _orthonormal_palm_basis(
        palm_center - Vector(wrist.position),
        Vector(index.position) - Vector(pinky.position),
    )


def _target_body_direction(armature, entries, semantic_id: str) -> Vector | None:
    start = _rest_head(armature, entries, semantic_id)
    child_id = _BODY_CHILD_ROLE.get(semantic_id)
    end = _rest_head(armature, entries, child_id) if child_id else None
    if start is not None and end is not None and (end - start).length > 1.0e-8:
        return end - start
    if semantic_id in {"HAND_L", "HAND_R"}:
        primary, _lateral = _target_palm_vectors(armature, entries, semantic_id[-1])
        if primary is not None and primary.length > 1.0e-8:
            return primary
    bone = _mapped_bone(armature, entries, semantic_id)
    if bone:
        return bone.bone.tail_local - bone.bone.head_local
    return None


def _target_torso_vectors(armature, entries) -> tuple[Vector | None, Vector | None]:
    left = _rest_head(armature, entries, "UPPER_ARM_L")
    right = _rest_head(armature, entries, "UPPER_ARM_R")
    pelvis = _rest_head(armature, entries, "PELVIS")
    chest = _rest_head(armature, entries, "CHEST")
    lateral = right - left if left is not None and right is not None else None
    up = chest - pelvis if pelvis is not None and chest is not None else None
    return lateral, up


def _target_body_reference(armature, entries, semantic_id: str) -> Vector | None:
    lateral, up = _target_torso_vectors(armature, entries)
    if semantic_id in {
        "SPINE_01",
        "SPINE_02",
        "CHEST",
        "NECK",
        "HEAD",
    }:
        return lateral
    if semantic_id.startswith("SHOULDER_"):
        return up
    adjacent = {
        "UPPER_ARM_L": "LOWER_ARM_L",
        "LOWER_ARM_L": "UPPER_ARM_L",
        "UPPER_ARM_R": "LOWER_ARM_R",
        "LOWER_ARM_R": "UPPER_ARM_R",
        "UPPER_LEG_L": "LOWER_LEG_L",
        "LOWER_LEG_L": "UPPER_LEG_L",
        "FOOT_L": "LOWER_LEG_L",
        "TOE_L": "LOWER_LEG_L",
        "UPPER_LEG_R": "LOWER_LEG_R",
        "LOWER_LEG_R": "UPPER_LEG_R",
        "FOOT_R": "LOWER_LEG_R",
        "TOE_R": "LOWER_LEG_R",
    }.get(semantic_id)
    if adjacent:
        return _target_body_direction(armature, entries, adjacent)
    if semantic_id in {"HAND_L", "HAND_R"}:
        _primary, palm_lateral = _target_palm_vectors(
            armature, entries, semantic_id[-1]
        )
        return palm_lateral
    return None


def _source_direction(points, start: str, end: str) -> Vector | None:
    first, second = points.get(start), points.get(end)
    if not first or not second or not first.valid or not second.valid:
        return None
    direction = Vector(second.position) - Vector(first.position)
    return direction if direction.length > 1.0e-8 else None


def _source_body_reference(points, semantic_id: str) -> Vector | None:
    if semantic_id in {
        "SPINE_01",
        "SPINE_02",
        "CHEST",
        "NECK",
        "HEAD",
    }:
        return _source_direction(points, "LEFT_SHOULDER", "RIGHT_SHOULDER")
    if semantic_id.startswith("SHOULDER_"):
        return _source_direction(points, "HIP_MID", "SHOULDER_MID")
    reference_pairs = {
        "UPPER_ARM_L": ("LEFT_ELBOW", "LEFT_WRIST"),
        "LOWER_ARM_L": ("LEFT_SHOULDER", "LEFT_ELBOW"),
        "HAND_L": ("LEFT_PINKY", "LEFT_INDEX"),
        "UPPER_ARM_R": ("RIGHT_ELBOW", "RIGHT_WRIST"),
        "LOWER_ARM_R": ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
        "HAND_R": ("RIGHT_PINKY", "RIGHT_INDEX"),
        "UPPER_LEG_L": ("LEFT_KNEE", "LEFT_ANKLE"),
        "LOWER_LEG_L": ("LEFT_HIP", "LEFT_KNEE"),
        "FOOT_L": ("LEFT_KNEE", "LEFT_ANKLE"),
        "TOE_L": ("LEFT_KNEE", "LEFT_ANKLE"),
        "UPPER_LEG_R": ("RIGHT_KNEE", "RIGHT_ANKLE"),
        "LOWER_LEG_R": ("RIGHT_HIP", "RIGHT_KNEE"),
        "FOOT_R": ("RIGHT_KNEE", "RIGHT_ANKLE"),
        "TOE_R": ("RIGHT_KNEE", "RIGHT_ANKLE"),
    }
    pair = reference_pairs.get(semantic_id)
    return _source_direction(points, *pair) if pair else None


def _source_head_orientation(
    points,
) -> tuple[Vector, Vector] | None:
    """Return provider-space head-up and lateral vectors from positions.

    Head roll and yaw are observable from the relative eye/ear positions.
    A provider neck joint is useful for the neck segment, but it is not a
    valid start point for a Blender head bone: eyes and ears sit physically
    forward of the cervical axis, and MHR70 has no skull-tip joint to form a
    true head segment. Use the image-plane face axis for head up instead,
    avoiding invented nod from this anatomical offset or shallow face depth.
    Explicit provider joint rotations are never copied.
    """
    left_ear = points.get("LEFT_EAR")
    right_ear = points.get("RIGHT_EAR")
    left_eye = points.get("LEFT_EYE")
    right_eye = points.get("RIGHT_EYE")
    shoulders = points.get("SHOULDER_MID")
    head_mid = points.get("HEAD_MID")
    lateral_samples = []
    for left, right in (
        (left_ear, right_ear),
        (left_eye, right_eye),
    ):
        if left is None or right is None or not left.valid or not right.valid:
            continue
        sample = Vector(right.position) - Vector(left.position)
        if sample.length > 1.0e-8:
            sample.normalize()
            if lateral_samples and sample.dot(lateral_samples[0]) < 0.0:
                sample.negate()
            lateral_samples.append(sample)
    if not lateral_samples:
        return None
    lateral = sum(lateral_samples, Vector())
    if lateral.length < 1.0e-8:
        return None
    lateral.normalize()

    # Provider coordinates are image-right X, image-down Y, and depth Z.
    # Rotate the observed face lateral 90 degrees in the image plane. The full
    # lateral (including depth) still carries face yaw into the two-vector
    # orientation solve, but no invalid neck-to-face segment can invent nod.
    up = Vector((lateral.y, -lateral.x, 0.0))
    if lateral.length < 1.0e-8 or up.length < 1.0e-8:
        return None
    # Face landmark ordering can flip the cross-product sign across provider
    # conventions. Align it to the observed torso-to-head direction.
    if shoulders and head_mid and shoulders.valid and head_mid.valid:
        torso_up = Vector(head_mid.position) - Vector(shoulders.position)
        if (
            torso_up.length > 1.0e-8
            and _canonical_vector(up).dot(_canonical_vector(torso_up)) < 0.0
        ):
            up.negate()
    return up, lateral


def _source_mhr_head_orientation(
    frame: CanonicalFrame,
) -> tuple[Vector, Vector] | None:
    """Build the SAM skull-up frame from articulated MHR127 positions.

    ``c_head_null`` is the end-site extension parented to ``c_head``.  Its
    position therefore supplies the head bone's primary (anatomical-up) axis;
    the eye centres supply the lateral axis.  Do not derive up by crossing the
    eye line with this vector: doing so treats the end site as a face-forward
    marker and folds an otherwise upright skull by roughly 90 degrees.
    """
    head = frame.auxiliary.get("MHR_HEAD")
    tip = frame.auxiliary.get("MHR_HEAD_TIP")
    left_eye = frame.auxiliary.get("MHR_EYE_L")
    right_eye = frame.auxiliary.get("MHR_EYE_R")
    if not all(
        item is not None and item.valid for item in (head, tip, left_eye, right_eye)
    ):
        return None
    primary = Vector(tip.position) - Vector(head.position)
    lateral = Vector(right_eye.position) - Vector(left_eye.position)
    if primary.length < 1.0e-6 or lateral.length < 1.0e-6:
        return None
    # On a near-frontal face, monocular crown depth is weakly constrained and
    # stylized SAM fits can turn a modest head roll into a pronounced nod.
    # Keep the observed image-plane crown direction authoritative and cap only
    # its camera-depth component. Profile views retain the full MHR depth.
    body_left_eye = frame.body.get("LEFT_EYE")
    body_right_eye = frame.body.get("RIGHT_EYE")
    body_left_shoulder = frame.body.get("LEFT_SHOULDER")
    body_right_shoulder = frame.body.get("RIGHT_SHOULDER")
    image_frontal = False
    if all(
        item is not None and item.valid
        for item in (
            body_left_eye,
            body_right_eye,
            body_left_shoulder,
            body_right_shoulder,
        )
    ):
        eye_width = math.dist(body_left_eye.image, body_right_eye.image)
        shoulder_width = math.dist(body_left_shoulder.image, body_right_shoulder.image)
        image_frontal = shoulder_width > 1.0e-6 and eye_width / shoulder_width > 0.25
    if image_frontal or abs(lateral.z) / lateral.length < 0.35:
        planar_length = math.hypot(primary.x, primary.y)
        if planar_length > 1.0e-6:
            depth_limit = planar_length * math.tan(math.radians(15.0))
            primary.z = min(max(primary.z, -depth_limit), depth_limit)
    axis = primary.normalized()
    projected = lateral - axis * lateral.dot(axis)
    if projected.length < lateral.length * 0.12:
        return None
    neck = frame.auxiliary.get("MHR_NECK")
    if neck and neck.valid:
        cervical = Vector(head.position) - Vector(neck.position)
        if cervical.length > 1.0e-6 and primary.dot(cervical) < 0.0:
            primary.negate()
    return primary, projected
