"""IK/FK limb ownership and IK control placement."""

from __future__ import annotations

import math

from mathutils import Matrix, Quaternion, Vector

from ...core.scene import SceneContext
from ...rig.profiles.common import profile_identifier
from ...rig.profiles.fk_switches import configure_fk_switches
from ..keyframes import _action_fcurves
from ..target_pose import TargetPose
from .common import evaluated_semantic_bone
from .frames import _target_body_direction
from .translation import PoseTranslationSolver


def prepare_limb_ownership(armature) -> None:
    """Populate metadata without changing or keying FK/IK switches."""
    configure_fk_switches(
        armature,
        profile_identifier(armature.mtp_rig_config),
    )


def limb_ownership(armature) -> dict[tuple[str, str], dict]:
    """Snapshot current per-limb FK/IK ownership without changing the rig."""
    config = armature.mtp_rig_config
    result = {}
    for chain_name, chain_id in (("arm", "ARM"), ("leg", "LEG")):
        for side_name, side in (("l", "L"), ("r", "R")):
            switch_name = getattr(config, f"{chain_name}_fk_switch_bone_{side_name}")
            property_name = getattr(
                config, f"{chain_name}_fk_switch_property_{side_name}"
            )
            switch = armature.pose.bones.get(switch_name)
            fk_value = float(
                getattr(
                    config,
                    f"{chain_name}_fk_switch_value_{side_name}",
                )
            )
            ik_value = float(
                getattr(
                    config,
                    f"{chain_name}_ik_switch_value_{side_name}",
                )
            )
            current = (
                float(switch[property_name])
                if switch and property_name and property_name in switch
                else fk_value
            )
            mode = "IK" if abs(current - ik_value) < abs(current - fk_value) else "FK"
            result[(chain_id, side)] = {
                "mode": mode,
                "switch": switch,
                "property": property_name,
                "value": current,
                "target": armature.pose.bones.get(
                    getattr(
                        config,
                        f"{chain_name}_ik_target_bone_{side_name}",
                    )
                ),
                "pole": armature.pose.bones.get(
                    getattr(
                        config,
                        f"{chain_name}_ik_pole_bone_{side_name}",
                    )
                ),
                "toe": (
                    armature.pose.bones.get(
                        getattr(
                            config,
                            f"leg_ik_toe_bone_{side_name}",
                        )
                    )
                    if chain_name == "leg"
                    else None
                ),
                "rotation_bend": (
                    armature.pose.bones.get(
                        ("upper_arm_ik." if chain_name == "arm" else "thigh_ik.") + side
                    )
                    if (
                        switch
                        and "pole_vector" in switch
                        and not bool(switch["pole_vector"])
                    )
                    else None
                ),
            }
    return result


def validate_limb_switch_interval(
    armature,
    start: float,
    end: float,
) -> str | None:
    """Reject a bake interval containing an existing ownership transition."""
    if end <= start:
        return None
    animation = armature.animation_data
    action = animation.action if animation else None
    if not action:
        return None
    for info in limb_ownership(armature).values():
        switch = info["switch"]
        property_name = info["property"]
        if not switch or not property_name:
            continue
        data_path = switch.path_from_id(f'["{property_name}"]')
        for curve in _action_fcurves(action):
            if curve.data_path != data_path:
                continue
            start_value = float(curve.evaluate(start))
            if abs(float(curve.evaluate(end)) - start_value) > 1.0e-5:
                return (
                    f"{switch.name} changes FK/IK ownership inside the "
                    "bake range; bake mode-consistent segments"
                )
            for point in curve.keyframe_points:
                frame = float(point.co.x)
                if (
                    start < frame <= end
                    and abs(float(curve.evaluate(frame)) - start_value) > 1.0e-5
                ):
                    return (
                        f"{switch.name} changes FK/IK ownership inside the "
                        "bake range; bake mode-consistent segments"
                    )
            # Bezier handles can overshoot or cross ownership thresholds
            # between key coordinates. Sample subframes across the interval,
            # including the actual integer frames a bake normally writes.
            sample = math.floor(start * 4.0 + 1.0) / 4.0
            while sample < end:
                if abs(float(curve.evaluate(sample)) - start_value) > 1.0e-5:
                    return (
                        f"{switch.name} changes FK/IK ownership inside the "
                        "bake range; bake mode-consistent segments"
                    )
                sample += 0.25
    return None


def _ik_pole_location(
    upper_matrix: Matrix,
    lower_matrix: Matrix,
    end_matrix: Matrix,
    pole,
    rest_geometry=None,
) -> Vector:
    root = upper_matrix.translation
    joint = lower_matrix.translation
    end = end_matrix.translation
    line = end - root
    rest_root = rest_joint = rest_end = None
    if rest_geometry is not None:
        rest_root, rest_joint, rest_end = rest_geometry
    if line.length < 1.0e-8:
        if rest_joint is not None:
            return joint + (pole.bone.matrix_local.translation - rest_joint)
        return joint.copy()
    axis = line.normalized()
    bend = joint - (root + axis * (joint - root).dot(axis))
    expected_bend = None
    if rest_root is not None and rest_joint is not None and rest_end is not None:
        rest_line = rest_end - rest_root
        if rest_line.length > 1.0e-8:
            rest_axis = rest_line.normalized()
            rest_bend = rest_joint - (
                rest_root + rest_axis * (rest_joint - rest_root).dot(rest_axis)
            )
            if rest_bend.length < 1.0e-8:
                rest_reference = pole.bone.matrix_local.translation - rest_joint
                rest_bend = rest_reference - (rest_axis * rest_reference.dot(rest_axis))
            if rest_bend.length > 1.0e-8:
                if rest_axis.dot(axis) < -0.9999:
                    transport = Quaternion(rest_bend.normalized(), math.pi)
                else:
                    transport = rest_axis.rotation_difference(axis)
                expected_bend = transport @ rest_bend
    straight_threshold = max(line.length * 1.0e-4, 1.0e-6)
    if bend.length < straight_threshold and expected_bend is not None:
        bend = expected_bend
    elif (
        expected_bend is not None
        and bend.length > 1.0e-8
        and bend.dot(expected_bend) < 0.0
    ):
        # Elbows and knees do not acquire a reverse bend merely because a
        # noisy observation crosses the straight-line singularity.
        bend.negate()
    if bend.length < 1.0e-8:
        bend = axis.cross(Vector((0.0, 0.0, 1.0)))
        if bend.length < 1.0e-8:
            bend = axis.cross(Vector((0.0, 1.0, 0.0)))
    rest_distance = 0.0
    if rest_joint is not None:
        rest_distance = (pole.bone.matrix_local.translation - rest_joint).length
    distance = max(
        rest_distance,
        line.length * 0.5,
        0.01,
    )
    return joint + bend.normalized() * distance


def _ik_rest_geometry(upper, lower):
    return (
        upper.bone.head_local.copy(),
        lower.bone.head_local.copy(),
        lower.bone.tail_local.copy(),
    )


def _ik_rest_pole_distance(pole, rest_geometry, chain_length):
    _rest_root, rest_joint, _rest_end = rest_geometry
    return max(
        (pole.bone.matrix_local.translation - rest_joint).length,
        float(chain_length) * 0.5,
        0.01,
    )


def _ik_channel_direction(
    target_pose: TargetPose,
    channel,
    pose_matrix: Matrix,
) -> Vector:
    """Return the anatomical direction represented by a generated control."""
    if (
        channel.desired_direction is not None
        and channel.desired_direction.length > 1.0e-8
    ):
        return channel.desired_direction.normalized()
    rest_rotation = channel.bone.bone.matrix_local.to_quaternion()
    delta = pose_matrix.to_quaternion() @ rest_rotation.inverted()
    rest_direction = (
        channel.rest_direction
        if channel.rest_direction is not None
        else _target_body_direction(
            target_pose.armature,
            target_pose.entries,
            channel.semantic_id,
        )
    )
    return (delta @ rest_direction).normalized()


def _rest_relative_effector_rotation(
    rest_lower: Quaternion,
    desired_lower: Quaternion,
    rest_effector: Quaternion,
) -> Quaternion:
    """Carry an undetected effector using a deterministic rest relationship."""
    relative = rest_lower.normalized().inverted() @ rest_effector.normalized()
    return (desired_lower.normalized() @ relative).normalized()


def _prepare_ik_output(target_pose: TargetPose) -> list[dict]:
    """Convert processed virtual FK chains into profile IK proposals."""
    proposals = []
    ownership = limb_ownership(target_pose.armature)
    semantics = {
        ("ARM", "L"): ("UPPER_ARM_L", "LOWER_ARM_L", "HAND_L"),
        ("ARM", "R"): ("UPPER_ARM_R", "LOWER_ARM_R", "HAND_R"),
        ("LEG", "L"): ("UPPER_LEG_L", "LOWER_LEG_L", "FOOT_L"),
        ("LEG", "R"): ("UPPER_LEG_R", "LOWER_LEG_R", "FOOT_R"),
    }
    target_pose.resolve()
    for key, roles in semantics.items():
        info = ownership[key]
        if info["mode"] != "IK":
            continue
        target = info["target"]
        pole = info["pole"]
        if not target or not pole:
            raise ValueError(
                f"{key[1]} {key[0].lower()} is in IK mode but its "
                "target or pole mapping is missing"
            )
        channels = [target_pose.channel_for(role) for role in roles]
        if (
            key[0] == "ARM"
            and channels[0] is None
            and channels[1] is None
            and channels[2] is not None
        ):
            end_channel = channels[2]
            end_matrix = target_pose._virtual_pose(end_channel.bone).copy()
            rest_offset = (
                end_channel.bone.bone.matrix_local.inverted_safe()
                @ target.bone.matrix_local
            )
            target_matrix = end_matrix @ rest_offset
            target_matrix.translation = target.matrix.translation
            proposals.append(
                {
                    "key": key,
                    "armature": target_pose.armature,
                    "entries": target_pose.entries,
                    "roles": roles,
                    "target": target,
                    "pole": pole,
                    "target_matrix": target_matrix,
                    "orientation_only": True,
                    "confidence": float(end_channel.confidence),
                }
            )
            target_pose.bones.pop(end_channel.bone.name, None)
            continue
        if channels[0] is not None and channels[1] is not None and channels[2] is None:
            upper_matrix = target_pose._virtual_pose(channels[0].bone).copy()
            lower_matrix = target_pose._virtual_pose(channels[1].bone).copy()
            output_upper = evaluated_semantic_bone(
                target_pose.armature,
                target_pose.entries,
                roles[0],
            )
            output_lower = evaluated_semantic_bone(
                target_pose.armature,
                target_pose.entries,
                roles[1],
            )
            if output_upper is None or output_lower is None:
                continue
            upper_direction = _ik_channel_direction(
                target_pose, channels[0], upper_matrix
            )
            lower_direction = _ik_channel_direction(
                target_pose, channels[1], lower_matrix
            )
            desired_root = upper_matrix.translation.copy()
            desired_joint = desired_root + upper_direction * float(
                output_upper.bone.length
            )
            desired_end = desired_joint + lower_direction * float(
                output_lower.bone.length
            )
            # No reliable hand/foot frame was detected. Carry the rest-pose
            # effector/lower-limb relationship with the solved lower limb.
            # Using the current pose here made output depend on the starting
            # pose and compounded changes across repeated applications.
            desired_target_rotation = _rest_relative_effector_rotation(
                channels[1].bone.bone.matrix_local.to_quaternion(),
                lower_matrix.to_quaternion(),
                target.bone.matrix_local.to_quaternion(),
            )
            target_matrix = Matrix.LocRotScale(
                desired_end,
                desired_target_rotation,
                target.matrix.to_scale(),
            )
            upper_points = Matrix.Translation(desired_root)
            lower_points = Matrix.Translation(desired_joint)
            end_points = Matrix.Translation(desired_end)
            rest_geometry = _ik_rest_geometry(
                output_upper,
                output_lower,
            )
            pole_distance = _ik_rest_pole_distance(
                pole,
                rest_geometry,
                float(output_upper.bone.length) + float(output_lower.bone.length),
            )
            proposal = {
                "key": key,
                "armature": target_pose.armature,
                "entries": target_pose.entries,
                "roles": roles,
                "target": target,
                "pole": pole,
                "target_matrix": target_matrix,
                "pole_location": _ik_pole_location(
                    upper_points,
                    lower_points,
                    end_points,
                    pole,
                    rest_geometry,
                ),
                "pole_rest_geometry": rest_geometry,
                "pole_distance": pole_distance,
                "desired_joint": desired_joint,
                "desired_end": desired_end,
                "upper_direction": upper_direction,
                "lower_direction": lower_direction,
                "upper_length": float(output_upper.bone.length),
                "lower_length": float(output_lower.bone.length),
                "confidence": min(
                    float(channel.confidence)
                    for channel in channels[:2]
                    if channel is not None
                ),
            }
            rotation_bend = info.get("rotation_bend")
            if rotation_bend:
                proposal["rotation_bend"] = rotation_bend
            proposals.append(proposal)
            for channel in channels[:2]:
                target_pose.bones.pop(channel.bone.name, None)
            continue
        if any(channel is None for channel in channels):
            continue
        upper_matrix, lower_matrix, end_matrix = (
            target_pose._virtual_pose(channel.bone).copy() for channel in channels
        )
        output_upper = evaluated_semantic_bone(
            target_pose.armature,
            target_pose.entries,
            roles[0],
        )
        output_lower = evaluated_semantic_bone(
            target_pose.armature,
            target_pose.entries,
            roles[1],
        )
        if output_upper is None or output_lower is None:
            continue

        # Generated FK controls are not always a spatially continuous
        # skeleton (their mechanism parents and offsets can move control
        # origins).  Reconstruct the desired joint positions from the solved
        # absolute rotations and the evaluated deform-chain lengths instead
        # of treating FK control locations as provider joints.
        upper_direction = _ik_channel_direction(target_pose, channels[0], upper_matrix)
        lower_direction = _ik_channel_direction(target_pose, channels[1], lower_matrix)
        desired_root = upper_matrix.translation.copy()
        desired_joint = desired_root + upper_direction * float(output_upper.bone.length)
        desired_end = desired_joint + lower_direction * float(output_lower.bone.length)
        upper_points = upper_matrix.copy()
        lower_points = lower_matrix.copy()
        end_points = end_matrix.copy()
        upper_points.translation = desired_root
        lower_points.translation = desired_joint
        end_points.translation = desired_end
        end_bone = channels[-1].bone
        rest_offset = (
            end_bone.bone.matrix_local.inverted_safe() @ target.bone.matrix_local
        )
        target_matrix = end_matrix @ rest_offset
        target_matrix.translation += desired_end - end_matrix.translation
        rest_geometry = _ik_rest_geometry(
            output_upper,
            output_lower,
        )
        pole_distance = _ik_rest_pole_distance(
            pole,
            rest_geometry,
            float(output_upper.bone.length) + float(output_lower.bone.length),
        )
        pole_location = _ik_pole_location(
            upper_points,
            lower_points,
            end_points,
            pole,
            rest_geometry,
        )
        proposal = {
            "key": key,
            "armature": target_pose.armature,
            "entries": target_pose.entries,
            "roles": roles,
            "target": target,
            "pole": pole,
            "target_matrix": target_matrix,
            "pole_location": pole_location,
            "pole_rest_geometry": rest_geometry,
            "pole_distance": pole_distance,
            "desired_joint": desired_joint,
            "desired_end": desired_end,
            "upper_direction": upper_direction,
            "lower_direction": lower_direction,
            "upper_length": float(output_upper.bone.length),
            "lower_length": float(output_lower.bone.length),
            "confidence": min(float(channel.confidence) for channel in channels),
        }
        rotation_bend = info.get("rotation_bend")
        if rotation_bend:
            proposal["rotation_bend"] = rotation_bend
        if key[0] == "LEG" and info.get("toe"):
            toe_channel = target_pose.channel_for(f"TOE_{key[1]}")
            if toe_channel:
                toe_matrix = target_pose._virtual_pose(toe_channel.bone).copy()
                proposal["toe"] = info["toe"]
                proposal["toe_matrix"] = (
                    toe_matrix
                    @ toe_channel.bone.bone.matrix_local.inverted_safe()
                    @ info["toe"].bone.matrix_local
                )
                target_pose.bones.pop(toe_channel.bone.name, None)
        proposals.append(proposal)
        for channel in channels:
            target_pose.bones.pop(channel.bone.name, None)
    target_pose.invalidate()
    return proposals


def _set_pose_translation_preserve_basis_rotation(
    bone,
    target_translation: Vector,
    *,
    attempts: int = 3,
    solver: PoseTranslationSolver | None = None,
    scene_context: SceneContext,
) -> None:
    """Place a control in evaluated pose space while preserving rotation."""
    (solver or PoseTranslationSolver(scene_context.update)).solve(
        bone,
        target_translation,
        attempts=attempts,
    )


def _commit_ik_output(
    proposals: list[dict], *, scene_context: SceneContext
) -> set[object]:
    changed = set()
    translation_solver = PoseTranslationSolver(scene_context.update)
    for proposal in proposals:
        target = proposal["target"]
        if proposal.get("orientation_only"):
            oriented = proposal["target_matrix"].copy()
            oriented.translation = target.matrix.translation
            target.matrix = oriented
            scene_context.update()
            changed.add(target)
            continue
        rotation_bend = proposal.get("rotation_bend")
        root = evaluated_semantic_bone(
            proposal["armature"],
            proposal["entries"],
            proposal["roles"][0],
        )
        if root is not None:
            previous_end = proposal["desired_end"].copy()
            proposal["desired_joint"] = (
                root.head + proposal["upper_direction"] * proposal["upper_length"]
            )
            proposal["desired_end"] = (
                proposal["desired_joint"]
                + proposal["lower_direction"] * proposal["lower_length"]
            )
            proposal["target_matrix"].translation += (
                proposal["desired_end"] - previous_end
            )
            upper_points = Matrix.Translation(root.head)
            lower_points = Matrix.Translation(proposal["desired_joint"])
            end_points = Matrix.Translation(proposal["desired_end"])
            proposal["pole_location"] = _ik_pole_location(
                upper_points,
                lower_points,
                end_points,
                proposal["pole"],
                proposal.get("pole_rest_geometry"),
            )
        # IK ownership is position-first.  Preserve the control's current
        # orientation while placing the effector and solving its bend plane;
        # monocular palm/foot roll must not displace a wrist or ankle.
        position_matrix = target.matrix.copy()
        position_matrix.translation = proposal["target_matrix"].translation
        target.matrix = position_matrix
        scene_context.update()
        pole = proposal["pole"]
        if rotation_bend is None:
            _set_pose_translation_preserve_basis_rotation(
                pole,
                proposal["pole_location"],
                solver=translation_solver,
                scene_context=scene_context,
            )
        lower = evaluated_semantic_bone(
            proposal["armature"],
            proposal["entries"],
            proposal["roles"][1],
        )
        end_output = evaluated_semantic_bone(
            proposal["armature"],
            proposal["entries"],
            proposal["roles"][2],
        )
        if lower and end_output:
            # Generated controls can have Child Of and mechanism-parent
            # offsets. Correct the effector in evaluated pose space instead
            # of assuming that assigning the visible control matrix is exact.
            for _attempt in range(3):
                error = proposal["desired_end"] - end_output.head
                if error.length < 1.0e-5:
                    break
                corrected = target.matrix.copy()
                corrected.translation += error
                target.matrix = corrected
                scene_context.update()
            if root and rotation_bend is not None:
                # With pole_vector disabled, Rigify exposes only local Y on
                # upper_arm_ik/thigh_ik.  That channel is the roll of the bend
                # plane around the root-to-effector line; assigning an
                # arbitrary global matrix is invalid because X/Z are locked.
                for _attempt in range(8):
                    end_error = proposal["desired_end"] - end_output.head
                    if end_error.length >= 1.0e-5:
                        corrected = target.matrix.copy()
                        corrected.translation += end_error
                        target.matrix = corrected
                        scene_context.update()
                    axis = proposal["desired_end"] - root.head
                    if axis.length < 1.0e-8:
                        break
                    axis.normalize()
                    current_bend = root.tail - (
                        root.head + axis * (root.tail - root.head).dot(axis)
                    )
                    desired_bend = proposal["desired_joint"] - (
                        root.head
                        + axis * (proposal["desired_joint"] - root.head).dot(axis)
                    )
                    if current_bend.length < 1.0e-8 or desired_bend.length < 1.0e-8:
                        break
                    current_bend.normalize()
                    desired_bend.normalize()
                    angle = math.atan2(
                        axis.dot(current_bend.cross(desired_bend)),
                        current_bend.dot(desired_bend),
                    )
                    if abs(angle) < math.radians(0.05) and end_error.length < 1.0e-5:
                        break
                    if abs(angle) >= math.radians(0.05):
                        rotation = rotation_bend.rotation_euler.copy()
                        rotation.y += angle
                        rotation_bend.rotation_euler = rotation
                        scene_context.update()
            if root and rotation_bend is None:
                # Alternate absolute effector and pole placement using only
                # detected/rest geometry. Rotating the current pole offset
                # toward the desired bend made this an iterative feedback
                # solver: applying the same frame again produced a new pose.
                for _attempt in range(3):
                    upper_points = Matrix.Translation(root.head)
                    lower_points = Matrix.Translation(proposal["desired_joint"])
                    end_points = Matrix.Translation(proposal["desired_end"])
                    pole_location = _ik_pole_location(
                        upper_points,
                        lower_points,
                        end_points,
                        pole,
                        proposal.get("pole_rest_geometry"),
                    )
                    _set_pose_translation_preserve_basis_rotation(
                        pole,
                        pole_location,
                        solver=translation_solver,
                        scene_context=scene_context,
                    )
                    error = proposal["desired_end"] - end_output.head
                    if error.length < 1.0e-5:
                        break
                    corrected = target.matrix.copy()
                    corrected.translation += error
                    target.matrix = corrected
                    scene_context.update()
            # Apply the available virtual hand/foot orientation only
            # after positional IK has converged, then repair any target-pivot
            # offset introduced by that orientation (common on foot rigs).
            oriented = proposal["target_matrix"].copy()
            oriented.translation = target.matrix.translation
            target.matrix = oriented
            scene_context.update()
            # Auto-Rig Pro parents each pole to its IK effector. Rotating the
            # effector changes the pole constraint's local-to-world Jacobian.
            # Reusing the inverse built before that rotation can move the pole
            # to the opposite side of the limb, producing an apparent
            # 180-degree knee or elbow flip that depends on the starting pose.
            translation_solver.invalidate(pole)
            for _attempt in range(3):
                error = proposal["desired_end"] - end_output.head
                if error.length < 1.0e-5:
                    break
                corrected = target.matrix.copy()
                corrected.translation += error
                target.matrix = corrected
                scene_context.update()
            if root and rotation_bend is None:
                # Target orientation can shift controls whose pivot is not at
                # the deform joint. Restore the same absolute bend plane once
                # more, then repair the endpoint without using current state.
                pole_location = _ik_pole_location(
                    Matrix.Translation(root.head),
                    Matrix.Translation(proposal["desired_joint"]),
                    Matrix.Translation(proposal["desired_end"]),
                    pole,
                    proposal.get("pole_rest_geometry"),
                )
                _set_pose_translation_preserve_basis_rotation(
                    pole,
                    pole_location,
                    solver=translation_solver,
                    scene_context=scene_context,
                )
                for _attempt in range(2):
                    error = proposal["desired_end"] - end_output.head
                    if error.length < 1.0e-5:
                        break
                    corrected = target.matrix.copy()
                    corrected.translation += error
                    target.matrix = corrected
                    scene_context.update()
        changed.add(target)
        if rotation_bend is None:
            changed.add(pole)
        if rotation_bend is not None:
            changed.add(rotation_bend)
        toe = proposal.get("toe")
        if toe is not None:
            toe.matrix = proposal["toe_matrix"]
            scene_context.update()
            changed.add(toe)
    return changed
