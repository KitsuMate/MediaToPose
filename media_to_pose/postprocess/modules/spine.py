"""Torso Curve: bend the spine chain along the detected torso curve."""

from __future__ import annotations

import math

from mathutils import Matrix, Quaternion, Vector

from ...retarget.target_pose import TargetPose
from ...retarget.target_pose import signed_twist_angle as _signed_twist_angle
from ...retarget.target_pose import (
    stable_orientation_delta as _stable_orientation_delta,
)
from ..base import CorrectionModule, _arc_point, _curve_sample, _hermite_sample


class SpineCorrection(CorrectionModule):
    module_id = "SPINE_CORRECTION"
    display_name = "Spine Correction"
    allows_endpoint_motion = True
    _SEMANTICS = ("SPINE_01", "SPINE_02", "CHEST")
    _LIMBS = (
        ("UPPER_ARM_L", "LOWER_ARM_L", "HAND_L"),
        ("UPPER_ARM_R", "LOWER_ARM_R", "HAND_R"),
        ("UPPER_LEG_L", "LOWER_LEG_L", "FOOT_L"),
        ("UPPER_LEG_R", "LOWER_LEG_R", "FOOT_R"),
    )

    def validate(self, armature, settings) -> list[str]:
        entries = {
            entry.semantic_id: entry for entry in armature.mtp_rig_config.mappings
        }
        names = []
        for semantic_id in self._SEMANTICS:
            entry = entries.get(semantic_id)
            if entry and entry.target_name and entry.target_name not in names:
                names.append(entry.target_name)
        return (
            ["Spine Correction has fewer than two distinct torso controls"]
            if len(names) < 2
            else []
        )

    @staticmethod
    def _valid_position(frame, name):
        point = frame.auxiliary.get(name)
        return Vector(point.position) if point and point.valid else None

    def _source_curve(self, pose: TargetPose):
        frame = pose.frame
        mhr = [self._valid_position(frame, f"MHR_SPINE_{index}") for index in range(4)]
        neck = self._valid_position(frame, "MHR_NECK")
        if all(point is not None for point in mhr) and neck is not None:
            return _curve_sample([*mhr, neck]), False
        body = frame.body
        hips = [body.get("LEFT_HIP"), body.get("RIGHT_HIP")]
        shoulders = [body.get("LEFT_SHOULDER"), body.get("RIGHT_SHOULDER")]
        if not all(point and point.valid for point in hips + shoulders):
            return None, True
        hip = (Vector(hips[0].position) + Vector(hips[1].position)) * 0.5
        shoulder = (Vector(shoulders[0].position) + Vector(shoulders[1].position)) * 0.5
        main = shoulder - hip
        if main.length < 1.0e-8:
            return None, True
        candidates = [
            body.get(name)
            for name in ("LEFT_EAR", "RIGHT_EAR", "LEFT_EYE", "RIGHT_EYE")
        ]
        valid_head = [item for item in candidates if item and item.valid]
        upper = main.normalized()
        if valid_head:
            head = sum((Vector(item.position) for item in valid_head), Vector()) / len(
                valid_head
            )
            observed = head - shoulder
            if observed.length > 1.0e-8:
                observed.normalize()
                angle = main.normalized().angle(observed)
                if angle > math.radians(45.0):
                    observed = main.normalized().slerp(
                        observed, math.radians(45.0) / angle
                    )
                upper = observed
        return _hermite_sample(hip, shoulder, main, upper * main.length * 0.65), True

    def process_frame(self, pose, settings, session) -> list[str]:
        samples, inferred = self._source_curve(pose)
        if not samples:
            return ["Spine Correction lacks usable torso observations"]
        # Sparse inferred curves are a conservative correction layered over
        # the original torso, so retain their pre-correction limb endpoints.
        # SAM's observed MHR curve replaces the original torso geometry: its
        # old endpoints were computed from the wrong shoulder/hip positions
        # and anchoring to them can make an otherwise valid limb unreachable.
        # SAM limb channels already contain provider-derived world directions;
        # resolving those from the corrected roots preserves their shape.
        pose.resolve()
        limb_anchors = []
        for roles in self._LIMBS if inferred else ():
            limb = [pose.channel_for(role) for role in roles]
            if any(channel is None for channel in limb):
                continue
            upper, lower, end = limb
            root = pose._virtual_pose(upper.bone).translation.copy()
            upper_direction = upper.desired_direction
            lower_direction = lower.desired_direction
            if (
                upper_direction is None
                or lower_direction is None
                or upper_direction.length < 1.0e-8
                or lower_direction.length < 1.0e-8
            ):
                continue
            first_length = max(float(upper.bone.length), 1.0e-6)
            second_length = max(float(lower.bone.length), 1.0e-6)
            joint = root + upper_direction.normalized() * first_length
            endpoint = joint + lower_direction.normalized() * second_length
            plane = (joint - root).cross(endpoint - joint)
            limb_anchors.append(
                (
                    roles,
                    limb,
                    endpoint,
                    joint,
                    first_length,
                    second_length,
                    plane,
                )
            )

        channels, seen = [], set()
        for semantic_id in self._SEMANTICS:
            channel = pose.channel_for(semantic_id)
            if channel and channel.bone.name not in seen:
                seen.add(channel.bone.name)
                channels.append(channel)
        warnings = []
        if inferred:
            warnings.append(
                "MediaPipe spine curvature is inferred from torso landmarks"
            )
        if len(channels) < 2:
            warnings.append(
                "Spine Correction has fewer than two distinct torso controls"
            )
        if not channels:
            return warnings

        # Preserve SAM's observed MHR centerline in target space. Re-aligning
        # this curve to a separately estimated 2D torso chord changed every
        # evaluated segment by the same global offset, despite the target pose
        # itself containing accurate 3D tangents.
        sample_origin = samples[0]
        target_samples = []
        for sample in samples:
            relative = sample - sample_origin
            if relative.length <= 1.0e-8:
                target_samples.append(Vector())
            else:
                target_samples.append(
                    pose.to_target_direction(relative) * relative.length
                )
        lengths = [max(channel.bone.length, 1.0e-6) for channel in channels]
        total, cursor = sum(lengths), 0.0
        curve_axis = target_samples[-1] - target_samples[0]
        if curve_axis.length > 1.0e-8:
            curve_axis.normalize()
        # Move the target torso with one common rest-to-observed rotation
        # before adding per-segment curvature. Independent absolute swing
        # solves can preserve the same centerline while giving Chest and
        # UpperChest different global deltas, which shears vertices weighted
        # across their skinning boundary.
        rigid_delta = Quaternion()
        rest_chain_axis = Vector((0.0, 0.0, 0.0))
        for channel, length in zip(channels, lengths, strict=False):
            if channel.rest_direction.length > 1.0e-8:
                rest_chain_axis += channel.rest_direction.normalized() * length
        if rest_chain_axis.length > 1.0e-8 and curve_axis.length > 1.0e-8:
            rigid_delta = rest_chain_axis.normalized().rotation_difference(curve_axis)
        body = pose.frame.body
        hip_lateral = shoulder_lateral = None
        if all(
            body.get(name) and body[name].valid for name in ("LEFT_HIP", "RIGHT_HIP")
        ):
            hip_lateral = Vector(body["RIGHT_HIP"].position) - Vector(
                body["LEFT_HIP"].position
            )
        if all(
            body.get(name) and body[name].valid
            for name in ("LEFT_SHOULDER", "RIGHT_SHOULDER")
        ):
            shoulder_lateral = Vector(body["RIGHT_SHOULDER"].position) - Vector(
                body["LEFT_SHOULDER"].position
            )
        for channel, length in zip(channels, lengths, strict=False):
            start_fraction = cursor / total
            cursor += length
            end_fraction = cursor / total
            direction = _arc_point(target_samples, end_fraction) - _arc_point(
                target_samples, start_fraction
            )
            if direction.length > 1.0e-8:
                direction.normalize()
            reference = None
            if hip_lateral is not None and shoulder_lateral is not None:
                reference = pose.to_target_direction(
                    hip_lateral.lerp(
                        shoulder_lateral,
                        (start_fraction + end_fraction) * 0.5,
                    )
                )
            # MediaPipe's independently reconstructed hip lateral can reverse
            # relative to the shoulder lateral in foreshortened, crossed-leg,
            # or stylized poses. A lateral axis has two valid signs for swing,
            # but choosing the opposite sign asks the spine for an otherwise
            # invisible 180-degree axial roll. The raw torso solve already has
            # a consistently labelled shoulder reference, so use it only to
            # choose the hemisphere of the inferred reference.
            baseline_reference = getattr(channel, "desired_reference", None)
            if reference is not None and baseline_reference is not None:
                axis = direction.normalized()
                candidate = reference - axis * reference.dot(axis)
                baseline = baseline_reference - axis * baseline_reference.dot(axis)
                if (
                    candidate.length > 1.0e-8
                    and baseline.length > 1.0e-8
                    and candidate.dot(baseline) < 0.0
                ):
                    reference.negate()
            # Positions own swing and lateral roll. The rest-relative MHR
            # frame contributes only its component around the fitted tangent,
            # avoiding direct transfer of MHR joint pre-rotations or axes.
            joint_index = min(
                max(int(round(((start_fraction + end_fraction) * 0.5) * 3.0)), 0),
                3,
            )
            observation = pose.frame.joint_frames.get(f"MHR_SPINE_{joint_index}")
            if reference is not None and observation and observation.valid:
                values = observation.rotation_matrix
                raw_delta = Matrix(
                    tuple(
                        tuple(float(values[row * 3 + column]) for column in range(3))
                        for row in range(3)
                    )
                )
                canonical = Matrix(
                    (
                        (1.0, 0.0, 0.0),
                        (0.0, 0.0, 1.0),
                        (0.0, -1.0, 0.0),
                    )
                )
                canonical_delta = canonical @ raw_delta @ canonical.inverted_safe()
                relative_delta = (
                    pose.source_basis.inverted_safe()
                    @ canonical_delta
                    @ pose.source_basis
                )
                target_delta = (
                    (
                        pose.target_basis
                        @ relative_delta
                        @ pose.target_basis.inverted_safe()
                    )
                    .to_quaternion()
                    .normalized()
                )
                twist = _signed_twist_angle(target_delta, direction.normalized())
                twist = min(max(twist, -math.radians(60.0)), math.radians(60.0))
                reference = (
                    Quaternion(direction.normalized(), twist * settings.strength)
                    @ reference
                )
            previous_rotation = (
                channel.desired_rotation.copy()
                if getattr(channel, "desired_rotation", None) is not None
                else None
            )
            rest_rotation = channel.bone.bone.matrix_local.to_3x3()
            transported_direction = rigid_delta @ channel.rest_direction
            transported_reference = (
                rigid_delta @ channel.rest_reference
                if channel.rest_reference is not None
                else None
            )
            local_delta = _stable_orientation_delta(
                transported_direction,
                direction,
                transported_reference,
                reference,
                reference_is_oriented=not inferred,
            )
            applied = local_delta is not None
            if applied:
                desired_rotation = (
                    local_delta.to_matrix() @ rigid_delta.to_matrix() @ rest_rotation
                )
                pose.set_desired_rotation(
                    channel.semantic_id,
                    channel.bone,
                    desired_rotation,
                    correction=channel.correction,
                    confidence=channel.confidence,
                    influence=min(max(channel.influence * settings.strength, 0.0), 1.0),
                    owner=self.module_id,
                )
                channel = pose.channel_for(channel.semantic_id)
                # Keep the original rest observations as invariants for
                # downstream endpoint reconstruction. Applying the transport
                # to these fields would make later correctors count it twice.
                channel.desired_direction = direction.copy()
                channel.desired_reference = (
                    reference.copy() if reference is not None else None
                )
            else:
                applied = pose.aim(
                    channel.semantic_id,
                    channel.bone,
                    direction,
                    rest_direction=channel.rest_direction,
                    desired_reference=reference,
                    rest_reference=channel.rest_reference,
                    correction=channel.correction,
                    confidence=channel.confidence,
                    influence=min(max(channel.influence * settings.strength, 0.0), 1.0),
                    owner=self.module_id,
                    # MHR supplies an observed axial frame. MediaPipe's inferred
                    # hip/shoulder lateral is only a bend-plane axis and must use
                    # the same closest-hemisphere policy as the base torso solve.
                    reference_is_oriented=not inferred,
                )
            # Sparse MediaPipe landmarks do not observe vertebrae. Even after
            # hemisphere correction, cap the *additional* inferred correction
            # rather than allowing an ambiguous reference to overturn a valid
            # raw torso solve. SAM's observed MHR centerline is not subject to
            # this conservative inference cap.
            maximum_correction = math.radians(45.0) if inferred else None
            if (
                applied
                and maximum_correction is not None
                and previous_rotation is not None
                and getattr(channel, "desired_rotation", None) is not None
            ):
                previous_q = previous_rotation.to_quaternion().normalized()
                proposed_q = channel.desired_rotation.to_quaternion().normalized()
                if previous_q.dot(proposed_q) < 0.0:
                    proposed_q.negate()
                correction_angle = previous_q.rotation_difference(proposed_q).angle
                if correction_angle > maximum_correction:
                    channel.desired_rotation = previous_q.slerp(
                        proposed_q,
                        maximum_correction / correction_angle,
                    ).to_matrix()
                    if inferred:
                        warnings.append(
                            f"{channel.semantic_id.replace('_', ' ').title()} "
                            "inferred bend was clamped to 45 degrees"
                        )
        pose.invalidate()
        pose.resolve()
        for (
            roles,
            limb,
            endpoint,
            old_joint,
            first_length,
            second_length,
            old_plane,
        ) in limb_anchors:
            upper, lower, _end = limb
            root = pose._virtual_pose(upper.bone).translation.copy()
            to_end = endpoint - root
            distance = to_end.length
            if distance < 1.0e-8:
                continue
            maximum = first_length + second_length - 1.0e-5
            minimum = abs(first_length - second_length) + 1.0e-5
            clamped_distance = min(max(distance, minimum), maximum)
            target = root + to_end.normalized() * clamped_distance
            if abs(clamped_distance - distance) > 1.0e-5:
                warnings.append(
                    f"{roles[0]} endpoint became unreachable after spine correction; correction was clamped"
                )
            axis = (target - root).normalized()
            along = (
                clamped_distance * clamped_distance
                + first_length * first_length
                - second_length * second_length
            ) / (2.0 * clamped_distance)
            height = math.sqrt(max(first_length * first_length - along * along, 0.0))
            projected = old_joint - (root + axis * (old_joint - root).dot(axis))
            if projected.length < 1.0e-7 and old_plane.length > 1.0e-7:
                projected = old_plane.normalized().cross(axis)
            if projected.length < 1.0e-7:
                projected = axis.orthogonal()
            bend = projected.normalized()
            joint = root + axis * along + bend * height
            upper_direction = joint - root
            lower_direction = target - joint
            pose.aim(
                upper.semantic_id,
                upper.bone,
                upper_direction,
                rest_direction=upper.rest_direction,
                desired_reference=lower_direction,
                rest_reference=upper.rest_reference,
                correction=upper.correction,
                confidence=upper.confidence,
                influence=upper.influence,
                owner=self.module_id,
            )
            pose.aim(
                lower.semantic_id,
                lower.bone,
                lower_direction,
                rest_direction=lower.rest_direction,
                desired_reference=upper_direction,
                rest_reference=lower.rest_reference,
                correction=lower.correction,
                confidence=lower.confidence,
                influence=lower.influence,
                owner=self.module_id,
            )
        return warnings
