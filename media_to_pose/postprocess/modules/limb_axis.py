"""Elbows, knees and limb twist: hinge axes and twist limits."""

from __future__ import annotations

import math

from mathutils import Quaternion, Vector

from ...target_pose import BoneChannel, TargetPose
from ...target_pose import orientation_frame as _orientation_frame
from ...target_pose import signed_twist_angle as _signed_twist_angle
from ..base import CorrectionModule, PostProcessingSession


class _AxisCorrection(CorrectionModule):
    """Shared axial-twist and hinge mechanics for one limb family."""

    _UPPER: tuple[str, ...] = ()
    _HINGES: tuple[str, ...] = ()
    _TWIST_LIMIT_FIELD = ""
    _USES_ARM_SURFACE = False
    _USES_PALM_ROLL = False

    @staticmethod
    def _hinge_axis(channel: BoneChannel) -> Vector | None:
        bone = channel.bone
        unlocked = [
            axis
            for axis, locked in zip(
                (
                    Vector((1.0, 0.0, 0.0)),
                    Vector((0.0, 1.0, 0.0)),
                    Vector((0.0, 0.0, 1.0)),
                ),
                bone.lock_rotation,
                strict=False,
            )
            if not locked
        ]
        if len(unlocked) == 1:
            return unlocked[0]
        if channel.rest_direction is None or channel.rest_reference is None:
            return None
        hinge_world = channel.rest_reference.cross(channel.rest_direction)
        if hinge_world.length < 1.0e-8:
            return None
        local = (
            bone.bone.matrix_local.to_3x3().inverted_safe() @ hinge_world.normalized()
        )
        components = (abs(local.x), abs(local.y), abs(local.z))
        index = components.index(max(components))
        axis = (
            Vector((1.0, 0.0, 0.0)),
            Vector((0.0, 1.0, 0.0)),
            Vector((0.0, 0.0, 1.0)),
        )[index]
        return axis if local[index] >= 0.0 else -axis

    def validate(self, armature, settings) -> list[str]:
        entries = {
            entry.semantic_id: entry for entry in armature.mtp_rig_config.mappings
        }
        warnings = []
        for semantic_id in self._HINGES:
            entry = entries.get(semantic_id)
            if not entry or not entry.target_name:
                warnings.append(f"{semantic_id} hinge target is missing")
                continue
            bone = armature.pose.bones.get(entry.target_name)
            if not bone:
                warnings.append(f"{semantic_id} hinge bone is missing")
        return warnings

    def _palm_forearm_twist(
        self,
        pose: TargetPose,
        side: str,
        settings,
        session: PostProcessingSession,
    ) -> tuple[float, float] | None:
        # Forearm roll is part of the body chain.  Keep it in the same
        # observation frame as the body-owned wrist/hand solve; the dedicated
        # hand detector is allowed to drive fingers only.  Mixing the body
        # hand axis with a separately cropped hand palm produced large,
        # opposing forearm/wrist rotations when those two detections disagreed.
        label = "LEFT" if side == "L" else "RIGHT"
        body = pose.frame.body
        forearm = pose.channel_for(f"LOWER_ARM_{side}")
        hand_channel = pose.channel_for(f"HAND_{side}")
        if (
            forearm is None
            or hand_channel is None
            or hand_channel.rest_reference is None
            or forearm.rest_direction is None
            or forearm.desired_direction is None
        ):
            return None
        index = body.get(f"{label}_INDEX")
        pinky = body.get(f"{label}_PINKY")
        if not index or not pinky or not index.valid or not pinky.valid:
            return None
        desired_lateral = pose.to_target_direction(
            Vector(index.position) - Vector(pinky.position)
        )
        # Palm width is the anatomical flexion axis of the wrist.  Project it
        # directly around the forearm instead of first "unbending" the hand
        # with a shortest-arc rotation.  The latter becomes ambiguous for a
        # strongly flexed wrist and used to reverse forearm roll by 180 degrees.
        rest_forearm_axis = forearm.rest_direction.normalized()
        desired_forearm_axis = forearm.desired_direction.normalized()
        rest_lateral = (
            hand_channel.rest_reference
            - rest_forearm_axis * hand_channel.rest_reference.dot(rest_forearm_axis)
        )
        desired_lateral = desired_lateral - desired_forearm_axis * desired_lateral.dot(
            desired_forearm_axis
        )
        if rest_lateral.length < 1.0e-8 or desired_lateral.length < 1.0e-8:
            return None
        transported = (
            rest_forearm_axis.rotation_difference(desired_forearm_axis)
            @ rest_lateral.normalized()
        )
        transported = (
            transported - desired_forearm_axis * transported.dot(desired_forearm_axis)
        ).normalized()
        desired_lateral.normalize()
        angle = math.atan2(
            desired_forearm_axis.dot(transported.cross(desired_lateral)),
            min(max(transported.dot(desired_lateral), -1.0), 1.0),
        )
        # atan2 wraps at +/-pi even though a video may cross that boundary
        # smoothly. Preserve the phase before applying the anatomical limit;
        # otherwise -181 degrees becomes +179 and the two clamped poses differ
        # by 180 degrees.
        angle = session.unwrap_forearm_twist(side, angle)
        # Confidence is deliberately not an enable/disable gate. A hard gate
        # made forearm roll switch between corrected and uncorrected branches
        # as hand confidence crossed one threshold, which was visible as wrist
        # jitter. The sequence pass uses this value as an observation weight
        # and bridges weak intervals from neighboring reliable rotations.
        confidence = min(
            float(index.confidence),
            float(pinky.confidence),
            float(hand_channel.confidence),
        )
        correction = angle * settings.palm_roll_strength
        return (
            min(
                max(correction, -settings.forearm_twist_limit),
                settings.forearm_twist_limit,
            ),
            min(max(confidence, 0.0), 1.0),
        )

    def _apply_axial_roll(
        self,
        channel: BoneChannel,
        angle: float,
        confidence: float,
    ) -> None:
        if channel.rest_direction is None or channel.desired_direction is None:
            return
        rest_axis = channel.rest_direction.normalized()
        desired_axis = channel.desired_direction.normalized()
        swing = rest_axis.rotation_difference(desired_axis)
        delta = Quaternion(desired_axis, angle) @ swing
        if channel.influence < 1.0:
            delta = Quaternion().slerp(delta, channel.influence)
        rest_rotation = channel.bone.bone.matrix_local.to_3x3()
        channel.desired_rotation = delta.to_matrix() @ rest_rotation
        channel.confidence = min(
            float(channel.confidence), min(max(float(confidence), 0.0), 1.0)
        )
        channel.owner = self.module_id

    def process_frame(self, pose, settings, session) -> list[str]:
        warnings = []
        # MHR70 observes both the posterior olecranon and anterior cubital
        # fossa. Their vector is much stronger axial-roll evidence than a
        # nearly straight shoulder-elbow-wrist plane.
        if self._USES_ARM_SURFACE:
            for side, label in (("L", "LEFT"), ("R", "RIGHT")):
                channel = pose.channel_for(f"UPPER_ARM_{side}")
                olecranon = pose.frame.auxiliary.get(f"{label}_OLECRANON")
                cubital = pose.frame.auxiliary.get(f"{label}_CUBITAL_FOSSA")
                if (
                    channel is not None
                    and olecranon
                    and cubital
                    and olecranon.valid
                    and cubital.valid
                ):
                    reference = pose.to_target_direction(
                        Vector(cubital.position) - Vector(olecranon.position)
                    )
                    if reference.length > 1.0e-8:
                        # Re-solve the rotation immediately. Updating only the
                        # metadata left the base elbow-plane rotation untouched,
                        # so SAM's stronger surface observation had no effect.
                        pose.aim(
                            channel.semantic_id,
                            channel.bone,
                            channel.desired_direction,
                            rest_direction=channel.rest_direction,
                            desired_reference=reference,
                            rest_reference=channel.rest_reference,
                            correction=channel.correction,
                            confidence=min(
                                channel.confidence,
                                olecranon.confidence,
                                cubital.confidence,
                            ),
                            influence=channel.influence,
                            owner=self.module_id,
                        )
        for semantic_id in self._UPPER:
            channel = pose.channel_for(semantic_id)
            if (
                channel is None
                or channel.desired_rotation is None
                or channel.rest_direction is None
                or channel.desired_direction is None
            ):
                continue
            rest_rotation = channel.bone.bone.matrix_local.to_3x3()
            full_delta = (
                (channel.desired_rotation @ rest_rotation.inverted_safe())
                .to_quaternion()
                .normalized()
            )
            swing = channel.rest_direction.rotation_difference(
                channel.desired_direction
            )
            residual_rotation = swing.inverted() @ full_delta
            twist = _signed_twist_angle(residual_rotation, channel.rest_direction)
            reliable = (
                channel.confidence >= settings.confidence_threshold
                and channel.desired_reference is not None
                and _orientation_frame(
                    channel.desired_direction,
                    channel.desired_reference,
                )
                is not None
            )
            if not reliable:
                continue
            limit = getattr(settings, self._TWIST_LIMIT_FIELD)
            twist = min(max(twist, -limit), limit)
            stabilized = swing @ Quaternion(channel.rest_direction.normalized(), twist)
            if channel.influence < 1.0:
                stabilized = Quaternion().slerp(stabilized, channel.influence)
            channel.desired_rotation = stabilized.to_matrix() @ rest_rotation
            channel.owner = self.module_id

        # Body-pose palm landmarks contain pronation/supination evidence that
        # the elbow-wrist line does not. Apply that roll around the mapped
        # forearm's own axis. Dedicated hand landmarks remain finger-only.
        rolled_forearms = set()
        if self._USES_PALM_ROLL:
            for side in ("L", "R"):
                observation = self._palm_forearm_twist(pose, side, settings, session)
                channel = pose.channel_for(f"LOWER_ARM_{side}")
                if observation is not None and channel is not None:
                    angle, confidence = observation
                    self._apply_axial_roll(channel, angle, confidence)
                    rolled_forearms.add(channel.semantic_id)

        if settings.hinge_projection:
            for semantic_id in self._HINGES:
                channel = pose.channel_for(semantic_id)
                if channel is None:
                    continue
                if semantic_id in rolled_forearms:
                    # Projecting the basis to a single hinge axis would erase
                    # the independently observed forearm pronation.
                    continue
                pose.replace_with_swing_only(channel, owner=self.module_id)
                axis = self._hinge_axis(channel)
                if axis is None:
                    warnings.append(
                        f"Could not infer hinge axis for {channel.bone.name}"
                    )
                    continue
                if getattr(pose, "diagnostic_mode", False):
                    direction_error = pose.hinge_projection_direction_error(
                        channel, axis
                    )
                    if direction_error > math.radians(0.5):
                        warnings.append(
                            f"Applied hinge projection for {channel.bone.name} "
                            f"despite {math.degrees(direction_error):.1f} degrees "
                            "of endpoint-direction change because the correction is enabled"
                        )
                channel.hinge_axis_local = axis
        pose.invalidate()
        return warnings


class ArmAxisCorrection(_AxisCorrection):
    module_id = "ARM_AXIS_CORRECTION"
    display_name = "Arm Axis Correction"
    _UPPER = ("UPPER_ARM_L", "UPPER_ARM_R")
    _HINGES = ("LOWER_ARM_L", "LOWER_ARM_R")
    _TWIST_LIMIT_FIELD = "upper_arm_twist_limit"
    _USES_ARM_SURFACE = True
    _USES_PALM_ROLL = True


class LegAxisCorrection(_AxisCorrection):
    module_id = "LEG_AXIS_CORRECTION"
    display_name = "Leg Axis Correction"
    _UPPER = ("UPPER_LEG_L", "UPPER_LEG_R")
    _HINGES = ("LOWER_LEG_L", "LOWER_LEG_R")
    _TWIST_LIMIT_FIELD = "thigh_twist_limit"
