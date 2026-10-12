"""Shoulders: clavicle elevation and arm-silhouette repair."""

from __future__ import annotations

import math
import re

from mathutils import Quaternion, Vector

from ..base import CorrectionModule, _smoothstep


class ShoulderCorrection(CorrectionModule):
    module_id = "SHOULDER_CORRECTION"
    display_name = "Shoulder Correction"
    # The optional silhouette repair deliberately has a tightly bounded
    # endpoint budget and reports its actual shift itself.
    allows_endpoint_motion = True

    @staticmethod
    def _side_from_name(name: str) -> str | None:
        lowered = name.lower()
        tokens = [token for token in re.split(r"[^a-z0-9]+", lowered) if token]
        if "left" in tokens or "l" in tokens:
            return "L"
        if "right" in tokens or "r" in tokens:
            return "R"
        return None

    @staticmethod
    def _image_elbow_angle(frame, side: str) -> float | None:
        prefix = "LEFT" if side == "L" else "RIGHT"
        points = [
            frame.body.get(f"{prefix}_{joint}")
            for joint in ("SHOULDER", "ELBOW", "WRIST")
        ]
        if any(point is None or not point.valid for point in points):
            return None
        shoulder, elbow, wrist = (Vector(point.image) for point in points)
        first = shoulder - elbow
        second = wrist - elbow
        if first.length < 1.0e-8 or second.length < 1.0e-8:
            return None
        angle = first.angle(second)
        return angle if math.isfinite(angle) else None

    def _repair_arm_silhouette(self, pose, settings) -> list[str]:
        """Bounded SAM 2D/3D reconciliation for visibly over-bent arms.

        A monocular image angle is not a complete 3D observation, so it is
        only allowed to extend a chain which SAM's articulated result bends
        substantially more.  The endpoint stays on its existing root-to-hand
        ray and can move by at most a small user-visible budget.
        """
        if not getattr(settings, "repair_arm_silhouette", True):
            return []
        limit_fraction = min(
            max(float(getattr(settings, "maximum_arm_extension", 0.02)), 0.0), 0.05
        )
        if limit_fraction <= 1.0e-7:
            return []
        if not any(
            point is not None and point.valid
            for point in (
                pose.frame.auxiliary.get("MHR_UPPER_ARM_L"),
                pose.frame.auxiliary.get("MHR_UPPER_ARM_R"),
            )
        ):
            return []
        warnings = []
        for side, label in (("L", "Left"), ("R", "Right")):
            # The full MHR observation is the gate that keeps this repair out
            # of MediaPipe's less reliable stylized-image path.
            mhr = pose.frame.auxiliary.get(f"MHR_UPPER_ARM_{side}")
            if mhr is None or not mhr.valid:
                continue
            image_angle = self._image_elbow_angle(pose.frame, side)
            upper = pose.channel_for(f"UPPER_ARM_{side}")
            lower = pose.channel_for(f"LOWER_ARM_{side}")
            if image_angle is None or upper is None or lower is None:
                continue
            pose.resolve()
            upper_pose = pose._virtual_pose(upper.bone)
            lower_pose = pose._virtual_pose(lower.bone)
            upper_length = max(float(upper.bone.bone.length), 1.0e-8)
            lower_length = max(float(lower.bone.bone.length), 1.0e-8)
            upper_direction = (
                upper_pose.to_3x3() @ Vector((0.0, 1.0, 0.0))
            ).normalized()
            lower_direction = (
                lower_pose.to_3x3() @ Vector((0.0, 1.0, 0.0))
            ).normalized()
            current_angle = (-upper_direction).angle(lower_direction)
            # Ignore small disagreement and cases where the image is more
            # bent.  The latter often represents real out-of-plane motion.
            disagreement = image_angle - current_angle
            if disagreement < math.radians(5.0):
                continue
            blend = min(max(0.60 * float(settings.strength), 0.0), 0.75)
            desired_angle = min(
                current_angle + disagreement * blend,
                math.radians(172.0),
            )
            desired_reach_sq = (
                upper_length * upper_length
                + lower_length * lower_length
                - 2.0 * upper_length * lower_length * math.cos(desired_angle)
            )
            desired_reach = math.sqrt(max(desired_reach_sq, 0.0))
            root = upper_pose.translation.copy()
            old_joint = root + upper_direction * upper_length
            old_end = old_joint + lower_direction * lower_length
            root_to_end = old_end - root
            current_reach = root_to_end.length
            total_length = upper_length + lower_length
            if current_reach < 1.0e-8:
                continue
            maximum_shift = total_length * limit_fraction
            reach = min(
                desired_reach,
                current_reach + maximum_shift,
                total_length * (1.0 - 1.0e-6),
            )
            shift = reach - current_reach
            if shift <= total_length * 1.0e-5:
                continue
            axis = root_to_end.normalized()
            along = (
                upper_length * upper_length
                - lower_length * lower_length
                + reach * reach
            ) / (2.0 * reach)
            height_sq = upper_length * upper_length - along * along
            height = math.sqrt(max(height_sq, 0.0))
            bend = old_joint - root
            bend -= axis * bend.dot(axis)
            if bend.length < 1.0e-8:
                reference = (
                    upper.desired_reference
                    if upper.desired_reference is not None
                    else upper.rest_reference
                )
                if reference is not None:
                    bend = reference - axis * reference.dot(axis)
            if bend.length < 1.0e-8:
                fallback = Vector((0.0, 0.0, 1.0))
                if abs(axis.dot(fallback)) > 0.95:
                    fallback = Vector((1.0, 0.0, 0.0))
                bend = fallback - axis * fallback.dot(axis)
            bend.normalize()
            end = root + axis * reach
            joint = root + axis * along + bend * height
            new_upper = (joint - root).normalized()
            new_lower = (end - joint).normalized()
            pose.aim(
                upper.semantic_id,
                upper.bone,
                new_upper,
                rest_direction=upper.rest_direction,
                desired_reference=upper.desired_reference,
                rest_reference=upper.rest_reference,
                correction=upper.correction,
                confidence=upper.confidence,
                influence=upper.influence,
                owner=self.module_id,
            )
            pose.aim(
                lower.semantic_id,
                lower.bone,
                new_lower,
                rest_direction=lower.rest_direction,
                desired_reference=new_upper,
                rest_reference=lower.rest_reference,
                correction=lower.correction,
                confidence=lower.confidence,
                influence=lower.influence,
                owner=self.module_id,
            )
            warnings.append(
                f"{label} arm silhouette repair moved the wrist/IK target "
                f"{shift / total_length * 100.0:.2f}% of arm length"
            )
            pose.resolve()
        return warnings

    def validate(self, armature, settings) -> list[str]:
        entries = {
            entry.semantic_id: entry for entry in armature.mtp_rig_config.mappings
        }
        warnings = []
        for side, label in (("L", "Left"), ("R", "Right")):
            upper = entries.get(f"UPPER_ARM_{side}")
            shoulder = entries.get(f"SHOULDER_{side}")
            if (
                upper
                and upper.target_name
                and (not shoulder or not shoulder.target_name)
            ):
                warnings.append(f"{label} shoulder correction has no mapped shoulder")
        return warnings

    def process_frame(self, pose, settings, session) -> list[str]:
        warnings = []
        rig_up = pose.target_basis @ Vector((0.0, 0.0, 1.0))
        auxiliary = pose.frame.auxiliary
        left_acromion = auxiliary.get("LEFT_ACROMION")
        right_acromion = auxiliary.get("RIGHT_ACROMION")
        acromion_mid = None
        if (
            left_acromion
            and right_acromion
            and left_acromion.valid
            and right_acromion.valid
        ):
            acromion_mid = (
                Vector(left_acromion.position) + Vector(right_acromion.position)
            ) * 0.5
        mhr_clavicles = {
            side: auxiliary.get(f"MHR_CLAVICLE_{side}") for side in ("L", "R")
        }
        mhr_upper_arms = {
            side: auxiliary.get(f"MHR_UPPER_ARM_{side}") for side in ("L", "R")
        }
        mhr_mid = None
        if all(point is not None and point.valid for point in mhr_clavicles.values()):
            mhr_mid = (
                Vector(mhr_clavicles["L"].position)
                + Vector(mhr_clavicles["R"].position)
            ) * 0.5
        for side, label in (("L", "Left"), ("R", "Right")):
            upper = pose.channel_for(f"UPPER_ARM_{side}")
            if not upper or upper.desired_direction is None:
                continue
            direction = upper.desired_direction.normalized()
            elevation = math.asin(
                min(max(direction.dot(rig_up.normalized()), -1.0), 1.0)
            )
            elevation = max(0.0, elevation)
            shoulder = pose.channel_for(f"SHOULDER_{side}")
            if shoulder is None:
                if elevation >= math.radians(60.0):
                    warnings.append(
                        f"{label} arm is elevated but its shoulder is unmapped"
                    )
                continue
            if shoulder.desired_direction is None:
                continue
            mhr_clavicle = mhr_clavicles.get(side)
            mhr_upper_arm = mhr_upper_arms.get(side)
            acromion = auxiliary.get(f"{'LEFT' if side == 'L' else 'RIGHT'}_ACROMION")
            # MHR exposes the actual articulated clavicle -> upper-arm joint
            # segment. The old midpoint -> clavicle vector represented only
            # the medial half of the shoulder girdle and omitted the outer
            # clavicle/acromion drop, displacing the arm root on stylized rigs.
            if (
                mhr_clavicle is not None
                and mhr_clavicle.valid
                and mhr_upper_arm is not None
                and mhr_upper_arm.valid
            ):
                observed = pose.to_target_direction(
                    Vector(mhr_upper_arm.position) - Vector(mhr_clavicle.position)
                )
                if observed.length > 1.0e-8:
                    shoulder.desired_direction = observed
                    shoulder.confidence = min(
                        shoulder.confidence,
                        mhr_clavicle.confidence,
                        mhr_upper_arm.confidence,
                    )
            elif mhr_mid is not None and mhr_clavicle is not None:
                observed = pose.to_target_direction(
                    Vector(mhr_clavicle.position) - mhr_mid
                )
                if observed.length > 1.0e-8:
                    shoulder.desired_direction = observed
                    shoulder.confidence = min(
                        shoulder.confidence, mhr_clavicle.confidence
                    )
            if (
                mhr_mid is None
                and acromion_mid is not None
                and acromion
                and acromion.valid
            ):
                observed = pose.to_target_direction(
                    Vector(acromion.position) - acromion_mid
                )
                if observed.length > 1.0e-8:
                    shoulder.desired_direction = observed
                    shoulder.confidence = min(
                        shoulder.confidence,
                        acromion.confidence,
                    )
            ramp = _smoothstep(
                settings.start_elevation,
                settings.full_influence_elevation,
                elevation,
            )
            contribution = 0.30 * settings.strength * ramp
            current = shoulder.desired_direction.normalized()
            difference = current.rotation_difference(direction)
            angle = min(
                difference.angle * contribution,
                settings.maximum_swing,
            )
            if angle > 1.0e-7:
                assisted = Quaternion(difference.axis, angle) @ current
                pose.aim(
                    shoulder.semantic_id,
                    shoulder.bone,
                    assisted,
                    rest_direction=shoulder.rest_direction,
                    desired_reference=shoulder.desired_reference,
                    rest_reference=shoulder.rest_reference,
                    correction=shoulder.correction,
                    confidence=shoulder.confidence,
                    influence=shoulder.influence,
                    owner=self.module_id,
                )

        warnings.extend(self._repair_arm_silhouette(pose, settings))
        return warnings
