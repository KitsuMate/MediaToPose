"""Provider-neutral temporary target-pose channels and Blender commit logic.

The retargeter builds this model, optional correction modules amend it, and
only :meth:`TargetPose.commit` writes pose or shape-key data.  Keeping the
model outside the post-processing package prevents the core retargeter from
depending on any particular correction implementation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from mathutils import Euler, Matrix, Quaternion, Vector


def _finite_vector(value: Vector) -> bool:
    return all(math.isfinite(float(component)) for component in value)


def orientation_frame(primary: Vector, reference: Vector | None) -> Matrix | None:
    if primary.length < 1.0e-8:
        return None
    y_axis = primary.normalized()
    if reference is None or reference.length < 1.0e-8:
        return None
    z_axis = reference - y_axis * reference.dot(y_axis)
    if z_axis.length < max(1.0e-6, reference.length * 0.12):
        return None
    z_axis.normalize()
    x_axis = y_axis.cross(z_axis)
    if x_axis.length < 1.0e-8:
        return None
    x_axis.normalize()
    z_axis = x_axis.cross(y_axis).normalized()
    return Matrix((x_axis, y_axis, z_axis)).transposed()


def stable_orientation_delta(
    rest_primary: Vector,
    desired_primary: Vector,
    rest_reference: Vector | None,
    desired_reference: Vector | None,
    *,
    reference_is_oriented: bool = False,
) -> Quaternion | None:
    """Solve swing plus a continuous, hemisphere-safe reference roll.

    A secondary reference becomes unreliable as it approaches the primary
    axis.  Treat its projected direction as an unoriented bend-plane axis,
    align it to parallel-transported rest roll, and fade its contribution in
    smoothly.  This prevents the old threshold from selecting a 180-degree
    axial branch for nearly straight limbs.
    """
    if (
        rest_primary.length < 1.0e-8
        or desired_primary.length < 1.0e-8
        or not _finite_vector(rest_primary)
        or not _finite_vector(desired_primary)
    ):
        return None
    rest_axis = rest_primary.normalized()
    desired_axis = desired_primary.normalized()
    swing = rest_axis.rotation_difference(desired_axis)
    if (
        rest_reference is None
        or desired_reference is None
        or rest_reference.length < 1.0e-8
        or desired_reference.length < 1.0e-8
        or not _finite_vector(rest_reference)
        or not _finite_vector(desired_reference)
    ):
        return swing
    rest_projected = rest_reference - rest_axis * rest_reference.dot(rest_axis)
    desired_projected = desired_reference - desired_axis * desired_reference.dot(
        desired_axis
    )
    rest_ratio = rest_projected.length / rest_reference.length
    desired_ratio = desired_projected.length / desired_reference.length
    if rest_projected.length < 1.0e-8 or desired_projected.length < 1.0e-8:
        return swing
    transported = swing @ rest_projected.normalized()
    observed = desired_projected.normalized()
    # Bend-plane references (elbow/knee planes) do not carry a trustworthy
    # arrow, so use their closest hemisphere. Anatomically ordered references
    # such as index-minus-pinky *are* oriented and must retain their sign;
    # discarding it turns a palm-over/palm-under distinction into a 180-degree
    # wrist and finger-roll error.
    if not reference_is_oriented and transported.dot(observed) < 0.0:
        observed.negate()
    roll = math.atan2(
        desired_axis.dot(transported.cross(observed)),
        min(max(transported.dot(observed), -1.0), 1.0),
    )
    reliability = min(rest_ratio, desired_ratio)
    reliability = min(max((reliability - 0.06) / 0.14, 0.0), 1.0)
    reliability = reliability * reliability * (3.0 - 2.0 * reliability)
    return (Quaternion(desired_axis, roll * reliability) @ swing).normalized()


def correction_quaternion(values: Sequence[float]) -> Quaternion:
    return Euler(tuple(float(value) for value in values), "XYZ").to_quaternion()


def respect_rotation_locks(bone, proposed: Quaternion) -> Quaternion:
    """Preserve locked local XYZ components from the pre-capture basis."""
    if not any(bone.lock_rotation):
        return proposed.normalized()
    current = bone.matrix_basis.to_quaternion().to_euler("XYZ")
    desired = proposed.normalized().to_euler("XYZ", current)
    for index, locked in enumerate(bone.lock_rotation):
        if locked:
            desired[index] = current[index]
    return desired.to_quaternion().normalized()


def project_twist(rotation: Quaternion, axis: Vector) -> Quaternion:
    axis = axis.normalized()
    vector = Vector((rotation.x, rotation.y, rotation.z))
    projected = axis * vector.dot(axis)
    result = Quaternion((rotation.w, projected.x, projected.y, projected.z))
    if result.magnitude < 1.0e-8:
        return Quaternion()
    result.normalize()
    return result


def signed_twist_angle(rotation: Quaternion, axis: Vector) -> float:
    twist = project_twist(rotation, axis)
    angle = twist.angle
    if angle > math.pi:
        angle -= math.tau
    if Vector(twist.axis).dot(axis) < 0.0:
        angle = -angle
    return angle


@dataclass
class BoneChannel:
    semantic_id: str
    bone: object
    desired_rotation: Matrix | None = None
    direct_basis_rotation: Quaternion | None = None
    location: Vector | None = None
    correction: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rest_direction: Vector | None = None
    desired_direction: Vector | None = None
    rest_reference: Vector | None = None
    desired_reference: Vector | None = None
    confidence: float = 1.0
    influence: float = 1.0
    hinge_axis_local: Vector | None = None
    owner: str = "RETARGET"
    properties: dict[str, float] = field(default_factory=dict)
    commit_priority: int = 100


@dataclass
class ShapeChannel:
    obj: object
    block: object
    value: float
    owner: str = "RETARGET"


class TargetPose:
    """A target-aware pose that can be amended before one Blender commit."""

    def __init__(
        self,
        armature,
        frame,
        entries,
        source_basis: Matrix,
        target_basis: Matrix,
        *,
        rig_plan=None,
    ):
        self.armature = armature
        self.frame = frame
        self.entries = entries
        self.source_basis = source_basis.copy()
        self.target_basis = target_basis.copy()
        self.rig_plan = rig_plan
        self.bones: dict[str, BoneChannel] = {}
        self.semantic_bones: dict[str, str] = {}
        self.shapes: dict[tuple[int, str], ShapeChannel] = {}
        self.warnings: list[str] = []
        self.applied_modules: list[str] = []
        self._resolved_basis: dict[str, Matrix] = {}
        self._resolved_pose: dict[str, Matrix] = {}
        self._revision = 0
        self._resolved_revision = -1
        self.diagnostic_mode = False
        self.commit_dependency_updates = 0

    @property
    def revision(self) -> int:
        return self._revision

    def invalidate(self) -> None:
        """Record one pose mutation and invalidate all derived transforms."""
        self._revision += 1
        self._resolved_basis.clear()
        self._resolved_pose.clear()

    @staticmethod
    def canonical_vector(value: Sequence[float]) -> Vector:
        return Vector((float(value[0]), float(value[2]), -float(value[1])))

    def to_target_direction(self, raw_direction: Vector) -> Vector:
        canonical = self.canonical_vector(raw_direction)
        relative = self.source_basis.inverted_safe() @ canonical
        result = self.target_basis @ relative
        return result.normalized() if result.length > 1.0e-8 else result

    def channel_for(self, semantic_id: str) -> BoneChannel | None:
        name = self.semantic_bones.get(semantic_id)
        return self.bones.get(name) if name else None

    def _channel(self, semantic_id: str, bone) -> BoneChannel:
        channel = self.bones.get(bone.name)
        if channel is None:
            channel = BoneChannel(semantic_id=semantic_id, bone=bone)
            self.bones[bone.name] = channel
        if semantic_id:
            channel.semantic_id = semantic_id
            self.semantic_bones[semantic_id] = bone.name
            if semantic_id.startswith(("SPINE", "CHEST", "NECK", "HEAD")):
                channel.commit_priority = 10
            elif semantic_id.startswith("SHOULDER"):
                channel.commit_priority = 20
            elif semantic_id.startswith(("UPPER_ARM", "UPPER_LEG")):
                channel.commit_priority = 30
            elif semantic_id.startswith(("LOWER_ARM", "LOWER_LEG")):
                channel.commit_priority = 40
            elif semantic_id.startswith(("HAND", "FOOT")):
                channel.commit_priority = 50
            elif semantic_id.startswith("TOE") or any(
                semantic_id.startswith(finger)
                for finger in (
                    "THUMB",
                    "INDEX",
                    "MIDDLE",
                    "RING",
                    "PINKY",
                )
            ):
                channel.commit_priority = 60
        return channel

    def aim(
        self,
        semantic_id: str,
        bone,
        desired_direction: Vector,
        *,
        rest_direction: Vector | None = None,
        desired_reference: Vector | None = None,
        rest_reference: Vector | None = None,
        correction: Sequence[float] = (0.0, 0.0, 0.0),
        confidence: float = 1.0,
        influence: float = 1.0,
        owner: str = "RETARGET",
        reference_is_oriented: bool = False,
    ) -> bool:
        if desired_direction.length < 1.0e-8:
            return False
        rest_rotation = bone.bone.matrix_local.to_3x3()
        rest_direction = (
            rest_direction.copy()
            if rest_direction is not None
            else rest_rotation @ Vector((0.0, 1.0, 0.0))
        )
        delta_q = stable_orientation_delta(
            rest_direction,
            desired_direction,
            rest_reference,
            desired_reference,
            reference_is_oriented=reference_is_oriented,
        )
        if delta_q is None:
            return False
        delta = delta_q.to_matrix()
        influence = min(max(float(influence), 0.0), 1.0)
        if influence < 1.0:
            delta_q = Quaternion().slerp(delta.to_quaternion().normalized(), influence)
            delta = delta_q.to_matrix()
        channel = self._channel(semantic_id, bone)
        channel.desired_rotation = delta @ rest_rotation
        channel.direct_basis_rotation = None
        channel.correction = tuple(float(value) for value in correction)
        channel.rest_direction = rest_direction.copy()
        channel.desired_direction = desired_direction.copy()
        channel.rest_reference = (
            rest_reference.copy() if rest_reference is not None else None
        )
        channel.desired_reference = (
            desired_reference.copy() if desired_reference is not None else None
        )
        channel.confidence = min(max(float(confidence), 0.0), 1.0)
        channel.influence = influence
        channel.owner = owner
        self.invalidate()
        return True

    def set_basis_rotation(
        self,
        semantic_id: str,
        bone,
        rotation: Quaternion,
        *,
        correction: Sequence[float] = (0.0, 0.0, 0.0),
        additive: bool = False,
        owner: str = "RETARGET",
        priority: int | None = None,
    ) -> None:
        channel = self._channel(semantic_id, bone)
        if additive:
            base = channel.direct_basis_rotation or bone.matrix_basis.to_quaternion()
            rotation = base @ rotation
        channel.direct_basis_rotation = rotation.normalized()
        channel.desired_rotation = None
        channel.correction = tuple(float(value) for value in correction)
        channel.owner = owner
        if priority is not None:
            channel.commit_priority = priority
        self.invalidate()

    def set_desired_rotation(
        self,
        semantic_id: str,
        bone,
        rotation: Matrix,
        *,
        correction: Sequence[float] = (0.0, 0.0, 0.0),
        confidence: float = 1.0,
        influence: float = 1.0,
        owner: str = "RETARGET",
    ) -> None:
        channel = self._channel(semantic_id, bone)
        influence = min(max(float(influence), 0.0), 1.0)
        desired = rotation.to_3x3()
        if influence < 1.0:
            rest = bone.bone.matrix_local.to_3x3()
            delta = desired @ rest.inverted_safe()
            blended = Quaternion().slerp(delta.to_quaternion().normalized(), influence)
            desired = blended.to_matrix() @ rest
        channel.desired_rotation = desired
        channel.direct_basis_rotation = None
        channel.correction = tuple(float(value) for value in correction)
        channel.confidence = min(max(float(confidence), 0.0), 1.0)
        channel.influence = influence
        channel.owner = owner
        self.invalidate()

    def set_location(
        self, semantic_id: str, bone, value: Vector, *, owner="RETARGET"
    ) -> None:
        channel = self._channel(semantic_id, bone)
        channel.location = value.copy()
        channel.owner = owner
        self.invalidate()

    def set_property(
        self,
        semantic_id: str,
        bone,
        property_name: str,
        value: float,
        *,
        owner="RETARGET",
    ) -> bool:
        channel = self._channel(semantic_id, bone)
        if property_name in channel.properties and channel.owner != owner:
            self.warnings.append(
                f"{bone.name}[{property_name}] is already owned by {channel.owner}"
            )
            return False
        channel.properties[property_name] = float(value)
        channel.owner = owner
        self.invalidate()
        return True

    def set_shape(self, obj, block, value: float, *, owner="RETARGET"):
        key = (block.as_pointer(), block.name)
        existing = self.shapes.get(key)
        if existing and existing.owner != owner:
            self.warnings.append(
                f"{obj.name}:{block.name} is already owned by {existing.owner}"
            )
            return
        self.shapes[key] = ShapeChannel(obj, block, float(value), owner)
        self.invalidate()

    def replace_with_swing_only(self, channel: BoneChannel, *, owner: str) -> None:
        if channel.rest_direction is None or channel.desired_direction is None:
            return
        rest_rotation = channel.bone.bone.matrix_local.to_3x3()
        swing = channel.rest_direction.rotation_difference(channel.desired_direction)
        if channel.influence < 1.0:
            swing = Quaternion().slerp(swing, channel.influence)
        channel.desired_rotation = swing.to_matrix() @ rest_rotation
        channel.desired_reference = None
        channel.owner = owner
        self.invalidate()

    def _virtual_pose(self, bone) -> Matrix:
        cached = self._resolved_pose.get(bone.name)
        if cached is not None:
            return cached
        parent_matrix = None
        parent_local = None
        if bone.parent:
            parent_matrix = self._virtual_pose(bone.parent)
            parent_local = bone.parent.bone.matrix_local
        kwargs = {}
        if parent_matrix is not None:
            kwargs = {
                "parent_matrix": parent_matrix,
                "parent_matrix_local": parent_local,
            }
        channel = self.bones.get(bone.name)
        if channel:
            # Retarget channels describe an absolute result relative to the
            # rig rest pose. Reading their current basis here makes the same
            # detection converge a little further on every Apply and lets an
            # arbitrary starting pose leak into IK and finger reconstruction.
            current_location, _ignored_rotation, current_scale = (
                bone.matrix_basis.decompose()
            )
            current_rotation = Quaternion()
            location = (
                channel.location if channel.location is not None else current_location
            )
            if channel.direct_basis_rotation is not None:
                rotation = channel.direct_basis_rotation
            elif channel.desired_rotation is not None:
                # Preserve the location produced by the temporary parent
                # hierarchy. Using bone.matrix.translation here reads the
                # current Blender pose, which can belong to a different
                # parent rotation and creates a spurious basis translation
                # when convert_local_to_pose() is inverted.
                base_pose = bone.bone.convert_local_to_pose(
                    Matrix.LocRotScale(
                        location,
                        current_rotation,
                        current_scale,
                    ),
                    bone.bone.matrix_local,
                    **kwargs,
                )
                desired_pose = Matrix.LocRotScale(
                    base_pose.translation,
                    channel.desired_rotation.to_quaternion(),
                    base_pose.to_scale(),
                )
                converted = bone.bone.convert_local_to_pose(
                    desired_pose,
                    bone.bone.matrix_local,
                    invert=True,
                    **kwargs,
                )
                rotation = converted.to_quaternion().normalized()
            else:
                rotation = current_rotation
            if channel.hinge_axis_local is not None:
                rotation = project_twist(rotation, channel.hinge_axis_local)
            rotation = rotation @ correction_quaternion(channel.correction)
            rotation = respect_rotation_locks(bone, rotation)
            basis = Matrix.LocRotScale(location, rotation, current_scale)
            self._resolved_basis[bone.name] = basis
        else:
            basis = bone.matrix_basis.copy()
        pose_matrix = bone.bone.convert_local_to_pose(
            basis,
            bone.bone.matrix_local,
            **kwargs,
        )
        self._resolved_pose[bone.name] = pose_matrix
        return pose_matrix

    def resolve(self) -> None:
        if self._resolved_revision == self._revision:
            return
        self._resolved_basis.clear()
        self._resolved_pose.clear()
        for channel in sorted(
            self.bones.values(),
            key=lambda item: (
                self.rig_plan.hierarchy_depth.get(item.bone.name, 0)
                if self.rig_plan is not None
                else len(item.bone.parent_recursive)
            ),
        ):
            self._virtual_pose(channel.bone)
        self._resolved_revision = self._revision

    @staticmethod
    def _copy_channel(channel: BoneChannel) -> BoneChannel:
        clone = replace(channel)
        for attribute in (
            "desired_rotation",
            "direct_basis_rotation",
            "location",
            "rest_direction",
            "desired_direction",
            "rest_reference",
            "desired_reference",
            "hinge_axis_local",
        ):
            value = getattr(channel, attribute)
            setattr(clone, attribute, value.copy() if value is not None else None)
        clone.properties = dict(channel.properties)
        return clone

    def snapshot_channels(self):
        """Capture correction-editable channels without copying Blender IDs."""
        return (
            {name: self._copy_channel(channel) for name, channel in self.bones.items()},
            dict(self.semantic_bones),
        )

    def restore_channels(self, snapshot) -> None:
        self.bones, self.semantic_bones = snapshot
        self.invalidate()

    def restore_channel_names(self, snapshot, names: set[str]) -> None:
        """Restore selected channels while preserving independent corrections."""
        snapshot_bones, snapshot_semantics = snapshot
        for name in names:
            previous = snapshot_bones.get(name)
            if previous is not None:
                self.bones[name] = self._copy_channel(previous)
        self.semantic_bones.update(snapshot_semantics)
        self.invalidate()

    def semantic_endpoints(self) -> dict[str, tuple[Vector, Vector, float]]:
        """Return virtual head/tail positions for every mapped pose channel."""
        self.resolve()
        result = {}
        for semantic_id, bone_name in self.semantic_bones.items():
            channel = self.bones.get(bone_name)
            if channel is None:
                continue
            matrix = self._virtual_pose(channel.bone)
            length = max(float(channel.bone.bone.length), 1.0e-8)
            result[semantic_id] = (
                matrix.translation.copy(),
                matrix @ Vector((0.0, length, 0.0)),
                length,
            )
        return result

    @staticmethod
    def maximum_endpoint_delta(before, after):
        """Return the largest normalized mapped head/tail displacement."""
        largest = (0.0, "", "")
        for semantic_id in before.keys() & after.keys():
            before_head, before_tail, length = before[semantic_id]
            after_head, after_tail, _after_length = after[semantic_id]
            for endpoint, distance in (
                ("head", (after_head - before_head).length / length),
                ("tail", (after_tail - before_tail).length / length),
            ):
                if distance > largest[0]:
                    largest = (distance, semantic_id, endpoint)
        return largest

    def hinge_projection_direction_error(
        self,
        channel: BoneChannel,
        axis: Vector,
    ) -> float:
        """Measure how much a proposed hinge projection moves a bone end."""
        previous_axis = channel.hinge_axis_local
        try:
            channel.hinge_axis_local = None
            self.invalidate()
            unprojected = self._virtual_pose(channel.bone).to_3x3()

            channel.hinge_axis_local = axis.copy()
            self.invalidate()
            projected = self._virtual_pose(channel.bone).to_3x3()
        finally:
            channel.hinge_axis_local = previous_axis
            self.invalidate()

        local_axis = Vector((0.0, 1.0, 0.0))
        unprojected_direction = unprojected @ local_axis
        projected_direction = projected @ local_axis
        if unprojected_direction.length < 1.0e-8 or projected_direction.length < 1.0e-8:
            return math.inf
        return unprojected_direction.angle(projected_direction)

    def iter_commit(self):
        """Commit channels in dependency-safe batches and yield after evaluation."""
        import bpy

        def update_scene() -> None:
            bpy.context.view_layer.update()
            self.commit_dependency_updates += 1

        self.resolve()
        self.commit_dependency_updates = 0
        # On a plain deform rig the virtual solver has already resolved every
        # local basis against the complete parent hierarchy. Re-evaluating the
        # dependency graph after every bone is both redundant and especially
        # costly for video. Generated control rigs retain the evaluated path
        # below because constraints can change a parent's effective matrix.
        if self.rig_plan is not None:
            constraint_free = self.rig_plan.constraint_free
        else:
            object_animation = self.armature.animation_data
            data_animation = self.armature.data.animation_data
            has_drivers = bool(
                object_animation
                and object_animation.drivers
                or data_animation
                and data_animation.drivers
            )
            constraint_free = not has_drivers and not any(
                bone.constraints for bone in self.armature.pose.bones
            )
        if constraint_free:
            changed = set()
            committed_channels = []
            for channel in sorted(
                self.bones.values(),
                key=lambda item: (
                    (
                        self.rig_plan.hierarchy_depth.get(item.bone.name, 0)
                        if self.rig_plan is not None
                        else len(item.bone.parent_recursive)
                    ),
                    item.commit_priority,
                ),
            ):
                basis = self._resolved_basis.get(channel.bone.name)
                if basis is None:
                    continue
                channel.bone.matrix_basis = basis.copy()
                for property_name, value in channel.properties.items():
                    if property_name in channel.bone:
                        channel.bone[property_name] = value
                changed.add(channel.bone)
                committed_channels.append(channel)
            self.armature.update_tag(refresh={"OBJECT"})
            update_scene()
            for channel in committed_channels:
                self._record_rotation_deviation(channel)
            changed_shapes = []
            for shape in self.shapes.values():
                shape.block.value = min(
                    max(shape.value, shape.block.slider_min),
                    shape.block.slider_max,
                )
                changed_shapes.append((shape.obj, shape.block))
            yield 1, 1
            return changed, changed_shapes

        def generated_commit_stage(channel: BoneChannel) -> int:
            semantic_id = channel.semantic_id
            if semantic_id in {"ROOT", "PELVIS", "SPINE_01", "SPINE_02", "CHEST"}:
                return 0
            if semantic_id == "NECK":
                return 1
            if semantic_id == "HEAD":
                return 2
            if semantic_id.startswith("SHOULDER_"):
                return 3
            if semantic_id.startswith(("UPPER_ARM_", "UPPER_LEG_")):
                return 4
            if semantic_id.startswith(("LOWER_ARM_", "LOWER_LEG_")):
                return 5
            if semantic_id.startswith(("HAND_", "FOOT_", "TOE_")):
                return 6
            return 7

        changed = set()
        ordered_channels = sorted(
            self.bones.values(),
            key=lambda item: (
                # Generated control rigs often connect controls through
                # constraints rather than Bone.parent. Anatomical dependency
                # order keeps c_head after c_neck/spine and hands after arms;
                # parent depth alone committed Auto-Rig Pro's head too early,
                # after which the neck update pitched it a second time.
                generated_commit_stage(item),
                len(item.bone.parent_recursive),
                item.commit_priority,
            ),
        )

        def dependency_group(channel: BoneChannel) -> tuple[int, int, int]:
            return (
                generated_commit_stage(channel),
                len(channel.bone.parent_recursive),
                channel.commit_priority,
            )

        group_start = 0
        group_total = len({dependency_group(item) for item in ordered_channels})
        group_current = 0
        while group_start < len(ordered_channels):
            group_key = dependency_group(ordered_channels[group_start])
            group_end = group_start + 1
            while (
                group_end < len(ordered_channels)
                and dependency_group(ordered_channels[group_end]) == group_key
            ):
                group_end += 1
            batch = ordered_channels[group_start:group_end]
            for channel in batch:
                bone = channel.bone
                location, current_rotation, scale = bone.matrix_basis.decompose()
                if channel.location is not None:
                    location = channel.location
                if channel.direct_basis_rotation is not None:
                    rotation = channel.direct_basis_rotation
                elif channel.desired_rotation is not None:
                    desired_pose = Matrix.LocRotScale(
                        bone.matrix.translation,
                        channel.desired_rotation.to_quaternion(),
                        bone.matrix.to_scale(),
                    )
                    kwargs = {}
                    if bone.parent:
                        kwargs = {
                            "parent_matrix": bone.parent.matrix,
                            "parent_matrix_local": (bone.parent.bone.matrix_local),
                        }
                    rotation = (
                        bone.bone.convert_local_to_pose(
                            desired_pose,
                            bone.bone.matrix_local,
                            invert=True,
                            **kwargs,
                        )
                        .to_quaternion()
                        .normalized()
                    )
                else:
                    rotation = current_rotation
                if channel.hinge_axis_local is not None:
                    rotation = project_twist(rotation, channel.hinge_axis_local)
                rotation = rotation @ correction_quaternion(channel.correction)
                rotation = respect_rotation_locks(bone, rotation)
                bone.matrix_basis = Matrix.LocRotScale(location, rotation, scale)
                changed.add(bone)
                for property_name, value in channel.properties.items():
                    if property_name in bone:
                        bone[property_name] = value
            self.armature.update_tag(refresh={"OBJECT"})
            update_scene()
            for channel in batch:
                self._record_rotation_deviation(channel)
            group_current += 1
            yield group_current, group_total
            group_start = group_end
        changed_shapes = []
        for shape in self.shapes.values():
            shape.block.value = min(
                max(shape.value, shape.block.slider_min),
                shape.block.slider_max,
            )
            changed_shapes.append((shape.obj, shape.block))
        return changed, changed_shapes

    def commit(self) -> tuple[set[object], list[tuple[object, object]]]:
        """Commit the resolved channels to Blender pose and shape-key data."""
        iterator = self.iter_commit()
        while True:
            try:
                next(iterator)
            except StopIteration as stopped:
                return stopped.value

    def _record_rotation_deviation(self, channel: BoneChannel) -> None:
        if channel.desired_rotation is None:
            return
        resolved = self._resolved_pose.get(channel.bone.name)
        if resolved is None:
            return
        actual = channel.bone.matrix.to_quaternion().normalized()
        # Compare against the fully resolved virtual pose, not the raw target.
        # The resolved matrix includes mapping-axis corrections, rotation locks,
        # and the solved parent hierarchy that commit() is expected to preserve.
        expected = resolved.to_quaternion().normalized()
        error = actual.rotation_difference(expected).angle
        if error <= math.radians(2.0):
            return
        message = (
            f"{channel.bone.name} evaluated rotation differs from the "
            f"solver by {math.degrees(error):.1f} degrees; check locks "
            "and active constraints"
        )
        if message not in self.warnings:
            self.warnings.append(message)
