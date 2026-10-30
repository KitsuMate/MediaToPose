"""Rig pose <-> SOMA skeleton, in both directions (Kimodo generation).

``bind`` reads the rig's rest once: the anatomical frame ``B`` (columns
right, forward, up in armature space), the rest alignment to SOMA, the scale
from SOMA's leg height to the rig's, and where SOMA's origin sits on the
rig's ground.

Kimodo points map to armature space as ``B @ (scale * C @ p + anchor)``;
the pelvis head gets a constant height offset measured at rest, so a SOMA
T-pose stands the rig exactly on its rest ground.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from mathutils import Matrix, Vector

from ..core.errors import MappingIncomplete
from ..core.motion import soma, soma_align
from ..core.scene import SceneContext
from ..rig import planning as rig_planning
from . import fast_pose
from .pose import _commit_target_pose
from .samples import _BakedBoneSample, _BakedFrameSample
from .solve.common import _mapped_bone
from .solve.frames import target_basis
from .solve.ik import _prepare_ik_output
from .target_pose import TargetPose

OWNER = "GENERATE"
_FEET = ("FOOT_L", "FOOT_R", "TOE_L", "TOE_R")


def _np(matrix) -> np.ndarray:
    return np.array([[float(value) for value in row] for row in matrix])


def _soma_leg_height() -> float:
    rest = soma.REST_POSITIONS
    hips = 0.5 * (
        rest[soma.JOINT_INDEX["LeftLeg"], 1] + rest[soma.JOINT_INDEX["RightLeg"], 1]
    )
    return float(hips - rest[:, 1].min())


@dataclass(frozen=True)
class SomaRigBinding:
    """How one rig, at one mapping revision, maps onto SOMA."""

    armature: object
    rig_plan: rig_planning.RigSolvePlan
    basis: np.ndarray
    bones: Mapping[int, object]
    semantics: Mapping[int, str]
    alignment: soma_align.SomaAlignment
    scale: float
    anchor: np.ndarray
    pelvis_offset: float
    pelvis: object
    root: object | None = None
    # Plain FK bones only (no constraints, IK, locks or drivers): posing can
    # skip the dependency graph (``fast_pose``) with the same result.
    fast: bool = False

    @property
    def entries(self):
        return self.rig_plan.entries

    @property
    def ground(self) -> float:
        return float(self.anchor[2])

    def to_armature(self, kimodo_points: np.ndarray) -> np.ndarray:
        """Armature-space points [..., 3] from Kimodo-space points."""
        anatomy = self.scale * np.asarray(kimodo_points) @ soma_align.C.T + self.anchor
        return anatomy @ self.basis.T

    def to_kimodo(self, armature_points: np.ndarray) -> np.ndarray:
        """Kimodo-space points [..., 3] from armature-space points."""
        anatomy = np.asarray(armature_points) @ self.basis - self.anchor
        return anatomy @ soma_align.C / self.scale

    def pelvis_target(self, hips: np.ndarray) -> Vector:
        point = self.to_armature(hips) + self.basis[:, 2] * self.pelvis_offset
        return Vector(tuple(float(value) for value in point))

    def pelvis_hips(self, head: Vector) -> np.ndarray:
        """SOMA Hips (Kimodo space) for a pelvis head in armature space."""
        point = np.array(head[:]) - self.basis[:, 2] * self.pelvis_offset
        return self.to_kimodo(point)


def _rest_head(bone) -> np.ndarray:
    return np.array(bone.bone.head_local[:])


def _rest_direction(armature, entries, semantic_id: str, bone) -> np.ndarray:
    for child_id in soma_align.RIG_CHILDREN.get(semantic_id, ()):
        child = _mapped_bone(armature, entries, child_id)
        if child is not None:
            direction = _rest_head(child) - _rest_head(bone)
            if np.linalg.norm(direction) > 1.0e-6:
                return direction
    return np.array(bone.bone.tail_local[:]) - _rest_head(bone)


def _fast_rig(rig_plan: rig_planning.RigSolvePlan, bones) -> bool:
    locked = any(
        any(bone.lock_rotation) or bone.lock_rotation_w or any(bone.lock_location)
        for bone in bones
    )
    # ``ik_chains`` lists every limb chain; IK controls are constraints,
    # which ``constraint_free`` already excludes.
    return bool(
        rig_plan.constraint_free and not locked and fast_pose.chain_is_plain(bones)
    )


def bind(armature, rig_plan: rig_planning.RigSolvePlan) -> SomaRigBinding:
    """Bind the rig's rest to SOMA. Raises ``MappingIncomplete`` when the
    body roles generation needs are not mapped."""
    entries = rig_plan.entries
    basis = _np(rig_plan.rest_basis.to_3x3())
    bones: dict[int, object] = {}
    semantics: dict[int, str] = {}
    directions: dict[str, np.ndarray] = {}
    for joint, semantic_id in soma_align.JOINT_SEMANTICS.items():
        bone = _mapped_bone(armature, entries, semantic_id)
        if bone is None:
            continue
        index = soma.JOINT_INDEX[joint]
        bones[index] = bone
        semantics[index] = semantic_id
        directions[semantic_id] = (
            _rest_direction(armature, entries, semantic_id, bone) @ basis
        )
    alignment = soma_align.align(directions)
    feet = [
        point @ basis
        for semantic_id in _FEET
        if (bone := _mapped_bone(armature, entries, semantic_id)) is not None
        for point in (_rest_head(bone), np.array(bone.bone.tail_local[:]))
    ]
    legs = [
        _rest_head(bones[soma.JOINT_INDEX[name]]) @ basis
        for name in ("LeftLeg", "RightLeg")
    ]
    ground = min(point[2] for point in feet)
    hip_center = 0.5 * (legs[0] + legs[1])
    leg_height = float(hip_center[2] - ground)
    if leg_height <= 1.0e-4:
        raise MappingIncomplete("The rig's legs have no height above its feet")
    scale = leg_height / _soma_leg_height()
    pelvis = bones[soma.ROOT]
    anchor = np.array((hip_center[0], hip_center[1], ground))
    pelvis_rest = _rest_head(pelvis) @ basis
    pelvis_offset = float(pelvis_rest[2] - (scale * soma.REST_HIP_HEIGHT + ground))
    root = _mapped_bone(armature, entries, "ROOT")
    return SomaRigBinding(
        armature=armature,
        rig_plan=rig_plan,
        basis=basis,
        bones=bones,
        semantics=semantics,
        alignment=alignment,
        scale=float(scale),
        anchor=anchor,
        pelvis_offset=pelvis_offset,
        pelvis=pelvis,
        root=root,
        fast=_fast_rig(rig_plan, [*bones.values(), *([root] if root else [])]),
    )


def current_rig_rotations(binding: SomaRigBinding) -> dict[int, np.ndarray]:
    """Anatomical rotations from rest of the mapped bones, as evaluated now."""
    basis = binding.basis
    result = {}
    for joint, bone in binding.bones.items():
        posed = _np(bone.matrix.to_3x3())
        rest = _np(bone.bone.matrix_local.to_3x3())
        delta = posed @ np.linalg.inv(rest)
        u, _s, vt = np.linalg.svd(delta)  # drop pose scale
        rotation = u @ vt
        result[joint] = basis.T @ rotation @ basis
    return result


def current_soma_pose(binding: SomaRigBinding) -> tuple[np.ndarray, np.ndarray]:
    """SOMA global rotations [30, 3, 3] and positions [30, 3] (Kimodo space)
    of the rig's evaluated pose, on SOMA's proportions."""
    globals_ = soma_align.rig_to_soma(current_rig_rotations(binding), binding.alignment)
    hips = binding.pelvis_hips(binding.pelvis.head)
    return globals_, soma_align.pose_positions(globals_, hips)


def _posed_root(binding: SomaRigBinding) -> Matrix:
    return target_basis(
        binding.armature, binding.entries, rest_basis=binding.rig_plan.rest_basis
    ).to_3x3()


def _store(sample: _BakedFrameSample, bone, *, location: bool, rotation: bool) -> None:
    previous = sample.bones.get(bone.name)
    loc, rot, scale = bone.matrix_basis.decompose()
    sample.bones[bone.name] = _BakedBoneSample(
        bone=bone,
        location=loc.copy(),
        rotation=rot.normalized(),
        scale=scale.copy(),
        key_location=location or bool(previous and previous.key_location),
        key_rotation=rotation or bool(previous and previous.key_rotation),
        semantic_id=previous.semantic_id if previous else "",
    )


def _root_matrix(
    binding: SomaRigBinding, smooth_root: np.ndarray, heading: float
) -> Matrix:
    """The ROOT control follows the model's smooth root on the ground and
    turns with the heading; the pelvis keeps the rest of the motion."""
    root = binding.root
    ground = np.array((smooth_root[0], 0.0, smooth_root[2]))
    turn = soma_align.C @ soma.rotation_y(heading) @ soma_align.C.T
    delta = binding.basis @ turn @ binding.basis.T
    matrix = Matrix(delta.tolist()).to_4x4() @ root.bone.matrix_local.to_3x3().to_4x4()
    matrix.translation = Vector(tuple(float(v) for v in binding.to_armature(ground)))
    return matrix


def _desired_rotations(binding: SomaRigBinding, global_rotations: np.ndarray):
    """Armature-space rotation [3x3] per mapped bone for one SOMA frame."""
    rotations = soma_align.soma_to_rig(global_rotations, binding.alignment)
    # SOMA rotations are in the world frame, which a turned ROOT does not
    # change, so the rest basis converts them.
    for joint, bone in binding.bones.items():
        delta = binding.basis @ rotations[joint] @ binding.basis.T
        yield joint, bone, Matrix(delta.tolist()) @ bone.bone.matrix_local.to_3x3()


def _apply_fast(
    binding: SomaRigBinding,
    global_rotations: np.ndarray,
    hips: np.ndarray,
    sample: _BakedFrameSample,
    smooth_root: np.ndarray | None,
    heading: float,
) -> None:
    """``apply_soma_frame`` for plain FK rigs, in matrix maths only."""
    chain = fast_pose.PoseChain(lambda bone: bone.matrix_basis)
    placed = {binding.pelvis.name}
    if binding.root is not None and smooth_root is not None:
        chain.set_pose(binding.root, _root_matrix(binding, smooth_root, heading))
        placed.add(binding.root.name)
    desired = sorted(
        _desired_rotations(binding, global_rotations),
        key=lambda item: len(item[1].parent_recursive),
    )
    for _joint, bone, rotation in desired:
        pose = rotation.to_4x4()
        pose.translation = (
            binding.pelvis_target(hips)
            if bone is binding.pelvis
            else chain.head_with_basis_location(bone)
        )
        chain.set_pose(bone, pose)
    bones = [bone for _joint, bone, _rotation in desired]
    if binding.root is not None and binding.root.name in placed:
        bones.append(binding.root)
    for bone in bones:
        location, rotation, scale = chain.basis(bone).decompose()
        sample.bones[bone.name] = _BakedBoneSample(
            bone=bone,
            location=location,
            rotation=rotation.normalized(),
            scale=scale,
            key_location=bone.name in placed,
            key_rotation=True,
        )


def apply_soma_frame(
    binding: SomaRigBinding,
    global_rotations: np.ndarray,
    hips: np.ndarray,
    sample: _BakedFrameSample,
    *,
    scene_context: SceneContext,
    smooth_root: np.ndarray | None = None,
    heading: float = 0.0,
) -> None:
    """Pose the rig like one SOMA frame and record the result in ``sample``.

    With a mapped ROOT control and ``smooth_root``, the root carries the
    ground path and heading.
    """
    if binding.fast:
        _apply_fast(binding, global_rotations, hips, sample, smooth_root, heading)
        return
    armature = binding.armature
    if binding.root is not None and smooth_root is not None:
        binding.root.matrix = _root_matrix(binding, smooth_root, heading)
        armature.update_tag(refresh={"OBJECT"})
        scene_context.update()
    # The pelvis head goes first: IK targets are solved against it.
    pelvis = binding.pelvis
    matrix = pelvis.matrix.copy()
    matrix.translation = binding.pelvis_target(hips)
    pelvis.matrix = matrix
    armature.update_tag(refresh={"OBJECT"})
    scene_context.update()
    rest_basis = Matrix(binding.basis.tolist())
    posed_root = _posed_root(binding)
    target_pose = TargetPose(
        armature,
        None,
        binding.entries,
        rest_basis,
        posed_root,
        rig_plan=binding.rig_plan,
    )
    for joint, bone, desired in _desired_rotations(binding, global_rotations):
        target_pose.set_desired_rotation(
            binding.semantics[joint], bone, desired, owner=OWNER
        )
    ik_proposals = _prepare_ik_output(target_pose)
    changed, _shapes = _commit_target_pose(
        target_pose, ik_proposals, scene_context=scene_context
    )
    ik_rotation = set()
    ik_location = set()
    for proposal in ik_proposals:
        ik_rotation.add(proposal["target"])
        if not proposal.get("orientation_only"):
            ik_location.add(proposal["target"])
            for key in ("pole",):
                if proposal.get(key) is not None:
                    ik_location.add(proposal[key])
            for key in ("rotation_bend", "toe"):
                if proposal.get(key) is not None:
                    ik_rotation.add(proposal[key])
    # Every mapped bone is stored on every frame, so each channel is keyed
    # (and warped) over the whole range.
    placed = {pelvis} | ({binding.root} if binding.root is not None else set())
    for bone in changed | placed | set(binding.bones.values()):
        _store(
            sample,
            bone,
            location=bone in placed or bone in ik_location,
            rotation=True if bone not in ik_location else bone in ik_rotation,
        )


__all__ = [
    "OWNER",
    "SomaRigBinding",
    "apply_soma_frame",
    "bind",
    "current_rig_rotations",
    "current_soma_pose",
]
