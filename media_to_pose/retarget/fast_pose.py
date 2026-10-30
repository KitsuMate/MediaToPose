"""Pose-space matrices of plain FK bones without the dependency graph.

For a bone that inherits its parent's rotation and scale, uses local
location and has no constraints, Blender's pose matrix is

    pose = parent_pose @ (parent_rest⁻¹ @ rest) @ basis

(``rest`` is ``bone.matrix_local``; a root bone uses ``rest`` alone). A
``PoseChain`` evaluates and inverts this in plain matrix maths, so posing
and reading many frames needs no scene update.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from mathutils import Matrix


def is_plain(bone) -> bool:
    """Whether ``bone`` (a pose bone) follows the plain FK formula."""
    data = bone.bone
    return (
        data.use_inherit_rotation
        and data.inherit_scale == "FULL"
        and data.use_local_location
        and not any(not constraint.mute for constraint in bone.constraints)
    )


def chain_is_plain(bones: Iterable) -> bool:
    """Whether ``bones`` and all their ancestors follow the plain formula."""
    seen = set()
    for bone in bones:
        current = bone
        while current is not None and current.name not in seen:
            if not is_plain(current):
                return False
            seen.add(current.name)
            current = current.parent
    return True


class PoseChain:
    """Pose matrices from per-bone bases, memoised until a basis changes.

    ``basis_of(bone)`` gives a bone's basis when it was not set here.
    """

    def __init__(self, basis_of: Callable[[object], Matrix]):
        self._basis_of = basis_of
        self._bases: dict[str, Matrix] = {}
        self._poses: dict[str, Matrix] = {}

    def basis(self, bone) -> Matrix:
        found = self._bases.get(bone.name)
        return found if found is not None else self._basis_of(bone)

    def _offset(self, bone) -> Matrix:
        rest = bone.bone.matrix_local
        if bone.parent is None:
            return rest
        return bone.parent.bone.matrix_local.inverted() @ rest

    def _frame(self, bone) -> Matrix:
        """Where the bone's basis applies: its parent pose and rest offset."""
        offset = self._offset(bone)
        return offset if bone.parent is None else self.pose(bone.parent) @ offset

    def pose(self, bone) -> Matrix:
        found = self._poses.get(bone.name)
        if found is None:
            found = self._frame(bone) @ self.basis(bone)
            self._poses[bone.name] = found
        return found

    def set_basis(self, bone, basis: Matrix) -> None:
        self._bases[bone.name] = basis.copy()
        self._poses.clear()

    def set_pose(self, bone, pose: Matrix) -> Matrix:
        """Set the basis that gives ``pose``; returns that basis."""
        basis = self._frame(bone).inverted() @ pose
        self.set_basis(bone, basis)
        return basis

    def head_with_basis_location(self, bone) -> Matrix:
        """The pose translation the bone's current basis location gives."""
        location = Matrix.Translation(self.basis(bone).to_translation())
        return (self._frame(bone) @ location).to_translation()


__all__ = ["PoseChain", "chain_is_plain", "is_plain"]
