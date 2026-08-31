"""Efficient evaluated-space placement for constrained pose controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import bpy
from mathutils import Matrix, Vector


@dataclass(frozen=True)
class TranslationSolveResult:
    converged: bool
    residual: float
    dependency_updates: int
    jacobian_builds: int


class PoseTranslationSolver:
    """Place several controls while reusing one Jacobian per control/frame."""

    def __init__(self, update: Optional[Callable[[], None]] = None):
        self._update = update or bpy.context.view_layer.update
        self._jacobians: dict[int, Matrix] = {}
        self.dependency_updates = 0
        self.jacobian_builds = 0

    def _evaluate(self, armature) -> None:
        armature.update_tag(refresh={"OBJECT"})
        self._update()
        self.dependency_updates += 1

    def invalidate(self, bone=None) -> None:
        """Discard a cached inverse after a control's constraint frame changes."""
        if bone is None:
            self._jacobians.clear()
            return
        self._jacobians.pop(int(bone.as_pointer()), None)

    @staticmethod
    def _well_conditioned(jacobian: Matrix) -> bool:
        columns = [
            Vector((jacobian[0][index], jacobian[1][index], jacobian[2][index]))
            for index in range(3)
        ]
        scale = columns[0].length * columns[1].length * columns[2].length
        if scale < 1.0e-12:
            return False
        normalized_volume = abs(float(jacobian.determinant())) / scale
        return normalized_volume >= 1.0e-4

    def _jacobian(self, bone, current: Vector) -> Optional[Matrix]:
        basis_location, basis_rotation, basis_scale = bone.matrix_basis.decompose()
        epsilon = max(float(bone.length) * 1.0e-4, 1.0e-5)
        columns = []
        for axis in range(3):
            trial = basis_location.copy()
            trial[axis] += epsilon
            bone.matrix_basis = Matrix.LocRotScale(
                trial,
                basis_rotation,
                basis_scale,
            )
            self._evaluate(bone.id_data)
            columns.append((bone.matrix.translation - current) / epsilon)
        bone.matrix_basis = Matrix.LocRotScale(
            basis_location,
            basis_rotation,
            basis_scale,
        )
        self._evaluate(bone.id_data)
        jacobian = Matrix(
            (
                (columns[0].x, columns[1].x, columns[2].x),
                (columns[0].y, columns[1].y, columns[2].y),
                (columns[0].z, columns[1].z, columns[2].z),
            )
        )
        self.jacobian_builds += 1
        return jacobian if self._well_conditioned(jacobian) else None

    def solve(
        self,
        bone,
        target_translation: Vector,
        *,
        attempts: int = 3,
        tolerance: float = 1.0e-6,
    ) -> TranslationSolveResult:
        desired = target_translation.copy()
        updates_before = self.dependency_updates
        jacobians_before = self.jacobian_builds

        # Matrix assignment already performs Blender's parent-space conversion
        # for ordinary controls. Only active constraints need the numerical
        # inverse used below.
        if not bone.constraints:
            matrix = bone.matrix.copy()
            matrix.translation = desired
            bone.matrix = matrix
            self._evaluate(bone.id_data)
            residual = float((desired - bone.matrix.translation).length)
            return TranslationSolveResult(
                residual < tolerance,
                residual,
                self.dependency_updates - updates_before,
                self.jacobian_builds - jacobians_before,
            )

        key = int(bone.as_pointer())
        previous_residual = float("inf")
        force_rebuild = key not in self._jacobians
        residual = float("inf")
        for _attempt in range(max(1, attempts)):
            self._evaluate(bone.id_data)
            current = bone.matrix.translation.copy()
            error = desired - current
            residual = float(error.length)
            if residual < tolerance:
                break

            jacobian = self._jacobians.get(key)
            if force_rebuild or jacobian is None:
                jacobian = self._jacobian(bone, current)
                force_rebuild = False
                if jacobian is None:
                    self._jacobians.pop(key, None)
                    break
                self._jacobians[key] = jacobian

            basis_location, basis_rotation, basis_scale = bone.matrix_basis.decompose()
            delta = jacobian.inverted() @ error
            maximum = max(float(bone.length) * 8.0, 0.25)
            if delta.length > maximum:
                delta = delta.normalized() * maximum
            bone.matrix_basis = Matrix.LocRotScale(
                basis_location + delta,
                basis_rotation,
                basis_scale,
            )
            self._evaluate(bone.id_data)
            residual = float((desired - bone.matrix.translation).length)
            # Rebuild only if the linearization stopped improving. Normal
            # correction passes keep using the one frame-local Jacobian.
            force_rebuild = residual >= previous_residual * 0.95
            previous_residual = residual

        return TranslationSolveResult(
            residual < tolerance,
            residual,
            self.dependency_updates - updates_before,
            self.jacobian_builds - jacobians_before,
        )
