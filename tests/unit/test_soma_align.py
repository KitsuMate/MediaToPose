"""Rest alignment between SOMA and a rig: round trips and A-pose rigs."""

from __future__ import annotations

import unittest

import numpy as np

from media_to_pose.core.errors import MappingIncomplete
from media_to_pose.core.motion import soma, soma_align


def _rotation(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    k = np.array(((0, -z, y), (z, 0, -x), (-y, x, 0)))
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def _soma_directions() -> dict[str, np.ndarray]:
    """A rig shaped exactly like SOMA's rest."""
    return {
        semantic: soma_align.soma_rest_direction(joint)
        for joint, semantic in soma_align.JOINT_SEMANTICS.items()
    }


class SomaAlignTests(unittest.TestCase):
    def test_axis_change_is_a_proper_rotation(self) -> None:
        self.assertAlmostEqual(float(np.linalg.det(soma_align.C)), 1.0)
        left_arm = soma_align.soma_rest_direction("LeftArm")
        self.assertLess(left_arm[0], 0.0)  # the character's left is -right

    def test_identical_rest_gives_identity_alignment(self) -> None:
        alignment = soma_align.align(_soma_directions())
        for rotation in alignment.rotations.values():
            np.testing.assert_allclose(rotation, np.eye(3), atol=1e-9)

    def test_a_pose_rig_reaches_the_t_pose(self) -> None:
        directions = _soma_directions()
        # Arms 45 degrees down on the rig.
        for side, sign in (("L", -1.0), ("R", 1.0)):
            down = np.array((sign, 0.0, -1.0)) / np.sqrt(2.0)
            for role in ("UPPER_ARM", "LOWER_ARM", "HAND"):
                directions[f"{role}_{side}"] = down
        alignment = soma_align.align(directions)
        rig = soma_align.soma_to_rig(
            np.broadcast_to(np.eye(3), (soma.JOINT_COUNT, 3, 3)), alignment
        )
        arm = soma.JOINT_INDEX["LeftArm"]
        lifted = rig[arm] @ directions["UPPER_ARM_L"]
        expected = soma_align.soma_rest_direction("LeftArm")
        np.testing.assert_allclose(
            lifted, expected / np.linalg.norm(expected), atol=1e-9
        )

    def test_round_trip_through_the_rig(self) -> None:
        directions = _soma_directions()
        directions["UPPER_LEG_L"] = np.array((0.1, 0.05, -1.0))
        alignment = soma_align.align(directions)
        rng = np.random.default_rng(3)
        globals_ = np.stack(
            [
                _rotation(rng.normal(size=3), rng.uniform(0.0, 1.5))
                for _ in range(soma.JOINT_COUNT)
            ]
        )
        # Joints without a rig bone follow their parent in the reverse map.
        for joint in range(soma.JOINT_COUNT):
            if joint not in alignment.rotations and soma.PARENTS[joint] >= 0:
                globals_[joint] = globals_[soma.PARENTS[joint]]
        back = soma_align.rig_to_soma(
            soma_align.soma_to_rig(globals_, alignment), alignment
        )
        np.testing.assert_allclose(back, globals_, atol=1e-9)

    def test_missing_required_role_is_reported(self) -> None:
        directions = _soma_directions()
        del directions["FOOT_R"]
        with self.assertRaises(MappingIncomplete) as raised:
            soma_align.align(directions)
        self.assertIn("FOOT_R", raised.exception.details["missing"])

    def test_pose_positions_keep_soma_proportions(self) -> None:
        identity = np.broadcast_to(np.eye(3), (soma.JOINT_COUNT, 3, 3))
        positions = soma_align.pose_positions(identity, np.zeros(3))
        np.testing.assert_allclose(positions, soma.REST_POSITIONS, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
