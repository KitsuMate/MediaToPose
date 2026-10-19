from __future__ import annotations

import unittest

import numpy as np

from media_to_pose.core.motion import soma


def _random_rotations(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    matrices, _ = np.linalg.qr(rng.normal(size=(*shape, 3, 3)))
    signs = np.sign(np.linalg.det(matrices))
    matrices[..., :, 0] *= signs[..., None]
    return matrices


class SkeletonTests(unittest.TestCase):
    def test_hierarchy_is_topologically_ordered(self):
        self.assertEqual(soma.JOINT_COUNT, 30)
        self.assertEqual(soma.PARENTS[soma.ROOT], -1)
        for index, parent in enumerate(soma.PARENTS):
            if index != soma.ROOT:
                self.assertLess(parent, index)

    def test_rest_pose_is_a_mirrored_t_pose(self):
        rest = soma.REST_POSITIONS
        for name in ("Arm", "ForeArm", "Hand", "Leg", "Shin", "Foot", "ToeBase"):
            left = rest[soma.JOINT_INDEX[f"Left{name}"]]
            right = rest[soma.JOINT_INDEX[f"Right{name}"]]
            self.assertGreater(left[0], 0.0)
            self.assertAlmostEqual(left[0], -right[0], delta=0.002)
            self.assertAlmostEqual(left[1], right[1], delta=0.002)
        # Arms horizontal, toes pointing forward.
        self.assertAlmostEqual(
            rest[soma.JOINT_INDEX["LeftArm"], 1], rest[soma.JOINT_INDEX["LeftHand"], 1]
        )
        toe = soma.JOINT_INDEX["LeftToeBase"]
        self.assertGreater(rest[toe, 2], rest[soma.JOINT_INDEX["LeftFoot"], 2])
        self.assertAlmostEqual(soma.REST_HIP_HEIGHT, 0.988708)

    def test_identity_forward_kinematics_gives_rest_pose(self):
        local = np.broadcast_to(np.eye(3), (2, soma.JOINT_COUNT, 3, 3))
        root = np.array([[0.0, 0.93, 0.0], [1.0, 0.9, -2.0]])
        global_rotations, positions = soma.forward_kinematics(local, root)
        np.testing.assert_allclose(global_rotations, local)
        np.testing.assert_allclose(positions, soma.REST_POSITIONS + root[:, None])

    def test_local_and_global_rotations_round_trip(self):
        rng = np.random.default_rng(3)
        local = _random_rotations(rng, (4, soma.JOINT_COUNT))
        global_rotations, positions = soma.forward_kinematics(local, np.zeros((4, 3)))
        np.testing.assert_allclose(
            soma.global_to_local(global_rotations), local, atol=1e-12
        )
        # Bone lengths survive any pose.
        for index, parent in enumerate(soma.PARENTS):
            if parent < 0:
                continue
            rest = np.linalg.norm(
                soma.REST_POSITIONS[index] - soma.REST_POSITIONS[parent]
            )
            posed = np.linalg.norm(positions[:, index] - positions[:, parent], axis=-1)
            np.testing.assert_allclose(posed, rest, atol=1e-12)

    def test_heading_follows_a_turn_about_up(self):
        # The captured rest pose is not perfectly symmetric (about 0.07 deg).
        rest = float(soma.heading_angle(soma.REST_POSITIONS))
        self.assertAlmostEqual(rest, 0.0, delta=0.002)
        for angle in (0.5, -2.0, 2.5):
            turned = soma.REST_POSITIONS @ soma.rotation_y(angle).T
            self.assertAlmostEqual(
                float(soma.heading_angle(turned)), angle + rest, places=12
            )
        # A quarter turn faces +X: (sin a, 0, cos a).
        facing = soma.rotation_y(np.pi / 2) @ np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(facing, [1.0, 0.0, 0.0], atol=1e-12)

    def test_axis_change_is_a_proper_rotation_to_right_forward_up(self):
        change = soma.KIMODO_TO_ANATOMICAL
        self.assertAlmostEqual(float(np.linalg.det(change)), 1.0)
        np.testing.assert_allclose(change @ change.T, np.eye(3))
        left_hand = soma.REST_POSITIONS[soma.JOINT_INDEX["LeftHand"]]
        head = soma.REST_POSITIONS[soma.JOINT_INDEX["Head"]]
        toe = soma.REST_POSITIONS[soma.JOINT_INDEX["LeftToeBase"]]
        foot = soma.REST_POSITIONS[soma.JOINT_INDEX["LeftFoot"]]
        self.assertLess((change @ left_hand)[0], 0.0)  # left is -right
        self.assertGreater((change @ head)[2], 0.0)  # up
        self.assertGreater((change @ (toe - foot))[1], 0.0)  # toes point forward

    def test_effector_chains_start_at_the_rotated_joint(self):
        self.assertEqual(
            soma.EFFECTOR_CHAINS["LEFT_HAND"][0], soma.JOINT_INDEX["LeftHand"]
        )
        self.assertEqual(soma.EFFECTOR_CHAINS["HIPS"], (soma.ROOT,))
        self.assertEqual(len(soma.CONTACT_JOINTS), 4)


if __name__ == "__main__":
    unittest.main()
