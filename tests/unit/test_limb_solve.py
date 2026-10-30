"""Limb rotations rebuilt from joint positions keep positions and drop twist."""

from __future__ import annotations

import unittest

import numpy as np

from media_to_pose.core.motion import limb_solve, soma

J = soma.JOINT_INDEX


def _rotation(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    k = np.array(((0, -z, y), (z, 0, -x), (-y, x, 0)))
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def _pose(local: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    rotations = np.broadcast_to(np.eye(3), (1, soma.JOINT_COUNT, 3, 3)).copy()
    for name, matrix in local.items():
        rotations[0, J[name]] = matrix
    return soma.forward_kinematics(rotations, np.zeros((1, 3)))


def _bone(child: str, parent: str) -> np.ndarray:
    return soma.REST_POSITIONS[J[child]] - soma.REST_POSITIONS[J[parent]]


def _twist_deg(rotation: np.ndarray, axis: np.ndarray) -> float:
    w = np.sqrt(max(0.0, 1.0 + np.trace(rotation))) / 2.0
    v = (
        np.array(
            (
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            )
        )
        / 4.0
    )
    axis = axis / np.linalg.norm(axis)
    return float(np.degrees(2.0 * np.arctan2(v @ axis, max(w, 1e-9))))


def _fk_positions(globals_: np.ndarray) -> np.ndarray:
    _rotations, positions = soma.forward_kinematics(
        soma.global_to_local(globals_), np.zeros((len(globals_), 3))
    )
    return positions


class LimbSolveTests(unittest.TestCase):
    def test_rest_pose_is_unchanged(self) -> None:
        globals_, positions = _pose({})
        solved = limb_solve.solve_limbs(globals_, positions)
        np.testing.assert_allclose(solved, globals_, atol=1e-9)

    def test_every_joint_position_is_kept(self) -> None:
        rng = np.random.default_rng(7)
        local = {
            name: _rotation(rng.normal(size=3), rng.uniform(0.2, 1.4))
            for name in soma.JOINT_NAMES
        }
        globals_, positions = _pose(local)
        solved = limb_solve.solve_limbs(globals_, positions)
        np.testing.assert_allclose(_fk_positions(solved), positions, atol=1e-9)

    def test_wrung_straight_leg_is_untwisted(self) -> None:
        thigh = _bone("LeftShin", "LeftLeg")
        shin = _bone("LeftFoot", "LeftShin")
        globals_, positions = _pose(
            {
                "Hips": _rotation((1.0, 0.0, 0.0), np.pi / 2),  # lying down
                "LeftLeg": _rotation(thigh, np.radians(150)),
                "LeftShin": _rotation(shin, np.radians(-135)),
            }
        )
        solved = limb_solve.solve_limbs(globals_, positions)
        local = soma.global_to_local(solved)[0]
        self.assertLess(abs(_twist_deg(local[J["LeftLeg"]], thigh)), 1.0)
        self.assertLess(abs(_twist_deg(local[J["LeftShin"]], shin)), 1.0)
        np.testing.assert_allclose(_fk_positions(solved), positions, atol=1e-9)

    def test_knees_and_elbows_bend_as_hinges(self) -> None:
        globals_, positions = _pose(
            {
                "LeftLeg": _rotation((1.0, 0.0, 0.0), np.radians(-70))
                @ _rotation(_bone("LeftShin", "LeftLeg"), np.radians(40)),
                "LeftShin": _rotation((1.0, 0.0, 0.0), np.radians(90)),
                "LeftArm": _rotation((0.0, 0.0, 1.0), np.radians(-75)),
                "LeftForeArm": _rotation((0.0, 1.0, 0.0), np.radians(-80)),
            }
        )
        solved = limb_solve.solve_limbs(globals_, positions)
        local = soma.global_to_local(solved)[0]
        for joint, hinge in (
            ("LeftShin", (1.0, 0.0, 0.0)),
            ("LeftForeArm", (0.0, -1.0, 0.0)),
        ):
            with self.subTest(joint=joint):
                rotation = local[J[joint]]
                # A pure hinge: the rotation axis is the rest hinge axis.
                angle = np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))
                self.assertGreater(np.degrees(angle), 30.0)
                axis = np.array(
                    (
                        rotation[2, 1] - rotation[1, 2],
                        rotation[0, 2] - rotation[2, 0],
                        rotation[1, 0] - rotation[0, 1],
                    )
                )
                axis /= np.linalg.norm(axis)
                self.assertGreater(abs(axis @ np.array(hinge)), 0.99)
        np.testing.assert_allclose(_fk_positions(solved), positions, atol=1e-9)

    def test_straight_arm_swinging_across_the_body_never_flips(self) -> None:
        # The straight left arm sweeps from the T-pose over the head to the
        # other side, past the opposite of its rest direction.
        frames = []
        for angle in np.linspace(0.0, np.radians(200), 60):
            _globals, positions = _pose({"LeftArm": _rotation((0.0, 0.0, 1.0), angle)})
            frames.append(positions[0])
        positions = np.array(frames)
        identity = np.broadcast_to(np.eye(3), (len(positions), soma.JOINT_COUNT, 3, 3))
        solved = limb_solve.solve_limbs(identity, positions)
        arm = solved[:, J["LeftArm"]]
        steps = [
            np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1)))
            for a, b in zip(arm, arm[1:], strict=False)
        ]
        self.assertLess(max(steps), 10.0)

    def test_heading_stays_continuous_through_a_roll(self) -> None:
        frames = []
        for angle in np.linspace(0.0, np.pi, 45):  # roll about the long axis
            _globals, positions = _pose(
                {
                    "Hips": _rotation((0.0, 0.0, 1.0), angle)
                    @ _rotation((1.0, 0.0, 0.0), np.pi / 2)
                }
            )
            frames.append(positions[0])
        positions = np.array(frames)
        heading = soma.steady_heading(positions)
        # The hip line's own angle jumps by about 180° mid-roll; the steady
        # heading spreads the turn over the roll.
        raw = np.unwrap(soma.heading_angle(positions))
        self.assertGreater(np.max(np.abs(np.diff(raw))), np.radians(90))
        self.assertLess(np.max(np.abs(np.diff(heading))), np.radians(15))
        self.assertTrue(np.all(np.isfinite(heading)))

    def test_over_straightened_knee_keeps_its_roll(self) -> None:
        # The knee bends normally, straightens, then over-straightens.
        frames = []
        for angle in np.radians(np.linspace(40.0, -12.0, 40)):
            _globals, positions = _pose({"LeftShin": _rotation((1.0, 0.0, 0.0), angle)})
            frames.append(positions[0])
        positions = np.array(frames)
        identity = np.broadcast_to(np.eye(3), (len(positions), soma.JOINT_COUNT, 3, 3))
        thigh = limb_solve.solve_limbs(identity, positions)[:, J["LeftLeg"]]
        steps = [
            np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1)))
            for a, b in zip(thigh, thigh[1:], strict=False)
        ]
        self.assertLess(max(steps), 5.0)

    def test_wrung_clavicle_follows_the_chest(self) -> None:
        clavicle = _bone("LeftArm", "LeftShoulder")
        globals_, positions = _pose(
            {"LeftShoulder": _rotation(clavicle, np.radians(166))}
        )
        solved = limb_solve.solve_limbs(globals_, positions)
        local = soma.global_to_local(solved)[0]
        self.assertLess(abs(_twist_deg(local[J["LeftShoulder"]], clavicle)), 1.0)
        np.testing.assert_allclose(_fk_positions(solved), positions, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
