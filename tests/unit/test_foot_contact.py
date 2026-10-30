"""Planting feet during contact intervals keeps bones and pins the foot."""

from __future__ import annotations

import unittest

import numpy as np

from media_to_pose.core.motion import foot_contact, soma

J = soma.JOINT_INDEX


def _rotation(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    k = np.array(((0, -z, y), (z, 0, -x), (-y, x, 0)))
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def _sliding_walk(frames: int = 30, drift: float = 0.005):
    """A bent-knee stance whose whole body drifts along x by ``drift`` per
    frame, with the left foot in contact on frames 5-20."""
    local = np.broadcast_to(np.eye(3), (frames, soma.JOINT_COUNT, 3, 3)).copy()
    local[:, J["LeftLeg"]] = _rotation((1.0, 0.0, 0.0), np.radians(-20))
    local[:, J["LeftShin"]] = _rotation((1.0, 0.0, 0.0), np.radians(40))
    hips = np.zeros((frames, 3))
    hips[:, 0] = np.arange(frames) * drift
    hips[:, 1] = soma.REST_HIP_HEIGHT - 0.03
    _globals, positions = soma.forward_kinematics(local, hips)
    contacts = np.zeros((frames, 4), dtype=bool)
    contacts[5:21, 0] = True
    return positions, contacts


def _length(positions, a: str, b: str) -> np.ndarray:
    return np.linalg.norm(positions[:, J[a]] - positions[:, J[b]], axis=-1)


class PlantFeetTests(unittest.TestCase):
    def test_sliding_foot_is_pinned_and_bones_kept(self) -> None:
        positions, contacts = _sliding_walk()
        planted, report = foot_contact.plant_feet(positions, contacts)
        ankle = planted[5:21, J["LeftFoot"]]
        np.testing.assert_allclose(
            ankle, np.broadcast_to(ankle[0], ankle.shape), atol=1e-6
        )
        self.assertGreater(report.sliding_before, 0.004)
        self.assertLess(report.sliding_after, 1e-6)
        for a, b in (
            ("LeftLeg", "LeftShin"),
            ("LeftShin", "LeftFoot"),
            ("LeftFoot", "LeftToeBase"),
        ):
            np.testing.assert_allclose(
                _length(planted, a, b), _length(positions, a, b), atol=1e-9
            )
        np.testing.assert_allclose(planted[:, J["Hips"]], positions[:, J["Hips"]])
        np.testing.assert_allclose(
            planted[:, J["RightFoot"]], positions[:, J["RightFoot"]]
        )

    def test_fades_in_and_out_without_a_pop(self) -> None:
        positions, contacts = _sliding_walk()
        planted, _report = foot_contact.plant_feet(positions, contacts)
        steps = np.linalg.norm(np.diff(planted[:, J["LeftFoot"]], axis=0), axis=-1)
        self.assertLess(steps.max(), 0.03)
        # Far from the interval the foot is untouched.
        np.testing.assert_allclose(
            planted[28:, J["LeftFoot"]], positions[28:, J["LeftFoot"]]
        )

    def test_a_guide_frame_anchors_its_interval(self) -> None:
        positions, contacts = _sliding_walk()
        planted, _report = foot_contact.plant_feet(
            positions, contacts, guide_frames=[7]
        )
        np.testing.assert_allclose(
            planted[5:21, J["LeftFoot"]][:, [0, 2]],
            np.broadcast_to(positions[7, J["LeftFoot"], [0, 2]], (16, 2)),
            atol=1e-6,
        )

    def test_never_pins_below_the_floor(self) -> None:
        positions, contacts = _sliding_walk()
        positions[:, :, 1] -= 0.1  # the model sank the whole body
        planted, _report = foot_contact.plant_feet(positions, contacts)
        rest = soma.REST_POSITIONS[J["LeftFoot"], 1] - soma.REST_POSITIONS[:, 1].min()
        self.assertGreaterEqual(planted[12, J["LeftFoot"], 1], rest - 1e-6)

    def test_no_contacts_change_nothing(self) -> None:
        positions, contacts = _sliding_walk()
        planted, report = foot_contact.plant_feet(positions, np.zeros_like(contacts))
        np.testing.assert_allclose(planted, positions)
        self.assertEqual(report.intervals, 0)

    def test_toe_stays_planted_while_the_heel_lifts(self) -> None:
        positions, contacts = _sliding_walk()
        contacts[:] = False
        contacts[5:16, 0] = True  # heel
        contacts[5:24, 1] = True  # toe, past heel-off
        planted, _report = foot_contact.plant_feet(positions, contacts)
        toe = planted[:, J["LeftToeBase"], [0, 2]]
        # Once the heel's fade is over, only the toe is pinned, at its
        # position on the toe run's middle frame.
        np.testing.assert_allclose(
            toe[20:24],
            np.broadcast_to(positions[14, J["LeftToeBase"], [0, 2]], (4, 2)),
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
