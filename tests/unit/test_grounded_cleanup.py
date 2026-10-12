from __future__ import annotations

import unittest

from media_to_pose.core.geometry import Vector
from media_to_pose.core.motion import grounded_cleanup


class GroundedCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.grounding = grounded_cleanup

    def _frames(self, roots, positions, motion_positions):
        frames = []
        for index, (root, position, motion) in enumerate(
            zip(roots, positions, motion_positions, strict=True)
        ):
            point = self.grounding.ContactPoint(
                position=Vector(position),
                image=Vector((0.5, 0.9)),
                confidence=1.0,
                direction=Vector((1.0, 0.0, 0.0)),
                motion_position=Vector(motion),
            )
            frames.append(
                self.grounding.GroundFrame(
                    timestamp=index / 30.0,
                    root=Vector(root),
                    points={"L_HEEL": point},
                    root_confidence=1.0,
                )
            )
        return frames

    def test_contact_accepts_world_space_plant_during_root_travel(self) -> None:
        roots = [(index * 0.02, 0.0, 0.0) for index in range(8)]
        positions = [(0.0, 0.0, 0.0)] * len(roots)
        motion = [(-root[0], 0.0, 0.0) for root in roots]
        frames = self._frames(roots, positions, motion)

        contacts, _ground = self.grounding.detect_contacts(
            frames, 1.0, Vector((0.0, 0.0, 1.0))
        )

        self.assertGreaterEqual(sum(value > 0.0 for value in contacts["L_HEEL"]), 6)

    def test_contact_accepts_relative_plant_when_root_observation_jitters(self) -> None:
        offsets = (0.0, 0.04, -0.03, 0.05, -0.02, 0.03, -0.01, 0.0)
        roots = [(value, 0.0, 0.0) for value in offsets]
        positions = [(value, 0.0, 0.0) for value in offsets]
        motion = [(0.0, 0.0, 0.0)] * len(roots)
        frames = self._frames(roots, positions, motion)

        contacts, _ground = self.grounding.detect_contacts(
            frames, 1.0, Vector((0.0, 0.0, 1.0))
        )

        self.assertGreaterEqual(sum(value > 0.0 for value in contacts["L_HEEL"]), 6)

    def test_foot_orientation_uses_target_up_axis(self) -> None:
        point = self.grounding.ContactPoint(
            position=Vector(),
            image=Vector(),
            confidence=1.0,
            direction=Vector((0.0, 0.0, 1.0)),
        )
        self.assertAlmostEqual(point.orientation_score(Vector((0.0, 0.0, 1.0))), 0.0)
        self.assertAlmostEqual(point.orientation_score(Vector((0.0, 1.0, 0.0))), 1.0)

    def test_planted_foot_reduces_root_jitter(self) -> None:
        offsets = (0.0, 0.08, -0.06, 0.07, -0.05, 0.04, -0.02, 0.0)
        roots = [(value, 0.0, 0.0) for value in offsets]
        positions = [(value, 0.0, 0.0) for value in offsets]
        motion = [(0.0, 0.0, 0.0)] * len(roots)
        frames = self._frames(roots, positions, motion)
        result = self.grounding.analyze_grounding(frames, 1.0, Vector((0.0, 0.0, 1.0)))
        solved = None
        for _iteration, roots in self.grounding.stabilize_root(
            frames,
            result.contacts,
            result.segments,
            1.0,
            Vector((0.0, 0.0, 1.0)),
            0.5,
        ):
            solved = roots

        self.assertIsNotNone(solved)
        self.assertLess(
            max(abs(value.x) for value in solved),
            max(abs(value) for value in offsets) * 0.35,
        )

    def test_airborne_vertical_travel_is_preserved(self) -> None:
        heights = (0.0, 0.0, 0.2, 0.6, 1.0, 0.6, 0.2, 0.0, 0.0)
        roots = [(0.0, 0.0, value) for value in heights]
        frames = self._frames(roots, roots, [(0.0, 0.0, 0.0)] * len(roots))
        solved = None
        for _iteration, candidate in self.grounding.stabilize_root(
            frames,
            {"L_HEEL": [0.0] * len(frames)},
            [],
            1.0,
            Vector((0.0, 0.0, 1.0)),
            0.5,
        ):
            solved = candidate

        self.assertIsNotNone(solved)
        original_travel = max(heights) - min(heights)
        solved_heights = [value.z for value in solved]
        solved_travel = max(solved_heights) - min(solved_heights)
        self.assertGreaterEqual(solved_travel, original_travel * 0.95)


if __name__ == "__main__":
    unittest.main()
