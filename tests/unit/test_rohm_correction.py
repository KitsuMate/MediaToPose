from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from media_to_pose import rohm_correction
from media_to_pose.geometry import Quaternion, Vector


class _IdentityRuntime:
    def normalize(self, value):
        return np.asarray(value, dtype=np.float32)

    def denormalize(self, value):
        return np.asarray(value, dtype=np.float32)

    def run_trajnet(self, condition, control, rng, *, control_gate):
        del control, rng, control_gate
        return np.asarray(condition, dtype=np.float32)

    def run_posenet(self, condition, rng):
        del rng
        result = np.asarray(condition, dtype=np.float32).copy()
        rotations = result[0, 154:280, 0, :].T.reshape(144, 21, 6)
        identity = np.asarray((1, 0, 0, 1, 0, 0), dtype=np.float32)
        missing = np.linalg.norm(rotations, axis=-1) < 1.0e-7
        rotations[missing] = identity
        result[0, 154:280, 0, :] = rotations.reshape(144, -1).T
        return result


class _RootRepairRuntime(_IdentityRuntime):
    def run_trajnet(self, condition, control, rng, *, control_gate):
        del control, rng, control_gate
        result = np.asarray(condition, dtype=np.float32).copy()
        # The second absolute trajectory item is root X (representation index 2).
        result[0, :, 1] = 0.0
        # A real TrajNet always reconstructs a valid global rotation even when
        # the entire root condition is masked.
        result[0, :, 4:10] = np.asarray((1, 0, 0, 1, 0, 0), dtype=np.float32)
        return result


class _PoseAndRootRepairRuntime(_RootRepairRuntime):
    def run_posenet(self, condition, rng):
        result = super().run_posenet(condition, rng)
        quarter_turn = np.asarray((0, -1, 1, 0, 0, 0), dtype=np.float32)
        result[0, 154:160, 0, :] = quarter_turn[:, None]
        return result


class _WildRootRuntime(_RootRepairRuntime):
    def run_trajnet(self, condition, control, rng, *, control_gate):
        result = super().run_trajnet(condition, control, rng, control_gate=control_gate)
        result[0, :, 1] = np.where(np.arange(result.shape[1]) % 2, -1.0, 1.0)
        return result


class RohmCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.correction = rohm_correction

    def test_rotation_6d_uses_first_two_columns(self) -> None:
        angle = np.deg2rad(37.0)
        matrix = np.asarray(
            (
                (np.cos(angle), -np.sin(angle), 0.0),
                (np.sin(angle), np.cos(angle), 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float32,
        )
        encoded = self.correction.matrix_to_rotation_6d(matrix)
        np.testing.assert_allclose(encoded, matrix[:, :2].reshape(6), atol=1e-7)
        np.testing.assert_allclose(
            self.correction.rotation_6d_to_matrix(encoded), matrix, atol=1e-6
        )

    def test_smplx_locals_round_trip_through_global_anatomical_hierarchy(
        self,
    ) -> None:
        locals_value = np.broadcast_to(np.eye(3), (2, 21, 3, 3)).copy()
        for frame, joint, angle in (
            (0, 0, 0.4),
            (0, 12, 0.3),
            (0, 15, -0.7),
            (1, 18, 0.9),
        ):
            cosine, sine = np.cos(angle), np.sin(angle)
            locals_value[frame, joint] = np.asarray(
                ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
            )

        global_value = self.correction.local_to_global_anatomical(locals_value)
        recovered = self.correction.global_anatomical_to_local(global_value)

        np.testing.assert_allclose(recovered, locals_value, atol=1.0e-6)
        self.assertFalse(
            np.allclose(global_value[0, 16], locals_value[0, 15], atol=1.0e-6)
        )

    def test_joint_representation_recovers_normalized_geometry(self) -> None:
        frames = 145
        joints = np.zeros((frames, 22, 3), dtype=np.float32)
        joints[:, :, 2] = np.linspace(0.0, 1.7, 22)
        joints[:, 1, 0] = 0.2
        joints[:, 2, 0] = -0.2
        joints[:, 16, 0] = -0.4
        joints[:, 17, 0] = 0.4
        joints[:, :, 1] = np.arange(frames, dtype=np.float32)[:, None] * 0.01
        rotations = np.broadcast_to(np.eye(3), (frames, 21, 3, 3)).copy()
        representation, context = self.correction.build_representation(
            joints, rotations, body_height=1.7
        )
        recovered = self.correction.recover_joints(representation, context)
        np.testing.assert_allclose(recovered, joints[:-1], atol=1e-5)
        self.assertEqual(representation.shape, (144, 294))
        np.testing.assert_array_equal(representation[:, 280:290], 0.0)

    def test_pose_and_root_correction_toggles_are_independent(self) -> None:
        source = np.zeros((144, 294), dtype=np.float32)
        source[:, 2] = 0.08
        source[:, 7:13] = np.asarray((1, 0, 0, 1, 0, 0), dtype=np.float32)
        identity = np.asarray((1, 0, 0, 1, 0, 0), dtype=np.float32)
        source[:, 154:280] = np.tile(identity, 21)
        visible = np.ones((144, 22), dtype=bool)
        pose_only = self.correction.correct_window(
            source,
            visible,
            _PoseAndRootRepairRuntime(),
            root_motion_mode="AUTOMATIC",
            correct_pose=True,
            correct_root=False,
        )
        root_only = self.correction.correct_window(
            source,
            visible,
            _PoseAndRootRepairRuntime(),
            root_motion_mode="AUTOMATIC",
            correct_pose=False,
            correct_root=True,
        )

        np.testing.assert_array_equal(pose_only.representation[:, :22], source[:, :22])
        self.assertFalse(pose_only.root_applied)
        self.assertFalse(
            np.array_equal(pose_only.representation[:, 154:160], source[:, 154:160])
        )
        np.testing.assert_array_equal(root_only.representation[:, 22:], source[:, 22:])
        self.assertTrue(root_only.root_applied)
        np.testing.assert_allclose(root_only.representation[:, 2], 0.0, atol=1.0e-7)

    def test_low_confidence_root_is_repaired_when_observations_exist(self) -> None:
        source = np.zeros((144, 294), dtype=np.float32)
        source[:, 2] = np.where(np.arange(144) % 2, -0.08, 0.08)
        source[:, 7:13] = np.asarray((1, 0, 0, 1, 0, 0), dtype=np.float32)
        identity = np.asarray((1, 0, 0, 1, 0, 0), dtype=np.float32)
        source[:, 154:280] = np.tile(identity, 21)
        visible = np.ones((144, 22), dtype=bool)
        visible[:, 0] = False
        root_observed = np.zeros(144, dtype=bool)
        root_observed[[0, -1]] = True

        result = self.correction.correct_window(
            source,
            visible,
            _RootRepairRuntime(),
            root_motion_mode="AUTOMATIC",
            root_observed=root_observed,
        )

        self.assertTrue(result.root_applied)
        np.testing.assert_allclose(result.representation[:, 2], 0.0, atol=1.0e-7)

    def test_finite_implausible_root_is_kept_with_quality_warning(self) -> None:
        source = np.zeros((144, 294), dtype=np.float32)
        source[:, 7:13] = np.asarray((1, 0, 0, 1, 0, 0), dtype=np.float32)
        source[:, 154:280] = np.tile(
            np.asarray((1, 0, 0, 1, 0, 0), dtype=np.float32), 21
        )

        result = self.correction.correct_window(
            source,
            np.ones((144, 22), dtype=bool),
            _WildRootRuntime(),
            root_motion_mode="AUTOMATIC",
        )

        self.assertTrue(result.root_applied)
        self.assertTrue(any("root step" in warning for warning in result.warnings))
        self.assertGreater(np.max(np.abs(result.representation[:, 2])), 0.5)

    def test_timestamped_pose_repair_preserves_trusted_frames_and_returns_globals(
        self,
    ) -> None:
        samples = []
        for frame in range(3):
            offsets = {}
            rotations = {}
            confidence = {}
            bones = {}
            for joint, semantic in enumerate(self.correction.SMPLX_SEMANTICS):
                offset = Vector((0.0, 0.0, joint * 1.7 / 21.0))
                if joint == 1:
                    offset.x = 0.2
                elif joint == 2:
                    offset.x = -0.2
                elif joint == 16:
                    offset.x = -0.4
                elif joint == 17:
                    offset.x = 0.4
                offsets[semantic] = offset
                rotations[semantic] = Quaternion()
                confidence[semantic] = 1.0
                bones[semantic] = SimpleNamespace(
                    semantic_id=semantic,
                    key_rotation=True,
                    key_location=False,
                    rotation=Quaternion(),
                    location=Vector(),
                )
            if frame == 1:
                confidence["UPPER_LEG_L"] = 0.0
            samples.append(
                SimpleNamespace(
                    bones=bones,
                    root_object=(
                        Vector((0.0, frame * 0.01, 0.0)),
                        Quaternion(),
                        Vector((1.0, 1.0, 1.0)),
                    ),
                    grounding=None,
                    body_height=1.7,
                    rohm_joint_offsets=offsets,
                    rohm_rotations=rotations,
                    rohm_confidence=confidence,
                    rohm_root_confidence=1.0,
                )
            )

        stats = self.correction.correct_samples(
            samples,
            (0.0, 1.0 / 30.0, 2.0 / 30.0),
            _PoseAndRootRepairRuntime(),
            root_motion_mode="OFF",
            correct_root=False,
            visibility_threshold=0.5,
        )

        self.assertEqual(stats.pose_global_rotations[0], {})
        self.assertTrue(stats.pose_global_rotations[1])
        self.assertEqual(stats.pose_global_rotations[2], {})
        self.assertEqual(
            set(stats.pose_global_rotations[1]),
            {"UPPER_LEG_L"},
        )
        self.assertAlmostEqual(
            stats.pose_global_rotations[1]["UPPER_LEG_L"].angle,
            np.deg2rad(self.correction.MAX_LEARNED_GLOBAL_POSE_DELTA_DEGREES),
            places=5,
        )
        self.assertTrue(
            all(
                sample.bones["UPPER_LEG_L"].rotation.angle < 1.0e-7
                for sample in samples
            )
        )

    def test_timestamped_adapter_returns_denoised_root_deltas(self) -> None:
        samples = []
        source_roots = []
        for frame in range(12):
            offsets = {}
            rotations = {}
            confidence = {}
            bones = {}
            for joint, semantic in enumerate(self.correction.SMPLX_SEMANTICS):
                offset = Vector((0.0, 0.0, joint * 1.7 / 21.0))
                if joint == 1:
                    offset.x = 0.2
                elif joint == 2:
                    offset.x = -0.2
                elif joint == 16:
                    offset.x = -0.4
                elif joint == 17:
                    offset.x = 0.4
                offsets[semantic] = offset
                rotations[semantic] = Quaternion()
                confidence[semantic] = 1.0
                bones[semantic] = SimpleNamespace(
                    semantic_id=semantic,
                    key_rotation=True,
                    key_location=False,
                    rotation=Quaternion(),
                    location=Vector(),
                )
            root = Vector((0.05 if frame % 2 else -0.05, frame * 0.01, 0.0))
            source_roots.append(root.copy())
            samples.append(
                SimpleNamespace(
                    bones=bones,
                    root_object=(
                        root.copy(),
                        Quaternion(),
                        Vector((1.0, 1.0, 1.0)),
                    ),
                    grounding=SimpleNamespace(root=root.copy()),
                    body_height=1.7,
                    rohm_joint_offsets=offsets,
                    rohm_rotations=rotations,
                    rohm_confidence=confidence,
                    # Below the masking threshold, but still an available root
                    # observation for the trajectory denoiser.
                    rohm_root_confidence=0.25,
                )
            )

        stats = self.correction.correct_samples(
            samples,
            tuple(frame / 30.0 for frame in range(len(samples))),
            _RootRepairRuntime(),
            root_motion_mode="AUTOMATIC",
            visibility_threshold=0.5,
        )

        corrected_x = [
            root.x + stats.root_deltas[index][0]
            for index, root in enumerate(source_roots)
        ]
        self.assertLess(max(corrected_x) - min(corrected_x), 1.0e-6)
        self.assertGreater(stats.corrected_source_frames, 0)
        self.assertTrue(any(abs(delta[0]) > 0.01 for delta in stats.root_deltas))

    def test_timestamped_adapter_pads_short_clip_without_changing_identity(
        self,
    ) -> None:
        samples = []
        for frame in range(3):
            offsets = {}
            rotations = {}
            confidence = {}
            bones = {}
            for joint, semantic in enumerate(self.correction.SMPLX_SEMANTICS):
                offset = Vector((0.0, 0.0, joint * 1.7 / 21.0))
                if joint == 1:
                    offset.x = 0.2
                elif joint == 2:
                    offset.x = -0.2
                elif joint == 16:
                    offset.x = -0.4
                elif joint == 17:
                    offset.x = 0.4
                offsets[semantic] = offset
                rotations[semantic] = Quaternion()
                confidence[semantic] = 1.0
                bones[semantic] = SimpleNamespace(
                    semantic_id=semantic,
                    key_rotation=True,
                    key_location=False,
                    rotation=Quaternion(),
                    location=Vector(),
                )
            samples.append(
                SimpleNamespace(
                    bones=bones,
                    root_object=(
                        Vector((0.0, frame * 0.01, 0.0)),
                        Quaternion(),
                        Vector((1.0, 1.0, 1.0)),
                    ),
                    grounding=None,
                    body_height=1.7,
                    rohm_joint_offsets=offsets,
                    rohm_rotations=rotations,
                    rohm_confidence=confidence,
                    rohm_root_confidence=1.0,
                )
            )
        stats = self.correction.correct_samples(
            samples,
            (0.0, 1.0 / 30.0, 2.0 / 30.0),
            _IdentityRuntime(),
            root_motion_mode="AUTOMATIC",
        )
        self.assertEqual(stats.windows, 1)
        self.assertEqual(stats.corrected_source_frames, 0)
        self.assertEqual(stats.model_frames, 145)

        iterator = self.correction.iter_correct_samples(
            samples,
            (0.0, 1.0 / 30.0, 2.0 / 30.0),
            _IdentityRuntime(),
            root_motion_mode="AUTOMATIC",
        )
        updates = []
        while True:
            try:
                updates.append(next(iterator))
            except StopIteration as stopped:
                cooperative_stats = stopped.value
                break
        self.assertEqual(cooperative_stats, stats)
        self.assertGreaterEqual(len(updates), 8)
        self.assertTrue(all(item[0].startswith("RoHM:") for item in updates))
        self.assertEqual(
            [item[1] for item in updates], sorted(item[1] for item in updates)
        )
        self.assertEqual(updates[-1][1], updates[-1][2])


if __name__ == "__main__":
    unittest.main()
