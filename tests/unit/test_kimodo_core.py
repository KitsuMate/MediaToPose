from __future__ import annotations

import unittest

import numpy as np

from media_to_pose.core.errors import BackendError, Cancelled, InvalidRequest
from media_to_pose.core.motion import soma
from media_to_pose.core.motion.kimodo import guides, rep, sampler


def _rotations(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    matrices, _ = np.linalg.qr(rng.normal(size=(*shape, 3, 3)))
    signs = np.sign(np.linalg.det(matrices))
    matrices[..., :, 0] *= signs[..., None]
    return matrices


def _stats(rng: np.random.Generator) -> rep.FeatureStats:
    return rep.FeatureStats(
        rng.normal(size=rep.FEATURE_COUNT), rng.uniform(0.1, 2.0, rep.FEATURE_COUNT)
    )


def _features(local: np.ndarray, root: np.ndarray, smooth: np.ndarray) -> np.ndarray:
    """Encode a pose the way upstream lays features out (velocities zero)."""
    global_rotations, positions = soma.forward_kinematics(local, root)
    count = len(root)
    features = np.zeros((count, rep.FEATURE_COUNT))
    features[:, rep.SMOOTH_ROOT] = smooth
    angle = soma.heading_angle(positions)
    features[:, rep.HEADING] = np.stack((np.cos(angle), np.sin(angle)), axis=-1)
    relative = positions.copy()
    relative[..., 0] -= smooth[:, None, 0]
    relative[..., 2] -= smooth[:, None, 2]
    features[:, rep.POSITIONS] = relative.reshape(count, -1)
    features[:, rep.ROTATIONS] = rep.matrix_to_rot6d(global_rotations).reshape(
        count, -1
    )
    features[:, rep.CONTACTS] = (0.9, 0.1, 0.6, 0.4)
    return features


class RepresentationTests(unittest.TestCase):
    def test_layout_covers_369_features(self):
        slices = (
            rep.SMOOTH_ROOT,
            rep.HEADING,
            rep.POSITIONS,
            rep.ROTATIONS,
            rep.VELOCITIES,
            rep.CONTACTS,
        )
        self.assertEqual(slices[0].start, 0)
        for before, after in zip(slices, slices[1:], strict=False):
            self.assertEqual(before.stop, after.start)
        self.assertEqual(slices[-1].stop, rep.FEATURE_COUNT)

    def test_six_d_round_trip_and_gram_schmidt(self):
        rng = np.random.default_rng(1)
        matrices = _rotations(rng, (5, 7))
        np.testing.assert_allclose(
            rep.rot6d_to_matrix(rep.matrix_to_rot6d(matrices)), matrices, atol=1e-12
        )
        noisy = rep.matrix_to_rot6d(matrices) + rng.normal(scale=0.05, size=(5, 7, 6))
        result = rep.rot6d_to_matrix(noisy)
        np.testing.assert_allclose(
            result @ np.swapaxes(result, -1, -2),
            np.broadcast_to(np.eye(3), result.shape),
            atol=1e-12,
        )
        np.testing.assert_allclose(np.linalg.det(result), 1.0)

    def test_statistics_normalise_with_epsilon(self):
        rng = np.random.default_rng(2)
        stats = _stats(rng)
        values = rng.normal(size=(4, rep.FEATURE_COUNT))
        normalised = stats.normalize(values)
        np.testing.assert_allclose(
            normalised, (values - stats.mean) / np.sqrt(stats.std**2 + 1e-5)
        )
        np.testing.assert_allclose(stats.denormalize(normalised), values)
        self.assertEqual(stats.normalize(values.astype(np.float32)).dtype, np.float32)
        joined = rep.FeatureStats.from_parts(
            np.zeros(5), np.ones(5), np.ones(364), np.full(364, 2.0)
        )
        self.assertEqual(joined.mean[5], 1.0)
        self.assertEqual(joined.std[0], 1.0)

    def test_decode_recovers_the_pose(self):
        rng = np.random.default_rng(3)
        local = _rotations(rng, (6, soma.JOINT_COUNT))
        root = rng.normal(size=(6, 3)) + (0.0, 0.9, 0.0)
        smooth = root + rng.normal(scale=0.05, size=(6, 3))
        smooth[:, 1] = root[:, 1]
        stats = _stats(rng)
        frames = rep.decode(stats.normalize(_features(local, root, smooth)), stats)
        global_rotations, positions = soma.forward_kinematics(local, root)
        np.testing.assert_allclose(frames.global_rotations, global_rotations, atol=1e-9)
        np.testing.assert_allclose(frames.positions, positions, atol=1e-9)
        np.testing.assert_allclose(frames.root_positions, root, atol=1e-9)
        np.testing.assert_allclose(frames.smooth_root, smooth, atol=1e-9)
        np.testing.assert_allclose(
            frames.heading, soma.heading_angle(positions), atol=1e-9
        )
        np.testing.assert_array_equal(frames.contacts[0], (True, False, True, False))
        moved = frames.translated(np.array((1.0, -2.0)))
        np.testing.assert_allclose(
            moved.root_positions - frames.root_positions, [[1.0, 0.0, -2.0]] * 6
        )
        self.assertEqual(frames.slice(2, 4).frame_count, 2)

    def test_decode_rejects_bad_frames(self):
        with self.assertRaises(BackendError):
            rep.decode(np.zeros((3, 10)))
        bad = np.zeros((2, rep.FEATURE_COUNT))
        bad[0, 0] = np.nan
        with self.assertRaises(BackendError):
            rep.decode(bad)


def _rest_pose(offset=(0.0, 0.93, 0.0)) -> np.ndarray:
    return soma.REST_POSITIONS + np.asarray(offset)


class GuideTests(unittest.TestCase):
    def test_full_body_guide_writes_positions_root_and_heading(self):
        pose = _rest_pose((1.0, 0.95, 2.0))
        observed, mask = guides.compile_guides([guides.FullBodyGuide(3, pose)], 10)
        self.assertEqual(int(mask.sum()), 3 + 2 + soma.JOINT_COUNT * 3)
        self.assertFalse(mask[:3].any() or mask[4:].any())
        np.testing.assert_allclose(observed[3, rep.SMOOTH_ROOT], (1.0, 0.95, 2.0))
        np.testing.assert_allclose(
            observed[3, rep.HEADING],
            (np.cos(soma.heading_angle(pose)), np.sin(soma.heading_angle(pose))),
        )
        relative = observed[3, rep.POSITIONS].reshape(soma.JOINT_COUNT, 3)
        np.testing.assert_allclose(
            relative, soma.REST_POSITIONS + (0.0, 0.95, 0.0), atol=1e-12
        )
        self.assertFalse(mask[:, rep.ROTATIONS].any())

    def test_effector_guide_writes_chain_positions_and_base_rotation(self):
        rng = np.random.default_rng(4)
        rotations = _rotations(rng, (soma.JOINT_COUNT,))
        guide = guides.EffectorGuide(
            0,
            frozenset({"LEFT_HAND"}),
            _rest_pose(),
            rotations,
            smooth_root_xz=(0.1, 0.2),
        )
        observed, mask = guides.compile_guides([guide], 4)
        self.assertEqual(int(mask.sum()), 3 + 2 + 2 * 3 + 6)
        hand = soma.JOINT_INDEX["LeftHand"]
        start = rep.ROTATIONS.start + 6 * hand
        np.testing.assert_allclose(
            observed[0, start : start + 6], rep.matrix_to_rot6d(rotations[hand])
        )
        np.testing.assert_allclose(observed[0, [0, 2]], (0.1, 0.2))
        tip = soma.JOINT_INDEX["LeftHandMiddleEnd"]
        position = observed[
            0, rep.POSITIONS.start + 3 * tip : rep.POSITIONS.start + 3 * tip + 3
        ]
        expected = _rest_pose()[tip] - (0.1, 0.0, 0.2)
        np.testing.assert_allclose(position, expected)

    def test_root_guide_and_first_write_wins(self):
        first = guides.RootGuide(2, (5.0, 6.0), heading=0.5)
        second = guides.FullBodyGuide(2, _rest_pose((9.0, 0.9, 9.0)))
        observed, mask = guides.compile_guides([first, second], 5)
        np.testing.assert_allclose(observed[2, [0, 2]], (5.0, 6.0))
        np.testing.assert_allclose(observed[2, rep.HEADING], (np.cos(0.5), np.sin(0.5)))
        # The later full-body guide still adds height and positions, relative
        # to the first root.
        self.assertTrue(mask[2, 1])
        hips = observed[2, rep.POSITIONS.start : rep.POSITIONS.start + 3]
        np.testing.assert_allclose(hips, (4.0, 0.9, 3.0))
        root_only, root_mask = guides.compile_guides(
            [guides.RootGuide(0, (1.0, 1.0))], 1
        )
        self.assertEqual(int(root_mask.sum()), 2)
        self.assertEqual(root_only[0, 0], 1.0)

    def test_sparse_effector_guide_pins_only_what_was_keyed(self):
        pose = _rest_pose((1.0, 0.95, 2.0))
        identity = np.broadcast_to(np.eye(3), (soma.JOINT_COUNT, 3, 3))
        guide = guides.EffectorGuide(
            4,
            frozenset({"LEFT_HAND", "HIPS"}),
            pose,
            identity,
            pin_root_position=False,
            pin_heading=False,
            rotation_only=frozenset({"HIPS"}),
        )
        observed, mask = guides.compile_guides([guide], 10)
        # No root position, height or heading; the hand chain's two joints
        # (positions) and two base rotations (hand, hips).
        self.assertFalse(mask[4, rep.SMOOTH_ROOT].any())
        self.assertFalse(mask[4, rep.HEADING].any())
        self.assertEqual(int(mask.sum()), 2 * 3 + 2 * 6)
        hips = rep.POSITIONS.start + 3 * soma.ROOT
        self.assertFalse(mask[4, hips : hips + 3].any())
        # The hand stays relative to the guide's own root.
        hand = soma.JOINT_INDEX["LeftHand"]
        start = rep.POSITIONS.start + 3 * hand
        np.testing.assert_allclose(
            observed[4, start : start + 3],
            pose[hand] - (pose[soma.ROOT, 0], 0.0, pose[soma.ROOT, 2]),
        )

    def test_guides_are_validated(self):
        with self.assertRaises(InvalidRequest):
            guides.compile_guides([guides.RootGuide(5, (0.0, 0.0))], 5)
        with self.assertRaises(InvalidRequest):
            guides.compile_guides([], 301)
        with self.assertRaises(InvalidRequest):
            guides.EffectorGuide(
                0,
                frozenset({"TAIL"}),
                _rest_pose(),
                np.broadcast_to(np.eye(3), (30, 3, 3)),
            )
        with self.assertRaises(InvalidRequest):
            guides.FullBodyGuide(0, np.zeros((29, 3)))

    def test_translate_and_transition_guides(self):
        guide = guides.FullBodyGuide(1, _rest_pose(), smooth_root_xz=(0.0, 0.0))
        moved = guides.translate_guide(guide, np.array((1.0, 2.0)), frame_shift=4)
        self.assertEqual(moved.frame, 5)
        np.testing.assert_allclose(
            moved.positions - guide.positions, np.broadcast_to((1.0, 0.0, 2.0), (30, 3))
        )
        rng = np.random.default_rng(5)
        local = _rotations(rng, (3, soma.JOINT_COUNT))
        root = np.array([[0.0, 0.9, 0.0], [0.1, 0.9, 0.0], [0.2, 0.9, 0.0]])
        frames = rep.decode(_features(local, root, root))
        transition = guides.guides_from_frames(frames, [0, 1, 2])
        self.assertEqual(len(transition), 6)
        observed, mask = guides.compile_guides(transition, 10)
        self.assertTrue(mask[:3, rep.POSITIONS].all())
        self.assertEqual(int(mask[:3, rep.ROTATIONS].sum()), 3 * 4 * 6)
        np.testing.assert_allclose(observed[1, [0, 2]], (0.1, 0.0))


class _RecordingDenoiser:
    def __init__(self, clean: np.ndarray):
        self.clean = clean
        self.calls: list[sampler.DenoiseInputs] = []

    def __call__(self, inputs: sampler.DenoiseInputs) -> np.ndarray:
        self.calls.append(inputs)
        return np.broadcast_to(self.clean, inputs.motion.shape).copy()


class SamplerTests(unittest.TestCase):
    def test_schedule_matches_upstream_values(self):
        plan = sampler.schedule(25)
        self.assertEqual(plan.timesteps[0], 0)
        self.assertEqual(plan.timesteps[1], 42)
        self.assertEqual(plan.timesteps[-1], 999)
        self.assertTrue(np.all(np.diff(plan.timesteps) > 0))
        self.assertAlmostEqual(float(plan.alphas_cumprod[0]), 0.99996, places=5)
        self.assertAlmostEqual(float(plan.alphas_cumprod[-1]) / 2.43e-9, 1.0, places=1)
        self.assertEqual(plan.alphas_cumprod_prev[0], 1.0)
        self.assertEqual(plan.alphas_cumprod.dtype, np.float32)
        np.testing.assert_array_equal(sampler.schedule(2).timesteps, (0, 999))
        with self.assertRaises(InvalidRequest):
            sampler.schedule(1)

    def test_ddim_last_step_returns_the_prediction(self):
        plan = sampler.schedule(10)
        rng = np.random.default_rng(6)
        noisy = rng.normal(size=(3, rep.FEATURE_COUNT)).astype(np.float32)
        clean = rng.normal(size=(3, rep.FEATURE_COUNT)).astype(np.float32)
        np.testing.assert_allclose(
            sampler.ddim_step(noisy, clean, plan, 0), clean, atol=1e-6
        )
        # At step i, x_t built from (clean, eps) maps to the same eps at i-1.
        index = 5
        noise = rng.normal(size=clean.shape).astype(np.float32)
        alpha = plan.alphas_cumprod[index]
        x_t = np.sqrt(alpha) * clean + np.sqrt(1 - alpha) * noise
        previous = plan.alphas_cumprod_prev[index]
        expected = np.sqrt(previous) * clean + np.sqrt(1 - previous) * noise
        np.testing.assert_allclose(
            sampler.ddim_step(x_t, clean, plan, index), expected, atol=1e-4
        )

    def test_guidance_combination(self):
        branches = np.stack((np.full(4, 3.0), np.full(4, 5.0), np.full(4, 1.0)))
        np.testing.assert_allclose(
            sampler.combine_guidance(branches, 2.0, 0.5), 1 + 2 * 2 + 0.5 * 4
        )

    def test_sampling_builds_three_branches_and_counts_down(self):
        frames = 8
        rng = np.random.default_rng(7)
        clean = rng.normal(size=(frames, rep.FEATURE_COUNT)).astype(np.float32)
        denoiser = _RecordingDenoiser(clean)
        observed = rng.normal(size=(frames, rep.FEATURE_COUNT)).astype(np.float32)
        mask = np.zeros((frames, rep.FEATURE_COUNT), dtype=bool)
        mask[0, :5] = True
        embedding = rng.normal(size=rep.TEXT_WIDTH)
        progress = []
        iterator = sampler.iter_sample(
            denoiser,
            sampler.initial_noise(3, frames),
            observed,
            mask,
            sampler.text_tokens(embedding),
            0.25,
            steps=5,
        )
        try:
            while True:
                progress.append(next(iterator))
        except StopIteration as stop:
            result = stop.value
        np.testing.assert_allclose(result, clean, atol=1e-5)
        self.assertEqual([p.step for p in progress], [1, 2, 3, 4, 5])
        timesteps = [int(call.timestep[0]) for call in denoiser.calls]
        self.assertEqual(timesteps, sorted(timesteps, reverse=True))
        self.assertEqual(timesteps[0], 999)
        first = denoiser.calls[0]
        self.assertEqual(first.motion.shape, (3, frames, rep.FEATURE_COUNT))
        self.assertFalse(
            first.constraint_mask[0].any() or first.constraint_mask[2].any()
        )
        np.testing.assert_array_equal(first.constraint_mask[1], mask)
        np.testing.assert_allclose(
            first.text_embedding[0, 0], embedding.astype(np.float32)
        )
        self.assertFalse(
            first.text_embedding[1:].any() or first.text_embedding[0, 1:].any()
        )
        self.assertTrue(first.motion_valid.all())
        np.testing.assert_allclose(first.first_heading_angle, 0.25)

    def test_sampling_is_reproducible_and_cancellable(self):
        noise = sampler.initial_noise(11, 4)
        np.testing.assert_array_equal(noise, sampler.initial_noise(11, 4))
        self.assertFalse(np.array_equal(noise, sampler.initial_noise(12, 4)))
        denoiser = _RecordingDenoiser(
            np.zeros((4, rep.FEATURE_COUNT), dtype=np.float32)
        )
        args = (
            denoiser,
            noise,
            np.zeros_like(noise),
            np.zeros(noise.shape, dtype=bool),
            sampler.text_tokens(None),
            0.0,
        )
        with self.assertRaises(Cancelled):
            sampler.sample(*args, steps=4, cancelled=lambda: len(denoiser.calls) >= 2)
        self.assertEqual(len(denoiser.calls), 2)
        with self.assertRaises(InvalidRequest):
            sampler.sample(
                lambda inputs: np.zeros((2, 4, rep.FEATURE_COUNT)), *args[1:], steps=3
            )


if __name__ == "__main__":
    unittest.main()
