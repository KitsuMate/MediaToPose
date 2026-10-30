from __future__ import annotations

import unittest

import numpy as np

from media_to_pose.core.errors import InvalidRequest
from media_to_pose.core.motion import guide_warp, retime, soma
from media_to_pose.core.motion import quaternions as quat
from media_to_pose.core.motion.kimodo import (
    generate,
    guides,
    prompt,
    rep,
    sampler,
    windows,
)


def _rotations(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    matrices, _ = np.linalg.qr(rng.normal(size=(*shape, 3, 3)))
    signs = np.sign(np.linalg.det(matrices))
    matrices[..., :, 0] *= signs[..., None]
    return matrices


def _rest_features(count: int) -> np.ndarray:
    """Unnormalised features of a T-pose standing still at the origin."""
    features = np.zeros((count, rep.FEATURE_COUNT))
    features[:, rep.SMOOTH_ROOT] = (0.0, 0.93, 0.0)
    features[:, rep.HEADING] = (1.0, 0.0)
    features[:, rep.POSITIONS] = (soma.REST_POSITIONS + (0.0, 0.93, 0.0)).reshape(-1)
    features[:, rep.ROTATIONS] = rep.matrix_to_rot6d(
        np.broadcast_to(np.eye(3), (30, 3, 3))
    ).reshape(-1)
    return features


class QuaternionTests(unittest.TestCase):
    def test_matrix_round_trip_and_algebra(self):
        rng = np.random.default_rng(1)
        matrices = _rotations(rng, (50,))
        values = quat.from_matrix(matrices)
        np.testing.assert_allclose(quat.to_matrix(values), matrices, atol=1e-12)
        self.assertTrue(np.all(values[:, 0] >= 0.0))
        np.testing.assert_allclose(
            quat.multiply(values, quat.inverse(values)),
            np.broadcast_to(quat.IDENTITY, values.shape),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            quat.to_matrix(quat.multiply(values[:-1], values[1:])),
            matrices[:-1] @ matrices[1:],
            atol=1e-12,
        )

    def test_slerp_and_continuity(self):
        quarter = quat.from_matrix(soma.rotation_y(np.pi / 2))
        half = quat.slerp(quat.IDENTITY, quarter, 0.5)
        self.assertAlmostEqual(float(quat.angle(half)), np.pi / 4)
        np.testing.assert_allclose(
            quat.slerp(quat.IDENTITY, -quarter, 1.0), quarter, atol=1e-12
        )
        flipped = np.stack((quarter, -quarter, quarter))
        self.assertTrue(
            np.all(
                np.sum(
                    quat.continuous(flipped)[1:] * quat.continuous(flipped)[:-1],
                    axis=-1,
                )
                > 0
            )
        )


class RetimeTests(unittest.TestCase):
    def test_scene_frames_map_to_thirty_fps(self):
        self.assertEqual(retime.model_index(120, 1, 24.0), 149)
        self.assertEqual(retime.model_frame_count(1, 120, 24.0), 150)
        self.assertEqual(retime.model_frame_count(1, 120, 30.0), 120)
        with self.assertRaises(InvalidRequest):
            retime.model_frame_count(10, 5, 24.0)
        with self.assertRaises(InvalidRequest):
            retime.model_index(1, 1, 0.0)

    def test_resample_interpolates_between_model_frames(self):
        features = _rest_features(4)
        features[:, rep.SMOOTH_ROOT.start] = (0.0, 1.0, 2.0, 3.0)
        turn = rep.matrix_to_rot6d(soma.rotation_y(np.pi / 2))
        features[3, rep.ROTATIONS.start : rep.ROTATIONS.start + 6] = turn
        frames = rep.decode(features)
        same = retime.resample(frames, [0, 1, 2, 3], 0, 30.0)
        np.testing.assert_allclose(
            same.root_positions, frames.root_positions, atol=1e-12
        )
        doubled = retime.resample(frames, [0, 1, 2, 5, 6, 7, 99], 0, 60.0)
        np.testing.assert_allclose(doubled.root_positions[:3, 0], (0.0, 0.5, 1.0))
        # Halfway between identity and a quarter turn of the hips.
        hips = quat.from_matrix(doubled.global_rotations[3, soma.ROOT])
        self.assertAlmostEqual(float(quat.angle(hips)), np.pi / 4, places=6)
        np.testing.assert_allclose(
            doubled.root_positions[-1], frames.root_positions[-1]
        )


class GuideWarpTests(unittest.TestCase):
    def test_vectors_hit_guides_and_fade(self):
        frames = np.arange(0, 31, dtype=float)
        values = np.zeros((31, 2))
        warped, deltas = guide_warp.warp_vectors(
            frames, values, {10.0: np.array([1.0, 0.0]), 20.0: np.array([1.0, 2.0])}, 3
        )
        np.testing.assert_allclose(warped[10], (1.0, 0.0))
        np.testing.assert_allclose(warped[20], (1.0, 2.0))
        np.testing.assert_allclose(warped[15], (0.0, 0.0))  # gap wider than 2 x 3
        self.assertTrue(0.0 < warped[11, 0] < 1.0)
        np.testing.assert_allclose(warped[:6], 0.0)
        np.testing.assert_allclose(deltas, [[1.0, 0.0], [1.0, 2.0]])
        # The fade is flat at the guide, so the warp adds no kink.
        fine = np.arange(0.0, 30.05, 0.1)
        smooth, _ = guide_warp.warp_vectors(
            fine, np.zeros((len(fine), 1)), {10.0: np.ones(1)}, 3
        )
        slope = np.abs(np.diff(smooth[:, 0]))
        at_guide = int(np.argmin(np.abs(fine - 10.0)))
        self.assertLess(slope[at_guide], 0.1 * slope.max())
        self.assertLess(slope[at_guide - 1], 0.1 * slope.max())

    def test_vectors_interpolate_across_gaps_without_blend_limit(self):
        frames = np.arange(0, 21, dtype=float)
        warped, _ = guide_warp.warp_vectors(
            frames, np.zeros((21, 1)), {5.0: np.ones(1), 15.0: np.ones(1)}, None
        )
        np.testing.assert_allclose(warped[5:16], 1.0)
        np.testing.assert_allclose(warped[:5], 1.0)  # held outside
        with self.assertRaises(InvalidRequest):
            guide_warp.warp_vectors(frames, np.zeros((21, 1)), {5.5: np.ones(1)}, None)

    def test_quaternions_hit_guides_exactly(self):
        rng = np.random.default_rng(2)
        frames = np.arange(0, 20, dtype=float)
        values = quat.continuous(quat.from_matrix(_rotations(rng, (20,))))
        target = quat.from_matrix(soma.rotation_y(0.7))
        warped, angles = guide_warp.warp_quaternions(frames, values, {8.0: target}, 4)
        np.testing.assert_allclose(
            quat.to_matrix(warped[8]), soma.rotation_y(0.7), atol=1e-9
        )
        np.testing.assert_allclose(
            quat.to_matrix(warped[:3]), quat.to_matrix(values[:3]), atol=1e-9
        )
        self.assertEqual(angles.shape, (1,))
        unchanged, none = guide_warp.warp_quaternions(frames, values, {}, 4)
        np.testing.assert_allclose(unchanged, values)
        self.assertEqual(len(none), 0)


class WindowTests(unittest.TestCase):
    def test_short_generation_is_one_window(self):
        self.assertEqual(windows.plan_windows(150), [windows.Window(0, 0, 150, False)])

    def test_long_generation_splits_with_transitions(self):
        plan = windows.plan_windows(700)
        self.assertEqual(len(plan), 3)
        self.assertFalse(plan[0].transition)
        for before, after in zip(plan, plan[1:], strict=False):
            self.assertTrue(after.transition)
            self.assertEqual(after.start, before.stop - windows.TRANSITION_FRAMES)
        self.assertEqual(plan[-1].stop, 700)
        self.assertTrue(all(w.frame_count <= rep.MAX_WINDOW_FRAMES for w in plan))

    def test_segments_never_share_a_window(self):
        segments = (windows.Segment(0, 100, "walk"), windows.Segment(100, 250, "run"))
        plan = windows.plan_windows(250, segments)
        self.assertEqual(
            [(w.segment, w.start, w.stop, w.transition) for w in plan],
            [(0, 0, 100, False), (1, 95, 250, True)],
        )
        with self.assertRaises(InvalidRequest):
            windows.plan_windows(
                250, (windows.Segment(0, 100), windows.Segment(120, 250))
            )
        with self.assertRaises(InvalidRequest):
            windows.plan_windows(1)

    def test_every_length_fits_the_model(self):
        for total in range(2, 1800):
            plan = windows.plan_windows(total)
            self.assertTrue(
                all(w.frame_count <= rep.MAX_WINDOW_FRAMES for w in plan), total
            )
            self.assertTrue(all(w.transition for w in plan[1:]), total)
        segments = (windows.Segment(0, 200, "walk"), windows.Segment(200, 795, "run"))
        plan = windows.plan_windows(795, segments)
        self.assertTrue(all(w.frame_count <= rep.MAX_WINDOW_FRAMES for w in plan))

    def test_first_segment_must_cover_the_transition(self):
        segments = (windows.Segment(0, 3, "wave"), windows.Segment(3, 100, "walk"))
        with self.assertRaises(InvalidRequest):
            windows.plan_windows(100, segments)

    def test_crossfade_keeps_length_and_ends(self):
        previous, current = np.zeros((10, 2)), np.ones((8, 2))
        result = windows.crossfade(previous, current, 5)
        self.assertEqual(len(result), 13)
        np.testing.assert_allclose(result[:5], 0.0)
        np.testing.assert_allclose(result[-3:], 1.0)
        self.assertTrue(np.all(np.diff(result[5:10, 0]) > 0))


def _copy_denoiser(base_normalised: np.ndarray):
    """Predict the guided values where the guide branch is masked, else a
    standing T-pose: the result shows exactly what the guides asked for."""

    def denoise(inputs: sampler.DenoiseInputs) -> np.ndarray:
        count = inputs.motion.shape[1]
        base = np.broadcast_to(base_normalised[0], (count, rep.FEATURE_COUNT))
        guided = np.where(inputs.constraint_mask[1], inputs.observed_motion[1], base)
        return np.stack((base, guided, base)).astype(np.float32)

    return denoise


class GenerateTests(unittest.TestCase):
    def setUp(self):
        self.stats = rep.FeatureStats(
            np.zeros(rep.FEATURE_COUNT), np.ones(rep.FEATURE_COUNT)
        )
        self.denoise = _copy_denoiser(self.stats.normalize(_rest_features(1)))

    def _request(self, count: int, guide_list, **kwargs) -> generate.GenerationRequest:
        return generate.GenerationRequest(
            frame_count=count,
            stats=self.stats,
            guides=tuple(guide_list),
            steps=4,
            text_weight=0.0,
            guide_weight=1.0,
            **kwargs,
        )

    def test_guides_land_where_asked_across_windows(self):
        target = soma.REST_POSITIONS + (2.0, 0.95, 3.0)
        request = self._request(
            400, [guides.FullBodyGuide(350, target)], start_xz=(0.5, 0.0)
        )
        progress = []
        iterator = generate.iter_generate(self.denoise, request)
        try:
            while True:
                progress.append(next(iterator))
        except StopIteration as stop:
            motion = stop.value
        self.assertEqual(len(motion.windows), 2)
        self.assertEqual(motion.frames.frame_count, 400)
        np.testing.assert_allclose(
            motion.frames.root_positions[350], (2.0, 0.95, 3.0), atol=1e-4
        )
        # Unguided frames stand where the generation started.
        np.testing.assert_allclose(
            motion.frames.root_positions[10], (0.5, 0.93, 0.0), atol=1e-4
        )
        self.assertEqual(progress[-1].window, 1)
        self.assertAlmostEqual(progress[-1].fraction, 1.0)

    def test_frame_zero_guide_sets_the_start(self):
        start = soma.REST_POSITIONS @ soma.rotation_y(0.3).T + (1.0, 0.9, -1.0)
        request = self._request(30, [guides.FullBodyGuide(0, start)])
        motion = generate.generate(self.denoise, request)
        np.testing.assert_allclose(
            motion.frames.root_positions[0], (1.0, 0.9, -1.0), atol=1e-4
        )

    def test_prompts_need_embeddings(self):
        segments = (
            windows.Segment(0, 20, "A person walks."),
            windows.Segment(20, 40, ""),
        )
        with self.assertRaises(InvalidRequest):
            generate.generate(self.denoise, self._request(40, [], segments=segments))
        embedding = {"A person walks.": np.ones(rep.TEXT_WIDTH)}
        motion = generate.generate(
            self.denoise, self._request(40, [], segments=segments, embeddings=embedding)
        )
        self.assertEqual(motion.frames.frame_count, 40)
        self.assertEqual(generate.window_prompts(segments), ["A person walks."])

    def test_requests_are_validated(self):
        with self.assertRaises(InvalidRequest):
            self._request(10, [guides.RootGuide(10, (0.0, 0.0))])
        with self.assertRaises(InvalidRequest):
            self._request(1, [])


def _fake_tokenize(text: str, special: bool) -> list[int]:
    ids = [ord(char) for char in text.replace(prompt.END_OF_TURN, "\x01")]
    return ([128000] if special else []) + ids


class PromptTests(unittest.TestCase):
    def test_sanitize_matches_upstream_examples(self):
        cases = {
            " A person is    walking.": "A person is walking.",
            "someone go forward": "Someone go forward.",
            "jumping)": "Jumping).",
            "-go": "Go.",
            "blocasdji  -----": "Blocasdji.",
            "": "",
            "a person walks. then RUNS to NYC!  and stops?": "A person walks. Then runs to nyc! And stops?",
        }
        for raw, expected in cases.items():
            self.assertEqual(prompt.sanitize(raw), expected)

    def test_encoder_inputs_pad_left_and_pool_the_prompt(self):
        inputs = prompt.encoder_inputs("walk", _fake_tokenize)
        self.assertEqual(inputs.text, "Walk.")
        self.assertEqual(inputs.input_ids.shape, (1, prompt.MAX_TOKENS))
        used = int(inputs.attention_mask.sum())
        self.assertEqual(inputs.input_ids[0, 0], prompt.PAD_TOKEN_ID)
        self.assertEqual(inputs.input_ids[0, -used], 128000)
        self.assertEqual(int(inputs.pooling_mask.sum()), len("Walk.") + 1)
        self.assertTrue(np.all(inputs.pooling_mask[0, -6:] == 1))
        self.assertFalse(inputs.truncated)

    def test_long_prompts_are_truncated_by_words(self):
        inputs = prompt.encoder_inputs("a person walks " * 10, _fake_tokenize)
        self.assertTrue(inputs.truncated)
        self.assertLessEqual(int(inputs.attention_mask.sum()), prompt.MAX_TOKENS)
        with self.assertRaises(InvalidRequest):
            prompt.encoder_inputs("   ", _fake_tokenize)


if __name__ == "__main__":
    unittest.main()
