"""Parity of the NumPy Kimodo code with upstream, from recorded reference runs.

`reference_60f.npz` is one 25-step separated-CFG run of upstream Kimodo
(fp32, CPU) with a root path, a full-body guide and a hands guide. Trace
arrays are in call order: step 0 is the first denoiser call (timestep 999).
`reference_tokens.npz` holds the upstream tokenizer ids for 30 strings.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from media_to_pose.core.motion.kimodo import guides, rep, sampler

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "kimodo"


def _load(name: str) -> dict[str, np.ndarray]:
    with np.load(FIXTURES / name, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


class ReferenceSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ref = _load("reference_60f.npz")

    def test_layout(self) -> None:
        ref = self.ref
        frames, steps = 60, int(ref["steps"])
        self.assertEqual(int(ref["schema_version"]), 1)
        self.assertEqual(ref["noise"].shape, (frames, rep.FEATURE_COUNT))
        self.assertEqual(ref["text_embedding"].shape, (rep.TEXT_WIDTH,))
        self.assertEqual(
            ref["trace_predicted_clean"].shape, (steps, frames, rep.FEATURE_COUNT)
        )
        self.assertEqual(
            ref["check_branches"].shape,
            (3, sampler.BRANCHES, frames, rep.FEATURE_COUNT),
        )
        self.assertEqual(ref["constraint_mask"].dtype, np.bool_)
        self.assertFalse(np.any(ref["observed_motion"][~ref["constraint_mask"]]))

    def test_timesteps_are_base_steps(self) -> None:
        plan = sampler.schedule(int(self.ref["steps"]))
        np.testing.assert_array_equal(plan.timesteps[::-1], self.ref["timesteps"])

    def test_single_steps(self) -> None:
        ref = self.ref
        plan = sampler.schedule(int(ref["steps"]))
        for row, step in enumerate(ref["check_steps"]):
            branches = ref["check_branches"][row].astype(np.float32)
            predicted = sampler.combine_guidance(branches, *ref["cfg_weights"])
            self.assertGreater(
                _cosine(predicted, ref["trace_predicted_clean"][step]), 0.9999
            )
            index = plan.steps - 1 - int(step)
            moved = sampler.ddim_step(
                ref["check_motion_in"][row], predicted, plan, index
            )
            self.assertGreater(_cosine(moved, ref["check_motion_out"][row]), 0.9999)

    def test_full_trace(self) -> None:
        ref = self.ref
        trace = ref["trace_predicted_clean"].astype(np.float32)
        seen: list[np.ndarray] = []

        def recorded(inputs: sampler.DenoiseInputs) -> np.ndarray:
            seen.append(inputs.motion[0].copy())
            return np.broadcast_to(trace[len(seen) - 1], inputs.motion.shape)

        final = sampler.sample(
            recorded,
            ref["noise"],
            ref["observed_motion"],
            ref["constraint_mask"],
            sampler.text_tokens(ref["text_embedding"]),
            float(ref["first_heading_angle"]),
            int(ref["steps"]),
        )
        for row, step in enumerate(ref["check_steps"]):
            self.assertGreater(_cosine(seen[step], ref["check_motion_in"][row]), 0.9999)
        self.assertGreater(_cosine(final, ref["check_motion_out"][-1]), 0.9999)

    def test_guide_mask(self) -> None:
        ref = self.ref
        compiled = [
            guides.RootGuide(int(frame), xz)
            for frame, xz in zip(ref["root_frames"], ref["root_xz"], strict=True)
        ]
        compiled.append(
            guides.FullBodyGuide(
                int(ref["full_body_frame"]), ref["full_body_positions"]
            )
        )
        compiled.append(
            guides.EffectorGuide(
                int(ref["hands_frame"]),
                frozenset({"LEFT_HAND", "RIGHT_HAND"}),
                ref["hands_positions"],
                ref["hands_rotations"],
            )
        )
        _, mask = guides.compile_guides(compiled, len(ref["noise"]))
        np.testing.assert_array_equal(mask, ref["constraint_mask"])

    def test_decode(self) -> None:
        ref = self.ref
        frames = rep.decode(ref["final_features"])
        np.testing.assert_allclose(
            frames.positions, ref["decoded_posed_joints"], atol=1e-4
        )
        np.testing.assert_allclose(
            frames.global_rotations, ref["decoded_global_rotations"], atol=1e-5
        )
        np.testing.assert_allclose(
            frames.root_positions, ref["decoded_root_positions"], atol=1e-5
        )
        np.testing.assert_allclose(
            frames.smooth_root, ref["decoded_smooth_root"], atol=1e-6
        )
        heading = ref["decoded_heading"]
        np.testing.assert_allclose(
            frames.heading, np.arctan2(heading[:, 1], heading[:, 0]), atol=1e-5
        )
        np.testing.assert_array_equal(frames.contacts, ref["decoded_foot_contacts"])


class ReferenceTokenTests(unittest.TestCase):
    def test_tokens(self) -> None:
        tokens = _load("reference_tokens.npz")
        count = len(tokens["strings"])
        self.assertEqual(count, 30)
        self.assertEqual(tokens["offsets"].shape, (count + 1,))
        self.assertEqual(int(tokens["offsets"][-1]), len(tokens["ids"]))
        self.assertTrue(np.all(tokens["pooled"] <= np.diff(tokens["offsets"])))
        first = tokens["ids"][tokens["offsets"][0] : tokens["offsets"][1]]
        self.assertEqual(int(first[0]), int(tokens["bos_id"]))
        self.assertEqual(int(first[-1]), int(tokens["eot_id"]))

    def test_embeddings(self) -> None:
        embeddings = _load("reference_embeddings.npz")
        vectors = embeddings["embeddings"]
        self.assertEqual(vectors.shape, (len(embeddings["prompts"]), rep.TEXT_WIDTH))
        self.assertTrue(np.all(np.isfinite(vectors)))
        # The parity run used the int4 embedding of the first prompt.
        ref = _load("reference_60f.npz")
        self.assertEqual(str(ref["prompt"]), str(embeddings["prompts"][0]))
        np.testing.assert_array_equal(ref["text_embedding"], vectors[0])


if __name__ == "__main__":
    unittest.main()
