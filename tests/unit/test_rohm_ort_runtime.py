from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from media_to_pose.backends.rohm import runtime as rohm_ort_runtime
from media_to_pose.core.errors import BackendError
from media_to_pose.core.motion import rohm_format


class RohmOrtRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = rohm_ort_runtime

    def test_visibility_vector_matches_upstream_channel_mapping(self) -> None:
        visible = np.ones((3, 22), dtype=bool)
        visible[1, 7] = False
        mask = rohm_format.visibility_vector(visible)
        self.assertEqual(mask.shape, (3, 294))
        self.assertEqual(mask[1, 22 + 7 * 3 : 22 + 8 * 3].tolist(), [0.0] * 3)
        self.assertEqual(mask[1, 154 + 6 * 6 : 154 + 7 * 6].tolist(), [0.0] * 6)
        self.assertEqual(mask[1, 290:292].tolist(), [0.0, 0.0])
        self.assertEqual(mask[0, 290:294].tolist(), [1.0] * 4)

    def test_visibility_rejects_wrong_joint_count(self) -> None:
        with self.assertRaisesRegex(BackendError, r"\[T, 22\]"):
            rohm_format.visibility_vector(np.ones((5, 21), dtype=bool))

    def test_diffusion_iterator_yields_after_every_onnx_call(self) -> None:
        class Session:
            def __init__(self):
                self.calls = 0

            def run(self, _outputs, feeds):
                self.calls += 1
                return [np.zeros_like(feeds["x_t"], dtype=np.float32)]

        session = Session()
        scheduler = {
            "posterior_mean_coef1": np.ones(3, dtype=np.float32),
            "posterior_mean_coef2": np.zeros(3, dtype=np.float32),
            "posterior_variance": np.zeros(3, dtype=np.float32),
        }
        iterator = self.runtime.RohmSession._iter_sample(
            session,
            scheduler,
            (1, 2),
            np.zeros((1, 2), dtype=np.float32),
            np.random.RandomState(10),
        )
        updates = []
        while True:
            try:
                updates.append(next(iterator))
            except StopIteration as stopped:
                result = stopped.value
                break

        self.assertEqual(updates, [(1, 3), (2, 3), (3, 3)])
        self.assertEqual(session.calls, 3)
        np.testing.assert_array_equal(result, np.zeros((1, 2), dtype=np.float32))

    def test_bundle_rejects_unmanifested_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text(
                '{"schema":1,"assets":{}}',
                encoding="utf-8",
            )
            (root / "normalizer.pkl").write_bytes(b"unsafe")
            with self.assertRaisesRegex(self.runtime.RohmRuntimeError, "unsafe file"):
                self.runtime.load_bundle(root)


if __name__ == "__main__":
    unittest.main()
