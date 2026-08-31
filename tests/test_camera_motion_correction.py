from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tests.blender_test_support import load_addon


class CameraMotionCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_addon().camera_motion_correction

    def _inputs(self, xs, *, times=None, depths=None):
        times = times or [index * 50 for index in range(len(xs))]
        depths = depths or [2.0] * len(xs)
        return tuple(
            self.module.CameraMotionInput(
                timestamp_ms=int(timestamp),
                root_translation=(float(x), 0.0, float(depth)),
                focal_length=1000.0,
            )
            for timestamp, x, depth in zip(times, xs, depths, strict=True)
        )

    def _track(self, inputs, translations, *, valid=None, cuts=None):
        valid = valid or [True] * len(inputs)
        cuts = cuts or [False] * len(inputs)
        return self.module.CameraMotionTrack(
            width=1000,
            height=600,
            frames=tuple(
                self.module.CameraTrackFrame(
                    timestamp_ms=item.timestamp_ms,
                    translation_x=float(translation),
                    confidence=1.0 if is_valid else 0.0,
                    tracks=100 if is_valid else 0,
                    inliers=80 if is_valid else 0,
                    residual=0.2 if is_valid else float("inf"),
                    valid=is_valid,
                    cut=cut,
                )
                for item, translation, is_valid, cut in zip(
                    inputs, translations, valid, cuts, strict=True
                )
            ),
        )

    def test_camera_pan_is_removed_from_lateral_root_only(self) -> None:
        inputs = self._inputs((0.0, 0.1, 0.2, 0.3))
        track = self._track(inputs, (0.0, 50.0, 50.0, 50.0))
        result = self.module.correction_offsets(inputs, track)
        self.assertEqual(result.corrected_intervals, 1)
        for expected, actual in zip(
            (0.0, -0.1, -0.2, -0.3), result.offsets_x, strict=True
        ):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_real_lateral_motion_and_forward_depth_are_preserved(self) -> None:
        lateral = self._inputs((0.0, 0.1, 0.2, 0.3))
        static_track = self._track(lateral, (0.0, 0.0, 0.0, 0.0))
        result = self.module.correction_offsets(lateral, static_track)
        self.assertEqual(result.offsets_x, (0.0, 0.0, 0.0, 0.0))

        forward = self._inputs((0.1, 0.1, 0.1, 0.1), depths=(2.2, 2.0, 1.8, 1.6))
        result = self.module.correction_offsets(
            forward, self._track(forward, (0.0, 0.0, 0.0, 0.0))
        )
        self.assertEqual(result.offsets_x, (0.0, 0.0, 0.0, 0.0))

    def test_short_gaps_interpolate_and_long_gaps_hold(self) -> None:
        short = self._inputs((0.0, 0.1, 0.2, 0.3))
        result = self.module.correction_offsets(
            short,
            self._track(
                short,
                (0.0, 50.0, 0.0, 50.0),
                valid=(True, True, False, True),
            ),
        )
        self.assertAlmostEqual(result.offsets_x[2], -0.2, places=6)

        long = self._inputs((0.0, 0.1, 0.2, 0.3), times=(0, 100, 700, 800))
        result = self.module.correction_offsets(
            long,
            self._track(
                long,
                (0.0, 50.0, 0.0, 50.0),
                valid=(True, True, False, True),
            ),
        )
        self.assertAlmostEqual(result.offsets_x[2], result.offsets_x[1], places=6)
        self.assertAlmostEqual(result.offsets_x[3], result.offsets_x[2], places=6)

    def test_offline_sidecar_is_checksum_validated_and_immutable(self) -> None:
        inputs = self._inputs((0.0, 0.1, 0.2))
        track = self._track(inputs, (0.0, 50.0, 50.0))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "camera_motion" / "current"
            key = self.module.cache_key(
                {"path": "video.mp4", "size": 123, "mtime_ns": 456}, inputs
            )
            self.module.store_sidecar(directory, key, track, np)
            restored = self.module.load_sidecar(directory, key, np)
            self.assertIsNotNone(restored)
            self.assertEqual(restored.frames, track.frames)

            manifest = json.loads(
                (directory / "manifest.json").read_text(encoding="utf8")
            )
            payload = directory / manifest["payload"]
            payload.write_bytes(payload.read_bytes() + b"corrupt")
            self.assertIsNone(self.module.load_sidecar(directory, key, np))

    def test_cache_key_rejects_timestamp_or_source_changes(self) -> None:
        source = {"path": "video.mp4", "size": 123, "mtime_ns": 456}
        first = self._inputs((0.0, 0.1, 0.2))
        second = self._inputs((0.0, 0.1, 0.2), times=(0, 51, 100))
        self.assertNotEqual(
            self.module.cache_key(source, first),
            self.module.cache_key(source, second),
        )
        changed = dict(source, mtime_ns=457)
        self.assertNotEqual(
            self.module.cache_key(source, first),
            self.module.cache_key(changed, first),
        )

    def test_masked_sparse_flow_recovers_background_pan(self) -> None:
        rng = np.random.default_rng(10)
        previous = np.zeros((480, 640), dtype=np.uint8)
        for x, y in rng.integers((5, 5), (635, 475), size=(1800, 2)):
            cv2.circle(previous, (int(x), int(y)), 1, 255, thickness=-1)
        current = cv2.warpAffine(
            previous,
            np.asarray(((1.0, 0.0, 12.0), (0.0, 1.0, -3.0))),
            (640, 480),
        )
        # A changing foreground person must not influence the background fit.
        cv2.rectangle(previous, (220, 40), (420, 450), 80, thickness=-1)
        cv2.circle(current, (360, 180), 100, 190, thickness=-1)
        mask = np.full(previous.shape, 255, dtype=np.uint8)
        cv2.rectangle(mask, (200, 20), (440, 470), 0, thickness=-1)
        step = self.module._estimate_step(previous, current, mask, mask, cv2, np)
        self.assertIsNotNone(step)
        self.assertAlmostEqual(step.translation_x, 12.0, delta=0.5)
        self.assertAlmostEqual(step.translation_y, -3.0, delta=0.5)
        self.assertGreaterEqual(step.inliers, self.module.MIN_INLIERS)


if __name__ == "__main__":
    unittest.main()
