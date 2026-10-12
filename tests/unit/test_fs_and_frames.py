"""Atomic writers and the pickle-free frame codec."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from media_to_pose.core import fs
from media_to_pose.core.cache import frames
from media_to_pose.core.types import (
    CameraIntrinsicsObservation,
    CanonicalFrame,
    JointFrameObservation,
    Landmark,
    RootMotionObservation,
)


def sample_frame(timestamp_ms: int = 40) -> CanonicalFrame:
    return CanonicalFrame(
        timestamp_ms=timestamp_ms,
        body={"NOSE": Landmark((0.1, 0.2, 0.3), (0.5, 0.4), 0.9)},
        hands={"LEFT": {"WRIST": Landmark((1.0, 2.0, 3.0), (0.1, 0.2), 0.8)}},
        auxiliary={"MHR_NECK": Landmark((0.0, 1.0, 0.0), (0.0, 0.0), 0.7)},
        joint_frames={"HEAD": JointFrameObservation((1, 0, 0, 0, 1, 0, 0, 0, 1), 0.6)},
        face_landmarks=[Landmark((0.0, 0.0, 0.1), (0.3, 0.3), 1.0)],
        blendshapes={"jawOpen": 0.25},
        camera=CameraIntrinsicsObservation(
            640, 480, 500.0, 500.0, 320.0, 240.0, 65.0, 51.0, source="MANUAL"
        ),
        root_motion=RootMotionObservation((0.0, 0.1, 2.0), focal_length=500.0),
        warnings=["test warning"],
    )


class AtomicWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_json_replaces_the_target_without_leftovers(self) -> None:
        path = self.root / "nested" / "value.json"
        fs.atomic_json(path, {"b": 1, "a": 2})
        fs.atomic_json(path, {"c": 3})
        self.assertEqual(json.loads(path.read_text("utf8")), {"c": 3})
        self.assertEqual([item.name for item in path.parent.iterdir()], ["value.json"])

    def test_failed_write_keeps_the_old_file(self) -> None:
        path = self.root / "value.json"
        fs.atomic_json(path, {"ok": True})
        with self.assertRaises(TypeError):
            fs.atomic_json(path, {"bad": object()})
        self.assertEqual(json.loads(path.read_text("utf8")), {"ok": True})
        self.assertEqual(len(list(self.root.iterdir())), 1)

    def test_npz_refuses_object_arrays(self) -> None:
        with self.assertRaises(TypeError):
            fs.atomic_npz(self.root / "a.npz", {"x": np.array([object()])})
        fs.atomic_npz(self.root / "b.npz", {"x": np.arange(3)})
        with np.load(self.root / "b.npz", allow_pickle=False) as archive:
            self.assertEqual(archive["x"].tolist(), [0, 1, 2])

    def test_replace_directory_swaps_and_cleans_up(self) -> None:
        target = self.root / "current"
        target.mkdir()
        (target / "old.txt").write_text("old")
        staging = self.root / "staging"
        staging.mkdir()
        (staging / "new.txt").write_text("new")
        fs.replace_directory(staging, target, self.root / ".previous")
        self.assertEqual(sorted(item.name for item in target.iterdir()), ["new.txt"])
        self.assertFalse((self.root / ".previous").exists())
        self.assertFalse(staging.exists())

    def test_replace_directory_restores_the_old_target_on_failure(self) -> None:
        target = self.root / "current"
        target.mkdir()
        (target / "old.txt").write_text("old")
        staging = self.root / "missing-staging"
        with self.assertRaises(OSError):
            fs.replace_directory(staging, target, self.root / ".previous")
        self.assertTrue((target / "old.txt").exists())

    def test_read_json_tolerates_missing_and_malformed_files(self) -> None:
        self.assertIsNone(fs.read_json(self.root / "missing.json"))
        (self.root / "bad.json").write_text("{")
        self.assertIsNone(fs.read_json(self.root / "bad.json"))


class FrameCodecTests(unittest.TestCase):
    def test_lossless_round_trip_keeps_the_face_mesh(self) -> None:
        frame = sample_frame()
        restored = frames.frame_from_plain(
            json.loads(json.dumps(frames.frame_to_plain(frame, compact=False)))
        )
        self.assertEqual(restored, frame)

    def test_compact_round_trip_moves_the_face_mesh_to_the_overlay(self) -> None:
        frame = sample_frame()
        restored = frames.frame_from_plain(frames.frame_to_plain(frame))
        self.assertEqual(restored.face_landmarks, [])
        self.assertEqual(restored.face_overlay, [(0.3, 0.3)])
        self.assertEqual(restored.body, frame.body)
        self.assertEqual(restored.camera, frame.camera)
        self.assertEqual(restored.root_motion, frame.root_motion)
        self.assertEqual(frame.face_landmarks[0].image, (0.3, 0.3))

    def test_chunks_are_npz_without_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunk.npz"
            source = [sample_frame(index * 40) for index in range(5)]
            frames.write_chunk(path, source, compact=False)
            with mock.patch("pickle.loads", side_effect=AssertionError("pickle")):
                restored = frames.read_chunk(path)
        self.assertEqual(restored, source)


if __name__ == "__main__":
    unittest.main()


class FrameSpoolTests(unittest.TestCase):
    def test_spool_round_trips_across_chunks_and_cleans_up(self) -> None:
        from media_to_pose.core.cache.stream import FrameSpool

        source = [sample_frame(index * 40) for index in range(7)]
        with FrameSpool(chunk_frames=3) as spool:
            for frame in source:
                spool.append(frame)
            self.assertEqual(len(spool), 7)
            self.assertEqual(list(spool), source)
            self.assertEqual(list(spool), source)
            directory = spool.directory
            self.assertEqual(len(list(directory.glob("chunk_*.npz"))), 3)
        self.assertFalse(directory.exists())
        with self.assertRaises(ValueError):
            spool.append(source[0])
