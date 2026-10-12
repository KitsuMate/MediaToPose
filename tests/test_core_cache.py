from __future__ import annotations

import importlib
import json
import queue
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import bpy

from tests.blender_test_support import (
    fixture_manifest,
    fixture_path,
    load_addon,
    verify_fixture,
)


class FixtureTests(unittest.TestCase):
    def test_manifest_and_hashes(self) -> None:
        manifest = fixture_manifest()
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["license"]["attribution"], "@Gartix")
        for fixture in manifest["fixture"]:
            self.assertTrue(verify_fixture(fixture["id"]).is_file())

    def test_image_dimensions(self) -> None:
        expected = {
            "sam_ref_pose_stylized": (3000, 4000),
            "mediapipe_ref_pose_realistic": (896, 1200),
        }
        for fixture_id, dimensions in expected.items():
            image = bpy.data.images.load(
                str(fixture_path(fixture_id)), check_existing=False
            )
            try:
                self.assertEqual(tuple(image.size), dimensions)
            finally:
                bpy.data.images.remove(image)


class MediaFrameSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        addon = load_addon()
        cls.media_io = __import__(
            f"{addon.__name__}.media_io",
            fromlist=["decode_video_frame_index_with_timestamp"],
        )

    def test_exact_source_frame_decode_returns_its_timestamp(self) -> None:
        image = object()

        class Capture:
            selected_frame = None
            released = False

            def isOpened(self):
                return True

            def set(self, property_id, value):
                self.selected_frame = (property_id, value)

            def read(self):
                return True, image

            def get(self, property_id):
                if property_id == 2:
                    return 166.7
                if property_id == 3:
                    return 24.0
                return 0.0

            def release(self):
                self.released = True

        capture = Capture()
        cv2 = SimpleNamespace(
            CAP_PROP_POS_FRAMES=1,
            CAP_PROP_POS_MSEC=2,
            CAP_PROP_FPS=3,
            VideoCapture=lambda _path: capture,
        )
        decoded, timestamp_ms = self.media_io.decode_video_frame_index_with_timestamp(
            "video.mp4",
            4,
            cv2,
        )
        self.assertIs(decoded, image)
        self.assertEqual(timestamp_ms, 167)
        self.assertEqual(capture.selected_frame, (cv2.CAP_PROP_POS_FRAMES, 4))
        self.assertTrue(capture.released)


class DetectionCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.addon = load_addon()
        cls.cache = importlib.import_module(f"{cls.addon.__name__}.core.cache.store")
        cls.options = importlib.import_module(
            f"{cls.addon.__name__}.session.capture_options"
        )
        cls.models = __import__(
            f"{cls.addon.__name__}.models", fromlist=["CanonicalFrame"]
        )

    def setUp(self) -> None:
        self.temporary = Path(tempfile.mkdtemp(prefix="mtp-cache-test-"))
        self.cache_directory = self.temporary / "cache"
        self.cache_directory.mkdir()
        self.root_patch = mock.patch.object(
            self.cache,
            "cache_root",
            side_effect=lambda create=True: self.cache_directory,
        )
        self.root_patch.start()

    def tearDown(self) -> None:
        self.cache.invalidate_ui_state()
        self.root_patch.stop()
        shutil.rmtree(self.temporary, ignore_errors=True)

    def test_current_cache_round_trip(self) -> None:
        media = self.models.MediaInfo(
            path=str(self.temporary / "source.jpg"),
            kind="VIDEO",
            fps=30.0,
            frame_count=3,
            duration_seconds=0.1,
            width=640,
            height=480,
        )
        writer = self.cache.SequenceWriter("key", {"capture": {}}, media, 1024 * 1024)
        for timestamp_ms in (0, 33, 66):
            writer.add(self.models.CanonicalFrame(timestamp_ms=timestamp_ms))
        directory, persisted = writer.finish()
        self.assertTrue(persisted)
        manifest = json.loads(
            (directory / self.cache.MANIFEST_NAME).read_text(encoding="utf8")
        )
        self.assertEqual(
            manifest["schema_version"],
            self.cache.CACHE_SCHEMA_VERSION,
        )
        frames = self.cache.iter_frames("key")
        self.assertIsNotNone(frames)
        self.assertEqual([frame.timestamp_ms for frame in frames], [0, 33, 66])

    def test_point_cache_compacts_face_mesh_and_preserves_pose_metadata(self) -> None:
        media = self.models.MediaInfo(
            path=str(self.temporary / "source.jpg"),
            kind="IMAGE",
            width=640,
            height=480,
        )
        frame = self.models.CanonicalFrame(
            timestamp_ms=0,
            body={
                "LEFT_SHOULDER": self.models.Landmark(
                    (0.0, 0.0, 0.0),
                    (0.25, 0.25),
                )
            },
            auxiliary={
                "LEFT_ACROMION": self.models.Landmark(
                    (0.1, 0.2, 0.3),
                    (0.3, 0.2),
                )
            },
            face_landmarks=[self.models.Landmark((0.0, 0.0, 0.0), (0.5, 0.5))],
            face_overlay=[(0.5, 0.5)],
            blendshapes={"jawOpen": 0.5},
            camera=self.models.CameraIntrinsicsObservation(
                width=640,
                height=480,
                fx=700.0,
                fy=705.0,
                cx=318.0,
                cy=241.0,
                horizontal_fov=49.1,
                vertical_fov=37.6,
                confidence=0.9,
                source="ANYCALIB",
                bundle_revision="a" * 40,
                precision="fp32",
            ),
            root_motion=self.models.RootMotionObservation(
                translation=(0.1, -0.2, 3.0),
                rotation_matrix=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                focal_length=900.0,
                scale=1.1,
                confidence=0.95,
            ),
        )
        self.cache.store_point("point", {"capture": {}}, media, 0, frame)
        restored = self.cache.load_frame("point", 0)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.face_landmarks, [])
        self.assertEqual(restored.face_overlay, [(0.5, 0.5)])
        self.assertEqual(
            restored.auxiliary["LEFT_ACROMION"].position,
            (0.1, 0.2, 0.3),
        )
        self.assertEqual(restored.root_motion.translation, (0.1, -0.2, 3.0))
        self.assertEqual(restored.root_motion.focal_length, 900.0)
        self.assertEqual(restored.camera.source, "ANYCALIB")
        self.assertAlmostEqual(restored.camera.fx, 700.0)

    def test_overflow_is_not_adopted_and_can_be_discarded(self) -> None:
        media = self.models.MediaInfo(path="unused", kind="VIDEO")
        writer = self.cache.SequenceWriter("overflow", {}, media, 1)
        writer.add(self.models.CanonicalFrame(timestamp_ms=0))
        directory, persisted = writer.finish()
        self.assertFalse(persisted)
        self.assertTrue(directory.exists())
        self.cache.discard_temporary(str(directory))
        self.assertFalse(directory.exists())

    def test_flush_uses_incremental_size(self) -> None:
        media = self.models.MediaInfo(path="unused", kind="VIDEO")
        writer = self.cache.SequenceWriter("key", {}, media, 1)
        writer.add(self.models.CanonicalFrame(timestamp_ms=0))
        with mock.patch.object(
            self.cache,
            "_path_size",
            side_effect=AssertionError("recursive scan used during flush"),
        ):
            writer._flush()
        self.assertGreater(writer.written_bytes, 0)
        self.assertTrue(writer.overflow)
        writer.discard()

    def test_ui_state_is_shared_across_redraws_until_invalidated(self) -> None:
        settings = SimpleNamespace(
            media_path="source.png",
            media_identity_json="",
            keep_detection_cache=True,
            provider="MEDIAPIPE",
            use_body=True,
            use_hands=True,
            use_face=False,
            mirror_input=False,
            smoothing=0.0,
            sam3d_model_variant="VITH",
            sam3d_device="AUTO",
        )
        source = self.cache.MediaSourceState(state="AVAILABLE")
        status = self.cache.CacheStatus(key="cache-key")
        with (
            mock.patch.object(
                self.options,
                "resolve_media_source",
                return_value=source,
            ) as resolve,
            mock.patch.object(
                self.options,
                "status_for_settings",
                return_value=status,
            ) as cache_status,
        ):
            self.cache.invalidate_ui_state()
            first = self.options.ui_state_for_settings(settings)
            second = self.options.ui_state_for_settings(settings)
            self.assertIs(first[0], second[0])
            self.assertIs(first[1], second[1])
            resolve.assert_called_once()
            cache_status.assert_called_once()

            self.cache.invalidate_ui_state()
            self.options.ui_state_for_settings(settings)
            self.assertEqual(resolve.call_count, 2)
            self.assertEqual(cache_status.call_count, 2)

    def test_preview_smoothing_does_not_fragment_raw_detection_cache(self) -> None:
        settings = SimpleNamespace(
            provider="MEDIAPIPE",
            use_body=True,
            use_hands=False,
            use_face=False,
            mirror_input=False,
            smoothing=0.8,
        )
        options = self.options.options_from_settings(settings)
        self.assertEqual(options["smoothing"], 0.8)
        self.assertEqual(options["confidence"], 0.35)
        self.assertNotIn(
            "smoothing",
            self.options._identity_options_from_settings(settings),
        )

    def test_cache_status_uses_size_accounted_during_writes(self) -> None:
        media = self.models.MediaInfo(path="unused", kind="VIDEO")
        writer = self.cache.SequenceWriter("key", {"capture": {}}, media, 0)
        writer.add(self.models.CanonicalFrame(timestamp_ms=0))
        directory, persisted = writer.finish()
        self.assertTrue(persisted)
        manifest = self.cache._read_manifest(directory)
        with mock.patch.object(
            self.cache,
            "_path_size",
            side_effect=AssertionError("recursive UI size scan used"),
        ):
            status = self.cache._status_from_manifest(manifest, matches=True)
        self.assertGreater(status.size_bytes, 0)

    def test_clear_cache_removes_camera_motion_sidecar(self) -> None:
        sidecar = self.cache.camera_motion_directory(create_root=True)
        sidecar.mkdir(parents=True)
        (sidecar / "track.npz").write_bytes(b"camera-track")
        self.assertGreater(self.cache.cache_usage_bytes(), 0)
        self.cache.clear_cache()
        self.assertFalse(sidecar.exists())


class CachePreviewWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.addon = load_addon()
        cls.preview = cls.addon.cache_preview

    def test_bounded_queue_coalesces_to_latest_request(self) -> None:
        target = queue.Queue(maxsize=1)
        self.preview._put_latest(target, "old")
        self.preview._put_latest(target, "new")
        self.assertEqual(target.qsize(), 1)
        self.assertEqual(target.get_nowait(), "new")

    def test_body_preview_connection_topology_and_order(self) -> None:
        self.assertEqual(
            self.preview.BODY_CONNECTIONS,
            (
                ("LEFT_EAR", "LEFT_EYE"),
                ("LEFT_EYE", "NOSE"),
                ("NOSE", "RIGHT_EYE"),
                ("RIGHT_EYE", "RIGHT_EAR"),
                ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
                ("LEFT_SHOULDER", "LEFT_ELBOW"),
                ("LEFT_ELBOW", "LEFT_WRIST"),
                ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
                ("RIGHT_ELBOW", "RIGHT_WRIST"),
                ("LEFT_SHOULDER", "LEFT_HIP"),
                ("RIGHT_SHOULDER", "RIGHT_HIP"),
                ("LEFT_HIP", "RIGHT_HIP"),
                ("LEFT_HIP", "LEFT_KNEE"),
                ("LEFT_KNEE", "LEFT_ANKLE"),
                ("LEFT_ANKLE", "LEFT_HEEL"),
                ("LEFT_ANKLE", "LEFT_FOOT_INDEX"),
                ("RIGHT_HIP", "RIGHT_KNEE"),
                ("RIGHT_KNEE", "RIGHT_ANKLE"),
                ("RIGHT_ANKLE", "RIGHT_HEEL"),
                ("RIGHT_ANKLE", "RIGHT_FOOT_INDEX"),
            ),
        )

    def test_worker_failure_has_initialized_result_metadata(self) -> None:
        request = self.preview._PreviewRequest(
            generation=7,
            key="cache-key",
            media_path="missing.png",
            source_index=37,
            mirror=False,
        )
        dependency_module = __import__(
            f"{self.addon.__name__}.dependency_bootstrap",
            fromlist=["import_runtime_dependencies"],
        )
        with mock.patch.object(
            dependency_module,
            "import_runtime_dependencies",
            side_effect=ImportError("injected dependency failure"),
        ):
            result = self.preview._render_overlay(request)
        self.assertEqual(result.generation, 7)
        self.assertEqual(result.media_path, "missing.png")
        self.assertEqual(result.source_index, -1)
        self.assertEqual(result.timestamp_ms, 0)
        self.assertIn("injected dependency failure", result.error)


if __name__ == "__main__":
    unittest.main()
