"""CaptureSession end to end with real MediaPipe inference on the fixtures."""

from __future__ import annotations

import importlib
import tempfile
import time
import unittest
from pathlib import Path

import bpy
from mathutils import Matrix

from tests.blender_test_support import fixture_path, register_addon

IMAGE = "mediapipe_ref_pose_realistic"


def _write_video(path: Path, frames: int = 8) -> None:
    import cv2
    import numpy as np

    image = cv2.imdecode(
        np.fromfile(str(fixture_path(IMAGE)), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    height, width = image.shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 24.0, (width, height)
    )
    try:
        for index in range(frames):
            shift = np.float32([[1, 0, index * 2], [0, 1, 0]])
            writer.write(cv2.warpAffine(image, shift, (width, height)))
    finally:
        writer.release()


class CaptureSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.addon = register_addon()
        bpy.ops.wm.open_mainfile(filepath=str(fixture_path("vroid_test_rig")))
        cls.addon.register()
        cls.armature = next(
            obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"
        )
        cls.addon.mapping.auto_map(cls.armature)
        rohm = cls.addon.post_processing.module_config(
            cls.armature.mtp_rig_config, "ROHM_CORRECTION"
        )
        rohm.enabled = False
        cls.directory = tempfile.TemporaryDirectory()
        cls.video = Path(cls.directory.name) / "clip.avi"
        _write_video(cls.video)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def setUp(self) -> None:
        for bone in self.armature.pose.bones:
            bone.matrix_basis = Matrix.Identity(4)
        bpy.context.view_layer.objects.active = self.armature
        bpy.context.scene.frame_set(1)
        self.requests = importlib.import_module(f"{self.addon.__name__}.core.requests")
        self.capture = importlib.import_module(
            f"{self.addon.__name__}.session.capture_session"
        )
        self.coordinator = self.addon.operation.COORDINATOR

    def _settings(self, media: Path, **overrides):
        values = {
            "provider": "MEDIAPIPE",
            "use_body": True,
            "use_hands": True,
            "media_path": str(media),
            "keep_detection_cache": False,
            "root_motion_mode": "IN_PLACE",
        }
        values.update(overrides)
        return self.requests.CaptureSettings(**values)

    def _prepare(self, settings, mode):
        context = self.addon.core.scene.SceneContext.from_context(bpy.context)
        return self.capture.CaptureSession.prepare(
            context, self.armature, settings, mode
        )

    def _run(self, session, *, stop_when=None):
        update = session.start()
        deadline = time.monotonic() + 120.0
        while not update.state.finished:
            if stop_when is not None and stop_when(update):
                return update
            self.assertLess(time.monotonic(), deadline, update.status)
            if update.state == "DETECTING":
                time.sleep(0.02)
            update = session.step(budget_seconds=0.05)
        self.assertTrue(session.wait())
        return update

    def test_image_pose_is_detected_and_applied(self) -> None:
        session = self._prepare(
            self._settings(fixture_path(IMAGE)), self.requests.CaptureMode.POSE
        )
        update = self._run(session)
        self.assertEqual(update.state, "SUCCEEDED", update.status)
        self.assertEqual(session.outcome.bones, 51)
        self.assertTrue(update.status.startswith("Pose applied to "))
        self.assertFalse(self.coordinator.is_busy())
        moved = sum(
            1
            for bone in self.armature.pose.bones
            if bone.matrix_basis != Matrix.Identity(4)
        )
        self.assertGreater(moved, 20)

    def _bake(self, *, keep_cache: bool):
        original = self.armature.animation_data_create().action
        session = self._prepare(
            self._settings(self.video, keep_detection_cache=keep_cache),
            self.requests.CaptureMode.KEYFRAMES,
        )
        update = self._run(session)
        self.addCleanup(self._restore_action, original)
        return session, update

    def _restore_action(self, original) -> None:
        action = self.armature.animation_data.action
        self.armature.animation_data.action = original
        if action is not None and action is not original and action.users == 0:
            bpy.data.actions.remove(action)

    def test_video_keyframes_through_the_frame_stream(self) -> None:
        session, update = self._bake(keep_cache=False)
        self.assertEqual(update.state, "SUCCEEDED", update.status)
        self.assertEqual(session.outcome.frames, 8)
        action = self.armature.animation_data.action
        self.assertEqual(action.name, f"{self.armature.name}_MediaToPose")
        keys = {
            round(point.co.x)
            for curve in self.addon.retarget.keyframes._action_fcurves(action)
            for point in curve.keyframe_points
        }
        self.assertTrue(keys)
        self.assertFalse(self.coordinator.is_busy())

    def test_video_keyframes_reuse_the_detection_cache(self) -> None:
        first, update = self._bake(keep_cache=True)
        self.assertEqual(update.state, "SUCCEEDED", update.status)
        self.assertFalse(first.outcome.cached)
        second, update = self._bake(keep_cache=True)
        self.assertEqual(update.state, "SUCCEEDED", update.status)
        self.assertTrue(second.outcome.cached)
        self.assertEqual(second.outcome.frames, first.outcome.frames)

    def test_cancel_during_apply_restores_the_original_action(self) -> None:
        original = self.armature.animation_data_create().action
        session = self._prepare(
            self._settings(self.video), self.requests.CaptureMode.KEYFRAMES
        )
        update = self._run(
            session, stop_when=lambda update: update.state == "APPLYING_SEQUENCE"
        )
        self.assertEqual(update.state, "APPLYING_SEQUENCE")
        session.cancel()
        update = session.step()
        self.assertEqual(update.state, "CANCELLED")
        self.assertTrue(session.wait())
        self.assertIs(self.armature.animation_data.action, original)
        self.assertFalse(self.coordinator.is_busy())

    def test_validation_errors_are_typed(self) -> None:
        errors = self.addon.core.errors
        with self.assertRaises(errors.MediaError):
            self._prepare(
                self._settings(Path(self.directory.name) / "missing.png"),
                self.requests.CaptureMode.POSE,
            )
        with self.assertRaises(errors.InvalidRequest):
            self._prepare(
                self._settings(fixture_path(IMAGE), use_body=False, use_hands=False),
                self.requests.CaptureMode.POSE,
            )


if __name__ == "__main__":
    unittest.main()
