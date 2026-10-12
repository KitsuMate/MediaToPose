from __future__ import annotations

import queue
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import bpy
from mathutils import Matrix

from tests.blender_test_support import load_addon, scene_context


class CaptureWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        addon = load_addon()
        cls.worker = addon.session.worker
        cls.progress = addon.session.progress
        cls.requests = addon.core.requests

    def _request(self, media, mode, source_index, options):
        return self.requests.CaptureRequest(
            provider_id="MEDIAPIPE",
            mode=mode,
            media_path=media.path,
            media_kind=media.kind,
            source_frame_index=source_index,
            use_body=True,
            use_hands=True,
            use_face=False,
            face_output="ARKIT",
            root_motion_mode="OFF",
            keep_original_root_position=False,
            keep_detection_cache=False,
            cache_limit_bytes=0,
            start_frame=1.0,
            scene_name="Scene",
            scene_pointer=1,
            options=options,
            media_info=media,
            cache_media_info=media,
        )

    def test_video_pose_detects_the_selected_source_frame(self) -> None:
        calls = []

        class Provider:
            def __init__(self, **options):
                calls.append(("init", options))

            def detect_video_frame_index(self, path, source_index):
                calls.append(("detect", path, source_index))
                return "selected-pose"

            def close(self):
                calls.append(("close",))

        job = SimpleNamespace(
            cancel=threading.Event(),
            messages=queue.Queue(),
            progress=lambda *_args: None,
            temp_path="",
        )
        media = SimpleNamespace(kind="VIDEO", path="clip.mp4")
        provider = SimpleNamespace(provider_class=Provider)
        request = self._request(media, "POSE", 17, {})
        with mock.patch.object(self.worker, "descriptor", return_value=provider):
            self.worker.run_detection(job, request)

        self.assertEqual(job.messages.get_nowait(), ("FRAME", "selected-pose"))
        self.assertIn(("detect", "clip.mp4", 17), calls)
        self.assertEqual(calls[-1], ("close",))

    def test_calibration_controls_are_not_forwarded_to_provider_constructor(
        self,
    ) -> None:
        calls = []

        class Provider:
            def __init__(
                self,
                *,
                model_directory,
                use_body,
                use_hands,
                use_face,
                mirror,
                confidence,
                smoothing,
                video_mode,
                camera_intrinsics=None,
            ):
                calls.append(("init", camera_intrinsics, video_mode))

            def detect_image(self, path):
                calls.append(("detect", path))
                return "pose"

            def close(self):
                calls.append(("close",))

        job = SimpleNamespace(
            cancel=threading.Event(),
            messages=queue.Queue(),
            progress=lambda *_args: None,
            temp_path="",
        )
        media = SimpleNamespace(kind="IMAGE", path="pose.png", width=0, height=0)
        options = {
            "model_directory": "models",
            "use_body": True,
            "use_hands": True,
            "use_face": False,
            "mirror": False,
            "confidence": 0.5,
            "smoothing": 0.35,
            "perspective_correction": "AUTO",
            "manual_fov_override": False,
            "manual_vertical_fov": 45.0,
            "anycalib_device": "AUTO",
            "camera_bundle_revision": "revision",
            "camera_precision": "fp32",
        }
        descriptor = SimpleNamespace(provider_class=Provider)
        request = self._request(media, "POSE", 0, options)
        with mock.patch.object(self.worker, "descriptor", return_value=descriptor):
            self.worker.run_detection(job, request)

        self.assertEqual(job.messages.get_nowait(), ("FRAME", "pose"))
        self.assertEqual(calls[0], ("init", None, False))
        self.assertEqual(calls[-1], ("close",))

    def test_sequence_progress_reserves_monotonic_range_for_rohm(self) -> None:
        modules = SimpleNamespace(
            modules=(
                SimpleNamespace(module_id="DETECTION_FILTER", enabled=True),
                SimpleNamespace(
                    module_id="ROHM_CORRECTION",
                    enabled=True,
                    learned_pose_residual=False,
                    root_trajectory_cleanup=True,
                ),
                SimpleNamespace(module_id="TEMPORAL_CORRECTION", enabled=True),
            )
        )
        request = SimpleNamespace(
            provider_id="MEDIAPIPE",
            mode="KEYFRAMES",
            media_kind="VIDEO",
            root_motion_mode="AUTOMATIC",
            postprocess_config=modules,
        )
        plan = self.progress.capture_progress_plan(request, cached=False)
        updates = (
            SimpleNamespace(phase="Loading detections", current=10, total=10),
            SimpleNamespace(phase="Solving animation", current=10, total=10),
            SimpleNamespace(
                phase="RoHM: first trajectory pass · window 1/1",
                current=100,
                total=2203,
            ),
            SimpleNamespace(
                phase="RoHM: final pose pass · window 1/1",
                current=2200,
                total=2203,
            ),
            SimpleNamespace(phase="Writing keyframes", current=10, total=10),
        )
        values = []
        previous = 0.0
        for update in updates:
            previous = self.progress.sequence_progress_percent(
                update,
                previous,
                plan,
            )
            values.append(previous)

        self.assertEqual(values, sorted(values))
        self.assertGreater(values[3], values[2])
        self.assertEqual(values[-1], 100.0)
        detection_span = plan.range_for("detection")[1] - plan.range_for("detection")[0]
        rohm_span = plan.range_for("rohm")[1] - plan.range_for("rohm")[0]
        self.assertAlmostEqual(rohm_span, detection_span * 2.0)

        sam_request = SimpleNamespace(**vars(request))
        sam_request.provider_id = "SAM3D_BODY"
        sam_plan = self.progress.capture_progress_plan(sam_request, cached=False)
        sam_detection_span = (
            sam_plan.range_for("detection")[1] - sam_plan.range_for("detection")[0]
        )
        self.assertGreater(sam_detection_span, detection_span)

    def test_rohm_progress_is_reserved_only_for_requested_outputs(self) -> None:
        rohm = SimpleNamespace(
            module_id="ROHM_CORRECTION",
            enabled=True,
            learned_pose_residual=False,
            root_trajectory_cleanup=False,
        )
        request = SimpleNamespace(
            provider_id="MEDIAPIPE",
            mode="KEYFRAMES",
            media_kind="VIDEO",
            root_motion_mode="AUTOMATIC",
            postprocess_config=SimpleNamespace(modules=(rohm,)),
        )

        self.assertFalse(self.progress.rohm_may_run(request))
        self.assertNotIn(
            "rohm",
            self.progress.capture_progress_plan(request, cached=True).ranges,
        )

        rohm.learned_pose_residual = True
        request.root_motion_mode = "OFF"
        self.assertTrue(self.progress.rohm_may_run(request))
        self.assertIn(
            "rohm",
            self.progress.capture_progress_plan(request, cached=True).ranges,
        )


class CaptureTransactionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.addon = load_addon()

    def setUp(self) -> None:
        data = bpy.data.armatures.new("TransactionRigData")
        self.armature = bpy.data.objects.new("TransactionRig", data)
        bpy.context.scene.collection.objects.link(self.armature)
        bpy.context.view_layer.objects.active = self.armature
        self.armature.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bone = data.edit_bones.new("Root")
        bone.head = (0.0, 0.0, 0.0)
        bone.tail = (0.0, 0.0, 1.0)
        bpy.ops.object.mode_set(mode="OBJECT")
        self.original_action = bpy.data.actions.new("OriginalAction")
        self.armature.animation_data_create().action = self.original_action
        bpy.context.scene.frame_set(7)

    def tearDown(self) -> None:
        if self.armature.animation_data:
            self.armature.animation_data.action = None
        bpy.data.objects.remove(self.armature, do_unlink=True)
        for action in tuple(bpy.data.actions):
            if action.name in {
                "OriginalAction",
                "TransactionRig_MediaToPose",
                "TransactionRig_MediaToPose_Staging",
            }:
                bpy.data.actions.remove(action)

    def test_failed_animation_restores_action_pose_frame_and_selection(self) -> None:
        transaction = self.addon.session.transaction.CaptureTransaction(
            scene_context(),
            self.armature,
            animation=True,
        )
        temporary = transaction.temporary_action
        self.armature.pose.bones["Root"].matrix_basis = Matrix.Translation(
            (1.0, 2.0, 3.0)
        )
        bpy.context.scene.frame_set(22)
        transaction.rollback()
        self.assertIs(
            self.armature.animation_data.action,
            self.original_action,
        )
        self.assertEqual(
            self.armature.pose.bones["Root"].matrix_basis,
            Matrix.Identity(4),
        )
        self.assertEqual(bpy.context.scene.frame_current, 7)
        self.assertIs(bpy.context.view_layer.objects.active, self.armature)
        self.assertTrue(self.armature.select_get())
        self.assertNotIn(temporary, bpy.data.actions[:])

    def test_success_adopts_staged_action(self) -> None:
        root = self.armature.pose.bones["Root"]
        root.rotation_mode = "XYZ"
        root.rotation_euler.x = 0.25
        root.keyframe_insert(data_path="rotation_euler", index=0, frame=2.0)
        transaction = self.addon.session.transaction.CaptureTransaction(
            scene_context(),
            self.armature,
            animation=True,
        )
        temporary = transaction.temporary_action
        transaction.commit()
        self.assertIs(self.armature.animation_data.action, temporary)
        self.assertEqual(temporary.name, "TransactionRig_MediaToPose")
        frames = {
            round(float(point.co.x), 4)
            for curve in self.addon.retarget.keyframes._action_fcurves(temporary)
            for point in curve.keyframe_points
        }
        self.assertIn(2.0, frames)

    def test_failed_animation_restores_shape_key_action(self) -> None:
        mesh = bpy.data.meshes.new("TransactionFaceData")
        mesh.from_pydata(
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            [],
            [(0, 1, 2)],
        )
        face = bpy.data.objects.new("TransactionFace", mesh)
        bpy.context.scene.collection.objects.link(face)
        modifier = face.modifiers.new("Armature", "ARMATURE")
        modifier.object = self.armature
        face.shape_key_add(name="Basis")
        smile = face.shape_key_add(name="Smile")
        original_value = float(smile.value)
        keys = mesh.shape_keys
        original = bpy.data.actions.new("OriginalShapeAction")
        keys.animation_data_create().action = original
        try:
            transaction = self.addon.session.transaction.CaptureTransaction(
                scene_context(),
                self.armature,
                animation=True,
            )
            temporary = next(
                staged
                for staged_keys, _original, staged in transaction.shape_action_transactions
                if staged_keys is keys
            )
            smile.value = 0.75
            smile.keyframe_insert(data_path="value", frame=7.0)
            transaction.rollback()
            self.assertIs(keys.animation_data.action, original)
            self.assertEqual(smile.value, original_value)
            self.assertNotIn(temporary, bpy.data.actions[:])
        finally:
            if keys.animation_data:
                keys.animation_data.action = None
            bpy.data.objects.remove(face, do_unlink=True)
            bpy.data.meshes.remove(mesh)
            if original in bpy.data.actions[:]:
                bpy.data.actions.remove(original)


if __name__ == "__main__":
    unittest.main()
