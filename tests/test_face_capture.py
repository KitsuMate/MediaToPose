"""A real face still through the command line, and the face close-up camera."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from tests.blender_test_support import fixture_path, open_rig, register_addon, run_cli


class FaceCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.addon = register_addon()
        cls.rig = open_rig(cls.addon, "facefx_face_rig", "FaceFX.Rig")

    def test_command_line_face_capture_metrics_and_sheet(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="mtp-face-")
        self.addCleanup(directory.cleanup)
        temporary = Path(directory.name)
        result_path = str(temporary / "result.json")
        code, output = run_cli(
            self.addon,
            "capture",
            "--media",
            str(fixture_path("face_smile_open")),
            "--armature",
            self.rig.name,
            "--parts",
            "face",
            "--face-output",
            "ARKIT",
            "--out",
            result_path,
            "--json",
        )
        self.assertEqual(code, 0, output)
        self.assertEqual(json.loads(output)["face_output"], "ARKIT")
        code, output = run_cli(
            self.addon,
            "evaluate",
            "--armature",
            self.rig.name,
            "--result",
            result_path,
            "--json",
        )
        self.assertEqual(code, 0, output)
        # One posed still: the rig shows exactly what was detected.
        self.assertLess(json.loads(output)["expression_fit"], 0.01)
        sheet = temporary / "face.png"
        code, output = run_cli(
            self.addon,
            "render",
            "--armature",
            self.rig.name,
            "--result",
            result_path,
            "--out",
            str(sheet),
            "--size",
            "96",
            "--face",
            "--json",
        )
        self.assertEqual(code, 0, output)
        image = bpy.data.images.load(str(sheet))
        self.addCleanup(bpy.data.images.remove, image)
        # The detected face and the rig's face side by side.
        self.assertEqual(tuple(image.size), (2 * 96, 96))
        without_face = json.loads(Path(result_path).read_text()) | {"face_output": ""}
        with self.assertRaises(self.addon.core.errors.InvalidRequest):
            self.addon.api.render_comparison(
                "unused.png", self.rig.name, result=without_face, face=True
            )


class FaceCameraTests(unittest.TestCase):
    """The close-up camera on VRoid, whose eye bones sit deep in the head."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.addon = register_addon()
        cls.rig = open_rig(cls.addon, "vroid_test_rig", "Armature")
        cls.face_render = cls.addon.evaluate.face_render

    def _camera(self, camera=None):
        made = self.face_render.face_camera(bpy.context.scene, self.rig, camera)
        if camera is None:
            data = made.data
            self.addCleanup(bpy.data.cameras.remove, data)
            self.addCleanup(bpy.data.objects.remove, made)
        bpy.context.view_layer.update()
        return made

    def test_face_camera_frames_the_whole_face_from_outside(self) -> None:
        scene = bpy.context.scene
        camera = self._camera()
        face = bpy.data.objects["Face"]
        corners = [face.matrix_world @ Vector(c) for c in face.bound_box]
        view = [world_to_camera_view(scene, camera, c) for c in corners]
        # In front of the face and seeing all of it across.
        self.assertTrue(all(v.z > 0.0 for v in view))
        self.assertTrue(all(-0.05 <= v.x <= 1.05 for v in view), view)
        # Top to bottom: the eyes and the chin (the bounding box also holds
        # the scalp, which may be cut).
        chin = min(
            (face.matrix_world @ vertex.co for vertex in face.data.vertices),
            key=lambda point: point.z,
        )
        eyes = [
            self.rig.matrix_world @ self.rig.pose.bones[name].head
            for name in ("J_Adj_L_FaceEye", "J_Adj_R_FaceEye")
        ]
        for point in [chin, *eyes]:
            coordinate = world_to_camera_view(scene, camera, point)
            self.assertTrue(0.0 <= coordinate.y <= 1.0, (point, coordinate))

    def test_face_camera_follows_the_posed_head(self) -> None:
        # Runs last: later tests see the restored head.
        self.addCleanup(bpy.context.view_layer.update)
        entries = self.addon.rig.mapping.mapping_by_id(self.rig.mtp_rig_config)
        head = self.rig.pose.bones[entries["HEAD"].target_name]
        self.addCleanup(
            setattr, self.rig.data, "pose_position", self.rig.data.pose_position
        )
        self.rig.data.pose_position = "POSE"
        camera = self._camera()
        original = camera.matrix_world.copy()
        pose = head.matrix.copy()
        basis, rotation_mode = head.matrix_basis.copy(), head.rotation_mode
        self.addCleanup(setattr, head, "matrix_basis", basis)
        self.addCleanup(setattr, head, "rotation_mode", rotation_mode)
        head.location.x += 0.35
        head.rotation_mode = "XYZ"
        head.rotation_euler.z += 0.4
        bpy.context.view_layer.update()
        world = self.rig.matrix_world
        delta = world @ head.matrix @ pose.inverted() @ world.inverted()
        self.assertIs(self._camera(camera), camera)
        expected = delta @ original
        self.assertLess((camera.location - expected.translation).length, 1e-5)
        difference = camera.matrix_world.to_quaternion().rotation_difference(
            expected.to_quaternion()
        )
        self.assertLess(difference.angle, 1e-3)


if __name__ == "__main__":
    unittest.main()
