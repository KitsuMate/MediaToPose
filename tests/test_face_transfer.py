"""Face transfer onto the FaceFX fixture: shape keys, face bones and gaze.

Each test applies synthetic blendshape coefficients and checks what moved,
so the numbers below encode what the visual review of the fixture showed.
"""

from __future__ import annotations

import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import bpy
from mathutils import Vector

from tests.blender_test_support import fixture_path, register_addon, scene_context


class FaceTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.addon = register_addon()
        bpy.ops.wm.open_mainfile(filepath=str(fixture_path("facefx_face_rig")))
        cls.addon.register()
        settings = bpy.context.scene.mtp_settings
        settings.use_face = True
        settings.use_body = False
        settings.use_hands = False
        cls.rig = bpy.data.objects["FaceFX.Rig"]
        settings.target_armature = cls.rig
        cls.addon.api.auto_map(cls.rig.name)
        cls.entries = cls.addon.rig.mapping.mapping_by_id(cls.rig.mtp_rig_config)
        cls.face = cls.addon.retarget.solve.face
        cls.meshes = [
            obj
            for obj in bpy.data.objects
            if obj.type == "MESH" and obj.data.shape_keys is not None
        ]

    def setUp(self) -> None:
        for obj in self.meshes:
            for block in obj.data.shape_keys.key_blocks:
                block.value = 0.0
        for bone in self.rig.pose.bones:
            bone.matrix_basis.identity()
        bpy.context.view_layer.update()

    def _apply(self, blendshapes: dict[str, float], face_output: str) -> None:
        frame = self.addon.core.types.CanonicalFrame(
            timestamp_ms=0, blendshapes=dict(blendshapes)
        )
        settings = SimpleNamespace(
            provider="MEDIAPIPE",
            use_body=False,
            use_hands=False,
            use_face=True,
            face_output=face_output,
        )
        self.addon.retarget.apply_frame(
            self.rig, frame, settings, post_process=False, scene_context=scene_context()
        )
        bpy.context.view_layer.update()

    def _world_offset(self, bone_name: str):
        bone = self.rig.pose.bones[bone_name]
        return self.rig.matrix_world.to_3x3() @ (
            bone.matrix.translation - bone.bone.matrix_local.translation
        )

    def test_detected_as_auto_rig_pro_with_every_face_control(self) -> None:
        described = self.addon.api.describe_rig(self.rig.name)
        self.assertEqual(described["detected_profile"], "AUTO_RIG_PRO")
        face = described["sections"]["FACE_RIG"]
        self.assertEqual(face["mapped"], face["enabled"])

    def test_describe_rig_face_block(self) -> None:
        described = self.addon.api.describe_rig(self.rig.name)
        face = described["face"]
        # The FaceFX mesh has every ARKit key except a lower lip shrug.
        self.assertEqual(face["expressions"]["ARKIT"]["unmapped"], ["mouthShrugLower"])
        self.assertEqual(len(face["expressions"]["ARKIT"]["mapped"]), 50)
        self.assertEqual(face["controls"]["FACE_LID_L"]["channel"], "LOCATION")
        self.assertEqual(face["unmapped_controls"], [])
        eye = face["eyes"]["EYE_L"]
        self.assertTrue(eye["look_at_target"])
        self.assertEqual(eye["tracked_by"], "c_eye.l")
        self.assertAlmostEqual(face["face_scale"], 0.0591, delta=0.002)
        self.assertGreater(
            described["shape_keys"]["FaceFX"]["keys_per_scheme"]["ARKIT"], 50
        )

    def test_face_completeness_is_null_when_face_capture_is_off(self) -> None:
        settings = bpy.context.scene.mtp_settings
        try:
            settings.use_face = False
            off = self.addon.api.describe_rig(self.rig.name)["completeness"]
        finally:
            settings.use_face = True
        on = self.addon.api.describe_rig(self.rig.name)["completeness"]
        self.assertIsNone(off["arkit"])
        self.assertIsNone(off["face"])
        self.assertGreater(on["arkit"], 0.9)
        self.assertEqual(on["face"], 1.0)

    def test_prefixed_arkit_names_and_correctives(self) -> None:
        scheme = self.addon.rig.face_schemes.shape_scheme
        self.assertEqual(scheme("facs_bs_EyeBlinkLeft"), "ARKIT")
        self.assertEqual(scheme("arkit_EyeBlinkLeft"), "ARKIT")
        self.assertEqual(scheme("blendShape.EyeBlinkLeft"), "ARKIT")
        self.assertEqual(scheme("blendShape.facs_cbs_EyeBlinkLeft"), "NONE")
        self.assertEqual(scheme("facs_ctrl_MouthPucker"), "ARKIT")
        self.assertEqual(scheme("facs_cbs_EyeFullCompressionLeft"), "NONE")
        self.assertEqual(scheme("facs_ctrl_vAA"), "NONE")
        self.assertEqual(scheme("AU12_L"), "FACS")
        self.assertEqual(scheme("AU45"), "FACS")
        for name in ("Fcl_EYE_Close_L", "Fcl_MTH_A", "vrm_blink", "Fcl_BRW_Angry"):
            self.assertEqual(scheme(name), "VRM", name)
        for name in ("Basis", "Smile_extra", "mouthSmileLeft_corrective"):
            with self.subTest(name=name):
                self.assertNotIn(scheme(name), {"FACS", "VRM"})

    def test_blink_drives_one_key_on_every_mesh_that_has_one(self) -> None:
        self._apply({"eyeBlinkLeft": 1.0}, "ARKIT")
        normalize = self.addon.rig.names.normalize_name
        for obj in self.meshes:
            blocks = obj.data.shape_keys.key_blocks
            blink = [
                b for b in blocks if normalize(b.name, shape=True) == "eyeblinkleft"
            ]
            driven = [b.name for b in blink if b.value > 0.5]
            with self.subTest(mesh=obj.name):
                # The face carries arkit_ and facs_bs_ copies; driving both
                # would close the eye twice as hard.
                self.assertEqual(len(driven), 1 if blink else 0, driven)
                if driven and any(b.name.startswith("arkit_") for b in blink):
                    self.assertTrue(driven[0].startswith("arkit_"))
            self.assertFalse(
                any(
                    b.value > 0.0
                    for b in blocks
                    if normalize(b.name, shape=True) == "eyeblinkright"
                )
            )

    def test_facs_jaw_action_units_follow_their_definitions(self) -> None:
        facs = self.face.facs_values
        parted = facs({"mouthLowerDownLeft": 0.8, "mouthLowerDownRight": 0.8})
        self.assertAlmostEqual(parted["AU25"], 0.8)
        self.assertEqual(parted["AU26"], 0.0)
        half = facs({"jawOpen": 0.5})
        self.assertEqual((half["AU26"], half["AU27"]), (0.5, 0.0))
        wide = facs({"jawOpen": 1.0, "mouthStretchLeft": 1.0})
        self.assertEqual(wide["AU27"], 1.0)
        self.assertEqual(facs({"mouthStretchLeft": 1.0})["AU27"], 0.0)

    def test_face_frame_measures_the_eyes_not_their_targets(self) -> None:
        frame = self.addon.rig.face_space.face_frame(self.rig, self.entries)
        self.assertAlmostEqual(frame.scale, 0.0591, delta=0.002)
        self.assertGreater(frame.up.z, 0.99)
        self.assertGreater(frame.lateral.x, 0.99)
        self.assertLess(frame.forward.y, -0.99)

    def test_bone_controls_move_in_face_space(self) -> None:
        self._apply(
            {"eyeBlinkLeft": 1.0, "browOuterUpLeft": 1.0, "mouthSmileLeft": 1.0},
            "BONES",
        )
        scale = self.addon.rig.face_space.face_frame(self.rig, self.entries).scale
        lid = self._world_offset("c_eyelid_top.l")
        brow = self._world_offset("c_eyebrow_01_end.l")
        corner = self._world_offset("c_lips_smile.l")
        # The jaw drags the mouth corners down through ARP's retain bones,
        # so it is checked on its own.
        self.setUp()
        self._apply({"jawOpen": 1.0}, "BONES")
        jaw = self._world_offset("c_jawbone.x")
        # Each moves mostly up or down, by its travel share of the face.
        self.assertAlmostEqual(lid.z, -0.34 * scale, delta=0.1 * scale)
        self.assertAlmostEqual(brow.z, 0.039 * scale, delta=0.01 * scale)
        self.assertAlmostEqual(corner.z, 0.116 * scale, delta=0.03 * scale)
        self.assertLess(abs(corner.x), 0.02 * scale)
        self.assertAlmostEqual(jaw.z, -0.93 * scale, delta=0.1 * scale)
        # ARP opens the jaw from the controller's position, not rotation.
        jaw_bone = self.rig.pose.bones["c_jawbone.x"]
        self.assertAlmostEqual(jaw_bone.rotation_quaternion.angle, 0.0, places=5)

    def test_bone_lids_follow_the_gaze(self) -> None:
        scale = self.addon.rig.face_space.face_frame(self.rig, self.entries).scale
        # The follow share goes through the lid's input range (full at 0.75);
        # lids close 0.34 eye distances and open (the negative side) 0.107.
        follow = self.face.LID_FOLLOW_GAZE / 0.75
        lower, lift = follow * 0.34 * scale, follow * 0.107 * scale
        self._apply({"eyeLookDownLeft": 1.0}, "BONES")
        down = self._world_offset("c_eyelid_top.l").z
        self.setUp()
        self._apply({"eyeLookUpLeft": 1.0}, "BONES")
        up = self._world_offset("c_eyelid_top.l").z
        self.assertAlmostEqual(down, -lower, delta=0.3 * lower)
        self.assertAlmostEqual(up, lift, delta=0.3 * lift)

    def _entry(self, semantic_id: str):
        entry = self.addon.rig.mapping.mapping_by_id(self.rig.mtp_rig_config)[
            semantic_id
        ]
        saved = {
            name: getattr(entry, name)
            for name in (*self.addon.rig.face_defaults.RANGE_FIELDS, "ranges_version")
        }
        saved.update(source=entry.source, manual_lock=entry.manual_lock)

        def restore() -> None:
            with self.addon.rig.locks.suppress_manual_locks():
                for name, value in saved.items():
                    setattr(entry, name, value)

        self.addCleanup(restore)
        return entry

    def test_shape_key_input_range_closes_a_weak_blink(self) -> None:
        entry = self._entry("ARKIT_eyeBlinkLeft")
        entry.input_full = 0.8
        self._apply({"eyeBlinkLeft": 0.8}, "ARKIT")
        block = bpy.data.objects["FaceFX"].data.shape_keys.key_blocks[entry.target_name]
        self.assertAlmostEqual(block.value, 1.0, places=4)
        # Editing a range is a manual edit: auto-mapping keeps it.
        self.assertTrue(entry.manual_lock)
        entry.output_full = 0.5
        self._apply({"eyeBlinkLeft": 0.8}, "ARKIT")
        self.assertAlmostEqual(block.value, 0.5, places=4)

    def test_bone_output_range_is_in_metres(self) -> None:
        entry = self._entry("FACE_LID_L")
        self.assertEqual(entry.ranges_version, 1)
        # Bone lids close fully from a detected blink of 0.75.
        self.assertAlmostEqual(entry.input_full, 0.75)
        morph = self.addon.rig.mapping.mapping_by_id(self.rig.mtp_rig_config)[
            "ARKIT_eyeBlinkLeft"
        ]
        self.assertAlmostEqual(morph.input_full, 1.0)
        entry.output_full = 0.012
        self._apply({"eyeBlinkLeft": 1.0}, "BONES")
        self.assertAlmostEqual(
            self._world_offset("c_eyelid_top.l").z, -0.012, delta=1e-4
        )

    def test_set_full_from_pose_reads_the_control(self) -> None:
        entry = self._entry("FACE_JAW_OPEN")
        config = self.rig.mtp_rig_config
        index = next(
            i
            for i, item in enumerate(config.mappings)
            if item.semantic_id == "FACE_JAW_OPEN"
        )
        saved_index = config.active_mapping_index
        self.addCleanup(setattr, config, "active_mapping_index", saved_index)
        config.active_mapping_index = index
        bpy.context.scene.mtp_settings.target_armature = self.rig
        jaw = self.rig.pose.bones["c_jawbone.x"]
        space = self.addon.rig.face_space.face_frame(self.rig, self.entries)
        jaw.location = self.addon.rig.face_space.to_bone(jaw, space.up * -0.02)
        result = bpy.ops.mtp.set_face_range_from_pose(side="FULL")
        self.assertEqual(result, {"FINISHED"})
        # The ARP jaw axis is -Z in face space, so 2 cm down reads +0.02.
        self.assertAlmostEqual(entry.output_full, 0.02, delta=1e-5)
        self.assertTrue(entry.manual_lock)
        bpy.ops.mtp.reset_face_ranges()
        scale = space.scale
        self.assertAlmostEqual(entry.output_full, 0.93 * scale, delta=1e-6)

    def test_mappings_from_0_2_0_are_migrated(self) -> None:
        entry = self._entry("ARKIT_mouthSmileLeft")
        lid = self._entry("FACE_LID_L")
        scale = self.addon.rig.face_space.face_frame(self.rig, self.entries).scale
        with self.addon.rig.locks.suppress_manual_locks():
            for item in (entry, lid):
                item.ranges_version = 0
            entry["gain"] = 2.0
            entry["offset"] = 0.1
            lid["clamp_max"] = 0.5
        self.addCleanup(lambda: [entry.pop(k, None) for k in ("gain", "offset")])
        self.addCleanup(lambda: lid.pop("clamp_max", None))
        self.addon.rig.mapping.ensure_mapping_entries(self.rig)
        self.assertEqual(entry.ranges_version, 1)
        self.assertAlmostEqual(entry.input_zero, 0.1)
        self.assertAlmostEqual(entry.input_full, 0.6)
        self.assertAlmostEqual(lid.input_full, 0.5)
        self.assertAlmostEqual(lid.output_full, 0.5 * 0.34 * scale, delta=1e-6)

    def test_schema_18_presets_are_migrated(self) -> None:
        presets = self.addon.rig.presets
        payload = presets.serialize_config(self.rig)
        self.assertEqual(payload["schema_version"], 19)
        legacy = json.loads(json.dumps(payload))
        legacy["schema_version"] = 18
        for item in legacy["mappings"]:
            for name in self.addon.rig.face_defaults.RANGE_FIELDS:
                item.pop(name, None)
            item.update(
                gain=1.0, offset=0.0, clamp_min=0.0, clamp_max=1.0, invert=False
            )
            if item["semantic_id"] == "ARKIT_jawOpen":
                item["gain"] = 4.0
        entry = self._entry("ARKIT_jawOpen")
        lid = self._entry("FACE_LID_L")
        scale = self.addon.rig.face_space.face_frame(self.rig, self.entries).scale
        self.addCleanup(presets.load_config, self.rig, payload)
        presets.load_config(self.rig, legacy)
        self.assertAlmostEqual(entry.input_full, 0.25)
        self.assertAlmostEqual(lid.output_full, 0.34 * scale, delta=1e-6)

    def test_fit_restores_pose_when_initial_evaluation_fails(self) -> None:
        module = self.addon.rig.face_fit
        bone = self.rig.pose.bones[0]
        bone.location.x = 0.123
        block = self.meshes[0].data.shape_keys.key_blocks[-1]
        block.value = 0.37
        self.rig.data.pose_position = "REST"
        try:
            with (
                mock.patch.object(
                    module._Meshes,
                    "positions",
                    side_effect=RuntimeError("evaluation failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "evaluation failed"),
            ):
                self.addon.api.fit_face_controls(self.rig.name, apply=False)
            self.assertAlmostEqual(bone.location.x, 0.123)
            self.assertAlmostEqual(block.value, 0.37)
            self.assertEqual(self.rig.data.pose_position, "REST")
        finally:
            self.rig.data.pose_position = "POSE"

    def test_fit_refuses_active_capture_and_invalid_controls(self) -> None:
        coordinator = self.addon.core.coordinator
        token = coordinator.COORDINATOR.acquire(
            coordinator.OperationKind.CAPTURE, "test"
        )
        try:
            with self.assertRaises(self.addon.core.errors.OperationBusy):
                self.addon.api.fit_face_controls(self.rig.name, apply=False)
        finally:
            coordinator.COORDINATOR.release(token)
        with self.assertRaises(self.addon.core.errors.InvalidRequest):
            self.addon.api.fit_face_controls(self.rig.name, controls=["typo"])
        report = self.addon.api.fit_face_controls(self.rig.name, controls=[])
        self.assertEqual(report["controls"], {})

    def test_fit_uses_pose_evaluation_and_restores_rest_mode(self) -> None:
        self.rig.data.pose_position = "REST"
        try:
            report = self.addon.api.fit_face_controls(
                self.rig.name, controls=["FACE_JAW_OPEN"], apply=False
            )
            self.assertTrue(report["controls"]["FACE_JAW_OPEN"]["full"]["accepted"])
            self.assertEqual(self.rig.data.pose_position, "REST")
        finally:
            self.rig.data.pose_position = "POSE"

    def test_default_bone_ranges_match_the_shape_keys(self) -> None:
        # The Auto-Rig Pro defaults were fitted on this rig: each control at
        # its default output is as close to its ARKit key as the best fit.
        lid = self.rig.pose.bones["c_eyelid_top.l"]
        lid.location.z = 0.003
        report = self.addon.api.fit_face_controls(self.rig.name, apply=False)
        self.assertAlmostEqual(lid.location.z, 0.003)  # pose restored
        controls = report["controls"]
        self.assertFalse(report["applied"])
        self.assertGreaterEqual(len(controls), 11)
        for control, sides in controls.items():
            for side, fit in sides.items():
                with self.subTest(control=control, side=side):
                    if not fit["accepted"]:
                        continue
                    self.assertLess(fit["rms_fit"], fit["rms_rest"])
                    self.assertLessEqual(
                        fit["rms_default"], fit["rms_fit"] * 1.05 + 2.0e-5
                    )
        # ARP's lip centre cannot follow ARKit's lip-side keys.
        self.assertFalse(controls["FACE_LIP_TOP"]["full"]["accepted"])
        self.assertTrue(controls["FACE_LID_L"]["full"]["accepted"])

    def test_fit_face_command_applies_accepted_fits(self) -> None:
        entry = self._entry("FACE_JAW_SIDE")
        lips = self._entry("FACE_LIP_TOP")
        with self.addon.rig.locks.suppress_manual_locks():
            entry.output_full = lips.output_full = 0.05
        output = io.StringIO()
        with redirect_stdout(output):
            code = self.addon.api.cli.execute(
                [
                    "fit-face",
                    "--armature",
                    self.rig.name,
                    "--controls",
                    "FACE_JAW_SIDE,FACE_LIP_TOP",
                    "--json",
                ]
            )
        self.assertEqual(code, 0, output.getvalue())
        printed = json.loads(output.getvalue())
        self.assertEqual(set(printed["controls"]), {"FACE_JAW_SIDE", "FACE_LIP_TOP"})
        scale = self.addon.rig.face_space.face_frame(self.rig, self.entries).scale
        self.assertAlmostEqual(entry.output_full, 0.082 * scale, delta=0.005 * scale)
        # A rejected fit leaves the default, not the value set before.
        self.assertAlmostEqual(lips.output_full, 0.05 * scale, delta=1e-6)

    def test_eye_gaze_also_drives_eye_bones_with_shape_keys(self) -> None:
        self._apply({"eyeLookOutLeft": 1.0, "eyeLookInRight": 1.0}, "ARKIT")
        target = self._world_offset("c_eye_target.l")
        # Positive coefficients look to the subject's left (+X here).
        self.assertGreater(target.x, 0.0)
        tracker = self.addon.rig.face_space.gaze_tracker(
            self.rig, self.rig.pose.bones["c_eye_target.l"]
        )
        self.assertIsNotNone(tracker)
        distance = (
            self.rig.data.bones["c_eye_target.l"].head_local - tracker.bone.head_local
        ).length
        expected = distance * math.tan(
            math.radians(self.addon.rig.face_defaults.GAZE_DEGREES)
        )
        self.assertAlmostEqual(target.x, expected, delta=0.1 * expected)


class VroidExpressionTests(unittest.TestCase):
    """VRM / VRoid expression keys driven from ARKit coefficients."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.addon = register_addon()
        bpy.ops.wm.open_mainfile(filepath=str(fixture_path("vroid_test_rig")))
        cls.addon.register()
        settings = bpy.context.scene.mtp_settings
        settings.use_face = True
        settings.face_output = "VRM"
        cls.rig = bpy.data.objects["Armature"]
        settings.target_armature = cls.rig
        cls.addon.api.auto_map(cls.rig.name)
        cls.keys = bpy.data.objects["Face"].data.shape_keys.key_blocks

    def setUp(self) -> None:
        for block in self.keys:
            block.value = 0.0

    def _apply(self, blendshapes: dict[str, float]) -> None:
        frame = self.addon.core.types.CanonicalFrame(
            timestamp_ms=0, blendshapes=dict(blendshapes)
        )
        settings = SimpleNamespace(
            provider="MEDIAPIPE",
            use_body=False,
            use_hands=False,
            use_face=True,
            face_output="VRM",
        )
        self.addon.retarget.apply_frame(
            self.rig, frame, settings, post_process=False, scene_context=scene_context()
        )

    def test_face_camera_frames_the_whole_face_from_outside(self) -> None:
        from bpy_extras.object_utils import world_to_camera_view

        scene = bpy.context.scene
        camera = self.addon.evaluate.face_render.face_camera(scene, self.rig)
        try:
            bpy.context.view_layer.update()
            face = bpy.data.objects["Face"]
            corners = [face.matrix_world @ Vector(c) for c in face.bound_box]
            view = [world_to_camera_view(scene, camera, c) for c in corners]
            # VRoid eye bones sit deep in the head; the camera must still be
            # in front of the face and see all of it.
            self.assertTrue(all(v.z > 0.0 for v in view))
            self.assertTrue(all(-0.05 <= v.x <= 1.05 for v in view), view)
            # Top to bottom: the eyes and the chin (the bounding box also
            # holds the scalp, which may be cut).
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
        finally:
            data = camera.data
            bpy.data.objects.remove(camera)
            bpy.data.cameras.remove(data)

    def test_face_camera_follows_posed_head_and_reuses_camera(self) -> None:
        scene = bpy.context.scene
        entries = self.addon.rig.mapping.mapping_by_id(self.rig.mtp_rig_config)
        head = self.rig.pose.bones[entries["HEAD"].target_name]
        position = self.rig.data.pose_position
        rotation_mode = head.rotation_mode
        self.rig.data.pose_position = "POSE"
        bpy.context.view_layer.update()
        camera = self.addon.evaluate.face_render.face_camera(scene, self.rig)
        bpy.context.view_layer.update()
        original = camera.matrix_world.copy()
        pose = head.matrix.copy()
        basis = head.matrix_basis.copy()
        try:
            head.location.x += 0.35
            head.rotation_mode = "XYZ"
            head.rotation_euler.z += 0.4
            bpy.context.view_layer.update()
            delta = self.rig.matrix_world @ head.matrix @ pose.inverted()
            delta = delta @ self.rig.matrix_world.inverted()
            reused = self.addon.evaluate.face_render.face_camera(
                scene, self.rig, camera
            )
            bpy.context.view_layer.update()
            self.assertIs(reused, camera)
            expected = delta @ original
            self.assertLess((camera.location - expected.translation).length, 1e-5)
            self.assertLess(
                camera.matrix_world.to_quaternion()
                .rotation_difference(expected.to_quaternion())
                .angle,
                1e-3,
            )
        finally:
            head.rotation_mode = rotation_mode
            head.matrix_basis = basis
            self.rig.data.pose_position = position
            bpy.context.view_layer.update()
            data = camera.data
            bpy.data.objects.remove(camera)
            bpy.data.cameras.remove(data)

    def test_every_vrm_expression_is_mapped(self) -> None:
        described = self.addon.api.describe_rig(self.rig.name)
        self.assertEqual(described["completeness"]["vrm"], 1.0)
        self.assertIn("VRM", described["shape_keys"]["Face"]["schemes"])

    def test_blinks_visemes_and_brows(self) -> None:
        self._apply({"eyeBlinkLeft": 1.0, "jawOpen": 1.0, "browInnerUp": 1.0})
        self.assertAlmostEqual(self.keys["Fcl_EYE_Close_L"].value, 1.0, places=3)
        self.assertEqual(self.keys["Fcl_EYE_Close_R"].value, 0.0)
        self.assertEqual(self.keys["Fcl_EYE_Close"].value, 0.0)
        self.assertAlmostEqual(self.keys["Fcl_MTH_A"].value, 1.0, places=3)
        self.assertAlmostEqual(self.keys["Fcl_BRW_Surprised"].value, 1.0, places=3)
        self.setUp()
        self._apply({"mouthPucker": 1.0, "mouthSmileLeft": 1.0, "mouthSmileRight": 1.0})
        self.assertAlmostEqual(self.keys["Fcl_MTH_U"].value, 1.0, places=3)
        self.assertAlmostEqual(self.keys["Fcl_MTH_Joy"].value, 1.0, places=3)
        self.assertEqual(self.keys["Fcl_MTH_A"].value, 0.0)


def _blink_clip(count: int = 40) -> list[dict[str, float]]:
    """A clip with a resting smile bias, jitter, and one quick blink."""
    clip = []
    for index in range(count):
        blink = {20: 0.7, 21: 0.95, 22: 0.9, 23: 0.6}.get(index, 0.05)
        clip.append(
            {
                "eyeBlinkLeft": blink,
                "eyeBlinkRight": blink,
                "mouthSmileLeft": 0.3 + 0.12 * math.sin(index * 2.5),
            }
        )
    return clip


class FaceSequenceTests(unittest.TestCase):
    """A video sequence through Face Expression Cleanup and temporal cleanup."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.addon = register_addon()
        bpy.ops.wm.open_mainfile(filepath=str(fixture_path("facefx_face_rig")))
        cls.addon.register()
        settings = bpy.context.scene.mtp_settings
        settings.use_face = True
        settings.use_body = False
        settings.use_hands = False
        cls.rig = bpy.data.objects["FaceFX.Rig"]
        settings.target_armature = cls.rig
        cls.addon.api.auto_map(cls.rig.name)
        cls.mesh = bpy.data.objects["FaceFX"]

    def _run(
        self, face_cleanup: bool, face_output: str = "ARKIT", *, clip=None
    ) -> dict[str, list[float]]:
        config = self.rig.mtp_rig_config
        module_config = self.addon.postprocess.module_config
        saved = [(item, item.enabled) for item in config.post_processing_modules]
        self.addCleanup(
            lambda: [setattr(item, "enabled", value) for item, value in saved]
        )
        for item in config.post_processing_modules:
            item.enabled = False
        module_config(config, "TEMPORAL_CORRECTION").enabled = True
        module_config(config, "FACE_CORRECTION").enabled = face_cleanup

        context = scene_context()
        step_ms = 1000.0 / context.fps
        frames = [
            self.addon.core.types.CanonicalFrame(
                timestamp_ms=round(index * step_ms), blendshapes=values
            )
            for index, values in enumerate(_blink_clip() if clip is None else clip)
        ]
        self.detections = [(1.0 + index, frame) for index, frame in enumerate(frames)]
        settings = SimpleNamespace(
            provider="MEDIAPIPE",
            use_body=False,
            use_hands=False,
            use_face=True,
            face_output=face_output,
            media_kind="VIDEO",
            root_motion_mode="OFF",
            keep_original_root_position=False,
            keep_detection_cache=False,
        )
        keys = self.mesh.data.shape_keys
        keys.animation_data_clear()
        self.addCleanup(keys.animation_data_clear)
        self.rig.animation_data_clear()
        self.addCleanup(self.rig.animation_data_clear)
        self.addCleanup(
            lambda: [b.matrix_basis.identity() for b in self.rig.pose.bones]
        )
        self.addon.retarget.apply_sequence(
            self.rig, frames, settings, 1.0, scene_context=context
        )
        normalize = self.addon.rig.names.normalize_name
        names = {}
        for role in ("eyeBlinkLeft", "mouthSmileLeft"):
            # The face carries arkit_ and facs_bs_ copies; arkit_ is keyed.
            matches = sorted(
                (
                    block.name
                    for block in keys.key_blocks
                    if normalize(block.name, shape=True) == role.lower()
                ),
                key=lambda name: not name.startswith("arkit_"),
            )
            names[role] = matches[0]
        scene = bpy.context.scene
        values: dict[str, list[float]] = {role: [] for role in names}
        for index in range(len(frames)):
            scene.frame_set(1 + index)
            for role, name in names.items():
                values[role].append(keys.key_blocks[name].value)
        return values

    def test_lost_face_keys_neutral_after_fading(self) -> None:
        # Fewer than ten detections avoids neutral calibration; the long gap
        # must end in explicit zero keys instead of holding the last smile.
        clip = [{"mouthSmileLeft": 0.8, "eyeBlinkLeft": 0.0}] * 3 + [{}] * 60
        values = self._run(face_cleanup=True, clip=clip)
        self.assertGreater(values["mouthSmileLeft"][0], 0.5)
        self.assertAlmostEqual(values["mouthSmileLeft"][-1], 0.0, places=6)

    def test_blink_survives_and_expressions_are_calibrated(self) -> None:
        values = self._run(face_cleanup=True)
        blink = values["eyeBlinkLeft"]
        # The blink keeps nearly its full depth at the detected frame.
        self.assertGreater(blink[21], 0.85)
        self.assertLess(max(blink[:15]), 0.05)
        smile = values["mouthSmileLeft"]
        roughness = sum(abs(smile[i + 1] - smile[i]) for i in range(len(smile) - 1))
        raw = [frame["mouthSmileLeft"] for frame in _blink_clip()]
        raw_roughness = sum(abs(raw[i + 1] - raw[i]) for i in range(len(raw) - 1))
        self.assertLess(roughness, 0.5 * raw_roughness)
        # Neutral calibration removed the resting smile.
        self.assertLess(sum(smile) / len(smile), 0.25)

    def test_without_face_cleanup_temporal_smoothing_still_runs(self) -> None:
        values = self._run(face_cleanup=False)
        smile = values["mouthSmileLeft"]
        # No calibration: the resting smile stays.
        self.assertGreater(sum(smile) / len(smile), 0.2)

    def _metrics(self, face_output: str):
        return self.addon.evaluate.metrics.evaluate(
            scene_context(),
            self.rig,
            frame_range=(1, len(self.detections)),
            detections=self.detections,
            face_output=face_output,
        )

    def test_face_metrics_on_shape_keys(self) -> None:
        self._run(face_cleanup=False)
        plain = self._metrics("ARKIT")
        self._run(face_cleanup=True)
        cleaned = self._metrics("ARKIT")
        self.assertEqual(cleaned.blink_recall, 1.0)
        # Cleanup smooths the jittery smile and removes its resting level.
        self.assertLess(cleaned.face_jitter, plain.face_jitter)
        self.assertGreater(cleaned.expression_fit, plain.expression_fit)
        self.assertLess(cleaned.asymmetry, 0.05)
        self.assertIsNone(
            self.addon.evaluate.metrics.evaluate(
                scene_context(), self.rig, frame_range=(1, 5)
            ).face_jitter
        )

    def test_face_metrics_on_bones(self) -> None:
        self._run(face_cleanup=True, face_output="BONES")
        measured = self._metrics("BONES")
        self.assertEqual(measured.blink_recall, 1.0)
        self.assertLess(measured.asymmetry, 0.05)
        self.assertIsNotNone(measured.expression_fit)


class FaceCaptureResultTests(unittest.TestCase):
    """A real still captured through the API reports its face values."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.addon = register_addon()
        bpy.ops.wm.open_mainfile(filepath=str(fixture_path("facefx_face_rig")))
        cls.addon.register()
        cls.rig = bpy.data.objects["FaceFX.Rig"]
        cls.addon.api.auto_map(cls.rig.name)

    def test_smile_result_lists_applied_and_unmapped_expressions(self) -> None:
        result = self.addon.api.capture(
            str(fixture_path("face_smile_open")),
            self.rig.name,
            parts={"face"},
            face_output="ARKIT",
            keep_cache=False,
        )
        self.assertTrue(result.succeeded, result.message)
        face = result.face
        self.assertEqual(face["output"], "ARKIT")
        self.assertEqual(face["frames_with_face"], 1)
        smile = face["expressions"]["mouthSmileLeft"]
        self.assertGreater(smile["peak"], 0.3)
        self.assertEqual(smile["peak"], smile["mean"])
        self.assertEqual(face["unmapped_expressions"], [])

    def test_disabled_mapping_is_reported_unmapped(self) -> None:
        entry = next(
            item
            for item in self.rig.mtp_rig_config.mappings
            if item.semantic_id == "ARKIT_mouthSmileRight"
        )
        entry.enabled = False
        self.addCleanup(setattr, entry, "enabled", True)
        result = self.addon.api.capture(
            str(fixture_path("face_smile_open")),
            self.rig.name,
            parts={"face"},
            face_output="ARKIT",
            keep_cache=False,
        )
        self.assertIn("mouthSmileRight", result.face["unmapped_expressions"])
        self.assertNotIn("mouthSmileRight", result.face["expressions"])

    def test_command_line_face_capture_metrics_and_render(self) -> None:
        def run(*argv: str) -> tuple[int, str]:
            output = io.StringIO()
            with redirect_stdout(output):
                code = self.addon.api.cli.execute(list(argv))
            return code, output.getvalue()

        with tempfile.TemporaryDirectory() as temporary:
            result_path = str(Path(temporary) / "result.json")
            code, output = run(
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
            self.assertIn("mouthSmileLeft", json.loads(output)["face"]["expressions"])
            code, output = run(
                "evaluate",
                "--armature",
                self.rig.name,
                "--result",
                result_path,
                "--json",
            )
            self.assertEqual(code, 0, output)
            metrics = json.loads(output)
            # One posed still: the rig shows exactly what was detected.
            self.assertLess(metrics["expression_fit"], 0.01)
            sheet = str(Path(temporary) / "face.png")
            code, output = run(
                "render",
                "--armature",
                self.rig.name,
                "--result",
                result_path,
                "--out",
                sheet,
                "--size",
                "96",
                "--face",
                "--json",
            )
            self.assertEqual(code, 0, output)
            self.assertTrue(Path(sheet).is_file())

    def test_face_comparison_sheet_has_source_rig_and_chart(self) -> None:
        result = self.addon.api.capture(
            str(fixture_path("face_surprise")),
            self.rig.name,
            parts={"face"},
            face_output="ARKIT",
            keep_cache=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            written = self.addon.api.render_comparison(
                str(Path(temporary) / "face.png"),
                self.rig.name,
                result=result,
                face=True,
                size=128,
            )
            image = bpy.data.images.load(written["path"])
            try:
                self.assertEqual(tuple(image.size), (3 * 128, 128))
            finally:
                bpy.data.images.remove(image)
        without_face = result.to_dict() | {"face": {}}
        with self.assertRaises(self.addon.core.errors.InvalidRequest):
            self.addon.api.render_comparison(
                "unused.png", self.rig.name, result=without_face, face=True
            )


if __name__ == "__main__":
    unittest.main()
