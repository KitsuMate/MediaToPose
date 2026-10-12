"""Face transfer onto the FaceFX fixture: shape keys, face bones and gaze.

Each test applies synthetic blendshape coefficients and checks what moved,
so the numbers below encode what the visual review of the fixture showed.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import bpy

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
        self.assertEqual(scheme("facs_ctrl_MouthPucker"), "ARKIT")
        self.assertEqual(scheme("facs_cbs_EyeFullCompressionLeft"), "NONE")
        self.assertEqual(scheme("facs_ctrl_vAA"), "NONE")
        self.assertEqual(scheme("AU12_L"), "FACS")

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
        frame = self.face.face_frame(self.rig, self.entries)
        self.assertAlmostEqual(frame.scale, 0.0591, delta=0.002)
        self.assertGreater(frame.up.z, 0.99)
        self.assertGreater(frame.lateral.x, 0.99)
        self.assertLess(frame.forward.y, -0.99)

    def test_bone_controls_move_in_face_space(self) -> None:
        self._apply(
            {"eyeBlinkLeft": 1.0, "browOuterUpLeft": 1.0, "mouthSmileLeft": 1.0},
            "BONES",
        )
        scale = self.face.face_frame(self.rig, self.entries).scale
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
        self.assertAlmostEqual(brow.z, 0.17 * scale, delta=0.05 * scale)
        self.assertAlmostEqual(corner.z, 0.17 * scale, delta=0.05 * scale)
        self.assertLess(abs(corner.x), 0.02 * scale)
        self.assertAlmostEqual(jaw.z, -0.55 * scale, delta=0.1 * scale)
        # ARP opens the jaw from the controller's position, not rotation.
        jaw_bone = self.rig.pose.bones["c_jawbone.x"]
        self.assertAlmostEqual(jaw_bone.rotation_quaternion.angle, 0.0, places=5)

    def test_bone_lids_follow_the_gaze(self) -> None:
        scale = self.face.face_frame(self.rig, self.entries).scale
        share = self.face.LID_FOLLOW_GAZE * 0.34 * scale
        self._apply({"eyeLookDownLeft": 1.0}, "BONES")
        down = self._world_offset("c_eyelid_top.l").z
        self.setUp()
        self._apply({"eyeLookUpLeft": 1.0}, "BONES")
        up = self._world_offset("c_eyelid_top.l").z
        self.assertAlmostEqual(down, -share, delta=0.3 * share)
        self.assertAlmostEqual(up, share, delta=0.3 * share)

    def test_eye_gaze_also_drives_eye_bones_with_shape_keys(self) -> None:
        self._apply({"eyeLookOutLeft": 1.0, "eyeLookInRight": 1.0}, "ARKIT")
        target = self._world_offset("c_eye_target.l")
        # Positive coefficients look to the subject's left (+X here).
        self.assertGreater(target.x, 0.0)
        tracker = self.face._gaze_tracker(
            self.rig, self.rig.pose.bones["c_eye_target.l"]
        )
        self.assertIsNotNone(tracker)
        distance = (
            self.rig.data.bones["c_eye_target.l"].head_local - tracker.bone.head_local
        ).length
        expected = distance * math.tan(self.face.GAZE_ANGLE)
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
        self, face_cleanup: bool, face_output: str = "ARKIT"
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
            for index, values in enumerate(_blink_clip())
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


if __name__ == "__main__":
    unittest.main()
