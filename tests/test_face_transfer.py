"""Face transfer onto the FaceFX and VRoid fixtures: keys, bones and gaze.

Each test applies synthetic blendshape coefficients and checks what moved,
so the numbers below encode what the visual review of the fixtures showed.
"""

from __future__ import annotations

import json
import math
import unittest

import bpy

from tests.blender_test_support import apply_face, open_rig, register_addon


def _reset_face(rig) -> None:
    for bone in rig.pose.bones:
        bone.matrix_basis.identity()
    for obj in bpy.data.objects:
        keys = obj.data.shape_keys if obj.type == "MESH" else None
        for block in keys.key_blocks if keys else ():
            block.value = 0.0
    bpy.context.view_layer.update()


class FaceTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.addon = register_addon()
        cls.rig = open_rig(
            cls.addon,
            "facefx_face_rig",
            "FaceFX.Rig",
            use_face=True,
            use_body=False,
            use_hands=False,
        )
        cls.entries = cls.addon.rig.mapping.mapping_by_id(cls.rig.mtp_rig_config)
        cls.defaults = cls.addon.rig.face_defaults
        cls.scale = cls.addon.rig.face_space.face_frame(cls.rig, cls.entries).scale

    def setUp(self) -> None:
        _reset_face(self.rig)

    def _apply(self, blendshapes: dict[str, float], face_output: str) -> None:
        apply_face(self.addon, self.rig, blendshapes, face_output)

    def _world_offset(self, bone_name: str):
        bone = self.rig.pose.bones[bone_name]
        return self.rig.matrix_world.to_3x3() @ (
            bone.matrix.translation - bone.bone.matrix_local.translation
        )

    def _travel(self, semantic_id: str, side: int = 0) -> float:
        """A control's default travel in metres (side 1: the negative side)."""
        return abs(self.defaults.FACE_CONTROL_TRAVEL[semantic_id][side]) * self.scale

    def _entry(self, semantic_id: str):
        """The mapping entry, restored after the test."""
        entry = self.entries[semantic_id]
        saved = {
            name: getattr(entry, name)
            for name in (*self.defaults.RANGE_FIELDS, "ranges_version")
        }
        saved.update(source=entry.source, manual_lock=entry.manual_lock)

        def restore() -> None:
            with self.addon.rig.locks.suppress_manual_locks():
                for name, value in saved.items():
                    setattr(entry, name, value)

        self.addCleanup(restore)
        return entry

    def test_describe_rig(self) -> None:
        described = self.addon.api.describe_rig(self.rig.name)
        self.assertEqual(described["detected_profile"], "AUTO_RIG_PRO")
        face = described["sections"]["FACE_RIG"]
        self.assertEqual(face["mapped"], face["enabled"])
        self.assertGreater(described["completeness"]["arkit"], 0.9)
        self.assertEqual(described["completeness"]["face"], 1.0)
        # Correctives and other unclassified keys are not a scheme.
        self.assertNotIn("NONE", described["shape_keys"]["FaceFX"]["schemes"])
        settings = bpy.context.scene.mtp_settings
        settings.use_face = False
        self.addCleanup(setattr, settings, "use_face", True)
        off = self.addon.api.describe_rig(self.rig.name)["completeness"]
        self.assertIsNone(off["arkit"])
        self.assertIsNone(off["face"])

    def test_prefixed_arkit_names_and_correctives(self) -> None:
        scheme = self.addon.rig.face_schemes.shape_scheme
        for name, expected in (
            ("facs_bs_EyeBlinkLeft", "ARKIT"),
            ("arkit_EyeBlinkLeft", "ARKIT"),
            ("blendShape.EyeBlinkLeft", "ARKIT"),
            ("facs_ctrl_MouthPucker", "ARKIT"),
            ("blendShape.facs_cbs_EyeBlinkLeft", "NONE"),
            ("facs_cbs_EyeFullCompressionLeft", "NONE"),
            ("facs_ctrl_vAA", "NONE"),
            ("AU12_L", "FACS"),
            ("AU45", "FACS"),
            ("Fcl_EYE_Close_L", "VRM"),
            ("Fcl_MTH_A", "VRM"),
            ("vrm_blink", "VRM"),
        ):
            with self.subTest(name=name):
                self.assertEqual(scheme(name), expected)
        for name in ("Basis", "Smile_extra", "mouthSmileLeft_corrective"):
            with self.subTest(name=name):
                self.assertNotIn(scheme(name), {"FACS", "VRM"})

    def test_blink_drives_one_key_on_every_mesh_that_has_one(self) -> None:
        self._apply({"eyeBlinkLeft": 1.0}, "ARKIT")
        normalize = self.addon.rig.names.normalize_name
        for obj in bpy.data.objects:
            keys = obj.data.shape_keys if obj.type == "MESH" else None
            if keys is None:
                continue
            blink = [
                b
                for b in keys.key_blocks
                if normalize(b.name, shape=True) == "eyeblinkleft"
            ]
            driven = [b.name for b in blink if b.value > 0.5]
            with self.subTest(mesh=obj.name):
                # The face carries arkit_ and facs_bs_ copies; driving both
                # would close the eye twice as hard.
                self.assertEqual(len(driven), 1 if blink else 0, driven)
                if driven and any(b.name.startswith("arkit_") for b in blink):
                    self.assertTrue(driven[0].startswith("arkit_"))

    def test_facs_jaw_action_units_follow_their_definitions(self) -> None:
        facs = self.addon.retarget.solve.face.facs_values
        parted = facs({"mouthLowerDownLeft": 0.8, "mouthLowerDownRight": 0.8})
        self.assertAlmostEqual(parted["AU25"], 0.8)
        self.assertEqual(parted["AU26"], 0.0)
        half = facs({"jawOpen": 0.5})
        self.assertEqual((half["AU26"], half["AU27"]), (0.5, 0.0))
        self.assertEqual(facs({"jawOpen": 1.0})["AU27"], 1.0)
        self.assertEqual(facs({"mouthStretchLeft": 1.0})["AU27"], 0.0)

    def test_face_frame_measures_the_eyes_not_their_targets(self) -> None:
        frame = self.addon.rig.face_space.face_frame(self.rig, self.entries)
        # The FaceFX eyes are 5.9 cm apart; their look-at targets are wider.
        self.assertAlmostEqual(frame.scale, 0.0591, delta=0.002)
        self.assertGreater(frame.up.z, 0.99)
        self.assertGreater(frame.lateral.x, 0.99)
        self.assertLess(frame.forward.y, -0.99)

    def test_bone_controls_move_in_face_space(self) -> None:
        self._apply(
            {"eyeBlinkLeft": 1.0, "browOuterUpLeft": 1.0, "mouthSmileLeft": 1.0},
            "BONES",
        )
        lid = self._world_offset("c_eyelid_top.l")
        brow = self._world_offset("c_eyebrow_01_end.l")
        corner = self._world_offset("c_lips_smile.l")
        # The jaw drags the mouth corners down through ARP's retain bones,
        # so it is checked on its own.
        _reset_face(self.rig)
        self._apply({"jawOpen": 1.0}, "BONES")
        jaw = self._world_offset("c_jawbone.x")
        # Each moves mostly up or down, by about its travel. Retain bones and
        # constraints keep the rig from matching the travel exactly.
        travel = self._travel
        self.assertAlmostEqual(lid.z, -travel("FACE_LID_L"), delta=0.1 * self.scale)
        self.assertAlmostEqual(
            brow.z, travel("FACE_BROW_OUTER_L"), delta=0.01 * self.scale
        )
        self.assertAlmostEqual(
            corner.z, travel("FACE_MOUTH_CORNER_L"), delta=0.03 * self.scale
        )
        self.assertLess(abs(corner.x), 0.02 * self.scale)
        self.assertAlmostEqual(jaw.z, -travel("FACE_JAW_OPEN"), delta=0.1 * self.scale)
        # ARP opens the jaw from the controller's position, not rotation.
        jaw_bone = self.rig.pose.bones["c_jawbone.x"]
        self.assertAlmostEqual(jaw_bone.rotation_quaternion.angle, 0.0, places=5)

    def test_bone_lids_follow_the_gaze(self) -> None:
        # The follow share goes through the lid's input range.
        full = self.defaults.FACE_CONTROL_INPUT_FULL["FACE_LID_L"]
        follow = self.addon.retarget.solve.face.LID_FOLLOW_GAZE / full
        lower = follow * self._travel("FACE_LID_L")
        lift = follow * self._travel("FACE_LID_L", side=1)
        self._apply({"eyeLookDownLeft": 1.0}, "BONES")
        down = self._world_offset("c_eyelid_top.l").z
        _reset_face(self.rig)
        self._apply({"eyeLookUpLeft": 1.0}, "BONES")
        up = self._world_offset("c_eyelid_top.l").z
        self.assertAlmostEqual(down, -lower, delta=0.3 * lower)
        self.assertAlmostEqual(up, lift, delta=0.3 * lift)

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
        entry.output_full = 0.012
        self._apply({"eyeBlinkLeft": 1.0}, "BONES")
        self.assertAlmostEqual(
            self._world_offset("c_eyelid_top.l").z, -0.012, delta=1e-4
        )

    def test_mappings_from_0_2_0_are_migrated(self) -> None:
        entry = self._entry("ARKIT_mouthSmileLeft")
        lid = self._entry("FACE_LID_L")
        with self.addon.rig.locks.suppress_manual_locks():
            for item in (entry, lid):
                item.ranges_version = self.defaults.RANGES_LEGACY
            entry["gain"] = 2.0
            entry["offset"] = 0.1
            lid["clamp_max"] = 0.5
        self.addCleanup(lambda: [entry.pop(k, None) for k in ("gain", "offset")])
        self.addCleanup(lambda: lid.pop("clamp_max", None))
        self.addon.rig.mapping.ensure_mapping_entries(self.rig)
        self.assertEqual(entry.ranges_version, self.defaults.RANGES_VERSION)
        self.assertAlmostEqual(entry.input_zero, 0.1)
        self.assertAlmostEqual(entry.input_full, 0.6)
        self.assertAlmostEqual(lid.input_full, 0.5)
        self.assertAlmostEqual(
            lid.output_full, 0.5 * self._travel("FACE_LID_L"), delta=1e-6
        )

    def test_schema_18_presets_are_migrated(self) -> None:
        presets = self.addon.rig.presets
        payload = presets.serialize_config(self.rig)
        legacy = json.loads(json.dumps(payload))
        legacy["schema_version"] = 18
        for item in legacy["mappings"]:
            for name in self.defaults.RANGE_FIELDS:
                item.pop(name, None)
            item.update(
                gain=1.0, offset=0.0, clamp_min=0.0, clamp_max=1.0, invert=False
            )
            if item["semantic_id"] == "ARKIT_jawOpen":
                item["gain"] = 4.0
        entry = self._entry("ARKIT_jawOpen")
        lid = self._entry("FACE_LID_L")
        self.addCleanup(presets.load_config, self.rig, payload)
        presets.load_config(self.rig, legacy)
        self.assertAlmostEqual(entry.input_full, 0.25)
        self.assertAlmostEqual(lid.output_full, self._travel("FACE_LID_L"), delta=1e-6)

    def test_auto_map_keeps_edited_ranges_until_forced(self) -> None:
        self.addon.api.auto_map(self.rig.name, force_profile=True)
        entry = self._entry("FACE_JAW_SIDE")
        default = entry.output_full
        # A range set without a manual lock, as fitting to the rig's keys
        # would: the source stays PROFILE.
        with self.addon.rig.locks.suppress_manual_locks():
            entry.output_full = default * 1.5
        self.addon.api.auto_map(self.rig.name)
        self.assertEqual(entry.source, "PROFILE")
        self.assertAlmostEqual(entry.output_full, default * 1.5)
        self.addon.api.auto_map(self.rig.name, force_profile=True)
        self.assertAlmostEqual(entry.output_full, default)

    def test_eye_gaze_also_drives_eye_bones_with_shape_keys(self) -> None:
        self._apply({"eyeLookOutLeft": 1.0, "eyeLookInRight": 1.0}, "ARKIT")
        target = self._world_offset("c_eye_target.l")
        tracker = self.addon.rig.face_space.gaze_tracker(
            self.rig, self.rig.pose.bones["c_eye_target.l"]
        )
        self.assertIsNotNone(tracker)
        distance = (
            self.rig.data.bones["c_eye_target.l"].head_local - tracker.bone.head_local
        ).length
        # Positive coefficients look to the subject's left (+X here), by the
        # default gaze angle.
        expected = distance * math.tan(math.radians(self.defaults.GAZE_DEGREES))
        self.assertAlmostEqual(target.x, expected, delta=0.1 * expected)


class VroidExpressionTests(unittest.TestCase):
    """VRM / VRoid expression keys driven from ARKit coefficients."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.addon = register_addon()
        cls.rig = open_rig(
            cls.addon, "vroid_test_rig", "Armature", use_face=True, face_output="VRM"
        )
        cls.keys = bpy.data.objects["Face"].data.shape_keys.key_blocks

    def _apply(self, blendshapes: dict[str, float]) -> None:
        _reset_face(self.rig)
        apply_face(self.addon, self.rig, blendshapes, "VRM")

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
        self._apply({"mouthPucker": 1.0, "mouthSmileLeft": 1.0, "mouthSmileRight": 1.0})
        self.assertAlmostEqual(self.keys["Fcl_MTH_U"].value, 1.0, places=3)
        self.assertAlmostEqual(self.keys["Fcl_MTH_Joy"].value, 1.0, places=3)
        self.assertEqual(self.keys["Fcl_MTH_A"].value, 0.0)


if __name__ == "__main__":
    unittest.main()
