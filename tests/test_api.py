"""The scripting API and command line on the VRoid fixture, without UI."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import bpy
from mathutils import Matrix

from tests.blender_test_support import fixture_path, register_addon
from tests.test_capture_session import IMAGE, _write_video

_DIRECTORY = tempfile.TemporaryDirectory()
ROOT = Path(_DIRECTORY.name)
VIDEO = ROOT / "clip.avi"
OPTIONS = {"provider": "MEDIAPIPE", "parts": "body,hands", "root_motion": "IN_PLACE"}


def setUpModule() -> None:
    _write_video(VIDEO, frames=10)


def tearDownModule() -> None:
    _DIRECTORY.cleanup()


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.addon = register_addon()
        bpy.ops.wm.open_mainfile(filepath=str(fixture_path("vroid_test_rig")))
        cls.addon.register()
        cls.api = cls.addon.api
        cls.armature = next(
            obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"
        )
        cls.addon.rig.mapping.auto_map(cls.armature)
        rohm = cls.addon.postprocess.module_config(
            cls.armature.mtp_rig_config, "ROHM_CORRECTION"
        )
        rohm.enabled = False
        cls.root = ROOT
        cls.video = VIDEO
        cls.options = OPTIONS
        cls.first = cls.api.capture(
            str(cls.video),
            cls.armature.name,
            keep_cache=True,
            result_path=cls.root / "first.json",
            **cls.options,
        )

    def setUp(self) -> None:
        bpy.context.view_layer.objects.active = self.armature
        bpy.context.scene.frame_set(1)

    def _reset_pose(self) -> None:
        for bone in self.armature.pose.bones:
            bone.matrix_basis = Matrix.Identity(4)

    def test_capture_result_describes_the_animation(self) -> None:
        result = self.first
        self.assertTrue(result.succeeded, result.message)
        self.assertEqual(result.frames_processed, 10)
        self.assertEqual(result.mode, "KEYFRAMES")
        self.assertTrue(result.action.endswith("_MediaToPose"))
        self.assertEqual(self.armature.animation_data.action.name, result.action)
        self.assertEqual(result.source_frame_range, [0, 10])
        self.assertEqual(len(result.confidence["per_frame"]), 10)
        self.assertGreater(result.confidence["per_part"]["body"], 0.5)
        modules = {step["module_id"] for step in result.cleanup_steps}
        self.assertIn("TEMPORAL_CORRECTION", modules)
        self.assertNotIn("ROHM_CORRECTION", modules)
        for warning in result.warnings:
            self.assertTrue(warning["code"])
        stored = json.loads((self.root / "first.json").read_text(encoding="utf8"))
        self.assertEqual(stored["schema_version"], 1)
        self.assertEqual(stored["action"], result.action)

    def test_evaluate_then_retarget_from_cache_with_new_settings(self) -> None:
        before = self.api.evaluate(self.armature.name, result=self.first)
        self.assertEqual(before.frames, 10)
        self.assertIsNotNone(before.landmark_fit_error_deg)
        self.assertGreaterEqual(before.jitter_mm, 0.0)
        smoothed = self.api.retarget_cached(
            str(self.video),
            self.armature.name,
            config={
                "TEMPORAL_CORRECTION": {
                    "enabled": True,
                    "settings": {"smooth_motion": True, "smoothing_strength": 1.0},
                }
            },
            **self.options,
        )
        self.assertTrue(smoothed.cached)
        temporal = next(
            step
            for step in smoothed.cleanup_steps
            if step["module_id"] == "TEMPORAL_CORRECTION"
        )
        self.assertEqual(temporal["parameters"]["smoothing_strength"], 1.0)
        after = self.api.evaluate(self.armature.name, result=smoothed)
        self.assertNotEqual(before.to_dict(), after.to_dict())
        # The rig's own settings are unchanged by a per-run config.
        stored = self.addon.postprocess.module_config(
            self.armature.mtp_rig_config, "TEMPORAL_CORRECTION"
        )
        self.assertNotEqual(stored.smoothing_strength, 1.0)

    def test_output_modes(self) -> None:
        animation = self.armature.animation_data
        current = animation.action
        name = current.name
        replaced = self.api.retarget_cached(
            str(self.video), self.armature.name, output="REPLACE", **self.options
        )
        self.assertEqual(replaced.action, name)
        self.assertEqual(animation.action.name, name)
        tracks = len(animation.nla_tracks)
        stripped = self.api.retarget_cached(
            str(self.video), self.armature.name, output="NLA_STRIP", **self.options
        )
        self.assertEqual(len(animation.nla_tracks), tracks + 1)
        self.assertEqual(stripped.nla_track, "Media to Pose")
        self.assertEqual(animation.action.name, name)
        strip = animation.nla_tracks[-1].strips[0]
        self.assertEqual(strip.action.name, stripped.action)
        animation.nla_tracks.remove(animation.nla_tracks[-1])

    def test_frame_range_keys_only_those_frames(self) -> None:
        result = self.api.retarget_cached(
            str(self.video), self.armature.name, frame_range=(3, 6), **self.options
        )
        self.assertEqual(result.frames_processed, 3)
        self.assertEqual(result.source_frame_range, [3, 6])
        timeline = [item["timeline_frame"] for item in result.confidence["per_frame"]]
        full = [item["timeline_frame"] for item in self.first.confidence["per_frame"]]
        self.assertEqual(timeline, full[3:6])

    def test_retarget_cached_without_detections_fails_cleanly(self) -> None:
        with self.assertRaises(self.api.InvalidRequest):
            self.api.retarget_cached(
                str(self.video), self.armature.name, provider="MEDIAPIPE", parts="body"
            )
        self.assertFalse(self.addon.core.coordinator.COORDINATOR.is_busy())

    def test_describe_rig_and_configuration(self) -> None:
        described = self.api.describe_rig(self.armature.name)
        self.assertEqual(described["armature"], self.armature.name)
        self.assertGreater(described["sections"]["BODY"]["coverage"], 0.8)
        self.assertIn("completeness", described)
        modules = self.api.list_postprocessors(self.armature.name)
        self.assertTrue(all("parameters" in module for module in modules))
        self.assertTrue(all("enabled" in module for module in modules))
        config = self.api.get_config(self.armature.name)
        stored = self.addon.postprocess.module_config(
            self.armature.mtp_rig_config, "TEMPORAL_CORRECTION"
        )
        enabled = stored.enabled
        changed = self.api.set_config(
            self.armature.name,
            {"post_processing": {"TEMPORAL_CORRECTION": {"smoothing_strength": 0.2}}},
        )
        module = next(
            item
            for item in changed["post_processing"]
            if item["module_id"] == "TEMPORAL_CORRECTION"
        )
        self.assertAlmostEqual(module["settings"]["smoothing_strength"], 0.2, places=5)
        self.assertEqual(module["enabled"], enabled)
        with self.assertRaises(self.api.InvalidRequest):
            self.api.set_config(
                self.armature.name,
                {"post_processing": {"TEMPORAL_CORRECTION": {"bogus": 1}}},
            )
        with self.assertRaises(self.api.InvalidRequest):
            self.api.set_config(self.armature.name, {"mappings": []})
        self.api.set_config(self.armature.name, config)
        self.assertEqual(self.api.get_config(self.armature.name), config)

    def test_providers_and_models(self) -> None:
        providers = {item["id"]: item for item in self.api.list_providers()}
        self.assertTrue(providers["MEDIAPIPE"]["available"])
        self.assertIn("body", providers["MEDIAPIPE"]["parts"])
        status = self.api.model_status()
        self.assertEqual(set(status), {"SAM3D_BODY", "ANYCALIB", "ROHM"})
        self.assertGreater(status["SAM3D_BODY"]["download_bytes"], 0)

    def test_detections_round_trip_through_files(self) -> None:
        exported = self.api.export_detections(self.root / "detections")
        self.assertEqual(exported["frames"], 10)
        self.assertTrue(Path(exported["npz"]).is_file())
        imported = self.api.import_detections(exported["json"])
        self.assertEqual(imported["cache_key"], exported["cache_key"])
        result = self.api.retarget_cached(
            str(self.video), self.armature.name, **self.options
        )
        self.assertEqual(result.frames_processed, 10)

    def test_render_comparison_writes_a_contact_sheet(self) -> None:
        written = self.api.render_comparison(
            self.root / "sheet.png",
            self.armature.name,
            result=self.first,
            count=2,
            size=128,
        )
        self.assertEqual(len(written["frames"]), 2)
        self.assertGreater(Path(written["path"]).stat().st_size, 1000)
        self.assertIsNone(bpy.data.objects.get("MediaToPose_Comparison"))

    def test_pose_capture_and_operator_execute(self) -> None:
        self._reset_pose()
        result = self.api.capture(
            str(fixture_path(IMAGE)), self.armature.name, parts="body", keep_cache=False
        )
        self.assertEqual(result.mode, "POSE")
        self.assertEqual(result.frames_processed, 1)
        settings = bpy.context.scene.mtp_settings
        settings.media_path = str(fixture_path(IMAGE))
        settings.target_armature = self.armature
        self._reset_pose()
        self.assertEqual(bpy.ops.mtp.apply_capture(mode="POSE"), {"FINISHED"})
        self.assertTrue(
            any(
                bone.matrix_basis != Matrix.Identity(4)
                for bone in self.armature.pose.bones
            )
        )


class CommandLineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.addon = register_addon()
        cls.cli = cls.addon.api.cli
        cls.armature = next(
            obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"
        )

    def _run(self, *argv: str) -> tuple[int, str]:
        output = io.StringIO()
        with redirect_stdout(output):
            code = self.cli.execute(list(argv))
        return code, output.getvalue()

    def test_json_commands(self) -> None:
        code, output = self._run(
            "describe-rig", "--armature", self.armature.name, "--json"
        )
        self.assertEqual(code, 0, output)
        self.assertEqual(json.loads(output)["armature"], self.armature.name)
        code, output = self._run("postprocessors", "--json")
        self.assertEqual(code, 0, output)
        self.assertTrue(json.loads(output))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "config.json"
            code, _output = self._run(
                "config", "get", "--armature", self.armature.name, "--out", str(target)
            )
            self.assertEqual(code, 0)
            code, output = self._run(
                "config", "set", str(target), "--armature", self.armature.name, "--json"
            )
            self.assertEqual(code, 0, output)

    def test_errors_map_to_exit_codes(self) -> None:
        code, output = self._run("describe-rig", "--armature", "Missing", "--json")
        self.assertEqual(code, self.addon.core.errors.InvalidRequest.exit_code)
        self.assertEqual(json.loads(output)["error"]["code"], "INVALID_REQUEST")
        code, _output = self._run("not-a-command")
        self.assertEqual(code, self.addon.core.errors.InvalidRequest.exit_code)
        code, _output = self._run("capture", "--json")
        self.assertNotEqual(code, 0)

    def test_capture_command_writes_the_result(self) -> None:
        target = ROOT / "cli_result.json"
        code, output = self._run(
            "capture",
            "--media",
            str(VIDEO),
            "--armature",
            self.armature.name,
            "--parts",
            OPTIONS["parts"],
            "--root-motion",
            OPTIONS["root_motion"],
            "--out",
            str(target),
            "--json",
        )
        self.assertEqual(code, 0, output)
        printed = json.loads(output)
        self.assertEqual(printed["status"], "SUCCEEDED")
        self.assertEqual(json.loads(target.read_text(encoding="utf8")), printed)
        code, output = self._run(
            "evaluate",
            "--armature",
            self.armature.name,
            "--result",
            str(target),
            "--json",
        )
        self.assertEqual(code, 0, output)
        self.assertEqual(json.loads(output)["frames"], 10)

    def test_command_is_registered(self) -> None:
        registration = self.addon.blender.registration
        self.assertIn(registration._CLI_KEY, bpy.app.driver_namespace)


if __name__ == "__main__":
    unittest.main()
