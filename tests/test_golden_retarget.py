"""Golden retarget snapshots on the VRoid fixture.

The cases replay recorded detections, so no inference runs. Each case stores
per-bone pose rotations and root location (still poses) or every written
keyframe (sequences) in ``tests/fixtures/golden/``. Run the Blender suite with
``--update-golden`` (or ``MTP_UPDATE_GOLDEN=1``) after an intended change and
review the JSON diff.
"""

from __future__ import annotations

import copy
import json
import math
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import bpy
from mathutils import Matrix, Quaternion

from tests.blender_test_support import (
    FIXTURES,
    fixture_path,
    register_addon,
    scene_context,
)

GOLDEN = FIXTURES / "golden"
SCHEMA_VERSION = 1
ROTATION_TOLERANCE_DEGREES = 0.1
LOCATION_TOLERANCE = 1.0e-3
KEY_VALUE_TOLERANCE = 2.0e-3
FRAME_TOLERANCE = 1.0e-4


def _update_requested() -> bool:
    return os.environ.get("MTP_UPDATE_GOLDEN") == "1"


def _round(values, digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values]


def _synthetic_sequence(source, count: int = 12):
    """Deterministic motion built from one recorded frame.

    The body sways sideways, the arms swing, and frame 7 carries a large
    wrist spike so the detection filter and outlier repair have work to do.
    """
    frames = []
    for index in range(count):
        frame = copy.deepcopy(source)
        frame.timestamp_ms = index * 40
        sway = 0.03 * math.sin(index * 0.6)
        swing = 0.05 * math.sin(index * 0.9)
        for name, point in frame.body.items():
            x, y, z = point.position
            offset = swing if "WRIST" in name or "ELBOW" in name else 0.0
            if "LEFT" in name:
                offset = -offset
            point.position = (x + sway, y + offset, z + 0.01 * math.cos(index))
        if index == 7 and "LEFT_WRIST" in frame.body:
            x, y, z = frame.body["LEFT_WRIST"].position
            frame.body["LEFT_WRIST"].position = (x + 0.6, y - 0.4, z)
        frames.append(frame)
    return frames


class GoldenRetargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.addon = register_addon()
        bpy.ops.wm.open_mainfile(filepath=str(fixture_path("vroid_test_rig")))
        cls.addon.register()
        cls.armature = next(
            obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"
        )
        cls.addon.mapping.auto_map(cls.armature)

    def setUp(self) -> None:
        for bone in self.armature.pose.bones:
            bone.matrix_basis = Matrix.Identity(4)
        config = self.armature.mtp_rig_config
        saved = [(item, item.enabled) for item in config.post_processing_modules]
        self.addCleanup(
            lambda: [setattr(item, "enabled", value) for item, value in saved]
        )
        # RoHM needs its external model bundle; keep the cases model-free.
        rohm = self.addon.post_processing.module_config(config, "ROHM_CORRECTION")
        rohm.enabled = False

    def _recorded(self, fixture_id: str):
        payload = json.loads(Path(fixture_path(fixture_id)).read_text("utf-8"))
        frame = self.addon.core.cache.frames.frame_from_plain(payload["frame"])
        settings = SimpleNamespace(
            provider=payload["provider"],
            use_body=True,
            use_hands=True,
            use_face=False,
            face_output="ARKIT",
            root_motion_mode="OFF",
            keep_original_root_position=False,
            keep_detection_cache=False,
        )
        return frame, settings

    def _pose_snapshot(self) -> dict:
        bones = {}
        for bone in self.armature.pose.bones:
            location, rotation, _scale = bone.matrix_basis.decompose()
            bones[bone.name] = {
                "rotation": _round(rotation),
                "location": _round(location),
            }
        return {"bones": bones}

    def _action_snapshot(self, action, result) -> dict:
        curves = {}
        for curve in self.addon.retarget.keyframes._action_fcurves(action):
            key = f"{curve.data_path}[{curve.array_index}]"
            curves[key] = [_round(point.co) for point in curve.keyframe_points]
        frames, bones, shapes, warnings = result
        return {
            "frames": frames,
            "bone_samples": bones,
            "shape_samples": shapes,
            "warnings": sorted(warnings),
            "curves": dict(sorted(curves.items())),
        }

    def _check(self, name: str, actual: dict) -> None:
        path = GOLDEN / f"{name}.json"
        payload = {"schema_version": SCHEMA_VERSION, "case": name, **actual}
        if _update_requested():
            GOLDEN.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, indent=1, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return
        if not path.is_file():
            self.fail(f"Missing golden snapshot {path}; run with --update-golden")
        expected = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(expected["schema_version"], SCHEMA_VERSION)
        if "bones" in expected:
            self._compare_pose(name, expected["bones"], actual["bones"])
        else:
            self._compare_action(name, expected, actual)

    def _compare_pose(self, name, expected, actual) -> None:
        self.assertEqual(set(expected), set(actual), name)
        for bone, values in expected.items():
            want = Quaternion(values["rotation"])
            have = Quaternion(actual[bone]["rotation"])
            angle = math.degrees(want.rotation_difference(have).angle)
            angle = min(angle, 360.0 - angle)
            self.assertLess(
                angle, ROTATION_TOLERANCE_DEGREES, f"{name}: {bone} rotation"
            )
            for want_axis, have_axis in zip(
                values["location"], actual[bone]["location"], strict=True
            ):
                self.assertAlmostEqual(
                    want_axis,
                    have_axis,
                    delta=LOCATION_TOLERANCE,
                    msg=f"{name}: {bone} location",
                )

    def _compare_action(self, name, expected, actual) -> None:
        for field in ("frames", "bone_samples", "shape_samples", "warnings"):
            self.assertEqual(expected[field], actual[field], f"{name}: {field}")
        self.assertEqual(set(expected["curves"]), set(actual["curves"]), name)
        for key, points in expected["curves"].items():
            have = actual["curves"][key]
            self.assertEqual(len(points), len(have), f"{name}: {key} key count")
            for (frame, value), (have_frame, have_value) in zip(
                points, have, strict=True
            ):
                self.assertAlmostEqual(frame, have_frame, delta=FRAME_TOLERANCE)
                self.assertAlmostEqual(
                    value,
                    have_value,
                    delta=KEY_VALUE_TOLERANCE,
                    msg=f"{name}: {key} at frame {frame}",
                )

    def _still(self, fixture_id: str, name: str) -> None:
        frame, settings = self._recorded(fixture_id)
        self.addon.retarget.prepare_limb_ownership(self.armature)
        self.addon.retarget.apply_frame(
            self.armature, frame, settings, scene_context=scene_context()
        )
        self._check(name, self._pose_snapshot())

    def _sequence(self, name: str, *, root_motion_mode: str, all_cleanup: bool):
        source, settings = self._recorded("mediapipe_ref_pose_realistic_output")
        settings.root_motion_mode = root_motion_mode
        config = self.armature.mtp_rig_config
        if all_cleanup:
            for module_id in (
                "SPINE_CORRECTION",
                "DETECTION_FILTER",
                "TEMPORAL_CORRECTION",
            ):
                self.addon.post_processing.module_config(
                    config, module_id
                ).enabled = True
            temporal = self.addon.post_processing.module_config(
                config, "TEMPORAL_CORRECTION"
            )
            state = (
                temporal.smooth_motion,
                temporal.ground_contacts,
                temporal.stabilize_hands,
            )
            temporal.smooth_motion = True
            temporal.ground_contacts = True
            temporal.stabilize_hands = True
            self.addCleanup(
                lambda: setattr(temporal, "smooth_motion", state[0])
                or setattr(temporal, "ground_contacts", state[1])
                or setattr(temporal, "stabilize_hands", state[2])
            )
        original = self.armature.animation_data_create().action
        staging = bpy.data.actions.new(f"Golden_{name}")
        self.armature.animation_data.action = staging
        scene = bpy.context.scene
        frame_before = scene.frame_current
        try:
            result = self.addon.retarget.apply_sequence(
                self.armature,
                iter(_synthetic_sequence(source)),
                settings,
                1.0,
                scene_context=scene_context(),
            )
            snapshot = self._action_snapshot(staging, result)
        finally:
            self.armature.animation_data.action = original
            bpy.data.actions.remove(staging)
            scene.frame_set(frame_before)
        self._check(name, snapshot)

    def test_still_mediapipe(self) -> None:
        self._still("mediapipe_ref_pose_realistic_output", "still_mediapipe")

    def test_still_sam(self) -> None:
        self._still("sam_ref_pose_stylized_output", "still_sam")

    def test_sequence_defaults_in_place(self) -> None:
        self._sequence(
            "sequence_defaults_off", root_motion_mode="OFF", all_cleanup=False
        )

    def test_sequence_all_cleanup_automatic(self) -> None:
        self._sequence(
            "sequence_cleanup_automatic",
            root_motion_mode="AUTOMATIC",
            all_cleanup=True,
        )


if __name__ == "__main__":
    unittest.main()
