"""Golden snapshots of face transfer.

Deterministic coefficients covering every ARKit name are applied to the
FaceFX face rig (ARKit shape keys and face bones) and to VRoid (VRM
expressions), and a short clip with a blink runs through the sequence
pipeline. The values every target ends up with are compared with
``tests/fixtures/golden/*.json``. Regenerate with ``--update-golden`` (or
``MTP_UPDATE_GOLDEN=1``) after an intended change and review the diff.
"""

from __future__ import annotations

import json
import math
import os
import unittest
from types import SimpleNamespace

import bpy

from tests.blender_test_support import (
    FIXTURES,
    fixture_path,
    register_addon,
    scene_context,
)

GOLDEN = FIXTURES / "golden"
SCHEMA_VERSION = 1
VALUE_TOLERANCE = 2.0e-3
LOCATION_TOLERANCE = 1.0e-4
ROTATION_TOLERANCE = 1.0e-3


def _coefficients(addon, phase: int = 0) -> dict[str, float]:
    """Every ARKit coefficient at a different, repeatable strength."""
    names = [
        name for name in addon.core.constants.FACE_BLENDSHAPES if name != "_neutral"
    ]
    return {
        name: round(0.8 * (((index + phase) * 37) % 100) / 100.0, 4)
        for index, name in enumerate(names)
    }


def _reset(rig) -> None:
    for bone in rig.pose.bones:
        bone.matrix_basis.identity()
    for obj in bpy.data.objects:
        keys = obj.data.shape_keys if obj.type == "MESH" else None
        for block in keys.key_blocks if keys else ():
            block.value = 0.0


def _shape_values() -> dict[str, float]:
    values = {}
    for obj in sorted(bpy.data.objects, key=lambda item: item.name):
        keys = obj.data.shape_keys if obj.type == "MESH" else None
        for block in keys.key_blocks[1:] if keys else ():
            if abs(block.value) > 1.0e-6:
                values[f"{obj.name}/{block.name}"] = round(block.value, 5)
    return values


def _bone_values(rig) -> dict[str, dict[str, list[float]]]:
    bones = {}
    for bone in rig.pose.bones:
        location, rotation, _scale = bone.matrix_basis.decompose()
        if location.length > 1.0e-7 or rotation.angle > 1.0e-6:
            bones[bone.name] = {
                "location": [round(v, 6) for v in location],
                "rotation": [round(v, 6) for v in rotation],
            }
    return dict(sorted(bones.items()))


class GoldenFaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.addon = register_addon()

    def _open(self, fixture_id: str, rig_name: str):
        bpy.ops.wm.open_mainfile(filepath=str(fixture_path(fixture_id)))
        self.addon.register()
        rig = bpy.data.objects[rig_name]
        self.addon.api.auto_map(rig.name)
        _reset(rig)
        return rig

    def _settings(self, face_output: str, **extra) -> SimpleNamespace:
        return SimpleNamespace(
            provider="MEDIAPIPE",
            use_body=False,
            use_hands=False,
            use_face=True,
            face_output=face_output,
            **extra,
        )

    def _still(self, rig, face_output: str) -> dict:
        frame = self.addon.core.types.CanonicalFrame(
            timestamp_ms=0, blendshapes=_coefficients(self.addon)
        )
        self.addon.retarget.apply_frame(
            rig,
            frame,
            self._settings(face_output),
            post_process=False,
            scene_context=scene_context(),
        )
        bpy.context.view_layer.update()
        return {"shapes": _shape_values(), "bones": _bone_values(rig)}

    def _check(self, name: str, actual: dict) -> None:
        path = GOLDEN / f"{name}.json"
        payload = {"schema_version": SCHEMA_VERSION, "case": name, **actual}
        if os.environ.get("MTP_UPDATE_GOLDEN") == "1":
            path.write_text(
                json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8"
            )
            return
        if not path.is_file():
            self.fail(f"Missing golden snapshot {path}; run with --update-golden")
        expected = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(expected["schema_version"], SCHEMA_VERSION)
        for key in ("shapes", "frames"):
            if key not in expected:
                continue
            self.assertEqual(set(expected[key]), set(actual[key]), f"{name}: {key}")
            for target, want in expected[key].items():
                have = actual[key][target]
                wants = want if isinstance(want, list) else [want]
                haves = have if isinstance(have, list) else [have]
                self.assertEqual(len(wants), len(haves), f"{name}: {target}")
                for a, b in zip(wants, haves, strict=True):
                    self.assertAlmostEqual(
                        a, b, delta=VALUE_TOLERANCE, msg=f"{name}: {target}"
                    )
        if "bones" in expected:
            self.assertEqual(
                set(expected["bones"]), set(actual["bones"]), f"{name}: bones"
            )
            for bone, want in expected["bones"].items():
                have = actual["bones"][bone]
                for a, b in zip(want["location"], have["location"], strict=True):
                    self.assertAlmostEqual(
                        a, b, delta=LOCATION_TOLERANCE, msg=f"{name}: {bone}"
                    )
                # q and -q are the same rotation.
                dot = abs(
                    sum(
                        x * y
                        for x, y in zip(want["rotation"], have["rotation"], strict=True)
                    )
                )
                self.assertAlmostEqual(
                    dot, 1.0, delta=ROTATION_TOLERANCE, msg=f"{name}: {bone}"
                )

    def test_still_face_arkit(self) -> None:
        rig = self._open("facefx_face_rig", "FaceFX.Rig")
        self._check("still_face_arkit", self._still(rig, "ARKIT"))

    def test_still_face_bones(self) -> None:
        rig = self._open("facefx_face_rig", "FaceFX.Rig")
        self._check("still_face_bones", self._still(rig, "BONES"))

    def test_still_face_vrm(self) -> None:
        rig = self._open("vroid_test_rig", "Armature")
        self._check("still_face_vrm", self._still(rig, "VRM"))

    def test_sequence_face_arkit(self) -> None:
        rig = self._open("facefx_face_rig", "FaceFX.Rig")
        config = rig.mtp_rig_config
        for item in config.post_processing_modules:
            item.enabled = item.module_id in {"FACE_CORRECTION", "TEMPORAL_CORRECTION"}
        context = scene_context()
        step_ms = 1000.0 / context.fps
        frames = []
        for index in range(24):
            values = _coefficients(self.addon, phase=index % 3)
            blink = {10: 0.7, 11: 0.95, 12: 0.6}.get(index, 0.05)
            values["eyeBlinkLeft"] = values["eyeBlinkRight"] = blink
            values["mouthSmileLeft"] = 0.3 + 0.1 * math.sin(index * 2.5)
            frames.append(
                self.addon.core.types.CanonicalFrame(
                    timestamp_ms=round(index * step_ms), blendshapes=values
                )
            )
        settings = self._settings(
            "ARKIT",
            media_kind="VIDEO",
            root_motion_mode="OFF",
            keep_original_root_position=False,
            keep_detection_cache=False,
        )
        self.addon.retarget.apply_sequence(
            rig, frames, settings, 1.0, scene_context=context
        )
        keys = bpy.data.objects["FaceFX"].data.shape_keys
        watched = (
            "arkit_EyeBlinkLeft",
            "arkit_MouthSmileLeft",
            "arkit_JawOpen",
            "arkit_BrowInnerUp",
        )
        names = [name for name in watched if name in keys.key_blocks]
        self.assertEqual(
            len(names), len(watched), [b.name for b in keys.key_blocks][:20]
        )
        sampled: dict[str, list[float]] = {name: [] for name in names}
        scene = bpy.context.scene
        for index in range(len(frames)):
            scene.frame_set(1 + index)
            for name in names:
                sampled[name].append(round(keys.key_blocks[name].value, 5))
        self._check("sequence_face_arkit", {"frames": sampled})


if __name__ == "__main__":
    unittest.main()
