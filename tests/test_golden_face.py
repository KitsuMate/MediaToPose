"""Golden snapshots of face transfer.

Deterministic coefficients covering every ARKit name are applied to the
FaceFX face rig (ARKit shape keys, face bones and generated FACS targets)
and to VRoid (VRM expressions), and a short clip with a blink runs through
the sequence pipeline. The values every target ends up with are compared
with ``tests/fixtures/golden/*.json``. Regenerate with ``--update-golden``
(or ``MTP_UPDATE_GOLDEN=1``) after an intended change and review the diff.
"""

from __future__ import annotations

import math
import unittest

import bpy

from tests.blender_test_support import (
    add_facs_targets,
    apply_face,
    assert_pose_close,
    face_settings,
    fixture_path,
    golden_snapshot,
    open_rig,
    register_addon,
    scene_context,
)

VALUE_TOLERANCE = 2.0e-3
LOCATION_TOLERANCE = 1.0e-4


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


class _GoldenFace(unittest.TestCase):
    fixture = ("facefx_face_rig", "FaceFX.Rig")

    @classmethod
    def setUpClass(cls) -> None:
        cls.addon = register_addon()
        cls.rig = open_rig(cls.addon, *cls.fixture)

    def setUp(self) -> None:
        _reset(self.rig)

    def _still(self, face_output: str) -> dict:
        apply_face(self.addon, self.rig, _coefficients(self.addon), face_output)
        return {"shapes": _shape_values(), "bones": _bone_values(self.rig)}

    def _check(self, name: str, actual: dict) -> None:
        expected = golden_snapshot(self, name, actual)
        if expected is None:
            return
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
            assert_pose_close(
                self,
                name,
                expected["bones"],
                actual["bones"],
                location_tolerance=LOCATION_TOLERANCE,
            )


class GoldenFaceFxTests(_GoldenFace):
    def test_still_face_arkit(self) -> None:
        actual = self._still("ARKIT")
        # The other meshes carry copies of the face's keys (with any prefix)
        # and follow it exactly; only the face itself is stored.
        normalize = self.addon.rig.names.normalize_name
        face = {
            normalize(key.split("/", 1)[1], shape=True): value
            for key, value in actual["shapes"].items()
            if key.startswith("FaceFX/")
        }
        for key, value in actual["shapes"].items():
            mesh, name = key.split("/", 1)
            if mesh != "FaceFX":
                self.assertAlmostEqual(
                    value, face[normalize(name, shape=True)], msg=key
                )
        actual["shapes"] = {
            key: value
            for key, value in actual["shapes"].items()
            if key.startswith("FaceFX/")
        }
        self._check("still_face_arkit", actual)

    def test_still_face_bones(self) -> None:
        self._check("still_face_bones", self._still("BONES"))

    def test_sequence_face_arkit(self) -> None:
        config = self.rig.mtp_rig_config
        for item in config.post_processing_modules:
            item.enabled = item.module_id == "TEMPORAL_CORRECTION"
        context = scene_context()
        step_ms = 1000.0 / context.fps
        frames = []
        for index in range(12):
            values = _coefficients(self.addon, phase=index % 3)
            blink = {6: 0.7, 7: 0.95, 8: 0.6}.get(index, 0.05)
            values["eyeBlinkLeft"] = values["eyeBlinkRight"] = blink
            values["mouthSmileLeft"] = 0.3 + 0.1 * math.sin(index * 2.5)
            frames.append(
                self.addon.core.types.CanonicalFrame(
                    timestamp_ms=round(index * step_ms), blendshapes=values
                )
            )
        settings = face_settings(
            "ARKIT",
            media_kind="VIDEO",
            root_motion_mode="OFF",
            keep_original_root_position=False,
            keep_detection_cache=False,
        )
        self.addon.retarget.apply_sequence(
            self.rig, frames, settings, 1.0, scene_context=context
        )
        keys = bpy.data.objects["FaceFX"].data.shape_keys
        self.addCleanup(keys.animation_data_clear)
        self.addCleanup(self.rig.animation_data_clear)
        names = (
            "arkit_EyeBlinkLeft",
            "arkit_MouthSmileLeft",
            "arkit_JawOpen",
            "arkit_BrowInnerUp",
        )
        sampled: dict[str, list[float]] = {name: [] for name in names}
        scene = bpy.context.scene
        for index in range(len(frames)):
            scene.frame_set(1 + index)
            for name in names:
                sampled[name].append(round(keys.key_blocks[name].value, 5))
        self._check("sequence_face_arkit", {"frames": sampled})


class GoldenFacsTests(_GoldenFace):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        add_facs_targets(cls.addon, cls.rig)

    def test_still_face_facs(self) -> None:
        actual = self._still("FACS")
        self.assertTrue(actual["shapes"], "FACS must drive shape keys")
        self._check("still_face_facs", actual)

    def test_real_face_reaches_facs_targets(self) -> None:
        result = self.addon.api.capture(
            str(fixture_path("face_laugh_jaw_open")),
            self.rig.name,
            parts={"face"},
            face_output="FACS",
            keep_cache=False,
        )
        self.assertTrue(result.succeeded, result.message)
        keys = bpy.data.objects["FaceFX"].data.shape_keys.key_blocks
        self.assertGreater(keys["AU26"].value, 0.1)


class GoldenVrmTests(_GoldenFace):
    fixture = ("vroid_test_rig", "Armature")

    def test_still_face_vrm(self) -> None:
        self._check("still_face_vrm", self._still("VRM"))


if __name__ == "__main__":
    unittest.main()
