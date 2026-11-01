"""Kimodo generation on the VRoid fixture with a stub denoiser (no model).

The stub predicts the statistics' mean, a standing T-pose, for every frame,
so the whole path runs: guides, sampling, retiming, posing, the guide warp
and keys, through the API and the command line.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import bpy
import numpy as np

from tests.blender_test_support import fixture_path, register_addon, run_cli


def _standing_stats(rep, soma):
    mean = np.zeros(rep.FEATURE_COUNT)
    height = soma.REST_HIP_HEIGHT
    mean[rep.SMOOTH_ROOT] = (0.0, height, 0.0)
    mean[rep.HEADING] = (1.0, 0.0)
    mean[rep.POSITIONS] = (soma.REST_POSITIONS + (0.0, height, 0.0)).reshape(-1)
    identity = np.broadcast_to(np.eye(3), (soma.JOINT_COUNT, 3, 3))
    mean[rep.ROTATIONS] = rep.matrix_to_rot6d(identity).reshape(-1)
    mean[rep.CONTACTS] = 1.0
    return rep.FeatureStats(mean, np.ones(rep.FEATURE_COUNT))


class _StubSession:
    def __init__(self, stats):
        self.stats = stats
        self.calls = 0

    def __call__(self, inputs):
        self.calls += 1
        return np.zeros(np.shape(inputs.motion), dtype=np.float32)


class GenerateTests(unittest.TestCase):
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
        name = cls.addon.__name__
        rep = __import__(f"{name}.core.motion.kimodo.rep", fromlist=["rep"])
        soma = __import__(f"{name}.core.motion.soma", fromlist=["soma"])
        cls.kimodo = __import__(f"{name}.backends.kimodo.backend", fromlist=["b"])
        cls.llm2vec = __import__(f"{name}.backends.llm2vec.backend", fromlist=["b"])
        cls.llm2vec_runtime = __import__(
            f"{name}.backends.llm2vec.runtime", fromlist=["r"]
        )
        cls.stub = _StubSession(_standing_stats(rep, soma))
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def setUp(self) -> None:
        animation = self.armature.animation_data
        if animation is not None:
            animation.action = None
            for track in tuple(animation.nla_tracks):
                animation.nla_tracks.remove(track)
        for bone in self.armature.pose.bones:
            bone.matrix_basis.identity()
        bpy.context.view_layer.objects.active = self.armature
        bpy.context.scene.frame_set(1)
        encoding = self.llm2vec_runtime.Encoding(
            np.zeros(4096, dtype=np.float32), "A person walks.", False
        )
        patches = (
            mock.patch.object(
                self.kimodo, "ensure_model_loaded", return_value=self.stub
            ),
            mock.patch.object(self.llm2vec, "encode_prompt", return_value=encoding),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def _bone(self, semantic: str):
        entries = self.addon.rig.mapping.mapping_by_id(self.armature.mtp_rig_config)
        return self.armature.pose.bones[entries[semantic].target_name]

    def test_dry_run_plans_without_a_model(self) -> None:
        self.stub.calls = 0
        plan = self.api.generate(
            self.armature.name,
            "A person walks.",
            frame_range=(1, 48),
            waypoints=[(1, 0.0, 0.0), (48, 0.0, -2.0)],
            dry_run=True,
        )
        self.assertEqual(plan.model_frames, 60)
        self.assertEqual([g["kind"] for g in plan.guides], ["ROOT", "ROOT"])
        self.assertTrue(plan.needs_text_encoder)
        self.assertEqual(self.stub.calls, 0)

    def test_generation_keys_the_range(self) -> None:
        result = self.api.generate(
            self.armature.name,
            "A person walks.",
            frame_range=(1, 24),
            steps=4,
            result_path=self.root / "walk.json",
        )
        self.assertTrue(result.succeeded)
        self.assertGreater(self.stub.calls, 0)
        self.assertGreater(result.keys_written, 0)
        payload = json.loads((self.root / "walk.json").read_text(encoding="utf8"))
        self.assertEqual(payload["model_frames"], 30)
        action = self.armature.animation_data.action
        self.assertIsNotNone(action)
        self.assertEqual(tuple(round(v) for v in action.frame_range), (1, 24))
        # The stub stands in a T-pose: the arms point sideways.
        bpy.context.scene.frame_set(12)
        arm = self._bone("UPPER_ARM_L")
        direction = (arm.tail - arm.head).normalized()
        self.assertGreater(abs(direction.x), 0.9)

    def test_keyed_pose_is_reached_exactly(self) -> None:
        arm = self._bone("UPPER_ARM_L")
        bpy.context.scene.frame_set(12)
        arm.rotation_mode = "QUATERNION"
        arm.rotation_quaternion = (0.8, 0.0, 0.6, 0.0)
        mapped = {
            self._bone(semantic).name
            for semantic in (
                "PELVIS",
                "CHEST",
                "HEAD",
                "UPPER_ARM_L",
                "UPPER_ARM_R",
                "LOWER_ARM_L",
                "LOWER_ARM_R",
                "HAND_L",
                "HAND_R",
                "UPPER_LEG_L",
                "UPPER_LEG_R",
                "LOWER_LEG_L",
                "LOWER_LEG_R",
                "FOOT_L",
                "FOOT_R",
                "NECK",
                "SPINE_01",
                "SHOULDER_L",
                "SHOULDER_R",
                "TOE_L",
                "TOE_R",
            )
        }
        for name in mapped:
            bone = self.armature.pose.bones[name]
            bone.rotation_mode = "QUATERNION"
            bone.keyframe_insert("rotation_quaternion", frame=12)
        keyed = arm.rotation_quaternion.copy()
        result = self.api.generate(
            self.armature.name, frame_range=(1, 24), steps=4, output="REPLACE"
        )
        self.assertEqual([g["kind"] for g in result.guides], ["FULL_BODY"])
        self.assertGreater(result.guides[0]["guide_warp_deg"], 1.0)
        bpy.context.scene.frame_set(12)
        self.assertLess(arm.rotation_quaternion.rotation_difference(keyed).angle, 1e-3)

    def test_partial_keys_pass_only_their_effectors(self) -> None:
        bpy.context.scene.frame_set(12)
        keyed = {}
        for semantic in ("UPPER_ARM_L", "LOWER_ARM_L", "HAND_L"):
            bone = self._bone(semantic)
            bone.rotation_mode = "QUATERNION"
            bone.rotation_quaternion = (0.9, 0.0, 0.0, 0.43589)
            bone.keyframe_insert("rotation_quaternion", frame=12)
            keyed[bone.name] = bone.rotation_quaternion.copy()
        head = self._bone("HEAD")
        head.rotation_mode = "QUATERNION"
        head.rotation_quaternion = (0.95, 0.31225, 0.0, 0.0)
        head.keyframe_insert("rotation_quaternion", frame=18)
        keyed[head.name] = head.rotation_quaternion.copy()
        result = self.api.generate(
            self.armature.name, frame_range=(1, 24), steps=4, output="REPLACE"
        )
        kinds = {g["frame"]: (g["kind"], g.get("effectors")) for g in result.guides}
        self.assertEqual(kinds[12.0], ("EFFECTOR", ["LEFT_HAND"]))
        self.assertEqual(kinds[18.0], ("KEYS", None))
        for frame, names in (
            (12, ("UPPER_ARM_L", "LOWER_ARM_L", "HAND_L")),
            (18, ("HEAD",)),
        ):
            bpy.context.scene.frame_set(frame)
            for semantic in names:
                bone = self._bone(semantic)
                with self.subTest(bone=semantic):
                    difference = bone.rotation_quaternion.rotation_difference(
                        keyed[bone.name]
                    )
                    self.assertLess(difference.angle, 1e-3)

    def test_fast_path_matches_the_scene_path(self) -> None:
        name = self.addon.__name__
        mod = lambda path: __import__(f"{name}.{path}", fromlist=["x"])  # noqa: E731
        soma_pose, generated = mod("retarget.soma_pose"), mod("retarget.generated")
        samples_mod, scene_mod = mod("retarget.samples"), mod("core.scene")
        mapping, planning = mod("rig.mapping"), mod("rig.planning")
        frames_mod, soma = mod("retarget.solve.frames"), mod("core.motion.soma")
        entries = mapping.mapping_by_id(self.armature.mtp_rig_config)
        plan = planning.compile_rig_solve_plan(
            self.armature,
            entries,
            lambda: frames_mod.target_rest_basis(self.armature, entries),
        )
        fast = soma_pose.bind(self.armature, plan)
        self.assertTrue(fast.fast)
        slow = replace(fast, fast=False)
        context = scene_mod.SceneContext(
            bpy.context.scene, bpy.context.view_layer, bpy.context.preferences
        )
        rng = np.random.default_rng(5)
        count = 12
        local = np.empty((count, soma.JOINT_COUNT, 3, 3))
        for frame in range(count):
            for joint in range(soma.JOINT_COUNT):
                axis = rng.normal(size=3)
                axis /= np.linalg.norm(axis)
                angle = rng.uniform(0.0, 0.6)
                k = np.array(
                    (
                        (0, -axis[2], axis[1]),
                        (axis[2], 0, -axis[0]),
                        (-axis[1], axis[0], 0),
                    )
                )
                local[frame, joint] = (
                    np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k
                )
        hips = np.zeros((count, 3))
        hips[:, 1] = soma.REST_HIP_HEIGHT
        hips[:, 0] = np.linspace(0.0, 0.3, count)
        rotations, _positions = soma.forward_kinematics(local, hips)
        smooth = np.stack((hips[:, 0], np.zeros(count), hips[:, 2]), axis=1)
        baked = {}
        for label, binding in (("fast", fast), ("slow", slow)):
            samples = []
            for frame in range(count):
                sample = samples_mod._BakedFrameSample(float(frame + 1), [])
                soma_pose.apply_soma_frame(
                    binding,
                    rotations[frame],
                    hips[frame],
                    sample,
                    scene_context=context,
                    smooth_root=smooth[frame],
                    heading=0.1 * frame,
                )
                samples.append(sample)
            contacts = np.zeros((count, 4), dtype=bool)
            contacts[2:10, 0] = True
            generated.plant_rig_feet(binding, samples, contacts, scene_context=context)
            baked[label] = samples
        for fast_sample, slow_sample in zip(baked["fast"], baked["slow"], strict=True):
            self.assertEqual(set(fast_sample.bones), set(slow_sample.bones))
            for bone_name, channel in fast_sample.bones.items():
                other = slow_sample.bones[bone_name]
                with self.subTest(bone=bone_name, frame=fast_sample.destination):
                    self.assertLess(
                        channel.rotation.rotation_difference(other.rotation).angle, 1e-4
                    )
                    self.assertLess((channel.location - other.location).length, 1e-5)
                    self.assertEqual(channel.key_location, other.key_location)

    def test_command_line_dry_run_and_errors(self) -> None:
        code, output = run_cli(
            self.addon,
            "generate",
            "--armature",
            self.armature.name,
            "--frame-range",
            "1:30",
            "--waypoint",
            "1:0,0",
            "--dry-run",
            "--json",
        )
        self.assertEqual(code, 0, output)
        self.assertTrue(json.loads(output.strip().splitlines()[-1])["dry_run"])
        code, _output = run_cli(
            self.addon, "generate", "--armature", "Missing", "--json"
        )
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
