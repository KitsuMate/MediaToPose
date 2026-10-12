from __future__ import annotations

import copy
import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import bpy
from mathutils import Matrix, Quaternion, Vector

from tests.blender_test_support import fixture_path, register_addon, scene_context


class VroidFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.addon = register_addon()
        bpy.ops.wm.open_mainfile(filepath=str(fixture_path("vroid_test_rig")))
        cls.addon.register()
        cls.armature = next(
            obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"
        )
        cls.addon.rig.mapping.auto_map(cls.armature)

    def setUp(self) -> None:
        for bone in self.armature.pose.bones:
            bone.matrix_basis = Matrix.Identity(4)
        # These fixture tests exercise retargeting and model-free cleanup.
        # RoHM is release-default but requires its external verified bundle.
        rohm = self.addon.postprocess.module_config(
            self.armature.mtp_rig_config, "ROHM_CORRECTION"
        )
        rohm_was_enabled = rohm.enabled
        rohm.enabled = False
        self.addCleanup(setattr, rohm, "enabled", rohm_was_enabled)

    def _recorded_frame(self, fixture_id: str):
        payload = json.loads(Path(fixture_path(fixture_id)).read_text(encoding="utf-8"))
        frame = self.addon.core.cache.frames.frame_from_plain(payload["frame"])
        settings = SimpleNamespace(
            provider=payload["provider"],
            use_body=True,
            use_hands=True,
            use_face=False,
            face_output="ARKIT",
        )
        return payload, frame, settings

    def _spine_quality(self, frame):
        entries = self.addon.rig.mapping.mapping_by_id(self.armature.mtp_rig_config)
        samples, inferred = self.addon.postprocess.CORRECTION_REGISTRY[
            "SPINE_CORRECTION"
        ]._source_curve(SimpleNamespace(frame=frame))
        channels = []
        for semantic_id in ("SPINE_01", "SPINE_02", "CHEST"):
            bone = self.addon.retarget.evaluated_semantic_bone(
                self.armature,
                entries,
                semantic_id,
            )
            if bone is not None and all(
                existing[1].name != bone.name for existing in channels
            ):
                channels.append((semantic_id, bone))
        self.assertTrue(samples)
        self.assertEqual(len(channels), 3)
        lengths = [max(float(bone.length), 1.0e-8) for _, bone in channels]
        total = sum(lengths)
        cursor = 0.0
        expected_cursor = channels[0][1].head.copy()
        source_basis = self.addon.retarget.source_root_basis(frame)
        target_basis = self.addon.retarget.target_basis(
            self.armature,
            entries,
        )
        anatomical = [bone for _, bone in channels]
        neck = self.addon.retarget.evaluated_semantic_bone(
            self.armature,
            entries,
            "NECK",
        )
        self.assertIsNotNone(neck)
        anatomical.append(neck)
        tangent_errors = []
        cumulative_errors = []
        for index, ((_semantic_id, bone), length) in enumerate(
            zip(channels, lengths, strict=False)
        ):
            start_fraction = cursor / total
            cursor += length
            end_fraction = cursor / total
            source_segment = self.addon.postprocess.base._arc_point(
                samples, end_fraction
            ) - self.addon.postprocess.base._arc_point(samples, start_fraction)
            intended = self.addon.retarget.solve.frames._to_target_direction(
                source_segment,
                source_basis,
                target_basis,
            ).normalized()
            endpoint = anatomical[index + 1].head
            actual = endpoint - bone.head
            tangent_errors.append(math.degrees(actual.angle(intended)))
            expected_cursor += intended * actual.length
            cumulative_errors.append((endpoint - expected_cursor).length / total)
        return inferred, tangent_errors, cumulative_errors

    def _assert_recorded_retarget(self, fixture_id: str, body_count: int) -> None:
        payload, frame, settings = self._recorded_frame(fixture_id)
        spine = self.addon.postprocess.module_config(
            self.armature.mtp_rig_config,
            "SPINE_CORRECTION",
        )
        spine_was_enabled = spine.enabled
        spine.enabled = True
        self.addCleanup(setattr, spine, "enabled", spine_was_enabled)
        self.assertEqual(
            sum(point.valid for point in frame.body.values()),
            body_count,
        )
        self.assertEqual(set(frame.hands), {"LEFT", "RIGHT"})
        self.addon.retarget.prepare_limb_ownership(self.armature)
        changed_bones, changed_shapes = self.addon.retarget.apply_frame(
            self.armature,
            frame,
            settings,
            scene_context=scene_context(),
        )
        self.assertEqual(changed_bones, 51)
        self.assertEqual(changed_shapes, 0)
        entries = self.addon.rig.mapping.mapping_by_id(self.armature.mtp_rig_config)
        for semantic_id in ("EYE_L", "EYE_R"):
            eye = self.armature.pose.bones[entries[semantic_id].target_name]
            self.assertEqual(eye.matrix_basis, Matrix.Identity(4))

        inferred, tangents, drift = self._spine_quality(frame)
        if payload["provider"] == "MEDIAPIPE":
            self.assertTrue(inferred)
            self.assertLess(max(tangents), 0.1)
            self.assertLess(max(drift), 1.0e-4)
        else:
            self.assertFalse(inferred)
            self.assertLess(max(tangents), 5.0)
            self.assertLess(max(drift), 0.05)

        head_errors = self.addon.retarget.diagnostics.normalized_head_position_errors(
            self.armature,
            frame,
        )
        self.assertLess(max(head_errors.values()), 0.01)
        finger_errors = self.addon.retarget.diagnostics.finger_world_position_errors(
            self.armature,
            entries,
            frame,
        )
        self.assertEqual(len(finger_errors), 30)
        self.assertLess(max(finger_errors.values()), 0.005)

    def test_image_pose_resets_root_location_unless_preserved(self) -> None:
        _payload, frame, settings = self._recorded_frame(
            "mediapipe_ref_pose_realistic_output"
        )
        entries = self.addon.rig.mapping.mapping_by_id(self.armature.mtp_rig_config)
        root = self.addon.retarget.solve.common._mapped_bone(
            self.armature, entries, "ROOT"
        )
        self.assertIsNotNone(root)

        root.location = (1.0, 2.0, 3.0)
        settings.keep_original_root_position = False
        self.addon.retarget.apply_frame(
            self.armature,
            frame,
            settings,
            scene_context=scene_context(),
        )
        self.assertLess(root.location.length, 1.0e-6)

        expected = Vector((1.0, 2.0, 3.0))
        root.location = expected
        settings.keep_original_root_position = True
        self.addon.retarget.apply_frame(
            self.armature,
            frame,
            settings,
            scene_context=scene_context(),
        )
        self.assertLess((root.location - expected).length, 1.0e-6)

    def test_expected_rig_structure(self) -> None:
        self.assertEqual(len(self.armature.data.bones), 145)
        meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
        self.assertEqual(len(meshes), 3)
        self.assertIn("T-Pose", bpy.data.actions)

    def test_rohm_root_application_keeps_grounding_in_sync(self) -> None:
        original_matrix = self.armature.matrix_world.copy()
        self.addCleanup(setattr, self.armature, "matrix_world", original_matrix)
        basis_location, basis_rotation, basis_scale = (
            self.armature.matrix_basis.decompose()
        )
        world_root = self.armature.matrix_world.translation.copy()
        point = self.addon.grounded_cleanup.ContactPoint(
            position=world_root + Vector((0.2, 0.0, -1.0)),
            image=Vector((0.5, 0.8)),
            confidence=1.0,
            motion_position=Vector((0.2, 0.0, -1.0)),
        )
        point_position = point.position.copy()
        grounding = self.addon.grounded_cleanup.GroundFrame(
            timestamp=0.0,
            root=world_root.copy(),
            points={"L_HEEL": point},
            joints={"PELVIS": world_root + Vector((0.0, 0.0, 0.8))},
            root_confidence=0.25,
        )
        sample = self.addon.retarget.samples._BakedFrameSample(
            destination=1.0,
            warning_sink=[],
            root_object=(
                basis_location.copy(),
                basis_rotation.copy(),
                basis_scale.copy(),
            ),
            grounding=grounding,
        )
        delta = Vector((0.3, -0.2, 0.1))

        updates = list(
            self.addon.retarget.rohm._iter_apply_rohm_root_deltas(
                self.armature,
                [sample],
                [tuple(delta)],
                "OBJECT",
                self.armature,
                scene_context=scene_context(),
            )
        )

        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0].phase, "RoHM: applying corrected root")
        self.assertEqual(sample.root_object[1], basis_rotation)
        self.assertEqual(sample.root_object[2], basis_scale)
        self.assertLess((grounding.root - (world_root + delta)).length, 1.0e-7)
        self.assertLess(
            (grounding.points["L_HEEL"].position - (point_position + delta)).length,
            1.0e-7,
        )
        self.assertLess(
            (
                grounding.joints["PELVIS"]
                - (world_root + Vector((0.0, 0.0, 0.8)) + delta)
            ).length,
            1.0e-7,
        )
        self.assertEqual(point.motion_position, Vector((0.2, 0.0, -1.0)))

    def test_all_thirty_finger_controls_map(self) -> None:
        entries = self.addon.rig.mapping.mapping_by_id(self.armature.mtp_rig_config)
        mapped = [
            entries[f"{finger}_{segment:02d}_{side}"].target_name
            for side in ("L", "R")
            for finger in ("THUMB", "INDEX", "MIDDLE", "RING", "PINKY")
            for segment in range(1, 4)
        ]
        self.assertEqual(len(mapped), 30)
        self.assertTrue(all(mapped))

    def test_rig_plan_and_plain_rig_commit_are_compiled(self) -> None:
        _payload, frame, settings = self._recorded_frame(
            "mediapipe_ref_pose_realistic_output"
        )
        first = self.addon.retarget.build_target_pose(
            self.armature,
            frame,
            settings,
        )
        second = self.addon.retarget.build_target_pose(
            self.armature,
            frame,
            settings,
        )
        self.assertIs(first.rig_plan, second.rig_plan)
        self.assertTrue(first.rig_plan.constraint_free)
        self.assertEqual(len(first.rig_plan.semantic_bones), 55)
        self.assertEqual(len(first.rig_plan.ik_chains), 4)
        live_entry = self.addon.rig.mapping.mapping_by_id(self.armature.mtp_rig_config)[
            "PELVIS"
        ]
        frozen_entry = first.rig_plan.entries["PELVIS"]
        original_gain = live_entry.gain
        live_entry.gain = original_gain + 0.25
        try:
            self.assertEqual(frozen_entry.gain, original_gain)
            self.assertNotEqual(frozen_entry.gain, live_entry.gain)
        finally:
            live_entry.gain = original_gain
        first.commit(scene_context().update)
        self.assertEqual(first.commit_dependency_updates, 1)

    def test_unconstrained_translation_avoids_jacobian_probes(self) -> None:
        bone = self.armature.pose.bones["J_Bip_C_Hips"]
        original = bone.matrix_basis.copy()
        try:
            solver = self.addon.retarget.solve.translation.PoseTranslationSolver(
                scene_context().update
            )
            result = solver.solve(
                bone,
                bone.matrix.translation + Vector((0.01, 0.0, 0.0)),
            )
            self.assertTrue(result.converged)
            self.assertEqual(result.jacobian_builds, 0)
            self.assertEqual(result.dependency_updates, 1)
        finally:
            bone.matrix_basis = original
            bpy.context.view_layer.update()

    def test_target_pose_mutation_invalidates_all_resolved_caches(self) -> None:
        _payload, frame, settings = self._recorded_frame(
            "mediapipe_ref_pose_realistic_output"
        )
        pose = self.addon.retarget.build_target_pose(
            self.armature,
            frame,
            settings,
        )
        pose.resolve()
        revision = pose.revision
        self.assertEqual(pose._resolved_revision, revision)
        channel = next(iter(pose.bones.values()))
        pose.set_location(
            channel.semantic_id,
            channel.bone,
            channel.bone.matrix_basis.translation + Vector((0.01, 0.0, 0.0)),
        )
        self.assertEqual(pose.revision, revision + 1)
        self.assertEqual(pose._resolved_basis, {})
        self.assertEqual(pose._resolved_pose, {})
        pose.semantic_endpoints()
        self.assertEqual(pose._resolved_revision, pose.revision)

    def test_body_pose_always_owns_main_hand_bones(self) -> None:
        for fixture_id in (
            "mediapipe_ref_pose_realistic_output",
            "sam_ref_pose_stylized_output",
        ):
            with self.subTest(fixture=fixture_id):
                _payload, frame, settings = self._recorded_frame(fixture_id)
                body_only_settings = copy.copy(settings)
                body_only_settings.use_hands = False
                body_only = self.addon.retarget.build_target_pose(
                    self.armature,
                    frame,
                    body_only_settings,
                    post_process=False,
                )
                body_and_fingers = self.addon.retarget.build_target_pose(
                    self.armature,
                    frame,
                    settings,
                    post_process=False,
                )

                for semantic_id in ("HAND_L", "HAND_R"):
                    expected = body_only.channel_for(semantic_id)
                    actual = body_and_fingers.channel_for(semantic_id)
                    self.assertIsNotNone(expected)
                    self.assertIsNotNone(actual)
                    self.assertLess(
                        expected.desired_rotation.to_quaternion()
                        .rotation_difference(actual.desired_rotation.to_quaternion())
                        .angle,
                        1.0e-7,
                    )

                self.assertIsNone(body_only.channel_for("INDEX_01_R"))
                self.assertIsNotNone(body_and_fingers.channel_for("INDEX_01_R"))

    def test_body_hand_keeps_observed_global_palm_orientation(self) -> None:
        _payload, frame, settings = self._recorded_frame("sam_ref_pose_stylized_output")
        pose = self.addon.retarget.build_target_pose(
            self.armature,
            frame,
            settings,
            post_process=False,
        )
        for semantic_id in ("HAND_L", "HAND_R"):
            channel = pose.channel_for(semantic_id)
            expected = self.addon.retarget.target_pose.stable_orientation_delta(
                channel.rest_direction,
                channel.desired_direction,
                channel.rest_reference,
                channel.desired_reference,
                reference_is_oriented=True,
            )
            rest = channel.bone.bone.matrix_local.to_3x3()
            actual = (
                (channel.desired_rotation @ rest.inverted_safe())
                .to_quaternion()
                .normalized()
            )
            self.assertIsNotNone(expected)
            self.assertLess(actual.rotation_difference(expected).angle, 1.0e-7)

    def test_finger_mapping_is_independent_of_body_provider_label(self) -> None:
        _payload, frame, settings = self._recorded_frame("sam_ref_pose_stylized_output")
        alternate = copy.copy(settings)
        alternate.provider = "MEDIAPIPE"

        sam_pose = self.addon.retarget.build_target_pose(
            self.armature,
            frame,
            settings,
            post_process=False,
        )
        mediapipe_pose = self.addon.retarget.build_target_pose(
            self.armature,
            frame,
            alternate,
            post_process=False,
        )

        for side in ("L", "R"):
            for finger in ("THUMB", "INDEX", "MIDDLE", "RING", "PINKY"):
                for segment in range(1, 4):
                    semantic_id = f"{finger}_{segment:02d}_{side}"
                    sam_channel = sam_pose.channel_for(semantic_id)
                    mediapipe_channel = mediapipe_pose.channel_for(semantic_id)
                    self.assertIsNotNone(sam_channel)
                    self.assertIsNotNone(mediapipe_channel)
                    self.assertLess(
                        sam_channel.desired_rotation.to_quaternion()
                        .rotation_difference(
                            mediapipe_channel.desired_rotation.to_quaternion()
                        )
                        .angle,
                        1.0e-7,
                    )

    def test_impossible_finger_fold_is_bounded_for_both_providers(self) -> None:
        for fixture_id in (
            "mediapipe_ref_pose_realistic_output",
            "sam_ref_pose_stylized_output",
        ):
            with self.subTest(fixture=fixture_id):
                _payload, frame, settings = self._recorded_frame(fixture_id)
                frame = copy.deepcopy(frame)
                hand = frame.hands["RIGHT"]
                mcp = Vector(hand["INDEX_FINGER_MCP"].position)
                pip = Vector(hand["INDEX_FINGER_PIP"].position)
                dip = Vector(hand["INDEX_FINGER_DIP"].position)
                incoming = pip - mcp
                hand["INDEX_FINGER_DIP"].position = tuple(
                    pip - incoming.normalized() * max((dip - pip).length, 1.0e-3)
                )

                pose = self.addon.retarget.build_target_pose(
                    self.armature,
                    frame,
                    settings,
                    post_process=False,
                )
                parent = pose.channel_for("INDEX_01_R")
                child = pose.channel_for("INDEX_02_R")
                self.assertIsNotNone(parent)
                self.assertIsNotNone(child)

                def channel_delta(channel):
                    rest = channel.bone.bone.matrix_local.to_3x3()
                    return (
                        (channel.desired_rotation @ rest.inverted_safe())
                        .to_quaternion()
                        .normalized()
                    )

                articulation = channel_delta(parent).inverted() @ channel_delta(child)
                articulation_degrees = math.degrees(articulation.angle)
                articulation_degrees = min(
                    articulation_degrees,
                    abs(360.0 - articulation_degrees),
                )
                self.assertLessEqual(
                    articulation_degrees,
                    130.001,
                )
                self.assertTrue(
                    any(
                        "Index 02 R articulation was limited" in warning
                        for warning in pose.warnings
                    )
                )

    def test_distal_finger_swing_is_constrained_to_hinge_plane(self) -> None:
        rest = Vector((0.0, 1.0, 0.0))
        palm_normal = Vector((0.0, 0.0, 1.0))
        sideways = Vector((1.0, 0.0, 0.0))
        mapper = (
            self.addon.retarget.solve.common._bounded_parent_relative_direction_delta
        )
        delta, constrained = mapper(
            rest,
            sideways,
            Quaternion(),
            math.radians(130.0),
            bend_reference=palm_normal,
        )
        result = (delta @ rest).normalized()
        lateral = rest.cross(palm_normal).normalized()
        off_plane = math.degrees(math.asin(result.dot(lateral)))

        self.assertTrue(constrained)
        self.assertAlmostEqual(off_plane, 25.0, places=4)

    def test_rotation_clamp_uses_shortest_quaternion_hemisphere(self) -> None:
        wrapped = Quaternion(Vector((0.0, 0.0, 1.0)), math.radians(350.0))
        bounded, constrained = self.addon.retarget.solve.common._clamp_rotation(
            wrapped,
            math.radians(45.0),
        )

        self.assertFalse(constrained)
        self.assertAlmostEqual(math.degrees(bounded.angle), 10.0, places=4)

    def test_sam_neck_uses_neutral_relative_rotation_frame(self) -> None:
        _payload, frame, settings = self._recorded_frame("sam_ref_pose_stylized_output")
        pose = self.addon.retarget.build_target_pose(
            self.armature,
            frame,
            settings,
            post_process=False,
        )
        channel = pose.channel_for("NECK")
        self.assertIsNotNone(channel)
        rest = channel.bone.bone.matrix_local.to_quaternion().normalized()
        actual = (
            channel.desired_rotation.to_quaternion().normalized() @ rest.inverted()
        ).normalized()
        expected = self.addon.retarget.solve.frames._joint_frame_target_delta(
            frame,
            "MHR_NECK",
            pose.source_basis,
            pose.target_basis,
        )
        self.assertIsNotNone(expected)
        self.assertLess(
            math.degrees(actual.rotation_difference(expected).angle),
            0.01,
        )
        self.assertLess(math.degrees(actual.angle), 45.0)

    def test_sam_neck_rotation_is_limited_without_limiting_head(self) -> None:
        _payload, frame, settings = self._recorded_frame("sam_ref_pose_stylized_output")
        frame = copy.deepcopy(frame)
        observation = frame.joint_frames["MHR_NECK"]
        extreme = Matrix.Rotation(math.radians(120.0), 3, "X")
        frame.joint_frames["MHR_NECK"] = type(observation)(
            rotation_matrix=tuple(value for row in extreme for value in row),
            confidence=observation.confidence,
            valid=True,
        )
        pose = self.addon.retarget.build_target_pose(
            self.armature,
            frame,
            settings,
            post_process=False,
        )
        neck = pose.channel_for("NECK")
        head = pose.channel_for("HEAD")
        rest = neck.bone.bone.matrix_local.to_quaternion().normalized()
        actual = (
            neck.desired_rotation.to_quaternion().normalized() @ rest.inverted()
        ).normalized()
        self.assertAlmostEqual(math.degrees(actual.angle), 45.0, places=3)
        self.assertIsNotNone(head.desired_rotation)
        self.assertIn("SAM neck rotation was limited to 45 degrees", pose.warnings)

    def test_sequence_streams_detection_and_keyframe_work(self) -> None:
        payload, source, settings = self._recorded_frame("sam_ref_pose_stylized_output")
        settings.root_motion_mode = "OFF"
        temporal = self.addon.postprocess.module_config(
            self.armature.mtp_rig_config, "TEMPORAL_CORRECTION"
        )
        temporal_state = (
            temporal.enabled,
            temporal.smooth_motion,
            temporal.ground_contacts,
            temporal.stabilize_hands,
        )
        temporal.enabled = True
        temporal.smooth_motion = True
        temporal.ground_contacts = False
        temporal.stabilize_hands = True
        settings.keep_detection_cache = False
        produced = []

        def frames():
            for index in range(3):
                frame = copy.deepcopy(source)
                frame.timestamp_ms = index * 33
                produced.append(index)
                yield frame

        original = self.armature.animation_data_create().action
        staging = bpy.data.actions.new("SequenceStreamingTest")
        self.armature.animation_data.action = staging
        try:
            iterator = self.addon.retarget.iter_apply_sequence(
                self.armature,
                frames(),
                settings,
                1.0,
                scene_context=scene_context(),
            )
            first = next(iterator)
            self.assertEqual(first.phase, "Loading detections")
            self.assertEqual(produced, [0])
            while True:
                progress = next(iterator)
                if progress.phase == "Streaming keyframes" and progress.current == 1:
                    break
            key_count = sum(
                len(curve.keyframe_points)
                for curve in self.addon.retarget.keyframes._action_fcurves(staging)
            )
            self.assertGreater(key_count, 0)
            self.assertEqual(produced, [0])
            saw_temporal = False
            while True:
                try:
                    progress = next(iterator)
                    saw_temporal |= progress.phase == "Smoothing animation"
                except StopIteration as stopped:
                    result = stopped.value
                    break
            self.assertTrue(saw_temporal)
            self.assertEqual(result[:3], (3, 153, 0))
            self.assertEqual(payload["provider"], "SAM3D_BODY")
            entries = self.addon.rig.mapping.mapping_by_id(self.armature.mtp_rig_config)
            hand_channels = {
                self.addon.retarget.solve.common._mapped_bone(
                    self.armature, entries, semantic_id
                )
                for semantic_id in entries
                if semantic_id.startswith(
                    (
                        "HAND_",
                        "THUMB_",
                        "INDEX_",
                        "MIDDLE_",
                        "RING_",
                        "PINKY_",
                    )
                )
            }
            hand_channels.discard(None)
            self.assertTrue(hand_channels)
            self.assertTrue(
                all(channel.rotation_mode == "QUATERNION" for channel in hand_channels)
            )
            hand_paths = {
                channel.path_from_id("rotation_quaternion") for channel in hand_channels
            }
            hand_rotation_curves = [
                curve
                for curve in self.addon.retarget.keyframes._action_fcurves(staging)
                if curve.data_path in hand_paths
            ]
            self.assertTrue(hand_rotation_curves)
            self.assertTrue(
                all(
                    point.interpolation == "LINEAR"
                    for curve in hand_rotation_curves
                    for point in curve.keyframe_points
                )
            )
            self.assertFalse(
                any(
                    curve.data_path.endswith(("rotation_euler", "rotation_axis_angle"))
                    and any(
                        channel.name in curve.data_path for channel in hand_channels
                    )
                    for curve in self.addon.retarget.keyframes._action_fcurves(staging)
                )
            )
        finally:
            (
                temporal.enabled,
                temporal.smooth_motion,
                temporal.ground_contacts,
                temporal.stabilize_hands,
            ) = temporal_state
            self.armature.animation_data.action = original
            bpy.data.actions.remove(staging)

    def test_automatic_root_motion_spools_and_falls_back_cleanly(self) -> None:
        _payload, source, settings = self._recorded_frame(
            "mediapipe_ref_pose_realistic_output"
        )
        settings.root_motion_mode = "AUTOMATIC"
        settings.keep_original_root_position = False
        temporal = self.addon.postprocess.module_config(
            self.armature.mtp_rig_config, "TEMPORAL_CORRECTION"
        )
        temporal_was_enabled = temporal.enabled
        temporal.enabled = False
        settings.keep_detection_cache = False
        frames = []
        for index in range(2):
            frame = copy.deepcopy(source)
            frame.timestamp_ms = index * 40
            frame.root_motion = None
            frames.append(frame)
        original = self.armature.animation_data_create().action
        staging = bpy.data.actions.new("SequenceSpoolTest")
        self.armature.animation_data.action = staging
        entries = self.addon.rig.mapping.mapping_by_id(self.armature.mtp_rig_config)
        root = self.addon.retarget.solve.common._mapped_bone(
            self.armature, entries, "ROOT"
        )
        self.assertIsNotNone(root)
        root.location = (1.0, 2.0, 3.0)
        try:
            result = self.addon.retarget.apply_sequence(
                self.armature,
                iter(frames),
                settings,
                1.0,
                scene_context=scene_context(),
            )
            self.assertEqual(result[:3], (2, 104, 0))
            self.assertTrue(any("baked in place" in warning for warning in result[3]))
            root_location_path = root.path_from_id("location")
            root_curves = {
                curve.array_index: curve
                for curve in self.addon.retarget.keyframes._action_fcurves(staging)
                if curve.data_path == root_location_path
            }
            self.assertEqual(set(root_curves), {0, 1, 2})
            self.assertTrue(
                all(abs(curve.evaluate(1.0)) < 1.0e-6 for curve in root_curves.values())
            )
        finally:
            temporal.enabled = temporal_was_enabled
            self.armature.animation_data.action = original
            bpy.data.actions.remove(staging)

    def test_automatic_root_motion_uses_sam_video_translation(self) -> None:
        _payload, source, _settings = self._recorded_frame(
            "sam_ref_pose_stylized_output"
        )
        samples = []
        for index in range(4):
            frame = copy.deepcopy(source)
            frame.timestamp_ms = index * 40
            observation = frame.root_motion
            observation.translation = (
                observation.translation[0] + index * 0.08,
                observation.translation[1],
                observation.translation[2],
            )
            samples.append(self.addon.retarget.root_motion._root_motion_sample(frame))
        entries = self.addon.rig.mapping.mapping_by_id(self.armature.mtp_rig_config)

        deltas, _yaws, mode, warnings = (
            self.addon.retarget.root_motion._root_motion_deltas(
                self.armature,
                entries,
                samples,
                "AUTOMATIC",
            )
        )

        self.assertEqual(mode, "AUTOMATIC")
        self.assertEqual(warnings, [])
        self.assertAlmostEqual(deltas[0].length, 0.0)
        self.assertGreater(deltas[-1].length, deltas[1].length)

    def test_ground_contacts_translate_ik_foot_without_rotating_it(self) -> None:
        class IKTarget:
            def __init__(self, matrix):
                self.name = "GroundContactTestIKTarget"
                self.parent_recursive = ()
                self._matrix = matrix.copy()

            @property
            def matrix(self):
                return self._matrix.copy()

            @matrix.setter
            def matrix(self, value):
                self._matrix = value.copy()

            @property
            def matrix_basis(self):
                return self._matrix.copy()

            @matrix_basis.setter
            def matrix_basis(self, value):
                self._matrix = value.copy()

        entries = self.addon.rig.mapping.mapping_by_id(self.armature.mtp_rig_config)
        upper = self.addon.retarget.evaluated_semantic_bone(
            self.armature, entries, "UPPER_LEG_L"
        )
        lower = self.addon.retarget.evaluated_semantic_bone(
            self.armature, entries, "LOWER_LEG_L"
        )
        foot = self.addon.retarget.evaluated_semantic_bone(
            self.armature, entries, "FOOT_L"
        )
        toe = self.addon.retarget.evaluated_semantic_bone(
            self.armature, entries, "TOE_L"
        )
        self.assertTrue(all(item is not None for item in (upper, lower, foot, toe)))

        sample = self.addon.retarget.samples._BakedFrameSample(1.0, [])
        for semantic_id, bone in (
            ("UPPER_LEG_L", upper),
            ("LOWER_LEG_L", lower),
            ("FOOT_L", foot),
            ("TOE_L", toe),
        ):
            location, rotation, scale = bone.matrix_basis.decompose()
            sample.bones[bone.name] = self.addon.retarget.samples._BakedBoneSample(
                bone=bone,
                location=location.copy(),
                rotation=rotation.normalized(),
                scale=scale.copy(),
                key_rotation=True,
                semantic_id=semantic_id,
            )
        ik_target = IKTarget(foot.matrix)
        target_location, target_rotation, target_scale = (
            ik_target.matrix_basis.decompose()
        )
        sample.bones[ik_target.name] = self.addon.retarget.samples._BakedBoneSample(
            bone=ik_target,
            location=target_location.copy(),
            rotation=target_rotation.normalized(),
            scale=target_scale.copy(),
            key_location=True,
            key_rotation=True,
        )
        current_heel = foot.head.copy()
        current_toe = toe.head.copy()
        initial_rotation = target_rotation.normalized()
        target_up = self.addon.retarget.target_basis(self.armature, entries) @ Vector(
            (0.0, 0.0, 1.0)
        )
        target_up.normalize()
        offset = Vector((0.015, 0.0, 0.0))
        planted_direction = Quaternion(target_up, math.radians(45.0)) @ (
            current_toe - current_heel
        )
        heel_target = current_heel + offset
        toe_target = heel_target + planted_direction
        point = self.addon.grounded_cleanup.ContactPoint(
            position=current_heel.copy(),
            image=Vector((0.5, 0.9)),
            confidence=1.0,
            direction=(current_toe - current_heel).normalized(),
            motion_position=Vector(),
        )
        sample.grounding = self.addon.grounded_cleanup.GroundFrame(
            timestamp=0.0,
            root=upper.head.copy(),
            points={"L_HEEL": point, "L_TOE": copy.deepcopy(point)},
            root_confidence=1.0,
        )
        result = SimpleNamespace(contacts={}, segments=[], warnings=[])
        with (
            mock.patch.object(
                self.addon.grounded_cleanup,
                "analyze_grounding",
                return_value=result,
            ),
            mock.patch.object(
                self.addon.retarget.grounding,
                "_contact_targets",
                return_value=[
                    {
                        "L_HEEL": (heel_target, 1.0),
                        "L_TOE": (toe_target, 1.0),
                    }
                ],
            ),
            mock.patch.object(
                self.addon.retarget.grounding,
                "limb_ownership",
                return_value={
                    ("LEG", "L"): {
                        "mode": "IK",
                        "target": ik_target,
                        "pole": None,
                        "rotation_bend": None,
                    }
                },
            ),
        ):
            list(
                self.addon.retarget.grounding._iter_ground_contacts_run(
                    self.armature,
                    entries,
                    [sample],
                    1.0,
                    "OFF",
                    "BONE",
                    upper,
                    target_up,
                    scene_context=scene_context(),
                )
            )
        actual_rotation = sample.bones[ik_target.name].rotation.normalized()
        self.assertLess(
            math.degrees(initial_rotation.rotation_difference(actual_rotation).angle),
            0.01,
        )

    def test_camera_motion_changes_only_root_location(self) -> None:
        entries = self.addon.rig.mapping.mapping_by_id(self.armature.mtp_rig_config)
        root_kind, root_target = self.addon.retarget.root_motion._sequence_root_target(
            self.armature, entries
        )
        self.assertEqual(root_kind, "BONE")
        foot = self.addon.retarget.evaluated_semantic_bone(
            self.armature, entries, "FOOT_L"
        )
        self.assertIsNotNone(foot)
        samples = []
        inputs = []
        for index, root_x in enumerate((0.0, 0.1)):
            sample = self.addon.retarget.samples._BakedFrameSample(float(index + 1), [])
            for semantic, bone in (("ROOT", root_target), ("FOOT_L", foot)):
                location, rotation, scale = bone.matrix_basis.decompose()
                if semantic == "FOOT_L":
                    rotation = Quaternion((0.2, 0.8, 0.1), math.radians(27.0))
                sample.bones[bone.name] = self.addon.retarget.samples._BakedBoneSample(
                    bone=bone,
                    location=location.copy(),
                    rotation=rotation.normalized(),
                    scale=scale.copy(),
                    key_location=semantic == "ROOT",
                    key_rotation=True,
                    semantic_id=semantic,
                )
            item = self.addon.camera_motion_correction.CameraMotionInput(
                timestamp_ms=index * 50,
                root_translation=(root_x, 0.0, 2.0),
                focal_length=1000.0,
            )
            sample.camera_motion_input = item
            inputs.append(item)
            samples.append(sample)
        track = self.addon.camera_motion_correction.CameraMotionTrack(
            width=1000,
            height=600,
            frames=(
                self.addon.camera_motion_correction.CameraTrackFrame(
                    timestamp_ms=0, valid=True
                ),
                self.addon.camera_motion_correction.CameraTrackFrame(
                    timestamp_ms=50,
                    translation_x=50.0,
                    confidence=1.0,
                    tracks=100,
                    inliers=80,
                    residual=0.2,
                    valid=True,
                ),
            ),
        )
        rotations = [
            {name: tuple(channel.rotation) for name, channel in sample.bones.items()}
            for sample in samples
        ]
        list(
            self.addon.retarget.camera_motion._iter_camera_motion_correction(
                self.armature,
                samples,
                SimpleNamespace(lateral_root_strength=1.0),
                "AUTOMATIC",
                root_kind,
                root_target,
                Vector((1.0, 0.0, 0.0)),
                Matrix.Identity(3),
                1.0,
                track,
                None,
                "Ready",
                True,
                scene_context=scene_context(),
            )
        )
        for expected, sample in zip(rotations, samples, strict=True):
            for name, channel in sample.bones.items():
                self.assertEqual(tuple(channel.rotation), expected[name])
        self.assertGreater(
            (
                samples[1].bones[root_target.name].location
                - samples[0].bones[root_target.name].location
            ).length,
            0.05,
        )

    def test_recorded_mediapipe_retarget_quality(self) -> None:
        self._assert_recorded_retarget(
            "mediapipe_ref_pose_realistic_output",
            33,
        )

    def test_recorded_sam_retarget_quality(self) -> None:
        self._assert_recorded_retarget(
            "sam_ref_pose_stylized_output",
            27,
        )


if __name__ == "__main__":
    unittest.main()
