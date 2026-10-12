from __future__ import annotations

import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import bpy
import numpy as np

from tests.blender_test_support import fixture_path, register_addon


class GenericMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.addon = register_addon()
        cls.mapping = cls.addon.rig.mapping

    def setUp(self) -> None:
        data = bpy.data.armatures.new("MappingTestData")
        self.armature = bpy.data.objects.new("MappingTest", data)
        bpy.context.scene.collection.objects.link(self.armature)
        bpy.context.view_layer.objects.active = self.armature
        self.armature.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        names = (
            "Root",
            "J_Bip_C_Hips",
            "J_Bip_C_Spine",
            "J_Bip_C_Chest",
            "J_Bip_C_UpperChest",
            "J_Bip_C_Neck",
            "J_Bip_C_Head",
            "J_Bip_L_Shoulder",
            "J_Bip_L_UpperArm",
            "J_Bip_L_LowerArm",
            "J_Bip_L_Hand",
            "J_Bip_R_Shoulder",
            "J_Bip_R_UpperArm",
            "J_Bip_R_LowerArm",
            "J_Bip_R_Hand",
            "J_Bip_L_UpperLeg",
            "J_Bip_L_LowerLeg",
            "J_Bip_L_Foot",
            "J_Bip_R_UpperLeg",
            "J_Bip_R_LowerLeg",
            "J_Bip_R_Foot",
        )
        for index, name in enumerate(names):
            bone = data.edit_bones.new(name)
            bone.head = (0.0, 0.0, index * 0.1)
            bone.tail = (0.0, 0.0, index * 0.1 + 0.08)
        bpy.ops.object.mode_set(mode="OBJECT")

    def tearDown(self) -> None:
        bpy.data.objects.remove(self.armature, do_unlink=True)

    def test_vroid_generic_names(self) -> None:
        self.mapping.auto_map(self.armature)
        entries = self.mapping.mapping_by_id(self.armature.mtp_rig_config)
        self.assertEqual(entries["PELVIS"].target_name, "J_Bip_C_Hips")
        self.assertEqual(entries["CHEST"].target_name, "J_Bip_C_UpperChest")
        self.assertEqual(entries["SHOULDER_L"].target_name, "J_Bip_L_Shoulder")
        self.assertEqual(entries["UPPER_ARM_L"].target_name, "J_Bip_L_UpperArm")
        self.assertEqual(entries["LOWER_ARM_R"].target_name, "J_Bip_R_LowerArm")
        self.assertEqual(entries["UPPER_LEG_L"].target_name, "J_Bip_L_UpperLeg")
        self.assertEqual(entries["LOWER_LEG_R"].target_name, "J_Bip_R_LowerLeg")

    def test_mapping_method_refresh_preserves_locks_and_discards_stale_results(
        self,
    ) -> None:
        self.mapping.auto_map(self.armature)
        entries = self.mapping.mapping_by_id(self.armature.mtp_rig_config)
        pelvis = entries["PELVIS"]
        pelvis.manual_lock = True
        pelvis.source = "MANUAL"
        stale = entries["CHEST"]
        with self.mapping.suppress_manual_locks():
            stale.target_name = "StaleAutomaticTarget"
            stale.source = "HEURISTIC"

        self.mapping.clear_unlocked_automatic_mappings(self.armature)

        self.assertEqual(pelvis.target_name, "J_Bip_C_Hips")
        self.assertTrue(pelvis.manual_lock)
        self.assertFalse(stale.target_name)
        self.assertEqual(stale.source, "NONE")

    def test_problem_arrow_selects_pose_bone(self) -> None:
        config = self.armature.mtp_rig_config
        entry = config.mappings.add()
        entry.semantic_id = "PELVIS"
        entry.section = "BODY"
        entry.target_kind = "BONE"
        entry.target_name = "J_Bip_C_Hips"
        config.active_mapping_index = 0
        bpy.context.scene.mtp_settings.target_armature = self.armature
        for pose_bone in self.armature.pose.bones:
            pose_bone.select = True

        result = bpy.ops.mtp.select_problem(mapping_index=0)

        self.assertEqual(result, {"FINISHED"})
        selected = [bone.name for bone in self.armature.pose.bones if bone.select]
        self.assertEqual(selected, ["J_Bip_C_Hips"])
        self.assertEqual(
            self.armature.data.bones.active.name,
            "J_Bip_C_Hips",
        )

    def test_heuristic_information_explains_mapping_confidence(self) -> None:
        entry = SimpleNamespace(
            target_kind="BONE",
            target_name="J_Bip_C_Hips",
            confidence=0.95,
        )
        self.assertEqual(
            self.mapping.heuristic_mapping_message(entry),
            'Auto-matched to bone "J_Bip_C_Hips" (95%). This percentage '
            "estimates the target name, side, and hierarchy match, not "
            "pose-detection accuracy.",
        )

        self.mapping.auto_map(self.armature)
        mapped = self.mapping.mapping_by_id(self.armature.mtp_rig_config)["PELVIS"]
        self.assertLess(mapped.confidence, 1.0)
        self.assertEqual(mapped.status, "INFO")
        _valid, warnings, _errors = self.mapping.validate_mapping(self.armature)
        self.assertEqual(mapped.status, "INFO")
        self.assertNotIn("warning", mapped.message.casefold())
        self.assertEqual(warnings, 0)
        self.assertIn(
            "auto-detected",
            self.armature.mtp_rig_config.validation_summary,
        )

    def test_head_and_eyes_are_body_mappings(self) -> None:
        self.mapping.ensure_mapping_entries(self.armature)
        entries = self.mapping.mapping_by_id(self.armature.mtp_rig_config)
        self.assertEqual(entries["HEAD"].section, "BODY")
        self.assertEqual(entries["EYE_L"].section, "BODY")
        self.assertEqual(entries["EYE_R"].section, "BODY")
        self.assertEqual(entries["EYE_L"].label, "Left Eye")
        self.assertEqual(entries["EYE_R"].label, "Right Eye")
        self.assertEqual(entries["FACE_JAW_OPEN"].section, "FACE_RIG")
        self.assertTrue(
            all(
                entry.semantic_id not in {"EYE_L", "EYE_R"}
                for entry in self.armature.mtp_rig_config.mappings
                if entry.section == "FACE_RIG"
            )
        )
        entries["EYE_L"].label = "Incorrect Left Label"
        entries["EYE_R"].label = "Incorrect Right Label"
        self.mapping.ensure_mapping_entries(self.armature)
        self.assertEqual(entries["EYE_L"].label, "Left Eye")
        self.assertEqual(entries["EYE_R"].label, "Right Eye")

    def test_disabled_facial_posing_suppresses_face_mapping_warnings(self) -> None:
        settings = bpy.context.scene.mtp_settings
        settings.target_armature = self.armature
        settings.provider = "MEDIAPIPE"
        self.mapping.ensure_mapping_entries(self.armature)
        entry = self.mapping.mapping_by_id(self.armature.mtp_rig_config)[
            "FACE_JAW_OPEN"
        ]
        with self.mapping.suppress_manual_locks():
            entry.target_name = "MissingJawControl"

        settings.use_face = False
        self.mapping.validate_mapping(self.armature, settings=settings)
        self.assertEqual(entry.status, "OPTIONAL")
        self.assertEqual(entry.message, "Facial posing is disabled")

        settings.use_face = True
        self.mapping.validate_mapping(self.armature, settings=settings)
        self.assertEqual(entry.status, "ERROR")
        self.assertEqual(entry.message, "Target no longer exists")

    def test_face_pass_keeps_gaze_and_expressions(self) -> None:
        provider_type = self.addon.providers.MediaPipeProvider
        provider = object.__new__(provider_type)
        provider.use_body = True
        provider.use_face = True
        provider.np = np
        point = SimpleNamespace(x=0.5, y=0.5, z=0.0)
        other_face = SimpleNamespace(x=0.1, y=0.1, z=0.0)
        result = SimpleNamespace(
            face_landmarks=[
                [other_face, other_face],
                [point, point],
            ],
            face_blendshapes=[
                [],
                [
                    SimpleNamespace(
                        category_name="eyeLookOutLeft",
                        display_name="",
                        score=0.75,
                    ),
                    SimpleNamespace(
                        category_name="mouthSmileLeft",
                        display_name="",
                        score=0.9,
                    ),
                ],
            ],
        )
        frame = self.addon.models.CanonicalFrame(timestamp_ms=0)
        frame.body["NOSE"] = self.addon.models.Landmark(
            position=(0.0, 0.0, 0.0),
            image=(0.5, 0.5),
        )

        provider._convert_face(result, frame)

        self.assertEqual(
            frame.blendshapes,
            {"eyeLookOutLeft": 0.75, "mouthSmileLeft": 0.9},
        )
        self.assertEqual(len(frame.face_overlay), 2)
        missing = self.addon.models.CanonicalFrame(timestamp_ms=0)
        provider._convert_face(None, missing)
        self.assertEqual(missing.warnings, ["No face detected"])

    def test_pose_wrist_matching_rejects_duplicate_hand_for_opposite_side(self) -> None:
        provider = object.__new__(self.addon.providers.MediaPipeProvider)
        provider.confidence = 0.5
        provider.mirror = False
        frame = self.addon.models.CanonicalFrame(timestamp_ms=0)
        frame.body.update(
            {
                "LEFT_ELBOW": self.addon.models.Landmark((0.0, 0.0, 0.0), (0.2, 0.5)),
                "LEFT_WRIST": self.addon.models.Landmark((0.0, 0.0, 0.0), (0.3, 0.5)),
                "RIGHT_ELBOW": self.addon.models.Landmark((0.0, 0.0, 0.0), (0.8, 0.5)),
                "RIGHT_WRIST": self.addon.models.Landmark((0.0, 0.0, 0.0), (0.7, 0.5)),
            }
        )

        def point(x, y):
            return SimpleNamespace(
                x=x,
                y=y,
                z=0.0,
                visibility=1.0,
                presence=1.0,
            )

        detailed = self._detailed_hand()
        world = [point(*landmark.position[:2]) for landmark in detailed.values()]
        result = SimpleNamespace(
            hand_world_landmarks=[
                world,
                world,
            ],
            hand_landmarks=[
                [point(0.30, 0.50) for _ in range(21)],
                [point(0.32, 0.51) for _ in range(21)],
            ],
            handedness=[
                [SimpleNamespace(category_name="Left", display_name="")],
                [SimpleNamespace(category_name="Right", display_name="")],
            ],
        )

        provider._convert_hands(result, frame)

        self.assertEqual(set(frame.hands), {"LEFT"})

    def test_mediapipe_pose_alignment_recovers_camera_translation(self) -> None:
        width, height = 1280, 720
        focal = float(width)
        expected = np.asarray((0.42, -0.18, 3.25), dtype=float)
        positions = {
            "LEFT_SHOULDER": (0.25, -0.55, -0.08),
            "RIGHT_SHOULDER": (-0.24, -0.54, 0.06),
            "LEFT_ELBOW": (0.48, -0.25, 0.11),
            "RIGHT_ELBOW": (-0.46, -0.24, -0.09),
            "LEFT_WRIST": (0.60, 0.02, 0.18),
            "RIGHT_WRIST": (-0.57, 0.04, -0.14),
            "LEFT_HIP": (0.16, 0.0, -0.03),
            "RIGHT_HIP": (-0.16, 0.0, 0.03),
            "LEFT_KNEE": (0.19, 0.52, 0.08),
            "RIGHT_KNEE": (-0.18, 0.50, -0.07),
            "LEFT_ANKLE": (0.20, 1.00, 0.13),
            "RIGHT_ANKLE": (-0.19, 0.98, -0.11),
        }
        frame = self.addon.models.CanonicalFrame(timestamp_ms=0)
        for name, position in positions.items():
            camera = np.asarray(position, dtype=float) + expected
            image = (
                0.5 + focal * camera[0] / camera[2] / width,
                0.5 + focal * camera[1] / camera[2] / height,
            )
            frame.body[name] = self.addon.models.Landmark(
                position,
                image,
                confidence=0.95,
            )
        frame.body["LEFT_WRIST"].image = (0.99, 0.01)

        observation = self.addon.providers.mediapipe_provider._estimate_root_motion(
            frame,
            width,
            height,
            np,
        )

        self.assertIsNotNone(observation)
        np.testing.assert_allclose(observation.translation, expected, atol=1.0e-6)
        self.assertAlmostEqual(observation.focal_length, focal)
        self.assertGreater(observation.confidence, 0.85)

    def test_mediapipe_pose_alignment_rejects_degenerate_observations(self) -> None:
        frame = self.addon.models.CanonicalFrame(timestamp_ms=0)
        for name in self.addon.providers.mediapipe_provider._ROOT_MOTION_LANDMARKS:
            frame.body[name] = self.addon.models.Landmark(
                (0.0, 0.0, 0.0),
                (0.5, 0.5),
            )

        observation = self.addon.providers.mediapipe_provider._estimate_root_motion(
            frame,
            1920,
            1080,
            np,
        )

        self.assertIsNone(observation)

    def _detailed_hand(self, image_x=0.5, *, ring_tip=None):
        positions = {
            "WRIST": (0.0, 0.0, 0.0),
            "THUMB_CMC": (0.25, 0.25, 0.0),
            "THUMB_MCP": (0.38, 0.45, 0.0),
            "THUMB_IP": (0.48, 0.65, 0.0),
            "THUMB_TIP": (0.55, 0.85, 0.0),
            "INDEX_FINGER_MCP": (0.4, 1.0, 0.0),
            "INDEX_FINGER_PIP": (0.43, 1.45, 0.0),
            "INDEX_FINGER_DIP": (0.46, 1.78, 0.0),
            "INDEX_FINGER_TIP": (0.48, 2.05, 0.0),
            "MIDDLE_FINGER_MCP": (0.0, 1.1, 0.0),
            "MIDDLE_FINGER_PIP": (0.0, 1.6, 0.0),
            "MIDDLE_FINGER_DIP": (0.0, 1.96, 0.0),
            "MIDDLE_FINGER_TIP": (0.0, 2.25, 0.0),
            "RING_FINGER_MCP": (-0.25, 1.0, 0.0),
            "RING_FINGER_PIP": (-0.28, 1.46, 0.0),
            "RING_FINGER_DIP": (-0.3, 1.8, 0.0),
            "RING_FINGER_TIP": ring_tip or (-0.32, 2.06, 0.0),
            "PINKY_MCP": (-0.5, 0.85, 0.0),
            "PINKY_PIP": (-0.58, 1.25, 0.0),
            "PINKY_DIP": (-0.62, 1.55, 0.0),
            "PINKY_TIP": (-0.65, 1.78, 0.0),
        }
        return {
            name: self.addon.models.Landmark(
                position,
                (image_x + position[0] * 0.02, 0.5 + position[1] * 0.02),
            )
            for name, position in positions.items()
        }

    def test_overlapping_pose_crops_keep_only_closest_hand(self) -> None:
        provider = object.__new__(self.addon.providers.MediaPipeProvider)
        provider.video_mode = False
        provider._trusted_hands = {}
        frame = self.addon.models.CanonicalFrame(timestamp_ms=0)
        frame.body.update(
            {
                "LEFT_ELBOW": self.addon.models.Landmark((0.0, 0.0, 0.0), (0.2, 0.5)),
                "LEFT_WRIST": self.addon.models.Landmark((0.0, 0.0, 0.0), (0.3, 0.5)),
                "RIGHT_ELBOW": self.addon.models.Landmark((0.0, 0.0, 0.0), (0.8, 0.5)),
                "RIGHT_WRIST": self.addon.models.Landmark((0.0, 0.0, 0.0), (0.7, 0.5)),
            }
        )
        first = self._detailed_hand(0.31)
        duplicate = self._detailed_hand(0.32)

        provider._assign_pose_guided_hands(
            [("LEFT", first), ("RIGHT", duplicate)], frame
        )

        self.assertEqual(set(frame.hands), {"LEFT"})
        self.assertTrue(any("same detailed hand" in item for item in frame.warnings))

    def test_random_access_video_frame_accepts_first_valid_hand(self) -> None:
        provider = object.__new__(self.addon.providers.MediaPipeProvider)
        provider.video_mode = True
        provider._trusted_hands = {}
        first = self.addon.models.CanonicalFrame(timestamp_ms=0)

        self.assertTrue(provider._trust_hand("RIGHT", self._detailed_hand(), first))

    def test_bad_finger_geometry_is_left_for_bounded_mapper(self) -> None:
        provider = object.__new__(self.addon.providers.MediaPipeProvider)
        provider.video_mode = False
        provider._trusted_hands = {}
        first_hand = self._detailed_hand()
        first = self.addon.models.CanonicalFrame(timestamp_ms=0)
        self.assertTrue(provider._trust_hand("RIGHT", first_hand, first))
        malformed = self._detailed_hand(ring_tip=(4.0, 4.0, 0.0))
        original_middle_tip = malformed["MIDDLE_FINGER_TIP"].position
        second = self.addon.models.CanonicalFrame(timestamp_ms=33)

        self.assertTrue(provider._trust_hand("RIGHT", malformed, second))

        self.assertEqual(
            malformed["RING_FINGER_TIP"].position,
            (4.0, 4.0, 0.0),
        )
        self.assertEqual(
            malformed["MIDDLE_FINGER_TIP"].position,
            original_middle_tip,
        )
        self.assertFalse(second.warnings)

    def test_short_hand_occlusion_holds_last_trusted_fingers(self) -> None:
        provider = object.__new__(self.addon.providers.MediaPipeProvider)
        provider.video_mode = False
        provider._trusted_hands = {}
        trusted = self._detailed_hand()
        first = self.addon.models.CanonicalFrame(timestamp_ms=0)
        self.assertTrue(provider._trust_hand("LEFT", trusted, first))
        missing = self.addon.models.CanonicalFrame(timestamp_ms=100)

        provider._hold_recent_hands(missing)

        self.assertIn("LEFT", missing.hands)
        self.assertLess(missing.hands["LEFT"]["WRIST"].confidence, 1.0)
        self.assertTrue(
            any("temporarily occluded" in item for item in missing.warnings)
        )

    def test_pose_guided_hand_crop_remaps_to_full_frame(self) -> None:
        provider = object.__new__(self.addon.providers.MediaPipeProvider)
        provider.cv2 = SimpleNamespace(flip=lambda image, _axis: image[:, ::-1])
        provider.np = np
        provider.mirror = False
        provider.confidence = 0.5
        provider.video_mode = False
        provider._mp_image = lambda image, apply_mirror=False: image
        provider._hand_crop_task = lambda _side: object()
        frame = self.addon.models.CanonicalFrame(timestamp_ms=0)
        frame.body.update(
            {
                "RIGHT_ELBOW": self.addon.models.Landmark(
                    (0.0, 0.0, 0.0), (0.60, 0.52)
                ),
                "RIGHT_WRIST": self.addon.models.Landmark(
                    (0.0, 0.0, 0.0), (0.70, 0.50)
                ),
                "RIGHT_THUMB": self.addon.models.Landmark(
                    (0.0, 0.0, 0.0), (0.74, 0.47)
                ),
                "RIGHT_INDEX": self.addon.models.Landmark(
                    (0.0, 0.0, 0.0), (0.77, 0.45)
                ),
                "RIGHT_PINKY": self.addon.models.Landmark(
                    (0.0, 0.0, 0.0), (0.76, 0.53)
                ),
            }
        )

        def point(x, y, z=0.0):
            return SimpleNamespace(
                x=x,
                y=y,
                z=z,
                visibility=1.0,
                presence=1.0,
            )

        result = SimpleNamespace(
            hand_world_landmarks=[[point(index * 0.01, 0.0) for index in range(21)]],
            hand_landmarks=[[point(0.5, 0.5) for _ in range(21)]],
        )
        provider._detect_task = lambda _task, _image, _timestamp: result
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        _crop, (x0, y0, x1, y1), _size = provider._pose_guided_hand_crop(
            image, frame, "RIGHT"
        )

        hand = provider._detect_hand_crop(image, frame, "RIGHT", 0)

        self.assertIsNotNone(hand)
        self.assertEqual(len(hand), 21)
        self.assertAlmostEqual(hand["WRIST"].image[0], (x0 + x1) * 0.5 / 600)
        self.assertAlmostEqual(hand["WRIST"].image[1], (y0 + y1) * 0.5 / 400)

    def test_temporal_palm_flip_does_not_suppress_a_valid_frame(self) -> None:
        provider = object.__new__(self.addon.providers.MediaPipeProvider)
        provider._trusted_hands = {}

        def landmark(position):
            return self.addon.models.Landmark(position, (0.5, 0.5))

        first = {
            "WRIST": landmark((0.0, 0.0, 0.0)),
            "MIDDLE_FINGER_MCP": landmark((0.0, 1.0, 0.0)),
            "INDEX_FINGER_MCP": landmark((1.0, 1.0, 0.0)),
            "PINKY_MCP": landmark((-1.0, 1.0, 0.0)),
        }
        flipped = {
            "WRIST": landmark((0.0, 0.0, 0.0)),
            "MIDDLE_FINGER_MCP": landmark((0.0, 1.0, 0.0)),
            "INDEX_FINGER_MCP": landmark((-1.0, 1.0, 0.0)),
            "PINKY_MCP": landmark((1.0, 1.0, 0.0)),
        }
        first_frame = self.addon.models.CanonicalFrame(timestamp_ms=0)
        flipped_frame = self.addon.models.CanonicalFrame(timestamp_ms=40)

        self.assertTrue(provider._trust_hand("RIGHT", first, first_frame))
        self.assertTrue(provider._trust_hand("RIGHT", flipped, flipped_frame))
        self.assertFalse(flipped_frame.warnings)

    def test_realistic_face_recovers_from_pose_guided_crop(self) -> None:
        provider = self.addon.providers.MediaPipeProvider(
            model_directory=str(Path(self.addon.__file__).parent / "assets" / "models"),
            use_body=True,
            use_hands=False,
            use_face=True,
            mirror=False,
            confidence=0.5,
            smoothing=0.0,
            video_mode=False,
        )
        try:
            frame = provider.detect_image(
                str(fixture_path("mediapipe_ref_pose_realistic"))
            )
        finally:
            provider.close()

        self.assertEqual(len(frame.body), 33)
        self.assertGreaterEqual(len(frame.face_landmarks), 468)
        self.assertTrue(frame.blendshapes)
        self.assertLess(
            math.dist(
                frame.body["NOSE"].image,
                frame.face_landmarks[1].image,
            ),
            0.08,
        )
        self.assertNotIn("No face detected", frame.warnings)
        self.assertTrue(
            any(
                "pose" in warning and "closer crop" in warning
                for warning in frame.warnings
            )
        )

    def test_reduced_threshold_pose_fallback_is_visible(self) -> None:
        provider = object.__new__(self.addon.providers.MediaPipeProvider)
        strict_task = object()
        fallback_task = object()
        missing = SimpleNamespace(pose_world_landmarks=[], pose_landmarks=[])
        recovered = SimpleNamespace(
            pose_world_landmarks=[[object()]],
            pose_landmarks=[[object()]],
        )
        provider._pose = strict_task
        provider._hands = None
        provider._face = None
        provider.confidence = 0.5
        provider.video_mode = False
        provider._mp_image = lambda _bgr: object()
        provider._fallback_pose_task = lambda: fallback_task
        provider._detect_task = (
            lambda task, _image, _timestamp: recovered
            if task is fallback_task
            else missing
        )
        used = {}

        def convert(_result, _frame, *, landmark_threshold):
            used["threshold"] = landmark_threshold

        provider._convert_pose = convert
        provider._smooth_frame = lambda frame: frame

        frame = provider._detect(None, 0)

        self.assertEqual(used["threshold"], 0.25)
        self.assertTrue(
            any("reduced confidence threshold" in warning for warning in frame.warnings)
        )

    def test_reduced_threshold_face_fallback_uses_best_candidate(self) -> None:
        provider = object.__new__(self.addon.providers.MediaPipeProvider)
        strict_task = object()
        fallback_task = object()
        point = SimpleNamespace(x=0.5, y=0.4, z=0.0)
        missing = SimpleNamespace(face_landmarks=[])
        recovered = SimpleNamespace(
            face_landmarks=[[point, point]],
            face_blendshapes=[[]],
        )
        provider._pose = None
        provider._hands = None
        provider._face = strict_task
        provider.use_face = True
        provider.confidence = 0.5
        provider.video_mode = False
        provider._previous = None
        provider._mp_image = lambda _bgr: object()
        provider._fallback_face_task = lambda: fallback_task
        provider._detect_face_crop = lambda _bgr, _frame, fallback: None
        provider._detect_task = (
            lambda task, _image, _timestamp: recovered
            if task is fallback_task
            else missing
        )
        provider._smooth_frame = lambda frame: frame

        frame = provider._detect(None, 0)

        self.assertEqual(len(frame.face_landmarks), 2)
        self.assertTrue(
            any("most suitable candidate" in warning for warning in frame.warnings)
        )

    def test_provider_identifier_is_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown capture provider"):
            self.addon.providers.descriptor("TYPO")

    def _map_names(self, label, names, profile="GENERIC"):
        data = bpy.data.armatures.new(f"{label}Data")
        armature = bpy.data.objects.new(label, data)
        if profile == "RIGIFY":
            data["rig_id"] = "test-rigify"
        elif profile == "AUTO_RIG_PRO":
            data["arp_updated"] = True
        bpy.context.scene.collection.objects.link(armature)
        bpy.ops.object.select_all(action="DESELECT")
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode="EDIT")
        for index, name in enumerate(names):
            bone = data.edit_bones.new(name)
            bone.head = (index * 0.01, 0.0, 0.0)
            bone.tail = (index * 0.01, 0.0, 0.1)
        bpy.ops.object.mode_set(mode="OBJECT")
        try:
            self.mapping.auto_map(armature)
            return {
                semantic_id: entry.target_name
                for semantic_id, entry in self.mapping.mapping_by_id(
                    armature.mtp_rig_config
                ).items()
            }
        finally:
            bpy.data.objects.remove(armature, do_unlink=True)

    def test_eye_mapping_across_supported_armature_types(self) -> None:
        scenarios = (
            (
                "Generic",
                ("Head", "LeftEye", "RightEye"),
                "GENERIC",
                ("LeftEye", "RightEye"),
            ),
            (
                "CharacterCreator",
                ("CC_Base_Head", "CC_Base_L_Eye", "CC_Base_R_Eye"),
                "GENERIC",
                ("CC_Base_L_Eye", "CC_Base_R_Eye"),
            ),
            (
                "Rigify",
                ("head", "eye.L", "eye.R"),
                "RIGIFY",
                ("eye.L", "eye.R"),
            ),
            (
                "AutoRigPro",
                ("c_head.x", "c_eye_target.l", "c_eye_target.r"),
                "AUTO_RIG_PRO",
                ("c_eye_target.l", "c_eye_target.r"),
            ),
            (
                "VRoidNames",
                ("J_Bip_C_Head", "J_Adj_L_FaceEye", "J_Adj_R_FaceEye"),
                "GENERIC",
                ("J_Adj_L_FaceEye", "J_Adj_R_FaceEye"),
            ),
        )
        for label, names, profile, expected in scenarios:
            with self.subTest(label=label):
                mapped = self._map_names(label, names, profile)
                self.assertEqual(mapped["EYE_L"], expected[0])
                self.assertEqual(mapped["EYE_R"], expected[1])

    def test_common_exporter_naming_profiles_are_side_safe(self) -> None:
        scenarios = (
            (
                "Mixamo",
                (
                    "mixamorig:Hips",
                    "mixamorig:Spine",
                    "mixamorig:Spine1",
                    "mixamorig:Spine2",
                    "mixamorig:LeftArm",
                    "mixamorig:LeftForeArm",
                    "mixamorig:LeftHand",
                    "mixamorig:LeftToeBase",
                    "mixamorig:LeftHandIndex1",
                ),
                {
                    "PELVIS": "mixamorig:Hips",
                    "UPPER_ARM_L": "mixamorig:LeftArm",
                    "LOWER_ARM_L": "mixamorig:LeftForeArm",
                    "TOE_L": "mixamorig:LeftToeBase",
                    "INDEX_01_L": "mixamorig:LeftHandIndex1",
                },
            ),
            (
                "Unreal",
                (
                    "root",
                    "pelvis",
                    "spine_01",
                    "spine_02",
                    "spine_03",
                    "clavicle_l",
                    "upperarm_l",
                    "lowerarm_l",
                    "hand_l",
                    "thigh_l",
                    "calf_l",
                    "foot_l",
                    "ball_l",
                    "index_01_l",
                ),
                {
                    "ROOT": "root",
                    "CHEST": "spine_03",
                    "SHOULDER_L": "clavicle_l",
                    "LOWER_LEG_L": "calf_l",
                    "TOE_L": "ball_l",
                    "INDEX_01_L": "index_01_l",
                },
            ),
            (
                "Daz",
                (
                    "hip",
                    "abdomenLower",
                    "abdomenUpper",
                    "chestUpper",
                    "lCollar",
                    "lShldrBend",
                    "lForearmBend",
                    "lHand",
                    "lThighBend",
                    "lShin",
                    "lFoot",
                    "lToe",
                    "lIndex1",
                ),
                {
                    "PELVIS": "hip",
                    "SPINE_01": "abdomenLower",
                    "UPPER_ARM_L": "lShldrBend",
                    "LOWER_ARM_L": "lForearmBend",
                    "UPPER_LEG_L": "lThighBend",
                    "INDEX_01_L": "lIndex1",
                },
            ),
            (
                "Mmd",
                (
                    "センター",
                    "下半身",
                    "上半身",
                    "上半身2",
                    "上半身3",
                    "首",
                    "頭",
                    "左肩",
                    "左腕",
                    "左ひじ",
                    "左手首",
                    "左足",
                    "左ひざ",
                    "左足首",
                    "左つま先",
                    "左人指１",
                ),
                {
                    "ROOT": "センター",
                    "PELVIS": "下半身",
                    "CHEST": "上半身3",
                    "UPPER_ARM_L": "左腕",
                    "LOWER_LEG_L": "左ひざ",
                    "TOE_L": "左つま先",
                    "INDEX_01_L": "左人指１",
                },
            ),
        )
        for label, names, expected in scenarios:
            with self.subTest(label=label):
                mapped = self._map_names(label, names)
                self.assertEqual(
                    {key: mapped[key] for key in expected},
                    expected,
                )

        one_sided = self._map_names(
            "OneSided",
            ("RightArm", "RightForeArm", "RightHand"),
        )
        self.assertEqual(one_sided["UPPER_ARM_L"], "")
        self.assertEqual(one_sided["LOWER_ARM_L"], "")
        self.assertEqual(one_sided["UPPER_ARM_R"], "RightArm")
        self.assertEqual(one_sided["LOWER_ARM_R"], "RightForeArm")


class VrmMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.addon = register_addon()
        cls.mapping = cls.addon.rig.mapping

    @staticmethod
    def _node(name):
        return SimpleNamespace(node=SimpleNamespace(bone_name=name))

    def test_vrm_one_metadata_maps_named_humanoid_bones(self) -> None:
        bones = SimpleNamespace(
            hips=self._node("hips"),
            spine=self._node("spine"),
            chest=self._node("chest"),
            upper_chest=self._node("upper_chest"),
            left_eye=self._node("unusual_left_eye_bone"),
            right_eye=self._node("unusual_right_eye_bone"),
            left_upper_arm=self._node("odd_arm"),
            left_thumb_metacarpal=self._node("thumb_a"),
            left_thumb_proximal=self._node("thumb_b"),
            left_thumb_distal=self._node("thumb_c"),
        )
        armature = SimpleNamespace(
            data=SimpleNamespace(
                vrm_addon_extension=SimpleNamespace(
                    spec_version="1.0",
                    vrm1=SimpleNamespace(humanoid=SimpleNamespace(human_bones=bones)),
                )
            )
        )
        result = self.mapping.vrm_humanoid_mapping(armature)
        self.assertEqual(result["PELVIS"], "hips")
        self.assertEqual(result["CHEST"], "upper_chest")
        self.assertEqual(result["UPPER_ARM_L"], "odd_arm")
        self.assertEqual(result["EYE_L"], "unusual_left_eye_bone")
        self.assertEqual(result["EYE_R"], "unusual_right_eye_bone")
        self.assertEqual(result["THUMB_01_L"], "thumb_a")
        self.assertEqual(result["THUMB_02_L"], "thumb_b")
        self.assertEqual(result["THUMB_03_L"], "thumb_c")

    def test_vrm_metadata_remains_part_of_the_generic_profile(self) -> None:
        self.assertEqual(
            set(self.addon.rig.profiles.common.PROFILE_LABELS),
            {"AUTO", "AUTO_RIG_PRO", "RIGIFY", "GENERIC"},
        )

    def test_automatic_mapping_refreshes_when_humanoid_metadata_appears(self) -> None:
        entry = SimpleNamespace(
            semantic_id="UPPER_LEG_L",
            manual_lock=False,
            target_name="",
        )
        config = SimpleNamespace(
            profile="AUTO",
            detected_profile="GENERIC",
            armature_signature="unchanged",
            mappings=[entry],
        )
        armature = SimpleNamespace(type="ARMATURE", mtp_rig_config=config)
        with (
            mock.patch.object(
                self.mapping,
                "detect_profile",
                return_value="GENERIC",
            ),
            mock.patch.object(
                self.mapping,
                "armature_signature",
                return_value="unchanged",
            ),
            mock.patch.object(
                self.mapping,
                "vrm_humanoid_mapping",
                return_value={"UPPER_LEG_L": "J_Bip_L_UpperLeg"},
            ),
            mock.patch.object(self.mapping, "auto_map") as auto_map,
        ):
            self.assertTrue(self.mapping.refresh_auto_mapping(armature))
        auto_map.assert_called_once_with(armature, settings=None)


if __name__ == "__main__":
    unittest.main()
