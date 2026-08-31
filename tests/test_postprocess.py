from __future__ import annotations

import importlib
import math
import unittest
from types import SimpleNamespace
from unittest import mock

from mathutils import Matrix, Quaternion, Vector

from tests.blender_test_support import load_addon


class _Collection(list):
    def add(self):
        value = SimpleNamespace()
        self.append(value)
        return value

    def remove(self, index):
        del self[index]


class PostProcessCompilationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.addon = load_addon()
        cls.post = cls.addon.post_processing

    def _config(self):
        modules = _Collection()
        defaults = {
            "strength": 1.0,
            "start_elevation": 0.0,
            "full_influence_elevation": 1.0,
            "maximum_swing": 1.0,
            "maximum_twist": 1.0,
            "repair_arm_silhouette": True,
            "maximum_arm_extension": 0.1,
            "hinge_projection": True,
            "confidence_threshold": 0.5,
            "upper_arm_twist_limit": 1.0,
            "forearm_twist_limit": 1.0,
            "palm_roll_strength": 1.0,
            "thigh_twist_limit": 1.0,
            "lateral_root_strength": 1.0,
            "reject_unreliable_limbs": True,
            "reject_unusable_frames": True,
            "maximum_motion_rejected": 30.0,
            "manual_bad_frames": "",
            "manual_bad_scope": "BODY",
            "smooth_motion": True,
            "ground_contacts": True,
            "smoothing_strength": 0.5,
            "stabilize_hands": True,
            "finger_max_angular_speed": math.radians(600.0),
            "wrist_max_angular_speed": math.radians(720.0),
            "hand_reacquire_delay": 0.3,
            "visibility_threshold": 0.5,
            "learned_pose_residual": False,
            "root_trajectory_cleanup": True,
        }
        for module_id, name, description in self.post.POST_PROCESSING_MODULES:
            item = SimpleNamespace(
                module_id=module_id,
                display_name=name,
                description=description,
                enabled=True,
                status="VALID",
                last_warning="",
                **defaults,
            )
            modules.append(item)
        return SimpleNamespace(
            detected_profile="GENERIC", post_processing_modules=modules
        )

    def test_configuration_is_immutable_and_ordered(self) -> None:
        source = self._config()
        compiled = self.post.compile_postprocess_config(source)
        self.assertEqual(compiled.detected_profile, "GENERIC")
        self.assertEqual(
            [module.module_id for module in compiled.modules],
            [item[0] for item in self.post.POST_PROCESSING_MODULES],
        )
        source.post_processing_modules[0].strength = 0.0
        self.assertEqual(compiled.modules[0].strength, 1.0)
        with self.assertRaises(TypeError):
            compiled.modules[0].values["strength"] = 0.0

    def test_required_cleanup_modules_and_optional_rohm_are_configured(self) -> None:
        constants = importlib.import_module(f"{self.addon.__name__}.constants")
        identifiers = [item[0] for item in self.post.POST_PROCESSING_MODULES]
        self.assertEqual(constants.SCHEMA_VERSION, 18)
        self.assertIn("SPINE_CORRECTION", identifiers)
        self.assertNotIn("SPINE_RECONSTRUCTION", identifiers)
        self.assertFalse(constants.POST_PROCESSING_DEFAULT_ENABLED["SPINE_CORRECTION"])
        self.assertTrue(
            constants.POST_PROCESSING_DEFAULT_ENABLED["CAMERA_MOTION_CORRECTION"]
        )
        self.assertFalse(constants.POST_PROCESSING_DEFAULT_ENABLED["ROHM_CORRECTION"])

    def test_temporal_processor_is_sequence_only(self) -> None:
        config = self._config()
        pose = SimpleNamespace(
            frame=SimpleNamespace(timestamp_ms=0),
            diagnostic_mode=False,
            applied_modules=[],
            warnings=[],
        )
        stages = [
            stage
            for stage, _current, _total in self.post.iter_pipeline(
                pose, config, module_filter={"TEMPORAL_CORRECTION"}
            )
        ]
        self.assertNotIn("Temporal Correction", stages)
        called = []
        context = self.post.SequenceProcessingContext(
            samples=(),
            timestamps=(),
            root_motion_mode="OFF",
            process_temporal=lambda settings: iter(
                (called.append(settings.module_id),)
            ),
            process_camera_motion=lambda settings: iter(
                (called.append(settings.module_id),)
            ),
            process_detection_filter=lambda settings: iter(
                (called.append(settings.module_id),)
            ),
            process_rohm=lambda settings: iter((called.append(settings.module_id),)),
        )
        list(self.post.iter_sequence_pipeline(context, config))
        self.assertEqual(
            called,
            [
                "CAMERA_MOTION_CORRECTION",
                "DETECTION_FILTER",
                "ROHM_CORRECTION",
                "TEMPORAL_CORRECTION",
            ],
        )

    def test_detection_filter_caps_only_new_rejections_and_repairs_boundary(
        self,
    ) -> None:
        motion_filter = importlib.import_module(f"{self.addon.__name__}.motion_filter")
        semantics = motion_filter.SMPLX_SEMANTICS
        samples = []
        for frame_index in range(3):
            confidence = {semantic: 1.0 for semantic in semantics}
            if frame_index == 0:
                for semantic in (
                    "SHOULDER_L",
                    "UPPER_ARM_L",
                    "LOWER_ARM_L",
                    "HAND_L",
                ):
                    confidence[semantic] = 0.1
            if frame_index == 1:
                confidence["TOE_R"] = 0.0
            rotation = Quaternion((0.7071068, 0.7071068, 0.0, 0.0))
            if frame_index:
                rotation = Quaternion()
            bones = {
                semantic: SimpleNamespace(
                    semantic_id=semantic,
                    key_rotation=True,
                    key_location=False,
                    rotation=rotation.copy(),
                    location=Vector(),
                    confidence=confidence[semantic],
                )
                for semantic in semantics
            }
            samples.append(
                SimpleNamespace(
                    bones=bones,
                    body_height=1.7,
                    rohm_joint_offsets={},
                    rohm_rotations={
                        semantic: bones[semantic].rotation.copy()
                        for semantic in semantics
                    },
                    rohm_confidence=confidence,
                    rohm_root_confidence=1.0,
                )
            )

        stats = motion_filter.filter_motion_observations(
            samples,
            (0.0, 1.0 / 30.0, 2.0 / 30.0),
            reject_unreliable_limbs=True,
            reject_unusable_frames=False,
            maximum_rejected_fraction=0.3,
        )

        self.assertEqual(stats.native_missing_observations, 1)
        self.assertEqual(stats.rejected_observations, 4)
        self.assertLessEqual(stats.rejected_observations, stats.rejection_budget)
        repaired = motion_filter.repair_body_gaps(
            samples,
            (0.0, 1.0 / 30.0, 2.0 / 30.0),
        )
        self.assertGreaterEqual(repaired, 4)
        self.assertLess(
            samples[0].bones["UPPER_ARM_L"].rotation.angle,
            math.radians(1.0),
        )

    def test_manual_source_frame_ranges_mask_requested_scope_outside_budget(
        self,
    ) -> None:
        motion_filter = importlib.import_module(f"{self.addon.__name__}.motion_filter")
        frames = motion_filter.parse_source_frame_ranges("1, 3-5, 99", 6)
        self.assertEqual(frames, frozenset({0, 2, 3, 4}))
        self.assertEqual(
            motion_filter.format_source_frame_ranges(frames),
            "1, 3-5",
        )
        samples = []
        for _index in range(6):
            confidence = {semantic: 1.0 for semantic in motion_filter.SMPLX_SEMANTICS}
            bones = {
                semantic: SimpleNamespace(
                    semantic_id=semantic,
                    key_rotation=True,
                    confidence=1.0,
                )
                for semantic in motion_filter.SMPLX_SEMANTICS
            }
            samples.append(
                SimpleNamespace(
                    bones=bones,
                    rohm_confidence=confidence,
                    rohm_root_confidence=1.0,
                )
            )

        rejected = motion_filter.apply_manual_motion_rejections(
            samples, frames, "ARM_L"
        )

        self.assertEqual(len(rejected), 4 * 4)
        self.assertEqual(samples[2].rohm_confidence["UPPER_ARM_L"], 0.0)
        self.assertEqual(samples[2].rohm_confidence["UPPER_ARM_R"], 1.0)
        self.assertEqual(samples[2].rohm_root_confidence, 1.0)

    def test_rohm_runs_only_for_enabled_masked_channels(self) -> None:
        retarget = self.addon.retarget
        samples = [
            SimpleNamespace(
                timestamp_seconds=index / 30.0,
                warning_sink=[],
                rohm_confidence={
                    semantic: 1.0 for semantic in retarget.motion_filter.SMPLX_SEMANTICS
                },
                rohm_root_confidence=value,
            )
            for index, value in enumerate((1.0, 0.0))
        ]
        settings = SimpleNamespace(
            enabled=True,
            visibility_threshold=0.5,
            learned_pose_residual=False,
            root_trajectory_cleanup=True,
        )

        self.assertEqual(
            retarget._rohm_requested_work(samples, settings, "AUTOMATIC"),
            (False, True),
        )
        samples[1].rohm_root_confidence = 1.0
        self.assertEqual(
            retarget._rohm_requested_work(samples, settings, "AUTOMATIC"),
            (False, False),
        )
        samples[1].rohm_confidence["UPPER_ARM_L"] = 0.0
        settings.learned_pose_residual = True
        self.assertEqual(
            retarget._rohm_requested_work(samples, settings, "OFF"),
            (True, False),
        )

    def test_detection_filter_rejects_confident_source_pose_collapse(self) -> None:
        motion_filter = importlib.import_module(f"{self.addon.__name__}.motion_filter")
        semantics = motion_filter.SMPLX_SEMANTICS
        stable = {
            "LEFT_SHOULDER": Vector((-0.3, 0.0, 1.4)),
            "RIGHT_SHOULDER": Vector((0.3, 0.0, 1.4)),
            "LEFT_ELBOW": Vector((-0.5, 0.0, 1.0)),
            "RIGHT_ELBOW": Vector((0.5, 0.0, 1.0)),
            "LEFT_WRIST": Vector((-0.6, 0.0, 0.6)),
            "RIGHT_WRIST": Vector((0.6, 0.0, 0.6)),
            "LEFT_HIP": Vector((-0.2, 0.0, 0.8)),
            "RIGHT_HIP": Vector((0.2, 0.0, 0.8)),
            "LEFT_KNEE": Vector((-0.2, 0.0, 0.4)),
            "RIGHT_KNEE": Vector((0.2, 0.0, 0.4)),
            "LEFT_ANKLE": Vector((-0.2, 0.0, 0.0)),
            "RIGHT_ANKLE": Vector((0.2, 0.0, 0.0)),
        }
        samples = []
        for frame_index in range(9):
            source_3d = {name: value.copy() for name, value in stable.items()}
            if frame_index == 4:
                for index, name in enumerate(source_3d):
                    source_3d[name] = Vector(
                        (
                            (-1.0 if index % 2 else 1.0) * (0.5 + index * 0.2),
                            0.0,
                            (index % 3) * 1.1,
                        )
                    )
            source_2d = {
                name: Vector((value.x * 0.2 + 0.5, 1.0 - value.z * 0.5))
                for name, value in source_3d.items()
            }
            bones = {
                semantic: SimpleNamespace(
                    semantic_id=semantic,
                    key_rotation=True,
                    key_location=False,
                    rotation=Quaternion(),
                    location=Vector(),
                    confidence=1.0,
                )
                for semantic in semantics
            }
            samples.append(
                SimpleNamespace(
                    bones=bones,
                    body_height=1.7,
                    rohm_joint_offsets={},
                    rohm_rotations={},
                    rohm_confidence={semantic: 1.0 for semantic in semantics},
                    rohm_root_confidence=1.0,
                    source_body_3d=source_3d,
                    source_body_2d=source_2d,
                )
            )

        stats = motion_filter.filter_motion_observations(
            samples,
            tuple(index / 30.0 for index in range(len(samples))),
            reject_unreliable_limbs=False,
            reject_unusable_frames=True,
            maximum_rejected_fraction=0.3,
        )

        self.assertIn(4, stats.rejected_root_frames)
        self.assertTrue(
            all(samples[4].rohm_confidence[semantic] == 0.0 for semantic in semantics)
        )

    def test_detection_filter_rejects_consecutive_confident_scale_collapse(
        self,
    ) -> None:
        motion_filter = importlib.import_module(f"{self.addon.__name__}.motion_filter")
        semantics = motion_filter.SMPLX_SEMANTICS
        stable = {
            "LEFT_SHOULDER": Vector((-0.3, 0.0, 1.4)),
            "RIGHT_SHOULDER": Vector((0.3, 0.0, 1.4)),
            "LEFT_ELBOW": Vector((-0.5, 0.0, 1.0)),
            "RIGHT_ELBOW": Vector((0.5, 0.0, 1.0)),
            "LEFT_WRIST": Vector((-0.6, 0.0, 0.6)),
            "RIGHT_WRIST": Vector((0.6, 0.0, 0.6)),
            "LEFT_HIP": Vector((-0.2, 0.0, 0.8)),
            "RIGHT_HIP": Vector((0.2, 0.0, 0.8)),
            "LEFT_KNEE": Vector((-0.2, 0.0, 0.4)),
            "RIGHT_KNEE": Vector((0.2, 0.0, 0.4)),
            "LEFT_ANKLE": Vector((-0.2, 0.0, 0.0)),
            "RIGHT_ANKLE": Vector((0.2, 0.0, 0.0)),
        }
        samples = []
        for frame_index in range(12):
            scale = 0.2 if 4 <= frame_index <= 6 else 1.0
            source_3d = {name: value * scale for name, value in stable.items()}
            source_2d = {
                name: Vector((value.x * 0.2 + 0.5, 1.0 - value.z * 0.5))
                for name, value in stable.items()
            }
            samples.append(
                SimpleNamespace(
                    bones={
                        semantic: SimpleNamespace(
                            semantic_id=semantic,
                            key_rotation=True,
                            key_location=False,
                            rotation=Quaternion(),
                            location=Vector(),
                            confidence=1.0,
                        )
                        for semantic in semantics
                    },
                    rohm_confidence={semantic: 1.0 for semantic in semantics},
                    rohm_root_confidence=1.0,
                    source_body_3d=source_3d,
                    source_body_2d=source_2d,
                )
            )

        stats = motion_filter.filter_motion_observations(
            samples,
            tuple(index / 30.0 for index in range(len(samples))),
            reject_unreliable_limbs=False,
            reject_unusable_frames=True,
            maximum_rejected_fraction=0.5,
        )

        self.assertTrue({4, 5, 6}.issubset(stats.rejected_root_frames))
        for frame_index in (4, 5, 6):
            self.assertTrue(
                all(
                    samples[frame_index].rohm_confidence[semantic] == 0.0
                    for semantic in semantics
                )
            )

    def test_detection_filter_rejects_consecutive_confident_bad_geometry(
        self,
    ) -> None:
        motion_filter = importlib.import_module(f"{self.addon.__name__}.motion_filter")
        semantics = motion_filter.SMPLX_SEMANTICS
        stable = {
            "LEFT_SHOULDER": Vector((-0.3, 0.0, 1.4)),
            "RIGHT_SHOULDER": Vector((0.3, 0.0, 1.4)),
            "LEFT_ELBOW": Vector((-0.5, 0.0, 1.0)),
            "RIGHT_ELBOW": Vector((0.5, 0.0, 1.0)),
            "LEFT_WRIST": Vector((-0.6, 0.0, 0.6)),
            "RIGHT_WRIST": Vector((0.6, 0.0, 0.6)),
            "LEFT_HIP": Vector((-0.2, 0.0, 0.8)),
            "RIGHT_HIP": Vector((0.2, 0.0, 0.8)),
            "LEFT_KNEE": Vector((-0.2, 0.0, 0.4)),
            "RIGHT_KNEE": Vector((0.2, 0.0, 0.4)),
            "LEFT_ANKLE": Vector((-0.2, 0.0, 0.0)),
            "RIGHT_ANKLE": Vector((0.2, 0.0, 0.0)),
        }
        samples = []
        for frame_index in range(12):
            source_3d = {name: value.copy() for name, value in stable.items()}
            if frame_index in {4, 5}:
                for joint_index, name in enumerate(source_3d):
                    source_3d[name] = Vector(
                        (
                            (-1.0 if joint_index % 2 else 1.0)
                            * (0.3 + joint_index * 0.35),
                            0.0,
                            (joint_index % 4) * 0.9,
                        )
                    )
            source_2d = {
                name: Vector((value.x * 0.2 + 0.5, 1.0 - value.z * 0.5))
                for name, value in source_3d.items()
            }
            samples.append(
                SimpleNamespace(
                    bones={
                        semantic: SimpleNamespace(
                            semantic_id=semantic,
                            key_rotation=True,
                            key_location=False,
                            rotation=Quaternion(),
                            location=Vector(),
                            confidence=1.0,
                        )
                        for semantic in semantics
                    },
                    rohm_confidence={semantic: 1.0 for semantic in semantics},
                    rohm_root_confidence=1.0,
                    source_body_3d=source_3d,
                    source_body_2d=source_2d,
                )
            )

        stats = motion_filter.filter_motion_observations(
            samples,
            tuple(index / 30.0 for index in range(len(samples))),
            reject_unreliable_limbs=False,
            reject_unusable_frames=True,
            maximum_rejected_fraction=0.5,
        )

        self.assertTrue({4, 5}.issubset(stats.rejected_root_frames))

    def test_fallback_interpolates_consecutive_internal_body_gaps(self) -> None:
        motion_filter = importlib.import_module(f"{self.addon.__name__}.motion_filter")
        semantic = "UPPER_ARM_L"
        samples = []
        for frame_index in range(5):
            angle = math.radians(90.0) if frame_index >= 3 else 0.0
            confidence = 0.0 if frame_index in {1, 2} else 1.0
            rotation = Quaternion((0.0, 0.0, 1.0), angle)
            channel = SimpleNamespace(
                semantic_id=semantic,
                key_rotation=True,
                key_location=False,
                rotation=rotation.copy(),
                location=Vector(),
                confidence=confidence,
            )
            samples.append(
                SimpleNamespace(
                    bones={semantic: channel},
                    rohm_confidence={semantic: confidence},
                    rohm_rotations={semantic: rotation.copy()},
                )
            )

        repaired = motion_filter.repair_body_gaps(
            samples,
            tuple(index / 30.0 for index in range(len(samples))),
        )

        self.assertEqual(repaired, 2)
        self.assertAlmostEqual(
            samples[1].bones[semantic].rotation.angle, math.radians(30.0), places=5
        )
        self.assertAlmostEqual(
            samples[2].bones[semantic].rotation.angle, math.radians(60.0), places=5
        )
        self.assertAlmostEqual(
            samples[1].rohm_rotations[semantic].angle, math.radians(30.0), places=5
        )
        self.assertAlmostEqual(
            samples[2].rohm_rotations[semantic].angle, math.radians(60.0), places=5
        )

    def test_detection_filter_repairs_body_even_when_root_ai_is_enabled(self) -> None:
        retarget = self.addon.retarget
        settings = SimpleNamespace(
            reject_unreliable_limbs=True,
            reject_unusable_frames=True,
            maximum_motion_rejected=30.0,
        )
        stats = retarget.motion_filter.MotionFilterStats(
            total_observations=22,
            native_missing_observations=0,
            rejected_observations=0,
            rejection_budget=6,
            rejected_limbs=0,
            rejected_frames=0,
            rejected=frozenset(),
            rejected_root_frames=frozenset(),
        )
        samples = [
            SimpleNamespace(
                timestamp_seconds=index / 30.0,
                warning_sink=[],
                rohm_confidence={
                    semantic: 1.0 for semantic in retarget.motion_filter.SMPLX_SEMANTICS
                },
                rohm_root_confidence=value,
            )
            for index, value in enumerate((1.0, 0.0))
        ]
        rohm_settings = SimpleNamespace(
            enabled=True,
            visibility_threshold=0.5,
            learned_pose_residual=False,
            root_trajectory_cleanup=True,
        )
        with (
            mock.patch.object(
                retarget.motion_filter,
                "filter_motion_observations",
                return_value=stats,
            ),
            mock.patch.object(
                retarget.motion_filter,
                "repair_body_gaps",
                return_value=3,
            ) as repair_body,
            mock.patch.object(retarget, "_repair_root_gaps_without_ai") as repair_root,
        ):
            list(
                retarget._iter_detection_filter(
                    samples,
                    settings,
                    rohm_settings=rohm_settings,
                    effective_mode="AUTOMATIC",
                )
            )

        repair_body.assert_called_once()
        repair_root.assert_not_called()

    def test_ground_contacts_require_owned_body_channels(self) -> None:
        retarget = load_addon().retarget
        face_only = [
            SimpleNamespace(
                bones={
                    "face": SimpleNamespace(semantic_id="JAW"),
                    "hand": SimpleNamespace(semantic_id="INDEX_FINGER_L"),
                }
            )
        ]
        self.assertFalse(retarget._has_groundable_body_channels(face_only))
        body = [
            SimpleNamespace(bones={"leg": SimpleNamespace(semantic_id="LOWER_LEG_L")})
        ]
        self.assertTrue(retarget._has_groundable_body_channels(body))

    def test_low_confidence_palm_roll_is_weighted_not_disabled(self) -> None:
        forearm = SimpleNamespace(
            rest_direction=Vector((0.0, 1.0, 0.0)),
            desired_direction=Vector((0.0, 1.0, 0.0)),
            confidence=0.9,
        )
        hand_channel = SimpleNamespace(
            rest_direction=Vector((0.0, 1.0, 0.0)),
            desired_direction=Vector((0.0, 1.0, 0.0)),
            rest_reference=Vector((1.0, 0.0, 0.0)),
            confidence=0.1,
        )

        def landmark(position):
            return SimpleNamespace(
                position=position,
                confidence=0.1,
                valid=True,
            )

        pose = SimpleNamespace(
            frame=SimpleNamespace(
                body={
                    "LEFT_INDEX": landmark((0.0, 0.0, 1.0)),
                    "LEFT_PINKY": landmark((0.0, 0.0, 0.0)),
                }
            ),
            channel_for=lambda semantic: {
                "LOWER_ARM_L": forearm,
                "HAND_L": hand_channel,
            }.get(semantic),
            to_target_direction=lambda value: value,
        )
        settings = SimpleNamespace(
            confidence_threshold=0.75,
            palm_roll_strength=1.0,
            forearm_twist_limit=math.pi,
        )

        observation = self.post.ArmAxisCorrection()._palm_forearm_twist(
            pose,
            "L",
            settings,
            self.post.PostProcessingSession(),
        )

        self.assertIsNotNone(observation)
        angle, confidence = observation
        self.assertGreater(abs(angle), math.radians(80.0))
        self.assertAlmostEqual(confidence, 0.1)

    def test_palm_roll_unwraps_across_signed_angle_boundary(self) -> None:
        forearm = SimpleNamespace(
            rest_direction=Vector((0.0, 1.0, 0.0)),
            desired_direction=Vector((0.0, 1.0, 0.0)),
            confidence=1.0,
        )
        hand_channel = SimpleNamespace(
            rest_reference=Vector((1.0, 0.0, 0.0)),
            confidence=1.0,
        )

        def pose_at(angle_degrees):
            lateral = Quaternion(
                Vector((0.0, 1.0, 0.0)),
                math.radians(angle_degrees),
            ) @ Vector((1.0, 0.0, 0.0))

            def landmark(position):
                return SimpleNamespace(
                    position=position,
                    confidence=1.0,
                    valid=True,
                )

            return SimpleNamespace(
                frame=SimpleNamespace(
                    body={
                        "LEFT_INDEX": landmark(lateral),
                        "LEFT_PINKY": landmark(Vector()),
                    }
                ),
                channel_for=lambda semantic: {
                    "LOWER_ARM_L": forearm,
                    "HAND_L": hand_channel,
                }.get(semantic),
                to_target_direction=lambda value: value,
            )

        settings = SimpleNamespace(
            palm_roll_strength=1.0,
            forearm_twist_limit=math.radians(90.0),
        )
        processor = self.post.ArmAxisCorrection()
        session = self.post.PostProcessingSession()
        session.prepare_frame(0)
        before = processor._palm_forearm_twist(pose_at(-170.0), "L", settings, session)
        session.prepare_frame(33)
        after = processor._palm_forearm_twist(pose_at(170.0), "L", settings, session)

        self.assertAlmostEqual(math.degrees(before[0]), -90.0)
        self.assertAlmostEqual(math.degrees(after[0]), -90.0)

    def test_forearm_phase_resets_when_sequence_rewinds(self) -> None:
        session = self.post.PostProcessingSession()
        session.prepare_frame(100)
        self.assertAlmostEqual(
            session.unwrap_forearm_twist("L", math.radians(-170.0)),
            math.radians(-170.0),
        )
        session.prepare_frame(50)
        self.assertAlmostEqual(
            session.unwrap_forearm_twist("L", math.radians(170.0)),
            math.radians(170.0),
        )

    def test_generated_ik_controls_inherit_limb_confidence(self) -> None:
        retarget = self.addon.retarget
        target, pole, bend, toe = object(), object(), object(), object()
        confidences = retarget._ik_bone_confidences(
            [
                {
                    "target": target,
                    "pole": pole,
                    "rotation_bend": bend,
                    "toe": toe,
                    "confidence": 0.18,
                }
            ]
        )
        self.assertEqual(
            confidences,
            {target: 0.18, pole: 0.18, bend: 0.18, toe: 0.18},
        )

    def test_ik_effector_rotation_invalidates_the_pole_inverse(self) -> None:
        retarget = self.addon.retarget

        class Control:
            def __init__(self, name):
                self.name = name
                self.matrix = Matrix.Identity(4)

        class Solver:
            instances = []

            def __init__(self):
                self.events = []
                self.instances.append(self)

            def solve(self, bone, _target, **_kwargs):
                self.events.append(("solve", bone.name))

            def invalidate(self, bone=None):
                self.events.append(("invalidate", bone.name if bone else None))

        target = Control("foot target")
        pole = Control("knee pole")
        roles = ("UPPER_LEG_L", "LOWER_LEG_L", "FOOT_L")
        root = SimpleNamespace(head=Vector((0.0, 0.0, 0.0)))
        lower = SimpleNamespace(head=Vector((0.0, 1.0, 0.0)))
        end = SimpleNamespace(head=Vector((0.0, 2.0, 0.0)))
        outputs = dict(zip(roles, (root, lower, end), strict=True))
        proposal = {
            "key": ("LEG", "L"),
            "armature": object(),
            "entries": {},
            "roles": roles,
            "target": target,
            "pole": pole,
            "target_matrix": Matrix.Translation((0.0, 2.0, 0.0)),
            "pole_location": Vector((1.0, 1.0, 0.0)),
            "desired_joint": Vector((0.0, 1.0, 0.0)),
            "desired_end": Vector((0.0, 2.0, 0.0)),
            "upper_direction": Vector((0.0, 1.0, 0.0)),
            "lower_direction": Vector((0.0, 1.0, 0.0)),
            "upper_length": 1.0,
            "lower_length": 1.0,
        }

        with (
            mock.patch.object(retarget, "PoseTranslationSolver", Solver),
            mock.patch.object(
                retarget,
                "evaluated_semantic_bone",
                side_effect=lambda _armature, _entries, role: outputs[role],
            ),
        ):
            retarget._commit_ik_output([proposal])

        events = Solver.instances[0].events
        invalidation = events.index(("invalidate", pole.name))
        self.assertIn(("solve", pole.name), events[:invalidation])
        self.assertIn(("solve", pole.name), events[invalidation + 1 :])

    def test_each_processor_failure_propagates_to_transaction_boundary(self) -> None:
        config = self._config()
        for module_id, module in self.post.CORRECTION_REGISTRY.items():
            if module.scope != "FRAME":
                continue
            pose = SimpleNamespace(
                frame=SimpleNamespace(timestamp_ms=0),
                diagnostic_mode=False,
                applied_modules=[],
                warnings=[],
            )
            iterator = self.post.iter_pipeline(
                pose,
                config,
                module_filter={module_id},
            )
            stage, current, total = next(iterator)
            self.assertTrue(stage)
            self.assertEqual((current, total), (1, 1))
            with (
                mock.patch.object(
                    module,
                    "process_frame",
                    side_effect=RuntimeError(f"injected {module_id}"),
                ),
                self.assertRaisesRegex(RuntimeError, module_id),
            ):
                next(iterator)


class AnimationCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.addon = load_addon()
        cls.cleanup = cls.addon.animation_cleanup

    def test_isolated_scalar_and_vector_spikes_are_rejected(self) -> None:
        noisy = [0.0, 1.0, 2.0, 20.0, 4.0, 5.0, 6.0]
        times = [float(index) for index in range(len(noisy))]
        cleaned, _count = self.cleanup.reduce_scalar_outliers(noisy, times)
        cleaned = self.cleanup.smooth_scalars(cleaned, times, None, 1.0)
        self.assertLessEqual(abs(cleaned[3] - 3.0), abs(noisy[3] - 3.0) * 0.2)
        self.assertLess(abs(cleaned[0] - noisy[0]), 1.0)
        vectors = [Vector((value, value * 0.5, 0.0)) for value in noisy]
        cleaned_vectors, _count = self.cleanup.reduce_vector_outliers(vectors, times)
        cleaned_vectors = self.cleanup.smooth_vectors(cleaned_vectors, times, None, 1.0)
        self.assertLessEqual(
            abs(cleaned_vectors[3].x - 3.0),
            abs(vectors[3].x - 3.0) * 0.2,
        )

    def test_quaternion_cleanup_keeps_short_continuous_hemisphere(self) -> None:
        crossing = [
            Quaternion((0.0, 0.0, 1.0), math.radians(degrees))
            for degrees in (170.0, 179.0, -179.0, -170.0)
        ]
        times = [float(index) for index in range(len(crossing))]
        cleaned = self.cleanup.smooth_quaternions(crossing, times, None, 0.7)
        for first, second in zip(cleaned, cleaned[1:], strict=False):
            self.assertGreaterEqual(first.dot(second), 0.0)
            self.assertLess(
                first.rotation_difference(second).angle,
                math.radians(25.0),
            )

    def test_isolated_half_turn_is_removed(self) -> None:
        samples = [
            Quaternion(),
            Quaternion((0.0, 1.0, 0.0), math.radians(3.0)),
            Quaternion((0.0, 1.0, 0.0), math.radians(180.0)),
            Quaternion((0.0, 1.0, 0.0), math.radians(5.0)),
            Quaternion((0.0, 1.0, 0.0), math.radians(7.0)),
        ]
        times = [float(index) for index in range(len(samples))]
        cleaned, _count = self.cleanup.reduce_quaternion_outliers(samples, times)
        cleaned = self.cleanup.smooth_quaternions(cleaned, times, None, 1.0)
        self.assertLess(cleaned[2].angle, math.radians(20.0))

    def test_short_hand_rotation_burst_is_bridged(self) -> None:
        values = [
            Quaternion((0.0, 1.0, 0.0), math.radians(degrees))
            for degrees in (0.0, 4.0, 82.0, -70.0, 7.0, 10.0)
        ]
        times = [index / 24.0 for index in range(len(values))]
        cleaned, count = self.cleanup.reduce_short_quaternion_bursts(
            values,
            times,
            minimum_excursion=math.radians(30.0),
        )
        self.assertGreaterEqual(count, 2)
        self.assertLess(
            max(
                math.degrees(left.rotation_difference(right).angle)
                for left, right in zip(cleaned, cleaned[1:], strict=False)
            ),
            20.0,
        )

    def test_sustained_fast_hand_turn_is_preserved(self) -> None:
        values = [
            Quaternion((0.0, 1.0, 0.0), math.radians(degrees))
            for degrees in (0.0, 25.0, 55.0, 90.0, 125.0, 155.0)
        ]
        times = [index / 24.0 for index in range(len(values))]
        cleaned, count = self.cleanup.reduce_short_quaternion_bursts(
            values,
            times,
            minimum_excursion=math.radians(30.0),
        )
        self.assertEqual(count, 0)
        self.assertLess(
            cleaned[-1].rotation_difference(values[-1]).angle,
            1.0e-6,
        )

    def test_distal_finger_speed_is_bounded(self) -> None:
        values = [
            Quaternion((0.0, 1.0, 0.0), math.radians(degrees))
            for degrees in (0.0, 8.0, 130.0, 142.0)
        ]
        times = [index / 24.0 for index in range(len(values))]
        cleaned, count = self.cleanup.limit_quaternion_speed(
            values,
            times,
            math.radians(1200.0),
        )
        self.assertGreaterEqual(count, 1)
        self.assertLessEqual(
            max(
                math.degrees(left.rotation_difference(right).angle)
                for left, right in zip(cleaned, cleaned[1:], strict=False)
            ),
            50.0001,
        )

    def test_time_aware_smoothing_preserves_linear_motion(self) -> None:
        times = [0.0, 0.03, 0.08, 0.12, 0.20, 0.27]
        values = [2.0 * value + 1.0 for value in times]
        cleaned = self.cleanup.smooth_scalars(values, times, None, 1.0)
        for expected, actual in zip(values, cleaned, strict=True):
            self.assertAlmostEqual(expected, actual, places=5)

    def test_vector_outlier_uses_vector_residual(self) -> None:
        times = [index / 30.0 for index in range(7)]
        values = [Vector((float(index), 0.0, 0.0)) for index in range(7)]
        values[3] = Vector((3.0, 20.0, 0.0))
        cleaned, count = self.cleanup.reduce_vector_outliers(values, times)
        self.assertEqual(count, 1)
        self.assertLess(cleaned[3].y, 1.0)

    def test_default_smoothing_halves_stationary_jitter_without_phase_shift(
        self,
    ) -> None:
        times = [index / 30.0 for index in range(61)]
        values = [0.1 if index % 2 else -0.1 for index in range(61)]
        cleaned = self.cleanup.smooth_scalars(values, times, None, 0.5)
        before = math.sqrt(sum(value * value for value in values) / len(values))
        after = math.sqrt(sum(value * value for value in cleaned) / len(cleaned))
        self.assertLessEqual(after, before * 0.5)

        event = [0.0] * 21
        for offset, value in enumerate((0.2, 0.65, 1.0, 0.65, 0.2)):
            event[8 + offset] = value
        event_times = [index / 30.0 for index in range(len(event))]
        stabilized = self.cleanup.smooth_scalars(event, event_times, None, 0.5)
        self.assertEqual(stabilized.index(max(stabilized)), 10)
        self.assertGreaterEqual(max(stabilized), 0.9)

    def test_quaternion_smoothing_stays_normalized(self) -> None:
        times = [index / 30.0 for index in range(12)]
        values = [
            Quaternion((0.2, 0.8, 0.1), math.radians(index * 7.0))
            for index in range(12)
        ]
        cleaned = self.cleanup.smooth_quaternions(values, times, None, 0.7)
        for value in cleaned:
            self.assertAlmostEqual(value.magnitude, 1.0, places=6)

    def test_bounded_quaternion_smoothing_does_not_overshoot_axis_range(self) -> None:
        times = [index / 24.0 for index in range(7)]
        values = [
            Quaternion((0.0, 0.0, 1.0), math.radians(degrees))
            for degrees in (0.0, 5.0, 20.0, 45.0, 70.0, 85.0, 90.0)
        ]

        cleaned = self.cleanup.smooth_quaternions_bounded(
            values,
            times,
            None,
            1.0,
        )

        for value in cleaned:
            angle = math.degrees(value.to_euler("XYZ").z)
            self.assertGreaterEqual(angle, -1.0e-5)
            self.assertLessEqual(angle, 90.0 + 1.0e-5)
            self.assertAlmostEqual(value.magnitude, 1.0, places=6)

    def test_anatomical_finger_hinge_allows_only_small_hyperextension(self) -> None:
        rest = Vector((1.0, 0.0, 0.0))
        bend = Vector((0.0, 1.0, 0.0))
        backward = Vector(
            (
                math.cos(math.radians(70.0)),
                -math.sin(math.radians(70.0)),
                0.0,
            )
        )

        result, constrained = (
            self.addon.retarget._bounded_parent_relative_direction_delta(
                rest,
                backward,
                Quaternion(),
                math.radians(105.0),
                bend_reference=bend,
                flexion_bounds=(math.radians(-15.0), math.radians(105.0)),
            )
        )
        direction = result @ rest
        flexion = math.degrees(math.atan2(direction.y, direction.x))

        self.assertTrue(constrained)
        self.assertAlmostEqual(flexion, -15.0, places=4)

    def test_hand_stabilizer_holds_sustained_detector_burst(self) -> None:
        times = [index / 25.0 for index in range(24)]
        angles = [0.0] * 5 + [82.0] * 13 + [0.0] * 6
        values = [Quaternion((0.0, 0.0, 1.0), math.radians(angle)) for angle in angles]

        cleaned, held = self.cleanup.stabilize_quaternion_jumps(
            values,
            times,
            math.radians(600.0),
            reacquire_seconds=0.8,
        )

        self.assertEqual(held, 13)
        self.assertLess(
            max(value.rotation_difference(values[0]).angle for value in cleaned),
            math.radians(0.01),
        )

    def test_hand_stabilizer_reacquires_a_persistent_new_pose_boundedly(self) -> None:
        times = [index / 25.0 for index in range(32)]
        values = [Quaternion() for _index in range(3)] + [
            Quaternion((0.0, 0.0, 1.0), math.radians(90.0)) for _index in range(29)
        ]

        cleaned, held = self.cleanup.stabilize_quaternion_jumps(
            values,
            times,
            math.radians(600.0),
            reacquire_seconds=0.8,
        )

        self.assertGreaterEqual(held, 20)
        self.assertLess(cleaned[20].rotation_difference(values[0]).angle, 1.0e-6)
        self.assertLess(
            cleaned[-1].rotation_difference(values[-1]).angle,
            math.radians(0.01),
        )
        self.assertLessEqual(
            max(
                cleaned[index - 1].rotation_difference(cleaned[index]).angle
                for index in range(1, len(cleaned))
            ),
            math.radians(24.0) + 1.0e-6,
        )

    def test_weak_wrist_interval_holds_then_eases_without_default_pose(self) -> None:
        times = [index * 0.1 for index in range(8)]
        values = [
            Quaternion((0.0, 1.0, 0.0), math.radians(degrees))
            for degrees in (10.0, 100.0, -80.0, 120.0, -110.0, 90.0, -70.0, 40.0)
        ]
        confidences = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        bridged = self.cleanup.bridge_low_confidence_quaternions(
            values, times, confidences
        )
        self.assertLess(
            bridged[3].rotation_difference(values[0]).angle,
            math.radians(0.1),
        )
        self.assertGreater(math.degrees(bridged[6].angle), 12.0)
        self.assertLess(math.degrees(bridged[6].angle), 40.0)
        for first, second in zip(bridged, bridged[1:], strict=False):
            self.assertGreaterEqual(first.dot(second), 0.0)

    def test_weak_ik_location_interval_uses_neighboring_motion(self) -> None:
        times = [index / 30.0 for index in range(7)]
        values = [Vector((float(index), 0.0, 0.0)) for index in range(7)]
        values[2].y = 20.0
        values[3].y = -20.0
        values[4].y = 15.0
        confidences = [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
        bridged = self.cleanup.bridge_low_confidence_vectors(values, times, confidences)
        for index in (2, 3, 4):
            self.assertAlmostEqual(bridged[index].y, 0.0, places=6)

    def test_finger_dropout_up_to_half_second_is_interpolated(self) -> None:
        retarget = load_addon().retarget
        timestamps = [index / 30.0 for index in range(8)]
        samples = [
            retarget._BakedFrameSample(
                destination=float(index),
                warning_sink=[],
                timestamp_seconds=timestamp,
            )
            for index, timestamp in enumerate(timestamps)
        ]
        bone = object()
        for index, degrees in ((0, 0.0), (7, 35.0)):
            samples[index].bones["finger"] = retarget._BakedBoneSample(
                bone=bone,
                location=Vector(),
                rotation=Quaternion((0.0, 1.0, 0.0), math.radians(degrees)),
                scale=Vector((1.0, 1.0, 1.0)),
                key_rotation=True,
                semantic_id="INDEX_01_L",
                confidence=0.9,
            )

        retarget._fill_short_bone_gaps(samples, timestamps)

        self.assertTrue(all("finger" in sample.bones for sample in samples))
        self.assertAlmostEqual(
            math.degrees(samples[4].bones["finger"].rotation.angle),
            20.0,
            places=4,
        )

    def test_timestamp_gaps_and_missing_channels_split_runs(self) -> None:
        retarget = load_addon().retarget
        timestamps = [0.0, 1.0 / 30.0, 2.0 / 30.0, 1.0, 1.0 + 1.0 / 30.0]
        self.assertEqual(
            retarget._continuous_runs(list(range(5)), timestamps),
            [[0, 1, 2], [3, 4]],
        )
        self.assertEqual(
            retarget._continuous_runs([0, 1, 3, 4], timestamps),
            [[0, 1], [3, 4]],
        )


if __name__ == "__main__":
    unittest.main()
