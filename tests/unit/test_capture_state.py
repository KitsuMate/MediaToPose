"""Side-effect-free capture readiness rules."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from media_to_pose.session import readiness as capture_state

MEDIAPIPE = SimpleNamespace(
    label="MediaPipe", capabilities=frozenset({"BODY", "HANDS", "FACE"})
)
SAM = SimpleNamespace(label="SAM 3D Body", capabilities=frozenset({"BODY", "HANDS"}))


def _settings(**overrides):
    values = {
        "provider": "MEDIAPIPE",
        "use_body": True,
        "use_hands": False,
        "use_face": False,
        "face_output": "ARKIT",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _entry(section: str, label: str, **overrides):
    values = {
        "section": section,
        "label": label,
        "required": True,
        "status": "VALID",
        "enabled": True,
        "target_name": label,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _armature(mappings, *, arkit=1.0, facs=1.0, face=1.0):
    return SimpleNamespace(
        mtp_rig_config=SimpleNamespace(
            mappings=mappings,
            arkit_complete=arkit,
            facs_complete=facs,
            face_complete=face,
        )
    )


class CaptureScopeTests(unittest.TestCase):
    def test_scope_needs_a_part_the_provider_supports(self) -> None:
        self.assertTrue(
            capture_state.enabled_capture_scope(_settings(), SAM.capabilities)
        )
        face_only = _settings(use_body=False, use_face=True)
        self.assertFalse(
            capture_state.enabled_capture_scope(face_only, SAM.capabilities)
        )
        self.assertTrue(
            capture_state.enabled_capture_scope(face_only, MEDIAPIPE.capabilities)
        )


@mock.patch.object(capture_state, "_fk_switch_issues", return_value=[])
@mock.patch.object(capture_state, "_descriptor", return_value=MEDIAPIPE)
class MappingIssueTests(unittest.TestCase):
    def test_uninitialized_rig(self, _descriptor, _fk) -> None:
        issues = capture_state.mapping_scope_issues(_armature([]), _settings())
        self.assertEqual(issues, ("Rig setup has not been initialized",))

    def test_missing_required_body_mappings_are_listed(self, _descriptor, _fk):
        mappings = [
            _entry("BODY", "Hips"),
            _entry("BODY", "Head", status="MISSING"),
            _entry("BODY", "Optional", required=False, status="MISSING"),
        ]
        issues = capture_state.mapping_scope_issues(_armature(mappings), _settings())
        self.assertEqual(issues, ("Missing required body mappings: Head",))

    def test_hands_need_at_least_one_finger_mapping(self, _descriptor, _fk):
        mappings = [_entry("BODY", "Hips"), _entry("HANDS", "Thumb", target_name="")]
        issues = capture_state.mapping_scope_issues(
            _armature(mappings), _settings(use_body=False, use_hands=True)
        )
        self.assertIn("No finger mappings are available", issues)

    def test_face_only_capture_needs_some_face_output(self, _descriptor, _fk):
        settings = _settings(use_body=False, use_face=True, face_output="FACS")
        issues = capture_state.mapping_scope_issues(
            _armature([_entry("BODY", "Hips")], facs=0.0), settings
        )
        self.assertEqual(issues, ("The selected FACS face output is fully missing",))

    def test_fk_switch_issues_are_appended(self, _descriptor, fk) -> None:
        fk.return_value = ["Arm IK switch is not FK"]
        issues = capture_state.mapping_scope_issues(
            _armature([_entry("BODY", "Hips")]), _settings()
        )
        self.assertEqual(issues, ("Arm IK switch is not FK",))
        fk.assert_called_once()


@mock.patch.object(capture_state, "_descriptor", return_value=MEDIAPIPE)
class FaceWarningTests(unittest.TestCase):
    def test_no_warning_without_face_capture(self, _descriptor) -> None:
        self.assertEqual(
            capture_state.face_output_warning(_armature([]), _settings()), ""
        )

    def test_partial_and_missing_outputs_warn(self, _descriptor) -> None:
        settings = _settings(use_face=True)
        self.assertEqual(
            capture_state.face_output_warning(_armature([], arkit=0.5), settings),
            "ARKit morphs are partially mapped (50%)",
        )
        self.assertEqual(
            capture_state.face_output_warning(_armature([], arkit=0.0), settings),
            "ARKit morphs are fully missing; face output will be skipped",
        )

    def test_provider_without_face_never_warns(self, descriptor) -> None:
        descriptor.return_value = SAM
        settings = _settings(use_face=True)
        self.assertEqual(
            capture_state.face_output_warning(_armature([], arkit=0.0), settings), ""
        )


if __name__ == "__main__":
    unittest.main()
