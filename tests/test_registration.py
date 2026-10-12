from __future__ import annotations

import unittest

import bpy
from _bpy_restrict_state import RestrictBlend

from tests.blender_test_support import load_addon


class RegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.addon = load_addon()

    def test_repeated_registration_and_cleanup(self) -> None:
        self.addon.register()
        self.addon.register()
        self.assertTrue(hasattr(bpy.types.Scene, "mtp_settings"))
        callbacks = [
            callback
            for callback in bpy.app.handlers.load_post
            if callback is self.addon.addon_lifecycle._initialize_scene_state
        ]
        self.assertEqual(len(callbacks), 1)
        self.assertIs(
            bpy.app.driver_namespace[self.addon.addon_lifecycle._STATE_TIMER_KEY],
            self.addon.addon_lifecycle._state_watch_timer,
        )

    def test_restricted_install_context_defers_scene_access(self) -> None:
        self.addon.unregister()
        with RestrictBlend():
            self.addon.register()
        self.assertTrue(self.addon.addon_lifecycle._scene_state_pending)
        self.assertTrue(self.addon.addon_lifecycle._reset_scene_state(bpy.data))

    def test_capture_and_cleanup_use_safe_defaults(self) -> None:
        scene_properties = self.addon.properties.MTPSceneSettings.bl_rna.properties
        module_properties = (
            self.addon.properties.MTPPostProcessingModuleConfig.bl_rna.properties
        )
        self.assertEqual(scene_properties["root_motion_mode"].default, "AUTOMATIC")
        self.assertFalse(scene_properties["keep_original_root_position"].default)
        self.assertTrue(module_properties["reject_unreliable_limbs"].default)
        self.assertTrue(module_properties["reject_unusable_frames"].default)
        self.assertTrue(module_properties["smooth_motion"].default)
        self.assertTrue(module_properties["ground_contacts"].default)
        self.assertTrue(module_properties["stabilize_hands"].default)
        self.assertFalse(module_properties["learned_pose_residual"].default)
        self.assertTrue(scene_properties["use_body"].default)
        self.assertFalse(scene_properties["use_hands"].default)
        self.assertFalse(scene_properties["use_face"].default)
        self.assertFalse(
            self.addon.post_processing.POST_PROCESSING_DEFAULT_ENABLED[
                "ROHM_CORRECTION"
            ]
        )

    def test_rohm_uses_managed_model_lifecycle_without_bypass_flags(self) -> None:
        scene_properties = self.addon.properties.MTPSceneSettings.bl_rna.properties
        module_properties = (
            self.addon.properties.MTPPostProcessingModuleConfig.bl_rna.properties
        )
        self.assertIn("rohm_device", scene_properties)
        self.assertNotIn("confidence_threshold", scene_properties)
        self.assertNotIn("missing_detection_policy", scene_properties)
        self.assertNotIn("correct_pose", module_properties)
        self.assertNotIn("correct_root", module_properties)
        self.assertNotIn("application_policy", module_properties)
        self.assertNotIn("rohm_local_bundle_path", scene_properties)
        self.assertNotIn("rohm_allow_unverified", scene_properties)


if __name__ == "__main__":
    unittest.main()
