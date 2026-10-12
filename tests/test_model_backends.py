"""Model backend operators and snapshots without installed models.

These tests never install or uninstall: uninstall would remove model folders
from the Blender profile the suite runs in.
"""

from __future__ import annotations

import unittest

import bpy

from tests.blender_test_support import register_addon


class ModelBackendOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.addon = register_addon()

    def test_every_backend_operator_is_registered(self) -> None:
        for name in (
            "anycalib_install",
            "anycalib_cancel_install",
            "anycalib_load",
            "anycalib_unload",
            "anycalib_uninstall",
            "rohm_install",
            "rohm_cancel_install",
            "rohm_load",
            "rohm_unload",
            "rohm_verify",
            "rohm_uninstall",
            "sam3d_setup",
            "sam3d_install",
            "sam3d_cancel_install",
            "sam3d_copy_install_diagnostics",
            "sam3d_open_install_log",
            "sam3d_dismiss_install_result",
            "sam3d_load",
            "sam3d_unload",
            "sam3d_uninstall",
        ):
            with self.subTest(operator=name):
                self.assertTrue(hasattr(bpy.ops.mtp, name))
                bpy.ops.mtp.__getattr__(name).get_rna_type()

    def test_cancel_without_a_running_install_is_a_no_op(self) -> None:
        for name in (
            "anycalib_cancel_install",
            "rohm_cancel_install",
            "sam3d_cancel_install",
        ):
            with self.subTest(operator=name):
                self.assertEqual(getattr(bpy.ops.mtp, name)(), {"CANCELLED"})

    def test_unload_without_a_loaded_model_succeeds_and_frees_the_lease(self) -> None:
        coordinator = self.addon.operation.COORDINATOR
        for name in ("anycalib_unload", "rohm_unload", "sam3d_unload"):
            with self.subTest(operator=name):
                self.assertEqual(getattr(bpy.ops.mtp, name)(), {"FINISHED"})
                self.assertFalse(coordinator.is_busy())

    def test_backend_snapshots_describe_missing_models(self) -> None:
        anycalib = self.addon.backends.anycalib.backend.backend_snapshot("CPU")
        rohm = self.addon.backends.rohm.backend.backend_snapshot("CPU")
        sam = self.addon.backends.sam3d.backend.backend_snapshot("VITH", "CPU")
        for snapshot in (anycalib, rohm):
            if not snapshot.installed:
                self.assertIn(snapshot.state, {"NOT_INSTALLED", "ERROR"})
        if not sam.installed:
            self.assertTrue(sam.message)
        for module in (
            self.addon.backends.anycalib.backend,
            self.addon.backends.rohm.backend,
            self.addon.backends.sam3d.backend,
        ):
            self.assertFalse(module.install_status_snapshot().running)

    def test_one_downloader_and_one_job_class_serve_every_backend(self) -> None:
        lifecycle = self.addon.backends.lifecycle
        for module in (
            self.addon.backends.anycalib.backend,
            self.addon.backends.rohm.backend,
            self.addon.backends.sam3d.backend,
        ):
            with self.subTest(module=module.__name__):
                self.assertIsInstance(module.BACKEND, lifecycle.ModelBackend)
                self.assertFalse(hasattr(module, "_Job"))
        self.assertIs(
            self.addon.backends.sam3d.backend.BackgroundJob, lifecycle.BackgroundJob
        )


if __name__ == "__main__":
    unittest.main()
