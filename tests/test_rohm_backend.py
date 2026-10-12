from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests.blender_test_support import PACKAGE_NAME, load_addon


class RohmBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        load_addon()
        cls.backend = importlib.import_module(f"{PACKAGE_NAME}.rohm_backend")

    def tearDown(self) -> None:
        self.backend.unload_model()

    def test_model_bundle_is_pinned_to_an_immutable_kitsumate_revision(
        self,
    ) -> None:
        specification = self.backend.MODEL_SPEC
        self.assertEqual(specification["repo"], "KitsuMate/rohm-onnx")
        self.assertTrue(self.backend.is_published())
        self.assertRegex(specification["revision"], r"^[0-9a-f]{40}$")
        self.assertEqual(
            specification["revision"],
            "21bb4c53d1265c522e3962a466c4bb9e41169a36",
        )
        self.assertEqual(
            specification["upstream_revision"],
            "57ba22cde1595285276ddb88c8c82b25bf0d9722",
        )

    def test_download_url_is_immutable_and_rejects_traversal(self) -> None:
        url = self.backend._hf_url(self.backend.MODEL_SPEC["manifest_path"])
        self.assertIn(self.backend.MODEL_SPEC["revision"], url)
        self.assertTrue(url.endswith("/bundles/fp32/manifest.json"))
        with self.assertRaisesRegex(RuntimeError, "Unsafe RoHM download path"):
            self.backend._hf_url("../checkpoint.pt")

    def test_default_storage_parent_matches_sam3(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            expected = temporary_path / "shared-models"
            with (
                mock.patch.object(
                    self.backend,
                    "_config_path",
                    side_effect=lambda **_kwargs: temporary_path
                    / "missing-storage.json",
                ),
                mock.patch.object(
                    self.backend.sam3d_backend,
                    "preferred_model_storage_parent",
                    return_value=expected,
                ),
            ):
                self.assertEqual(self.backend.configured_storage_parent(), expected)

    def test_model_device_choices_are_stable(self) -> None:
        self.assertEqual(
            [item[0] for item in self.backend.rohm_device_enum_items()],
            ["AUTO", "GPU", "CPU"],
        )

    def test_loaded_snapshot_reports_provider_precision_and_sizes(self) -> None:
        diagnostic = SimpleNamespace(
            requested_mode="GPU",
            actual_ep="WebGpuExecutionProvider",
            actual_device="WEBGPU",
            fallback=False,
            precision="fp32",
            download_bytes=222_030_103,
            installed_bytes=222_030_103,
            upstream_revision="57ba22cde1595285276ddb88c8c82b25bf0d9722",
        )
        session = SimpleNamespace(diagnostics=diagnostic)
        self.backend.BACKEND.set_session(session)
        snapshot = self.backend.backend_snapshot("AUTO")
        self.assertEqual(snapshot.actual_ep, "WebGpuExecutionProvider")
        self.assertEqual(snapshot.precision, "fp32")
        self.assertEqual(snapshot.download_bytes, 222_030_103)
        self.assertEqual(snapshot.installed_bytes, 222_030_103)


if __name__ == "__main__":
    unittest.main()
