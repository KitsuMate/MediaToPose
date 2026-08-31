from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.blender_test_support import load_addon


class AnyCalibBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addon = load_addon()
        cls.runtime = cls.addon.anycalib_ort_runtime

    def _bundle(self, directory: Path, *, graph_path="model.onnx"):
        graph = directory / "model.onnx"
        graph.write_bytes(b"onnx-placeholder")
        manifest = {
            "contract": self.runtime.BUNDLE_CONTRACT,
            "runner": self.runtime.RUNNER,
            "precision": "fp32",
            "upstream_revision": self.runtime.UPSTREAM_REVISION,
            "checkpoint_revision": self.runtime.CHECKPOINT_REVISION,
            "graph": {
                "path": graph_path,
                "bytes": graph.stat().st_size,
                "sha256": hashlib.sha256(graph.read_bytes()).hexdigest(),
            },
        }
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")

    def test_manifest_hash_and_revision_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._bundle(root)
            bundle = self.runtime.load_bundle(root)
            self.assertEqual(bundle.precision, "fp32")
            (root / "model.onnx").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "byte size|SHA-256"):
                self.runtime.load_bundle(root)

    def test_manifest_rejects_escaping_graph_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._bundle(root, graph_path="../model.onnx")
            with self.assertRaisesRegex(RuntimeError, "Unsafe"):
                self.runtime.load_bundle(root, verify_files=False)

    def test_manifest_can_be_parsed_before_graph_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._bundle(root)
            (root / "model.onnx").unlink()

            bundle = self.runtime.load_bundle_manifest(root)

            self.assertEqual(bundle.graph.path, "model.onnx")
            with self.assertRaisesRegex(RuntimeError, "graph is missing"):
                self.runtime.load_bundle(root, verify_files=False)

    def test_published_legacy_contract_name_remains_compatible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._bundle(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf8"))
            manifest["contract"] = "vision_to_pose.anycalib_onnx.v1"
            manifest_path.write_text(json.dumps(manifest), encoding="utf8")

            bundle = self.runtime.load_bundle_manifest(root)

            self.assertEqual(bundle.precision, "fp32")

    def test_target_shapes_cover_portrait_landscape_and_square(self):
        for height, width in ((1000, 500), (500, 1000), (800, 800)):
            target = self.runtime.target_size(height, width)
            self.assertEqual(target[0] % 14, 0)
            self.assertEqual(target[1] % 14, 0)
            self.assertGreater(target[0], 0)
            self.assertGreater(target[1], 0)

    def test_public_bundle_is_pinned_to_an_immutable_kitsumate_commit(self):
        spec = self.addon.anycalib_backend.MODEL_SPEC
        self.assertEqual(spec["repo"], "KitsuMate/anycalib-onnx")
        self.assertTrue(self.addon.anycalib_backend.is_published())
        self.assertNotEqual(spec["revision"], "main")
        for value in spec["variants"].values():
            self.assertGreater(value["download_bytes"], 0)
            self.assertNotIn("pinhole", value["manifest_path"].casefold())

    def test_numpy_preprocessing_is_finite_and_exact_target_shape(self):
        image = np.arange(96 * 64 * 3, dtype=np.uint8).reshape(96, 64, 3)
        tensor, scale, shift = self.runtime.prepare_image(image, object(), np)
        target = self.runtime.target_size(96, 64)
        self.assertEqual(tensor.shape, (1, 3, *target))
        self.assertTrue(np.isfinite(tensor).all())
        self.assertGreater(scale[0], 0.0)
        self.assertGreater(scale[1], 0.0)
        self.assertEqual(len(shift), 2)


if __name__ == "__main__":
    unittest.main()
