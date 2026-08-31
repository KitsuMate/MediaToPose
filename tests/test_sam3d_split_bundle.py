from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tests.blender_test_support import load_addon


class SamSplitBundleContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = load_addon().sam3d_ort_runtime

    @staticmethod
    def _record(path: Path) -> dict[str, object]:
        return {
            "path": path.relative_to(path.parents[1]).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }

    def _split_bundle(self, directory: str) -> Path:
        root = Path(directory)
        graph_root = root / "graphs"
        artifact_root = root / "mhr"
        graph_root.mkdir()
        artifact_root.mkdir()
        backbone = graph_root / "backbone.onnx"
        decoder = graph_root / "decoder_body_layer0.onnx"
        lbs = artifact_root / "body_model.lbs"
        backbone.write_bytes(b"backbone")
        decoder.write_bytes(b"decoder")
        lbs.write_bytes(b"lbs")
        graphs = [
            {
                **self._record(backbone),
                "role": "backbone",
                "inputs": {"image": "image_tensor"},
                "outputs": {"features": "feature_tensor"},
            },
            {
                **self._record(decoder),
                "role": "decoder.body.layer.0",
                "inputs": {
                    "tokens": "tokens",
                    "image_tokens": "image_tokens",
                    "token_pe": "token_pe",
                    "image_pe": "image_pe",
                },
                "outputs": {"tokens_out": "tokens_out"},
            },
        ]
        manifest = {
            "schema_version": self.runtime.BUNDLE_SCHEMA_VERSION,
            "contract": self.runtime.BUNDLE_CONTRACT,
            "runner": self.runtime.SPLIT_NUMPY_RUNNER,
            "variant": "VITH",
            "model_revision": "a" * 40,
            "precision": "mixed-fp16-fp32",
            "graphs": graphs,
            "artifacts": [self._record(lbs)],
            "neutral_rotations": {
                str(index): [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]
                for index in sorted(self.runtime.REQUIRED_NEUTRAL_ROTATION_INDICES)
            },
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf8")
        return path

    def test_split_bundle_loads_every_graph_and_artifact_by_role(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._split_bundle(directory)
            bundle = self.runtime.load_bundle_manifest(
                path,
                expected_variant="VITH",
                expected_model_revision="a" * 40,
            )
            self.assertEqual(bundle.runner, self.runtime.SPLIT_NUMPY_RUNNER)
            self.assertEqual(
                [graph.role for graph in bundle.graphs],
                ["backbone", "decoder.body.layer.0"],
            )
            self.assertEqual(
                bundle.graph_for_role("decoder.body.layer.0").outputs,
                {"tokens_out": "tokens_out"},
            )
            self.assertEqual(
                bundle.graph_path_for_role("backbone"),
                Path(directory).resolve() / "graphs" / "backbone.onnx",
            )
            self.assertEqual(
                [item.path for item in bundle.files],
                [
                    "graphs/backbone.onnx",
                    "graphs/decoder_body_layer0.onnx",
                    "mhr/body_model.lbs",
                ],
            )
            with self.assertRaisesRegex(
                self.runtime.BundleContractError, "no graph with role"
            ):
                bundle.graph_for_role("decoder.hand.layer.0")

    def test_published_legacy_contract_name_remains_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._split_bundle(directory)
            manifest = json.loads(path.read_text(encoding="utf8"))
            manifest["contract"] = "vision-to-pose.sam3d.raw.v1"
            path.write_text(json.dumps(manifest), encoding="utf8")

            bundle = self.runtime.load_bundle_manifest(path)

            self.assertEqual(bundle.variant, "VITH")

    def test_split_bundle_requires_multiple_uniquely_named_graphs(self) -> None:
        invalid_roles = ("backbone", "BACKBONE", "../backbone", "backbone..body")
        for role in invalid_roles:
            with (
                self.subTest(role=role),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = self._split_bundle(directory)
                manifest = json.loads(path.read_text(encoding="utf8"))
                manifest["graphs"][1]["role"] = role
                if role in {"backbone", "BACKBONE"}:
                    expected = "duplicate graph roles"
                else:
                    expected = "dotted identifier"
                path.write_text(json.dumps(manifest), encoding="utf8")
                with self.assertRaisesRegex(self.runtime.BundleContractError, expected):
                    self.runtime.load_bundle_manifest(path, verify_files=False)

        with tempfile.TemporaryDirectory() as directory:
            path = self._split_bundle(directory)
            manifest = json.loads(path.read_text(encoding="utf8"))
            manifest["graphs"] = manifest["graphs"][:1]
            path.write_text(json.dumps(manifest), encoding="utf8")
            with self.assertRaisesRegex(
                self.runtime.BundleContractError, "must declare multiple graphs"
            ):
                self.runtime.load_bundle_manifest(path, verify_files=False)

    def test_split_bundle_rejects_ambiguous_or_unsafe_tensor_mappings(self) -> None:
        invalid_inputs = (
            {},
            {"../image": "image"},
            {"image": 4},
            {"image": " image"},
            {"image": "same", "mask": "same"},
            {"image": "image\nname"},
        )
        for inputs in invalid_inputs:
            with (
                self.subTest(inputs=inputs),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = self._split_bundle(directory)
                manifest = json.loads(path.read_text(encoding="utf8"))
                manifest["graphs"][0]["inputs"] = inputs
                path.write_text(json.dumps(manifest), encoding="utf8")
                with self.assertRaises(self.runtime.BundleContractError):
                    self.runtime.load_bundle_manifest(path, verify_files=False)

    def test_split_bundle_rejects_reused_paths_and_non_onnx_graphs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._split_bundle(directory)
            manifest = json.loads(path.read_text(encoding="utf8"))
            manifest["artifacts"][0] = {
                key: manifest["graphs"][0][key] for key in ("path", "sha256", "bytes")
            }
            path.write_text(json.dumps(manifest), encoding="utf8")
            with self.assertRaisesRegex(
                self.runtime.BundleContractError, "duplicate file paths"
            ):
                self.runtime.load_bundle_manifest(path, verify_files=False)

        with tempfile.TemporaryDirectory() as directory:
            path = self._split_bundle(directory)
            manifest = json.loads(path.read_text(encoding="utf8"))
            manifest["graphs"][0]["path"] = "graphs/backbone.bin"
            path.write_text(json.dumps(manifest), encoding="utf8")
            with self.assertRaisesRegex(self.runtime.BundleContractError, "ONNX file"):
                self.runtime.load_bundle_manifest(path, verify_files=False)

    def test_split_bundle_rejects_duplicate_json_mapping_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._split_bundle(directory)
            text = path.read_text(encoding="utf8")
            text = text.replace(
                '"inputs": {"image": "image_tensor"}',
                '"inputs": {"image": "image_tensor", "image": "shadow"}',
                1,
            )
            path.write_text(text, encoding="utf8")
            with self.assertRaisesRegex(
                self.runtime.BundleContractError, "repeats object key 'image'"
            ):
                self.runtime.load_bundle_manifest(path, verify_files=False)


if __name__ == "__main__":
    unittest.main()
