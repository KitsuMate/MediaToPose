"""LLM2Vec text encoder backend: pins, licence, memory check, cache and session.

The tokenizer and ONNX Runtime are fakes; nothing is downloaded.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from media_to_pose.backends.llm2vec import backend, runtime
from media_to_pose.core.cache.embeddings import EmbeddingCache
from media_to_pose.core.errors import (
    DependencyMissing,
    InsufficientResources,
    LicenseRequired,
    ModelMissing,
)
from media_to_pose.core.motion.kimodo import prompt as prompt_layout


def fake_tokenize(text: str, add_special_tokens: bool) -> list[int]:
    ids = [1000 + len(word) for word in text.split()]
    return [128000, *ids] if add_special_tokens else ids


class FakeSession:
    def __init__(self):
        self.feeds = None

    def run(self, names, feeds):
        self.feeds = feeds
        value = float(feeds["pooling_mask"].sum())
        return [np.full((1, 4096), value, dtype=np.float32)]


class FakeOrt:
    def __init__(self):
        self.created: list[tuple[str, list[str]]] = []
        self.session = FakeSession()

    def SessionOptions(self):  # noqa: N802 - ONNX Runtime's name
        return SimpleNamespace()

    def InferenceSession(self, path, *, sess_options, providers):  # noqa: N802
        self.created.append((path, providers))
        return self.session


def make_bundle(root: Path) -> Path:
    files = []
    for relative in (runtime.GRAPH, runtime.GRAPH_DATA, runtime.TOKENIZER):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(relative.encode()).hexdigest(),
            }
        )
    manifest = {"schema_version": 1, "revision": backend.REVISION, "files": files}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    return root


class TemporaryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)


class PinTests(unittest.TestCase):
    def test_files_are_pinned_with_their_licences(self) -> None:
        self.assertRegex(backend.REVISION, re.compile(r"[0-9a-f]{40}"))
        paths = {item.path for item in backend.bundle_files()}
        self.assertTrue(set(backend.LICENSE.files) <= paths)
        self.assertTrue({runtime.GRAPH, runtime.GRAPH_DATA, runtime.TOKENIZER} <= paths)
        for item in backend.bundle_files():
            self.assertRegex(item.sha256, re.compile(r"[0-9a-f]{64}"))
        self.assertGreater(backend.DOWNLOAD_BYTES, 5_800_000_000)
        self.assertIn("Built with Meta Llama 3", backend.LICENSE.summary)

    def test_missing_tokenizers_package_is_a_dependency_error(self) -> None:
        dependencies = backend.runtime.dependencies
        with (
            mock.patch.object(dependencies, "_module_is_available", return_value=False),
            self.assertRaises(DependencyMissing),
        ):
            dependencies.import_tokenizers()


class EncoderTests(TemporaryTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.bundle = runtime.load_bundle(make_bundle(self.root))
        self.ort = FakeOrt()
        patcher = mock.patch.object(
            runtime.resources, "available_ram_bytes", return_value=32 * 1024**3
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _encoder(self, **kwargs) -> runtime.LLM2VecEncoder:
        return runtime.LLM2VecEncoder(
            self.bundle, ort_module=self.ort, tokenize=fake_tokenize, **kwargs
        )

    def test_encode_runs_on_cpu_and_releases_the_session(self) -> None:
        encoder = self._encoder()
        vector = encoder.encode("a person waves")
        self.assertEqual((vector.shape, vector.dtype), ((4096,), np.float32))
        self.assertFalse(encoder.loaded)
        path, providers = self.ort.created[0]
        self.assertTrue(path.endswith("model_int4_b128_fp32act_s64.onnx"))
        self.assertEqual(providers, ["CPUExecutionProvider"])
        feeds = self.ort.session.feeds
        for name in ("input_ids", "attention_mask", "pooling_mask"):
            self.assertEqual(feeds[name].shape, (1, prompt_layout.MAX_TOKENS))
            self.assertEqual(feeds[name].dtype, np.int64)
        expected = len(fake_tokenize("A person waves." + "<|eot_id|>", False))
        self.assertEqual(float(vector[0]), expected)

    def test_keep_loaded_holds_the_session(self) -> None:
        encoder = self._encoder(keep_loaded=True)
        encoder.encode("one")
        encoder.encode("two")
        self.assertTrue(encoder.loaded)
        self.assertEqual(len(self.ort.created), 1)
        encoder.close()
        self.assertFalse(encoder.loaded)

    def test_memory_shortfall_is_raised_before_loading(self) -> None:
        encoder = self._encoder()
        with (
            mock.patch.object(
                runtime.resources, "available_ram_bytes", return_value=2 * 1024**3
            ),
            self.assertRaises(InsufficientResources) as caught,
        ):
            encoder.encode("a person waves")
        self.assertEqual(caught.exception.details["needed_bytes"], 6 * 1024**3)
        self.assertEqual(self.ort.created, [])


class BackendTests(TemporaryTestCase):
    def setUp(self) -> None:
        super().setUp()
        for name, value in (
            ("configured_storage_parent", self.root),
            ("_config_path", self.root / "config" / "storage.json"),
        ):
            patcher = mock.patch.object(backend, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(backend.shutdown)
        self.cache = EmbeddingCache(self.root / "embeddings")

    def test_licence_is_required_before_installing(self) -> None:
        with self.assertRaises(LicenseRequired) as caught:
            backend.start_install(online_access=True)
        self.assertEqual(
            caught.exception.details["accept_license"],
            f"meta-llama-3-community-license@{backend.REVISION}",
        )
        self.assertEqual(backend.backend_snapshot().license_state, "REQUIRED")

    def test_accepted_install_downloads_the_code_listed_bundle(self) -> None:
        with mock.patch.object(
            backend.lifecycle, "install_pinned_bundle", return_value="done"
        ) as install:
            job = backend.start_install(
                online_access=True,
                accept_license=backend.LICENSE.accept_value(backend.REVISION),
            )
            assert job.thread is not None
            job.thread.join(timeout=5)
        bundle = install.call_args.args[1]
        self.assertEqual(bundle.manifest_path, "")
        self.assertEqual(bundle.list_files(self.root), backend.bundle_files())
        self.assertEqual(backend.backend_snapshot().license_state, "ACCEPTED")

    def test_empty_prompt_needs_no_encoder(self) -> None:
        with mock.patch.object(backend, "ensure_encoder") as ensure:
            self.assertIsNone(backend.encode_prompt("  ", cache=self.cache))
        ensure.assert_not_called()

    def test_missing_encoder_is_reported(self) -> None:
        with self.assertRaises(ModelMissing):
            backend.encode_prompt("A person waves", cache=self.cache)

    def test_cached_prompt_never_loads_the_encoder(self) -> None:
        encoder = mock.Mock()
        encoder.encode_prompt.return_value = runtime.Encoding(
            np.ones(4096, dtype=np.float32), "A person waves.", False
        )
        with mock.patch.object(backend, "ensure_encoder", return_value=encoder):
            first = backend.encode_prompt("a person waves", cache=self.cache)
        encoder.encode_prompt.assert_called_once_with("A person waves.")
        with mock.patch.object(backend, "ensure_encoder") as ensure:
            again = backend.encode_prompt("  a person   waves ", cache=self.cache)
        ensure.assert_not_called()
        assert first is not None and again is not None
        np.testing.assert_array_equal(first.vector, again.vector)
        key = backend.embedding_key("A person waves.")
        self.assertEqual(key.encoder_revision, backend.REVISION)
        self.assertEqual(key.template_version, prompt_layout.TEMPLATE_VERSION)

    def test_installed_encoder_reports_ready_on_cpu(self) -> None:
        backend.STORAGE.ensure(self.root / "media_to_pose_llm2vec")
        make_bundle(self.root / "media_to_pose_llm2vec" / "active")
        snapshot = backend.backend_snapshot()
        self.assertEqual((snapshot.state, snapshot.actual_device), ("READY", "cpu"))
        self.assertEqual(snapshot.min_ram_bytes, 6 * 1024**3)
        self.assertEqual(snapshot.attribution, "Built with Meta Llama 3")


if __name__ == "__main__":
    unittest.main()
