"""Kimodo denoiser backend: pinned files, variants, licence, install and session.

ONNX Runtime and downloads are replaced by fakes; no model is fetched.
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

from media_to_pose.backends import licenses, lifecycle
from media_to_pose.backends.kimodo import backend, runtime
from media_to_pose.core.errors import (
    InsufficientResources,
    LicenseRequired,
    OnlineAccessDisabled,
)
from media_to_pose.core.motion.kimodo import rep
from media_to_pose.core.motion.kimodo.sampler import DenoiseInputs

HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


def _write(root: Path, relative: str, data: bytes) -> lifecycle.BundleFile:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return lifecycle.BundleFile(relative, len(data), hashlib.sha256(data).hexdigest())


def _npy(array: np.ndarray) -> bytes:
    with tempfile.TemporaryFile() as handle:
        np.save(handle, array)
        handle.seek(0)
        return handle.read()


def make_bundle(root: Path, variant: str = "fp16") -> Path:
    """A bundle folder shaped like an install, with a fake graph."""
    files = [
        _write(root, f"onnx/model_{variant}.onnx", b"graph"),
        _write(root, "LICENSE", b"licence"),
        _write(root, "NOTICE", b"notice"),
    ]
    sizes = {"global_root": 5, "body": 364}
    for name, relative in runtime.STATS_FILES.items():
        part = "global_root" if name.startswith("global_root") else "body"
        value = np.full(sizes[part], 2.0 if name.endswith("std") else 0.5)
        files.append(_write(root, relative, _npy(value)))
    manifest = {
        "schema_version": 1,
        "revision": backend.REVISION,
        "files": [vars(item) for item in files],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    return root


class FakeSession:
    def __init__(self, path, *, sess_options, providers, dtype="tensor(float16)"):
        self.path = path
        self.providers = providers
        self.options = sess_options
        self.fail = False
        self.feeds = None
        types = {
            "motion": dtype,
            "observed_motion": dtype,
            "motion_valid": "tensor(bool)",
            "constraint_mask": "tensor(bool)",
            "text_embedding": dtype,
            "timestep": "tensor(int64)",
            "first_heading_angle": dtype,
        }
        self._inputs = [SimpleNamespace(name=k, type=v) for k, v in types.items()]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return [SimpleNamespace(name="predicted_clean_motion")]

    def run(self, names, feeds):
        if self.fail:
            raise RuntimeError("device lost")
        self.feeds = feeds
        return [feeds["motion"] * 0.5]


class FakeOrt:
    GraphOptimizationLevel = SimpleNamespace(ORT_ENABLE_ALL=99)

    def __init__(self, providers=("WebGpuExecutionProvider", "CPUExecutionProvider")):
        self.providers = list(providers)
        self.sessions: list[FakeSession] = []
        self.fail_gpu = False

    def get_available_providers(self):
        return self.providers

    def SessionOptions(self):  # noqa: N802 - ONNX Runtime's name
        return SimpleNamespace()

    def InferenceSession(self, path, *, sess_options, providers):  # noqa: N802
        session = FakeSession(path, sess_options=sess_options, providers=providers)
        session.fail = self.fail_gpu and providers[0] != "CPUExecutionProvider"
        self.sessions.append(session)
        return session


def _inputs(frames: int = 4) -> dict[str, np.ndarray]:
    return {
        "motion": np.ones((3, frames, 369), dtype=np.float32),
        "observed": np.zeros((3, frames, 369), dtype=np.float32),
        "mask": np.zeros((3, frames, 369), dtype=bool),
        "valid": np.ones((3, frames), dtype=bool),
        "text": np.zeros((3, 50, 4096), dtype=np.float32),
        "timestep": np.full(3, 999, dtype=np.int64),
        "heading": np.zeros(3, dtype=np.float32),
    }


class TemporaryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)


class PinTests(unittest.TestCase):
    def test_revisions_and_hashes_are_pinned(self) -> None:
        self.assertRegex(backend.REVISION, HEX40)
        for variant in (
            backend.KimodoVariant.FP16,
            backend.KimodoVariant.FP32,
            backend.KimodoVariant.INT8,
        ):
            files = backend.bundle_files(variant)
            self.assertEqual(sum(item.path.endswith(".onnx") for item in files), 1)
            for item in files:
                self.assertRegex(item.sha256, HEX64)
                self.assertGreater(item.bytes, 0)

    def test_every_file_comes_from_the_pinned_kitsumate_repository(self) -> None:
        self.assertTrue(backend.REPOSITORY.startswith("KitsuMate/"))
        for item in backend.bundle_files(backend.KimodoVariant.FP16):
            self.assertIn(item.repository, {"", backend.REPOSITORY})
            self.assertIn(item.revision, {"", backend.REVISION})
        stats = [
            item
            for item in backend.bundle_files(backend.KimodoVariant.FP32)
            if item.path.startswith("stats/")
        ]
        self.assertEqual(
            sorted(item.path for item in stats), sorted(runtime.STATS_FILES.values())
        )
        licence_files = {item.path for item in backend.SHARED_FILES}
        self.assertTrue(set(backend.LICENSE.files) <= licence_files)

    def test_auto_variant_follows_the_device(self) -> None:
        def resolve(providers, device="AUTO"):
            with mock.patch.object(
                backend.ort_runtime,
                "available_execution_providers",
                return_value=providers,
            ):
                return backend.resolve_variant("AUTO", device)

        gpu = ("WebGpuExecutionProvider", "CPUExecutionProvider")
        self.assertIs(resolve(gpu), backend.KimodoVariant.FP16)
        self.assertIs(resolve(gpu, "CPU"), backend.KimodoVariant.FP32)
        self.assertIs(resolve(("CPUExecutionProvider",)), backend.KimodoVariant.FP32)
        self.assertIs(backend.resolve_variant("int8"), backend.KimodoVariant.INT8)


class InstallTests(TemporaryTestCase):
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

    def _install(self, **kwargs):
        with mock.patch.object(
            backend.lifecycle, "install_pinned_bundle", return_value="installed"
        ) as install:
            job = backend.start_install("INT8", **kwargs)
            assert job.thread is not None
            job.thread.join(timeout=5)
        return job, install

    def test_licence_is_required_before_anything_starts(self) -> None:
        with self.assertRaises(LicenseRequired) as caught:
            self._install(online_access=True)
        details = caught.exception.details
        self.assertEqual(
            details["accept_license"],
            f"nvidia-open-model-license@{backend.REVISION}",
        )
        self.assertIn("nvidia-open-model-license", details["url"])

    def test_offline_install_is_refused(self) -> None:
        with self.assertRaises(OnlineAccessDisabled):
            self._install(
                online_access=False,
                accept_license=backend.LICENSE.accept_value(backend.REVISION),
            )

    def test_accepted_install_records_the_licence_and_downloads(self) -> None:
        job, install = self._install(
            online_access=True,
            accept_license=backend.LICENSE.accept_value(backend.REVISION),
            source=licenses.AcceptanceSource.CLI,
        )
        self.assertEqual(job.snapshot().outcome, "DONE")
        _job, bundle, storage, root, _config = install.call_args.args
        self.assertEqual(install.call_args.kwargs, {"online_access": True})
        self.assertIs(storage, backend.STORAGE)
        self.assertEqual(root, self.root / "media_to_pose_kimodo")
        self.assertEqual(bundle.repository, backend.REPOSITORY)
        self.assertEqual(bundle.revision, backend.REVISION)
        self.assertEqual(
            bundle.list_files(root), backend.bundle_files(backend.KimodoVariant.INT8)
        )
        accepted = backend.license_acceptance()
        assert accepted is not None
        self.assertEqual(accepted["source"], "CLI")
        snapshot = backend.backend_snapshot("INT8", "CPU")
        self.assertEqual(snapshot.license_state, "ACCEPTED")
        self.assertEqual(snapshot.state, "NOT_INSTALLED")
        self.assertEqual(
            snapshot.download_bytes, backend.download_bytes(backend.KimodoVariant.INT8)
        )

    def test_installed_bundle_reports_ready(self) -> None:
        backend.STORAGE.ensure(self.root / "media_to_pose_kimodo")
        make_bundle(self.root / "media_to_pose_kimodo" / "active", "fp32")
        with mock.patch.object(
            backend.ort_runtime,
            "available_execution_providers",
            return_value=("CPUExecutionProvider",),
        ):
            snapshot = backend.backend_snapshot("AUTO", "AUTO")
        self.assertEqual((snapshot.state, snapshot.variant), ("READY", "FP32"))
        self.assertTrue(snapshot.fallback)
        self.assertEqual(snapshot.license_state, "REQUIRED")
        self.assertEqual(snapshot.min_ram_bytes, 2 * 1024**3)


class BundleTests(TemporaryTestCase):
    def test_bundle_loads_and_statistics_join(self) -> None:
        bundle = runtime.load_bundle(make_bundle(self.root))
        self.assertEqual(
            (bundle.variant, bundle.graph), ("FP16", "onnx/model_fp16.onnx")
        )
        stats = runtime.load_stats(bundle)
        self.assertEqual(stats.mean.shape, (rep.FEATURE_COUNT,))
        self.assertEqual(stats.mean[0], 0.5)

    def test_altered_file_fails_verification(self) -> None:
        make_bundle(self.root)
        (self.root / "NOTICE").write_bytes(b"NOTICE")
        with self.assertRaises(runtime.KimodoRuntimeError):
            runtime.load_bundle(self.root)
        runtime.load_bundle(self.root, verify_files=False)

    def test_pickled_statistics_are_refused(self) -> None:
        bundle = runtime.load_bundle(make_bundle(self.root))
        path = self.root / runtime.STATS_FILES["body_std"]
        np.save(path, np.array([{"x": 1}], dtype=object), allow_pickle=True)
        with self.assertRaises(runtime.KimodoRuntimeError):
            runtime.load_stats(bundle)


class SessionTests(TemporaryTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.bundle = runtime.load_bundle(make_bundle(self.root))
        self.ort = FakeOrt()

    def _session(self, mode="AUTO", **kwargs):
        with mock.patch.object(
            runtime.ort_runtime,
            "available_execution_providers",
            side_effect=lambda ort: tuple(ort.get_available_providers()),
        ):
            return runtime.KimodoSession(
                self.bundle, mode, ort_module=self.ort, sys_platform="win32", **kwargs
            )

    def test_inputs_take_the_graph_dtypes(self) -> None:
        session = self._session()
        self.assertEqual(session.diagnostics.actual_device, "webgpu")
        created = self.ort.sessions[0]
        self.assertEqual(
            created.providers, ["WebGpuExecutionProvider", "CPUExecutionProvider"]
        )
        self.assertEqual(created.options.graph_optimization_level, 99)
        output = session.denoise(**_inputs())
        self.assertEqual(output.dtype, np.float32)
        self.assertEqual(output.shape, (3, 4, 369))
        self.assertEqual(created.feeds["motion"].dtype, np.float16)
        self.assertEqual(created.feeds["motion_valid"].dtype, np.bool_)
        self.assertEqual(created.feeds["timestep"].dtype, np.int64)

    def test_session_is_the_samplers_denoise_function(self) -> None:
        session = self._session()
        values = _inputs()
        inputs = DenoiseInputs(
            motion=values["motion"],
            observed_motion=values["observed"],
            motion_valid=values["valid"],
            constraint_mask=values["mask"],
            text_embedding=values["text"],
            timestep=values["timestep"],
            first_heading_angle=values["heading"],
        )
        np.testing.assert_allclose(session(inputs), 0.5)

    def test_auto_falls_back_to_cpu_after_an_accelerator_failure(self) -> None:
        self.ort.fail_gpu = True
        session = self._session()
        output = session.denoise(**_inputs())
        self.assertEqual(output.shape, (3, 4, 369))
        self.assertEqual(session.diagnostics.actual_ep, "CPUExecutionProvider")
        self.assertTrue(session.diagnostics.fallback)
        self.assertEqual(self.ort.sessions[-1].providers, ["CPUExecutionProvider"])

    def test_required_gpu_does_not_fall_back(self) -> None:
        self.ort.fail_gpu = True
        session = self._session("GPU")
        with self.assertRaises(runtime.KimodoRuntimeError):
            session.denoise(**_inputs())

    def test_wrong_shapes_are_refused(self) -> None:
        session = self._session()
        values = _inputs(frames=1)
        with self.assertRaises(runtime.KimodoRuntimeError):
            session.denoise(**values)


class MemoryTests(unittest.TestCase):
    def test_shortfall_suggests_int8_when_it_fits(self) -> None:
        with (
            mock.patch.object(
                backend.resources, "available_ram_bytes", return_value=1536 * 1024**2
            ),
            self.assertRaises(InsufficientResources) as caught,
        ):
            backend._require_ram(backend.KimodoVariant.FP32)
        self.assertEqual(caught.exception.details["fitting_variant"], "INT8")


if __name__ == "__main__":
    unittest.main()
