from __future__ import annotations

import hashlib
import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests.blender_test_support import load_addon


class SamBackendContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.addon = load_addon()
        cls.sam = cls.addon.backends.sam3d.backend

    def test_model_specs_use_kitsumate_and_require_an_immutable_revision(self) -> None:
        self.assertEqual(tuple(self.sam.MODEL_VARIANTS), ("VITH",))
        specification = self.sam.MODEL_VARIANTS["VITH"]
        self.assertEqual(specification["repo"], "KitsuMate/sam-3d-body-onnx")
        self.assertTrue(
            self.addon.backends.sam3d.ort_runtime.is_immutable_revision(
                specification["revision"]
            )
        )

    def test_device_modes_are_stable_and_invalid_values_are_rejected(self) -> None:
        runtime = self.addon.backends.sam3d.ort_runtime
        self.assertEqual(
            [item[0] for item in self.sam.sam3d_device_enum_items()],
            ["AUTO", "GPU", "CPU"],
        )
        for mode in ("AUTO", "GPU", "CPU"):
            self.assertEqual(runtime.normalize_mode(mode), mode)
        for invalid in ("CUDA", "MPS", "DIRECTML", "WEBGPU", "unknown"):
            with (
                self.subTest(mode=invalid),
                self.assertRaisesRegex(runtime.OrtRuntimeError, "Unsupported"),
            ):
                runtime.normalize_mode(invalid)

    def test_backend_readiness_is_cached_until_install_state_changes(self) -> None:
        specification = self.sam.MODEL_VARIANTS["VITH"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = root / "graph.onnx"
            graph.write_bytes(b"onnx")
            bundle = mock.Mock(
                root=root,
                files=(mock.Mock(path=graph.name, bytes=graph.stat().st_size),),
                model_revision=specification["model_revision"],
                precision="fp16",
            )
            manifest = {
                "runtime_manifest_version": self.sam.RUNTIME_MANIFEST_VERSION,
                "backend_version": self.sam.BACKEND_VERSION,
                "backend_kind": "onnxruntime",
                "variant": "VITH",
                "bundle_revision": specification["revision"],
                "bundle_manifest": "bundle.json",
                "available_providers": ["CPUExecutionProvider"],
                "requested_mode": "CPU",
                "actual_ep": "CPUExecutionProvider",
                "actual_device": "cpu",
            }
            with (
                mock.patch.object(self.sam, "_read_json", return_value=manifest),
                mock.patch.object(
                    self.sam,
                    "installed_model_directory",
                    return_value=root,
                ),
                mock.patch.object(
                    self.addon.backends.sam3d.ort_runtime,
                    "load_bundle_manifest",
                    return_value=bundle,
                ) as load_bundle,
            ):
                self.sam.invalidate_backend_snapshot()
                try:
                    first = self.sam.backend_snapshot("VITH", "CPU")
                    second = self.sam.backend_snapshot("VITH", "CPU")
                    self.assertTrue(first.installed)
                    self.assertIs(first, second)
                    load_bundle.assert_called_once()

                    self.sam.invalidate_backend_snapshot()
                    self.sam.backend_snapshot("VITH", "CPU")
                    self.assertEqual(load_bundle.call_count, 2)
                finally:
                    self.sam.invalidate_backend_snapshot()

    def test_execution_provider_policy_is_platform_specific_with_cpu_fallback(self):
        runtime = self.addon.backends.sam3d.ort_runtime
        providers = (
            "CPUExecutionProvider",
            "WebGpuExecutionProvider",
            "CoreMLExecutionProvider",
        )
        self.assertEqual(
            runtime.select_execution_provider("AUTO", providers, sys_platform="win32"),
            ("WebGpuExecutionProvider", "webgpu", False),
        )
        self.assertEqual(
            runtime.select_execution_provider("AUTO", providers, sys_platform="darwin"),
            ("CoreMLExecutionProvider", "coreml", False),
        )
        self.assertEqual(
            runtime.select_execution_provider(
                "AUTO",
                ("CPUExecutionProvider",),
                sys_platform="linux",
            ),
            ("CPUExecutionProvider", "cpu", True),
        )
        with self.assertRaisesRegex(runtime.OrtRuntimeError, "GPU mode requires"):
            runtime.select_execution_provider(
                "GPU",
                ("CPUExecutionProvider",),
                sys_platform="linux",
            )

    def _bundle(self, directory: str, *, artifact: bool = False):
        root = Path(directory)
        backbone = root / "backbone.onnx"
        decoder = root / "decoder.onnx"
        backbone.write_bytes(b"fake-backbone")
        decoder.write_bytes(b"fake-decoder")
        artifacts = []
        if artifact:
            weights = root / "backbone.onnx.data"
            weights.write_bytes(b"fake-weights")
            artifacts.append(
                {
                    "path": weights.name,
                    "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                    "bytes": weights.stat().st_size,
                }
            )
        manifest = {
            "schema_version": 1,
            "contract": self.addon.backends.sam3d.ort_runtime.BUNDLE_CONTRACT,
            "runner": self.addon.backends.sam3d.ort_runtime.SPLIT_NUMPY_RUNNER,
            "variant": "VITH",
            "model_revision": "a" * 40,
            "precision": "fp16",
            "graphs": [
                {
                    "role": "backbone",
                    "path": backbone.name,
                    "sha256": hashlib.sha256(backbone.read_bytes()).hexdigest(),
                    "bytes": backbone.stat().st_size,
                    "inputs": {"image": "image"},
                    "outputs": {"features": "features"},
                },
                {
                    "role": "decoder.body.layer.0",
                    "path": decoder.name,
                    "sha256": hashlib.sha256(decoder.read_bytes()).hexdigest(),
                    "bytes": decoder.stat().st_size,
                    "inputs": {"tokens": "tokens"},
                    "outputs": {"tokens_out": "tokens_out"},
                },
            ],
            "artifacts": artifacts,
            "neutral_rotations": {
                str(index): [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                for index in sorted(
                    self.addon.backends.sam3d.ort_runtime.REQUIRED_NEUTRAL_ROTATION_INDICES
                )
            },
        }
        path = root / "bundle.json"
        path.write_text(json.dumps(manifest), encoding="utf8")
        return path

    def test_bundle_manifest_verifies_graph_and_external_data(self) -> None:
        runtime = self.addon.backends.sam3d.ort_runtime
        with tempfile.TemporaryDirectory() as directory:
            bundle = runtime.load_bundle_manifest(
                self._bundle(directory, artifact=True),
                expected_variant="VITH",
                expected_model_revision="a" * 40,
            )
            self.assertEqual(len(bundle.files), 3)
            self.assertEqual(bundle.precision, "fp16")
            (Path(directory) / "backbone.onnx.data").write_bytes(b"tampered")
            with self.assertRaisesRegex(runtime.BundleContractError, "size differs"):
                runtime.load_bundle_manifest(Path(directory) / "bundle.json")

    def test_bundle_manifest_rejects_escaping_artifact(self) -> None:
        runtime = self.addon.backends.sam3d.ort_runtime
        with tempfile.TemporaryDirectory() as directory:
            path = self._bundle(directory)
            manifest = json.loads(path.read_text(encoding="utf8"))
            manifest["artifacts"] = [
                {"path": "../escape", "sha256": "0" * 64, "bytes": 1}
            ]
            path.write_text(json.dumps(manifest), encoding="utf8")
            with self.assertRaisesRegex(runtime.BundleContractError, "Unsafe"):
                runtime.load_bundle_manifest(path, verify_files=False)

    def test_bundle_requires_every_provider_neutral_rotation(self) -> None:
        runtime = self.addon.backends.sam3d.ort_runtime
        with tempfile.TemporaryDirectory() as directory:
            path = self._bundle(directory)
            manifest = json.loads(path.read_text(encoding="utf8"))
            missing = min(runtime.REQUIRED_NEUTRAL_ROTATION_INDICES)
            del manifest["neutral_rotations"][str(missing)]
            path.write_text(json.dumps(manifest), encoding="utf8")
            with self.assertRaisesRegex(
                runtime.BundleContractError,
                f"missing provider-consumed MHR joints: {missing}",
            ):
                runtime.load_bundle_manifest(path, verify_files=False)

    def test_bundle_rejects_improper_neutral_rotation(self) -> None:
        runtime = self.addon.backends.sam3d.ort_runtime
        with tempfile.TemporaryDirectory() as directory:
            path = self._bundle(directory)
            manifest = json.loads(path.read_text(encoding="utf8"))
            index = min(runtime.REQUIRED_NEUTRAL_ROTATION_INDICES)
            manifest["neutral_rotations"][str(index)] = [
                -1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
            path.write_text(json.dumps(manifest), encoding="utf8")
            with self.assertRaisesRegex(runtime.BundleContractError, "proper rotation"):
                runtime.load_bundle_manifest(path, verify_files=False)

    def test_bundle_rejects_nonfinite_and_nonorthonormal_neutral_rotations(
        self,
    ) -> None:
        runtime = self.addon.backends.sam3d.ort_runtime
        invalid_matrices = {
            "non-finite": [
                float("nan"),
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            "orthonormal": [
                2.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        }
        for message, matrix in invalid_matrices.items():
            with (
                self.subTest(violation=message),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = self._bundle(directory)
                manifest = json.loads(path.read_text(encoding="utf8"))
                index = min(runtime.REQUIRED_NEUTRAL_ROTATION_INDICES)
                manifest["neutral_rotations"][str(index)] = matrix
                path.write_text(json.dumps(manifest), encoding="utf8")
                with self.assertRaisesRegex(runtime.BundleContractError, message):
                    runtime.load_bundle_manifest(path, verify_files=False)

    def test_every_ort_output_has_exact_dtype_shape_and_finite_values(self) -> None:
        runtime = self.addon.backends.sam3d.ort_runtime
        people = 2
        valid = {
            key: np.zeros((people, *suffix), dtype=dtype)
            for key, (dtype, suffix) in runtime.OUTPUT_TENSOR_CONTRACTS.items()
        }
        runtime._validate_output_tensors(np, valid, people)
        for key, (dtype, suffix) in runtime.OUTPUT_TENSOR_CONTRACTS.items():
            with self.subTest(output=key, violation="dtype"):
                invalid = dict(valid)
                invalid[key] = np.zeros((people, *suffix), dtype=np.float64)
                with self.assertRaisesRegex(runtime.OrtRuntimeError, "dtype float32"):
                    runtime._validate_output_tensors(np, invalid, people)
            with self.subTest(output=key, violation="person axis"):
                invalid = dict(valid)
                invalid[key] = np.zeros((people + 1, *suffix), dtype=dtype)
                with self.assertRaisesRegex(
                    runtime.OrtRuntimeError, "leading person axis"
                ):
                    runtime._validate_output_tensors(np, invalid, people)
            with self.subTest(output=key, violation="tensor suffix"):
                invalid = dict(valid)
                wrong_suffix = (1,) if not suffix else (suffix[0] + 1, *suffix[1:])
                invalid[key] = np.zeros((people, *wrong_suffix), dtype=dtype)
                with self.assertRaisesRegex(runtime.OrtRuntimeError, "must have shape"):
                    runtime._validate_output_tensors(np, invalid, people)
            with self.subTest(output=key, violation="finite"):
                invalid = dict(valid)
                value = valid[key].copy()
                value.flat[0] = np.nan
                invalid[key] = value
                with self.assertRaisesRegex(runtime.OrtRuntimeError, "non-finite"):
                    runtime._validate_output_tensors(np, invalid, people)

    def test_auto_webgpu_prefers_nvidia_adapter(self) -> None:
        runtime = self.addon.backends.sam3d.ort_runtime

        class Hardware:
            type = "GPU"

            def __init__(self, vendor_id, device_id):
                self.vendor_id = vendor_id
                self.device_id = device_id

        class Device:
            ep_name = "WebGpuExecutionProvider"

            def __init__(self, vendor_id, device_id):
                self.device = Hardware(vendor_id, device_id)

        class FakeOrt:
            @staticmethod
            def get_available_providers():
                return ["WebGpuExecutionProvider", "CPUExecutionProvider"]

            @staticmethod
            def get_ep_devices():
                return [Device(0x1002, 5688), Device(0x10DE, 9373)]

        diagnostics = runtime.probe_runtime("AUTO", ort_module=FakeOrt())
        self.assertEqual(diagnostics.actual_device, "webgpu")
        self.assertEqual(diagnostics.hardware_vendor_id, 0x10DE)
        self.assertEqual(diagnostics.hardware_device_id, 9373)

    def test_webgpu_plugin_registration_can_retry_after_transient_failure(self) -> None:
        runtime = self.addon.backends.sam3d.ort_runtime

        class FakeOrt:
            attempts = 0
            registered = False

            @classmethod
            def get_available_providers(cls):
                providers = ["CPUExecutionProvider"]
                if cls.registered:
                    providers.insert(0, "WebGpuExecutionProvider")
                return providers

            @classmethod
            def register_execution_provider_library(cls, _name, _path):
                cls.attempts += 1
                if cls.attempts == 1:
                    raise RuntimeError("injected transient failure")
                cls.registered = True

        plugin = mock.Mock()
        plugin.get_library_path.return_value = "webgpu-provider"
        identity = id(FakeOrt)
        runtime._REGISTERED_ORT_MODULES.discard(identity)
        try:
            with (
                mock.patch.object(runtime.sys, "platform", "win32"),
                mock.patch.dict(
                    runtime.sys.modules,
                    {"onnxruntime_ep_webgpu": plugin},
                ),
            ):
                with self.assertRaisesRegex(
                    runtime.OrtRuntimeError, "Could not register"
                ):
                    runtime._register_webgpu_plugin(FakeOrt)
                self.assertNotIn(identity, runtime._REGISTERED_ORT_MODULES)
                runtime._register_webgpu_plugin(FakeOrt)
                self.assertIn(identity, runtime._REGISTERED_ORT_MODULES)
        finally:
            runtime._REGISTERED_ORT_MODULES.discard(identity)
        self.assertEqual(FakeOrt.attempts, 2)

    def test_sam_runtime_loads_before_mediapipe_dependencies(self) -> None:
        provider_module = importlib.import_module(
            f"{self.addon.__name__}.providers.sam3d_body_provider"
        )
        calls = []
        load_arguments = []

        class Estimator:
            neutral_rotations = {}

        class Diagnostics:
            actual_device = "cpu"

        def load_model(*_args, **kwargs):
            calls.append("model")
            load_arguments.append(kwargs)
            return Estimator(), Diagnostics(), "cpu"

        def dependencies():
            calls.append("dependencies")
            return object(), object(), object()

        with (
            mock.patch.object(provider_module, "_load_model", load_model),
            mock.patch.object(
                provider_module,
                "import_runtime_dependencies",
                dependencies,
            ),
            mock.patch.object(
                provider_module.SAM3DBodyProvider,
                "_create_locator",
                return_value=None,
            ),
        ):
            provider_module.SAM3DBodyProvider(
                use_body=True,
                use_hands=True,
                runtime_directory="runtime",
                source_directory="source",
                backend_directory="backend",
                checkpoint_path="checkpoint",
            )
            provider_module.SAM3DBodyProvider(
                use_body=True,
                use_hands=True,
                verify_accelerator_execution=True,
                runtime_directory="runtime",
                source_directory="source",
                backend_directory="backend",
                checkpoint_path="checkpoint",
            )
        self.assertEqual(
            calls,
            ["model", "dependencies", "model", "dependencies"],
        )
        self.assertEqual(
            [arguments["verify_accelerator_execution"] for arguments in load_arguments],
            [False, True],
        )
        self.assertFalse(hasattr(provider_module.SAM3DBodyProvider, "torch"))

    def test_provider_accelerator_verification_is_forwarded_and_cache_keyed(
        self,
    ) -> None:
        provider_module = importlib.import_module(
            f"{self.addon.__name__}.providers.sam3d_body_provider"
        )

        class Diagnostics:
            actual_device = "webgpu"

        first = mock.Mock(diagnostics=Diagnostics())
        second = mock.Mock(diagnostics=Diagnostics())
        paths = {
            "backend_directory": "backend",
            "bundle_manifest_path": "bundle.json",
            "bundle_revision": "a" * 40,
            "model_revision": "b" * 40,
        }
        previous = (
            provider_module._MODEL_KEY,
            provider_module._ESTIMATOR,
            provider_module._RUNTIME_DIAGNOSTICS,
        )
        provider_module._MODEL_KEY = None
        provider_module._ESTIMATOR = None
        provider_module._RUNTIME_DIAGNOSTICS = None
        try:
            with mock.patch.object(
                provider_module.sam3d_ort_runtime,
                "load_estimator",
                side_effect=(first, second),
            ) as load_estimator:
                normal = provider_module._load_model("VITH", "GPU", paths)
                cached = provider_module._load_model("VITH", "GPU", paths)
                verified = provider_module._load_model(
                    "VITH",
                    "GPU",
                    paths,
                    verify_accelerator_execution=True,
                )

            self.assertIs(normal[0], first)
            self.assertIs(cached[0], first)
            self.assertIs(verified[0], second)
            self.assertEqual(load_estimator.call_count, 2)
            self.assertFalse(
                load_estimator.call_args_list[0].kwargs["verify_accelerator_execution"]
            )
            self.assertTrue(
                load_estimator.call_args_list[1].kwargs["verify_accelerator_execution"]
            )
            first.close.assert_called_once_with()
        finally:
            provider_module.unload_model_cache()
            (
                provider_module._MODEL_KEY,
                provider_module._ESTIMATOR,
                provider_module._RUNTIME_DIAGNOSTICS,
            ) = previous

    def test_provider_refreshes_profile_diagnostics_after_inference(self) -> None:
        provider_module = importlib.import_module(
            f"{self.addon.__name__}.providers.sam3d_body_provider"
        )

        class Diagnostics:
            def __init__(self, *, verified: bool) -> None:
                self.actual_device = "webgpu"
                self.accelerator_nodes_verified = verified

        initial = Diagnostics(verified=False)
        verified = Diagnostics(verified=True)

        class Estimator:
            diagnostics = initial

            def process_one_image(self, *_args, **_kwargs):
                self.diagnostics = verified
                return ["prediction"]

        class Cv2:
            COLOR_BGR2RGB = object()

            @staticmethod
            def cvtColor(image, _conversion):
                return image

        estimator = Estimator()
        provider = provider_module.SAM3DBodyProvider.__new__(
            provider_module.SAM3DBodyProvider
        )
        provider.estimator = estimator
        provider.runtime = initial
        provider.device = initial.actual_device
        provider.cv2 = Cv2()
        provider.use_hands = True
        image = np.zeros((2, 3, 3), dtype=np.uint8)

        previous_estimator = provider_module._ESTIMATOR
        previous_diagnostics = provider_module._RUNTIME_DIAGNOSTICS
        provider_module._ESTIMATOR = estimator
        provider_module._RUNTIME_DIAGNOSTICS = initial
        try:
            with (
                mock.patch.object(provider, "_input_image", return_value=image),
                mock.patch.object(provider, "_bbox", return_value=[(0, 0, 3, 2)]),
                mock.patch.object(provider, "_convert", return_value="frame"),
            ):
                self.assertEqual(provider._detect(image, 7), "frame")
            self.assertIs(provider.runtime, verified)
            self.assertIs(provider_module.runtime_diagnostics(), verified)
            self.assertTrue(provider.runtime.accelerator_nodes_verified)
            self.assertEqual(provider.device, "webgpu")
        finally:
            provider_module._ESTIMATOR = previous_estimator
            provider_module._RUNTIME_DIAGNOSTICS = previous_diagnostics

    def test_download_hash_mismatch_removes_partial_file(self) -> None:
        class Response:
            headers = {"Content-Length": "3"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _size):
                value, self.value = getattr(self, "value", b"bad"), b""
                return value

        job = self.sam.BackgroundJob("download")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "source.zip"
            with (
                mock.patch.object(
                    self.sam.urllib.request,
                    "urlopen",
                    return_value=Response(),
                ),
                self.assertRaisesRegex(RuntimeError, "SHA-256"),
            ):
                self.sam._download_file(
                    "https://example.invalid/source.zip",
                    destination,
                    job,
                    0.0,
                    1.0,
                    "Source",
                    "Download",
                    expected_sha256="0" * 64,
                )
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(directory).glob("*.part")), [])

    def test_download_transport_failure_removes_partial_file(self) -> None:
        class Response:
            headers = {"Content-Length": "6"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _size):
                if not hasattr(self, "read_once"):
                    self.read_once = True
                    return b"abc"
                raise TimeoutError("injected timeout")

        job = self.sam.BackgroundJob("download")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "source.zip"
            with (
                mock.patch.object(
                    self.sam.urllib.request,
                    "urlopen",
                    return_value=Response(),
                ),
                self.assertRaisesRegex(TimeoutError, "injected timeout"),
            ):
                self.sam._download_file(
                    "https://example.invalid/source.zip",
                    destination,
                    job,
                    0.0,
                    1.0,
                    "Source",
                    "Download",
                )
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(directory).glob("*.part")), [])

    def test_installation_lock_blocks_a_second_process_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.sam._InstallationLock(root)
            second = self.sam._InstallationLock(root)
            first.acquire()
            try:
                with self.assertRaisesRegex(RuntimeError, "Another Blender"):
                    second.acquire()
            finally:
                first.release()
            second.acquire()
            second.release()
            self.assertFalse((root / ".install.lock").exists())

    def test_failed_activation_restores_current_install(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_parent = root / "runtime"
            model_parent = root / "model"
            runtime_parent.mkdir()
            model_parent.mkdir()
            runtime_staging = runtime_parent / "staging"
            runtime_current = runtime_parent / "active"
            runtime_backup = runtime_parent / "previous"
            model_staging = model_parent / "staging"
            model_current = model_parent / "active"
            model_backup = model_parent / "previous"
            for path, value in (
                (runtime_staging, "new-runtime"),
                (runtime_current, "old-runtime"),
                (model_staging, "new-model"),
                (model_current, "old-model"),
            ):
                path.mkdir()
                (path / "value.txt").write_text(value, encoding="utf-8")

            real_replace = os.replace

            def replace(source, destination):
                if Path(source) == runtime_staging:
                    raise OSError("injected runtime activation failure")
                return real_replace(source, destination)

            with (
                mock.patch("os.replace", side_effect=replace),
                self.assertRaisesRegex(OSError, "injected"),
            ):
                self.sam._activate_staged_install(
                    runtime_staging,
                    runtime_current,
                    runtime_backup,
                    model_staging,
                    model_current,
                    model_backup,
                )
            self.assertEqual(
                (runtime_current / "value.txt").read_text(encoding="utf-8"),
                "old-runtime",
            )
            self.assertEqual(
                (model_current / "value.txt").read_text(encoding="utf-8"),
                "old-model",
            )
            self.assertEqual(
                (model_staging / "value.txt").read_text(encoding="utf-8"),
                "new-model",
            )

    def test_activation_lock_failure_keeps_current_and_download_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_parent = root / "runtime"
            model_parent = root / "model"
            runtime_parent.mkdir()
            model_parent.mkdir()
            runtime_staging = runtime_parent / "staging"
            runtime_current = runtime_parent / "active"
            runtime_backup = runtime_parent / "previous"
            model_staging = model_parent / "download"
            model_current = model_parent / "active"
            model_backup = model_parent / "previous"
            for path, value in (
                (runtime_staging, "new-runtime"),
                (runtime_current, "old-runtime"),
                (model_staging, "downloaded-model"),
                (model_current, "old-model"),
            ):
                path.mkdir()
                (path / "value.txt").write_text(value, encoding="utf-8")

            real_replace = os.replace
            calls = []

            def replace(source, destination):
                calls.append((Path(source), Path(destination)))
                if Path(source) == runtime_current:
                    raise PermissionError(5, "injected Windows access denial")
                return real_replace(source, destination)

            with (
                mock.patch("os.replace", side_effect=replace),
                self.assertRaises(PermissionError),
            ):
                self.sam._activate_staged_install(
                    runtime_staging,
                    runtime_current,
                    runtime_backup,
                    model_staging,
                    model_current,
                    model_backup,
                )

            self.assertEqual(calls, [(runtime_current, runtime_backup)])
            self.assertEqual(
                (runtime_current / "value.txt").read_text(encoding="utf-8"),
                "old-runtime",
            )
            self.assertEqual(
                (model_current / "value.txt").read_text(encoding="utf-8"),
                "old-model",
            )
            self.assertEqual(
                (model_staging / "value.txt").read_text(encoding="utf-8"),
                "downloaded-model",
            )

    def test_verified_model_files_are_reused_after_a_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            source_manifest = self._bundle(str(source))
            runtime_root = root / "runtime"
            model_root = root / "models"
            layout = self.sam._InstallLayout(
                root=runtime_root,
                staging=runtime_root / "staging",
                backup=runtime_root / "previous",
                storage_parent=root,
                storage_root=model_root,
                model_staging=model_root / ".download",
                model_current=model_root / "active",
                model_backup=model_root / ".previous",
                previous_storage_root=None,
            )
            self.sam._prepare_ort_staging(layout)
            for name in ("backbone.onnx", "decoder.onnx"):
                destination = layout.model_staging / name
                destination.write_bytes((source / name).read_bytes())

            downloads = []

            def download(_url, destination, *_args, **_kwargs):
                downloads.append(Path(destination))
                Path(destination).write_bytes(source_manifest.read_bytes())

            specification = {
                "repo": "KitsuMate/sam-3d-body-onnx",
                "revision": "b" * 40,
                "manifest_path": "onnx/vith/manifest.json",
                "model_revision": "a" * 40,
            }
            job = self.sam.BackgroundJob("retry")
            with (
                mock.patch.object(self.sam, "_download_file", side_effect=download),
                mock.patch.object(self.sam, "_check_bundle_storage"),
            ):
                bundle = self.sam._download_ort_bundle(
                    job,
                    layout,
                    "VITH",
                    specification,
                )

            self.assertEqual(downloads, [layout.model_staging / "bundle.json"])
            self.assertEqual(
                {item.path for item in bundle.files},
                {"backbone.onnx", "decoder.onnx"},
            )


class SamProviderConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.addon = load_addon()
        cls.provider_module = importlib.import_module(
            f"{cls.addon.__name__}.providers.sam3d_body_provider"
        )

    def _provider(self):
        provider = self.provider_module.SAM3DBodyProvider.__new__(
            self.provider_module.SAM3DBodyProvider
        )
        provider.np = np
        provider.use_body = True
        provider.use_hands = True
        provider.confidence = 0.5
        provider.smoothing = 0.0
        provider._previous = None
        provider._locator_fallback_used = False
        provider._neutral_rotations = {
            index: np.eye(3) for index in self.provider_module._MHR_JOINTS.values()
        }
        return provider

    @staticmethod
    def _prediction(joints, rotations):
        keypoints_3d = np.asarray(
            [[float(index), index + 100.0, index + 200.0] for index in range(70)],
            dtype=np.float32,
        )
        keypoints_2d = np.asarray(
            [[index * 2.0, index * 3.0, 0.9] for index in range(70)],
            dtype=np.float32,
        )
        return {
            "pred_keypoints_3d": keypoints_3d,
            "pred_keypoints_2d": keypoints_2d,
            "pred_cam_t": np.asarray((0.2, -0.1, 3.5)),
            "focal_length": np.asarray((1200.0,)),
            # MHR exposes intrinsic ZYX Euler angles as [rz, ry, rx].
            "global_rot": np.asarray((np.pi / 2.0, 0.0, 0.0)),
            "scale_params": np.asarray((1.05,)),
            "pred_joint_coords": joints,
            "pred_global_rots": rotations,
        }

    def test_mhr_output_converts_to_canonical_body_hands_and_spine(self) -> None:
        joints = np.asarray(
            [[index, index + 0.25, index + 0.5] for index in range(127)],
            dtype=np.float32,
        )
        rotations = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 127, axis=0)
        frame = self._provider()._convert(
            [self._prediction(joints, rotations)],
            width=280,
            height=350,
            timestamp_ms=125,
        )
        self.assertEqual(frame.timestamp_ms, 125)
        self.assertEqual(frame.body["LEFT_SHOULDER"].position[0], 75.0)
        self.assertEqual(frame.body["RIGHT_WRIST"].position[0], 41.0)
        self.assertEqual(frame.hands["RIGHT"]["INDEX_FINGER_MCP"].position[0], 28.0)
        self.assertEqual(frame.hands["LEFT"]["INDEX_FINGER_TIP"].position[0], 46.0)
        self.assertEqual(frame.auxiliary["MHR_SPINE_0"].position[0], 34.0)
        self.assertEqual(frame.auxiliary["MHR_HEAD_TIP"].position[0], 126.0)
        self.assertTrue(frame.joint_frames["MHR_HEAD"].valid)
        self.assertEqual(frame.root_motion.translation, (0.2, -0.1, 3.5))
        self.assertEqual(frame.root_motion.focal_length, 1200.0)
        self.assertTrue(
            np.allclose(
                np.asarray(frame.root_motion.rotation_matrix).reshape(3, 3),
                np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
                atol=1.0e-7,
            )
        )

    def test_degenerate_spine_and_non_orthonormal_frame_are_rejected(self) -> None:
        joints = np.asarray(
            [[index, index + 0.25, index + 0.5] for index in range(127)],
            dtype=np.float32,
        )
        joints[35] = joints[34]
        rotations = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 127, axis=0)
        rotations[115, 0, 0] = 2.0
        frame = self._provider()._convert(
            [self._prediction(joints, rotations)],
            width=280,
            height=350,
            timestamp_ms=126,
        )
        self.assertFalse(frame.auxiliary["MHR_SPINE_0"].valid)
        self.assertFalse(frame.joint_frames["MHR_HEAD"].valid)
        self.assertTrue(any("degenerate" in item for item in frame.warnings))


if __name__ == "__main__":
    unittest.main()
