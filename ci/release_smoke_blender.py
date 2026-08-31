"""Exercise an installed Media to Pose archive inside Blender."""

from __future__ import annotations

import base64
import importlib
import json
import os
import queue
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import bpy

MODULE_NAME = "bl_ext.user_default.media_to_pose"
TINY_ONNX_MODEL = base64.b64decode(
    "CAgSHHZpc2lvbi10by1wb3NlLXJlbGVhc2Utc21va2U6bAoRCgF4CgRiaWFzEgF5IgNBZGQ"
    "SEXZ0cF9yZWxlYXNlX3Ntb2tlKhoIAQgDEAEiDAAAAD8AAIC/AAAAQEIEYmlhc1oTCgF4Eg4"
    "KDAgBEggKAggBCgIIA2ITCgF5Eg4KDAgBEggKAggBCgIIA0IECgAQDQ=="
)
EXPECTED_ORT_VERSIONS = {
    "win32": "1.27.0",
    "linux": "1.27.0",
    "darwin": "1.23.2",
}


def _assert_registered(*, import_dependencies: bool = False) -> object:
    addon = importlib.import_module(MODULE_NAME)
    extension_root = Path(os.environ["BLENDER_USER_EXTENSIONS"]).resolve()
    module_path = Path(addon.__file__).resolve()
    if not module_path.is_relative_to(extension_root):
        raise AssertionError(
            f"Loaded source checkout instead of archive: {module_path}"
        )
    if not hasattr(bpy.types.Scene, "mtp_settings"):
        raise AssertionError("Installed extension did not register Scene.mtp_settings")
    lookup = bpy.types.bpy_struct.bl_rna_get_subclass_py
    if lookup("MTP_AddonPreferences", None) is None:
        raise AssertionError(
            "Installed extension did not register centralized model preferences"
        )
    expected_version = os.environ.get("MTP_EXPECTED_VERSION", "")
    if not expected_version:
        raise AssertionError("Expected extension version was not provided")
    actual_version = addon.constants.extension_version()
    if actual_version != expected_version:
        raise AssertionError(
            f"Installed extension version is {actual_version}, expected "
            f"{expected_version}"
        )
    correction_ids = {item[0] for item in addon.constants.POST_PROCESSING_MODULES}
    required_corrections = {"ROHM_CORRECTION"}
    if not required_corrections.issubset(correction_ids):
        raise AssertionError("Installed extension does not expose RoHM correction")
    required_rohm_operators = {
        "MTP_OT_rohm_install",
        "MTP_OT_rohm_cancel_install",
        "MTP_OT_rohm_load",
        "MTP_OT_rohm_unload",
        "MTP_OT_rohm_verify",
        "MTP_OT_rohm_uninstall",
    }
    missing_rohm_operators = sorted(
        name for name in required_rohm_operators if not hasattr(bpy.types, name)
    )
    if missing_rohm_operators:
        raise AssertionError(
            f"Installed extension is missing RoHM lifecycle operators: {missing_rohm_operators}"
        )
    available, message = addon.dependency_bootstrap.ensure_bundled_dependencies()
    if not available:
        raise AssertionError(message)
    if import_dependencies:
        dependencies = addon.dependency_bootstrap.import_runtime_dependencies()
        _assert_native_dependency_origins(*dependencies[:2])
    return addon


def _assert_native_dependency_origins(cv2: object, mediapipe: object) -> None:
    extension_root = Path(os.environ["BLENDER_USER_EXTENSIONS"]).resolve()
    for name, module in (("OpenCV", cv2), ("MediaPipe", mediapipe)):
        module_file = getattr(module, "__file__", None)
        if not module_file:
            raise AssertionError(f"{name} does not expose its loaded module path")
        module_path = Path(module_file).resolve()
        if not module_path.is_relative_to(extension_root):
            raise AssertionError(
                f"Loaded {name} outside Blender extension wheels: {module_path}"
            )


def _toggle_registration() -> object:
    disabled = bpy.ops.preferences.addon_disable(module=MODULE_NAME)
    if disabled != {"FINISHED"}:
        raise AssertionError(f"Extension disable failed: {disabled}")
    if hasattr(bpy.types.Scene, "mtp_settings"):
        raise AssertionError("Extension left Scene.mtp_settings after disable")
    enabled = bpy.ops.preferences.addon_enable(module=MODULE_NAME)
    if enabled != {"FINISHED"}:
        raise AssertionError(f"Extension enable failed: {enabled}")
    return _assert_registered()


def _exercise_fake_capture(addon: object) -> None:
    operators = addon.operators_capture

    class FakeProvider:
        closed = False

        def __init__(self, **_options):
            del _options

        def detect_image(self, _path):
            return addon.models.CanonicalFrame(timestamp_ms=0)

        def close(self):
            type(self).closed = True

    class FakeJob:
        def __init__(self):
            self.cancel = threading.Event()
            self.messages = queue.Queue()
            self.temp_path = ""
            self.progress_updates = []

        def progress(self, current, total, status):
            self.progress_updates.append((current, total, status))

    media = addon.models.MediaInfo(path="ignored.png", kind="IMAGE")
    job = FakeJob()
    original_descriptor = operators.descriptor
    operators.descriptor = lambda _provider_id: SimpleNamespace(
        provider_class=FakeProvider
    )
    try:
        operators._worker(
            job,
            "MEDIAPIPE",
            {},
            media,
            media,
            "POSE",
            0.0,
            "unused-cache-key",
            {},
            False,
            1024,
        )
    finally:
        operators.descriptor = original_descriptor
    kind, frame = job.messages.get_nowait()
    if kind != "FRAME" or frame.timestamp_ms != 0:
        raise AssertionError(f"Unexpected fake capture result: {kind!r}")
    if not FakeProvider.closed:
        raise AssertionError("Capture worker did not close its provider")
    if (
        not job.progress_updates
        or job.progress_updates[-1][0] != job.progress_updates[-1][1]
        or job.progress_updates[-1][2] != "Detection complete"
    ):
        raise AssertionError("Capture worker did not publish completion progress")


def _exercise_onnxruntime(addon: object) -> None:
    runtime = addon.dependency_bootstrap.import_onnxruntime()
    diagnostics = addon.dependency_bootstrap.onnxruntime_diagnostics()
    expected_version = EXPECTED_ORT_VERSIONS.get(sys.platform)
    if diagnostics["version"] != expected_version:
        raise AssertionError(
            f"Unexpected ONNX Runtime version: {diagnostics['version']}; "
            f"expected {expected_version}"
        )
    available_providers = diagnostics["available_providers"]
    expected_accelerator = diagnostics["expected_accelerator"]
    if "CPUExecutionProvider" not in available_providers:
        raise AssertionError("Bundled ONNX Runtime has no CPU execution provider")
    if expected_accelerator not in available_providers:
        raise AssertionError(
            f"Bundled ONNX Runtime is missing {expected_accelerator}: "
            f"{available_providers}"
        )
    runtime_path = Path(runtime.__file__).resolve()
    extension_root = Path(os.environ["BLENDER_USER_EXTENSIONS"]).resolve()
    if not runtime_path.is_relative_to(extension_root):
        raise AssertionError(
            f"Loaded ONNX Runtime outside Blender wheels: {runtime_path}"
        )

    numpy = importlib.import_module("numpy")
    expected = numpy.asarray([[1.5, 1.0, 5.0]], dtype=numpy.float32)
    model_input = numpy.asarray([[1.0, 2.0, 3.0]], dtype=numpy.float32)
    session = runtime.InferenceSession(
        TINY_ONNX_MODEL, providers=["CPUExecutionProvider"]
    )
    actual = session.run(["y"], {"x": model_input})[0]
    if not numpy.allclose(actual, expected, rtol=0.0, atol=1e-6):
        raise AssertionError(f"Tiny ONNX inference mismatch: {actual!r}")
    diagnostics["smoke_session_providers"] = tuple(session.get_providers())

    if os.environ.get("MTP_REQUIRE_ACCELERATOR") == "1":
        options = runtime.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(
            Path(tempfile.gettempdir()) / "mtp-ort-accelerator"
        )
        ep_devices = [
            item
            for item in getattr(runtime, "get_ep_devices", lambda: ())()
            if str(getattr(item, "ep_name", "")) == expected_accelerator
        ]

        def device_rank(item):
            hardware = getattr(item, "device", None)
            vendor_id = int(getattr(hardware, "vendor_id", 0) or 0)
            return {0x10DE: 300, 0x1002: 200, 0x8086: 100}.get(vendor_id, 0)

        selected_device = max(ep_devices, key=device_rank) if ep_devices else None
        add_devices = getattr(options, "add_provider_for_devices", None)
        if selected_device is not None and callable(add_devices):
            add_devices(
                [selected_device],
                {"preferredLayout": "NHWC", "enableGraphCapture": "0"},
            )
            accelerated = runtime.InferenceSession(
                TINY_ONNX_MODEL, sess_options=options
            )
            hardware = selected_device.device
            diagnostics["selected_accelerator_device"] = {
                "device_id": int(hardware.device_id),
                "metadata": dict(hardware.metadata),
                "type": str(hardware.type),
                "vendor": str(hardware.vendor),
                "vendor_id": int(hardware.vendor_id),
            }
        else:
            accelerated = runtime.InferenceSession(
                TINY_ONNX_MODEL,
                sess_options=options,
                providers=[expected_accelerator, "CPUExecutionProvider"],
            )
        accelerated_output = accelerated.run(["y"], {"x": model_input})[0]
        if not numpy.allclose(accelerated_output, expected, rtol=0.0, atol=1e-6):
            raise AssertionError(
                f"Accelerated ONNX inference mismatch: {accelerated_output!r}"
            )
        profile_path = Path(accelerated.end_profiling())
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
        finally:
            profile_path.unlink(missing_ok=True)
        kernel_providers = sorted(
            {
                str(event.get("args", {}).get("provider", ""))
                for event in profile
                if event.get("args", {}).get("provider")
            }
        )
        if expected_accelerator not in kernel_providers:
            raise AssertionError(
                f"Tiny ONNX graph did not execute on {expected_accelerator}: "
                f"{kernel_providers}"
            )
        diagnostics["accelerator_kernel_providers"] = kernel_providers
    diagnostics["runtime_path"] = str(runtime_path)
    print("MTP_ORT_DIAGNOSTICS=" + json.dumps(diagnostics, sort_keys=True))


def _exercise_rohm_bundle(addon: object) -> None:
    configured = os.environ.get("MTP_ROHM_BUNDLE", "").strip()
    if not configured:
        return
    root = Path(configured).resolve(strict=True)
    bundle = addon.rohm_ort_runtime.load_bundle(
        root,
        expected_upstream_revision=addon.rohm_backend.MODEL_SPEC["upstream_revision"],
    )
    runtime = addon.dependency_bootstrap.import_onnxruntime()
    session = addon.rohm_ort_runtime.RohmSession(
        bundle,
        "CPU",
        ort_module=runtime,
    )
    diagnostics = session.diagnostics
    if diagnostics.actual_ep != "CPUExecutionProvider":
        raise AssertionError(
            f"RoHM CPU smoke selected an unexpected provider: {diagnostics.actual_ep}"
        )
    print(
        "MTP_ROHM_RUNTIME_OK="
        + json.dumps(
            {
                "assets": len(bundle.assets),
                "bytes": diagnostics.installed_bytes,
                "precision": diagnostics.precision,
                "provider": diagnostics.actual_ep,
                "upstream_revision": diagnostics.upstream_revision,
            },
            sort_keys=True,
        )
    )


def main() -> None:
    addon = _assert_registered()
    if "--verify-only" not in sys.argv:
        addon = _toggle_registration()
        _exercise_fake_capture(addon)
    cv2, mediapipe, _numpy = addon.dependency_bootstrap.import_runtime_dependencies()
    _assert_native_dependency_origins(cv2, mediapipe)
    _exercise_onnxruntime(addon)
    _exercise_rohm_bundle(addon)
    if addon.operation.COORDINATOR.is_busy():
        raise AssertionError("Release smoke left an operation lease active")
    print("MTP_RELEASE_SMOKE_OK")


if __name__ == "__main__":
    main()
