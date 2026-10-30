"""Inspect Python dependencies activated by Blender's extension manager."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from .errors import DependencyMissing

REQUIRED_MODULES = ("mediapipe", "cv2", "onnxruntime")
EXPECTED_ONNX_ACCELERATOR = {
    "win32": "WebGpuExecutionProvider",
    "linux": "WebGpuExecutionProvider",
    "darwin": "CoreMLExecutionProvider",
}


@dataclass(frozen=True)
class DependencySnapshot:
    checked: bool = False
    available: bool = False
    message: str = "Dependency status has not been checked"


_DEPENDENCY_SNAPSHOT = DependencySnapshot()


def dependency_snapshot() -> DependencySnapshot:
    """Return the last probe without importing or extracting from UI draw."""
    return _DEPENDENCY_SNAPSHOT


def _publish_snapshot(available: bool, message: str) -> tuple[bool, str]:
    global _DEPENDENCY_SNAPSHOT
    _DEPENDENCY_SNAPSHOT = DependencySnapshot(
        checked=True,
        available=bool(available),
        message=str(message),
    )
    return available, message


def _module_is_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def ensure_bundled_dependencies() -> tuple[bool, str]:
    """Report whether Blender activated the extension's declared wheels."""
    if all(_module_is_available(name) for name in REQUIRED_MODULES):
        return _publish_snapshot(
            True, "Blender-managed dependency wheels are available"
        )
    missing = [name for name in REQUIRED_MODULES if not _module_is_available(name)]
    return _publish_snapshot(
        False,
        "Blender did not activate the packaged dependency wheels: "
        + ", ".join(missing)
        + ". Reinstall and enable the complete Media to Pose extension.",
    )


def refresh_dependency_snapshot() -> DependencySnapshot:
    ensure_bundled_dependencies()
    return dependency_snapshot()


def bundled_runtime_path() -> str:
    """Return dependency roots suitable for a helper subprocess PYTHONPATH."""
    available, _message = ensure_bundled_dependencies()
    if not available:
        return ""
    roots = []
    for name in REQUIRED_MODULES:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        if not spec:
            continue
        locations = list(spec.submodule_search_locations or ())
        if locations:
            root = str(Path(locations[0]).resolve().parent)
        elif spec.origin:
            root = str(Path(spec.origin).resolve().parent)
        else:
            continue
        if root not in roots:
            roots.append(root)
    return os.pathsep.join(roots)


def import_runtime_dependencies():
    """Return imported cv2, MediaPipe, and Blender's NumPy."""
    available, message = ensure_bundled_dependencies()
    if not available:
        raise DependencyMissing(message)
    try:
        cv2 = importlib.import_module("cv2")
        mediapipe = importlib.import_module("mediapipe")
        numpy = importlib.import_module("numpy")
    except Exception as exc:
        raise DependencyMissing(
            f"Bundled MediaPipe dependencies could not be loaded: {exc}"
        ) from exc
    return cv2, mediapipe, numpy


def import_camera_dependencies():
    """Import only OpenCV and NumPy for model-free camera tracking."""
    missing = [name for name in ("cv2", "numpy") if not _module_is_available(name)]
    if missing:
        raise DependencyMissing(
            "Blender did not activate the packaged camera-tracking wheels: "
            + ", ".join(missing)
        )
    try:
        return importlib.import_module("cv2"), importlib.import_module("numpy")
    except Exception as exc:
        raise DependencyMissing(
            f"Camera-tracking dependencies could not be loaded: {exc}"
        ) from exc


def import_onnxruntime():
    """Import the platform-bundled ONNX Runtime distribution."""
    available, message = ensure_bundled_dependencies()
    if not available:
        raise DependencyMissing(message)
    try:
        return importlib.import_module("onnxruntime")
    except Exception as exc:
        raise DependencyMissing(
            f"Bundled ONNX Runtime could not be loaded: {exc}"
        ) from exc


def import_tokenizers():
    """Import Hugging Face ``tokenizers`` for the optional LLM2Vec encoder.

    Not in ``REQUIRED_MODULES``: the wheels are bundled, but only prompts
    need them, so capture keeps working when they fail to activate.
    """
    if not _module_is_available("tokenizers"):
        raise DependencyMissing(
            "Prompts need the bundled 'tokenizers' package, which Blender did "
            "not activate; reinstall Media to Pose or generate without a prompt"
        )
    try:
        return importlib.import_module("tokenizers")
    except Exception as exc:
        raise DependencyMissing(f"'tokenizers' could not be loaded: {exc}") from exc


def onnxruntime_diagnostics() -> dict[str, object]:
    """Describe the bundled runtime without attempting hardware inference."""
    runtime = import_onnxruntime()
    available_providers = tuple(runtime.get_available_providers())
    return {
        "version": str(runtime.__version__),
        "available_providers": available_providers,
        "expected_accelerator": EXPECTED_ONNX_ACCELERATOR.get(sys.platform),
        "platform": sys.platform,
    }
