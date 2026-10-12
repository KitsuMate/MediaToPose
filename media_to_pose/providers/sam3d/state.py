"""The loaded SAM 3D Body estimator, shared by every capture and the model panel."""

from __future__ import annotations

import gc
import threading
from contextlib import suppress

from ...backends.sam3d import backend as sam3d_backend
from ...backends.sam3d import ort_runtime as sam3d_ort_runtime

_MODEL_LOCK = threading.RLock()
_MODEL_KEY = None
_ESTIMATOR = None
_RUNTIME_DIAGNOSTICS = None


def model_loaded() -> bool:
    return _ESTIMATOR is not None


def unload_model_cache() -> None:
    global _MODEL_KEY, _ESTIMATOR, _RUNTIME_DIAGNOSTICS
    with _MODEL_LOCK:
        estimator = _ESTIMATOR
        _MODEL_KEY = None
        _ESTIMATOR = None
        _RUNTIME_DIAGNOSTICS = None
        if estimator is not None:
            with suppress(AttributeError, RuntimeError):
                estimator.close()
        gc.collect()


def runtime_diagnostics():
    """Return immutable diagnostics for the currently loaded ORT session."""
    return _RUNTIME_DIAGNOSTICS


def _refresh_runtime_diagnostics(estimator):
    """Publish the estimator's latest immutable diagnostics snapshot."""
    global _RUNTIME_DIAGNOSTICS
    diagnostics = estimator.diagnostics
    with _MODEL_LOCK:
        # An inference may finish while another thread replaces the shared
        # estimator.  Never let the retired estimator overwrite the new
        # cache's diagnostics in that case.
        if estimator is _ESTIMATOR:
            _RUNTIME_DIAGNOSTICS = diagnostics
    return diagnostics


def dependency_status(options=None):
    options = options or {}
    variant = str(options.get("variant", "VITH"))
    device = str(options.get("device", "AUTO"))
    snapshot = sam3d_backend.backend_snapshot(variant, device)
    return snapshot.installed, snapshot.message


def cache_fingerprint(options):
    variant = str(options.get("variant", "VITH"))
    spec = sam3d_backend.model_spec(variant)
    return {
        "adapter_version": sam3d_backend.ADAPTER_VERSION,
        "bundle_contract": sam3d_ort_runtime.BUNDLE_CONTRACT,
        "model_repo": spec["repo"],
        "bundle_revision": spec["revision"],
        "model_revision": spec.get("model_revision", ""),
        "model_precision": spec.get("precision", ""),
        "model_variant": variant,
    }


def _load_model(
    variant: str,
    requested_device: str,
    paths: dict[str, str],
    *,
    verify_accelerator_execution: bool = False,
):
    global _MODEL_KEY, _ESTIMATOR, _RUNTIME_DIAGNOSTICS
    with _MODEL_LOCK:
        requested_mode = sam3d_ort_runtime.normalize_mode(requested_device)
        verify_accelerator_execution = bool(verify_accelerator_execution)
        key = (
            variant,
            requested_mode,
            paths.get("backend_directory", ""),
            paths.get("bundle_manifest_path", ""),
            paths.get("bundle_revision", ""),
            paths.get("model_revision", ""),
            verify_accelerator_execution,
        )
        if _ESTIMATOR is not None and key == _MODEL_KEY:
            return (
                _ESTIMATOR,
                _RUNTIME_DIAGNOSTICS,
                _RUNTIME_DIAGNOSTICS.actual_device,
            )
        unload_model_cache()
        estimator = sam3d_ort_runtime.load_estimator(
            paths,
            variant,
            requested_mode,
            verify_accelerator_execution=verify_accelerator_execution,
        )
        _ESTIMATOR = estimator
        _MODEL_KEY = key
        _RUNTIME_DIAGNOSTICS = estimator.diagnostics
        return estimator, estimator.diagnostics, estimator.diagnostics.actual_device


def load_model_cache(
    variant: str,
    requested_device: str,
    paths: dict[str, str] | None = None,
    *,
    verify_accelerator_execution: bool = False,
) -> str:
    """Load and retain the selected estimator without starting capture."""
    _estimator, _diagnostics, device_name = _load_model(
        variant,
        requested_device,
        paths or sam3d_backend.runtime_paths(),
        verify_accelerator_execution=verify_accelerator_execution,
    )
    return device_name
