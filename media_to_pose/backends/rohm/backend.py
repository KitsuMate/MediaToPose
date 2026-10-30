"""Download, validate, and own the optional immutable RoHM ONNX bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ...core.coordinator import OperationKind
from ...core.errors import Cancelled, ModelMissing, MtpError
from .. import lifecycle
from ..lifecycle import (
    BackgroundJob,
    BundleFile,
    JobKind,
    JobSnapshot,
    ManagedStorage,
    ModelBackend,
    PinnedBundle,
)
from ..sam3d import backend as sam3d_backend
from ..sam3d import ort_runtime as sam3d_ort_runtime
from . import runtime as rohm_ort_runtime

HF_ENDPOINT = lifecycle.HF_ENDPOINT
MODEL_STORAGE_FOLDER = "media_to_pose_rohm"
MODEL_STORAGE_MARKER = lifecycle.STORAGE_MARKER
MODEL_SPEC = {
    "repo": "KitsuMate/rohm-onnx",
    "revision": "21bb4c53d1265c522e3962a466c4bb9e41169a36",
    "upstream_revision": "57ba22cde1595285276ddb88c8c82b25bf0d9722",
    "manifest_path": "bundles/fp32/manifest.json",
    "precision": "fp32",
    "download_bytes": 222_030_103,
}
STORAGE = ManagedStorage(MODEL_STORAGE_FOLDER, "rohm_models", "RoHM")


@dataclass(frozen=True)
class BackendSnapshot:
    installed: bool = False
    state: str = "NOT_INSTALLED"
    message: str = "RoHM is not installed"
    requested_mode: str = "AUTO"
    actual_ep: str = ""
    actual_device: str = ""
    fallback: bool = False
    precision: str = ""
    download_bytes: int = 0
    installed_bytes: int = 0
    upstream_revision: str = ""
    model_directory: str = ""


class RohmBackend(ModelBackend):
    label = "RoHM"
    install_operation = OperationKind.ROHM_INSTALL
    load_operation = OperationKind.ROHM_MODEL_LOAD
    unload_operation = OperationKind.ROHM_MODEL_UNLOAD
    uninstall_operation = OperationKind.ROHM_UNINSTALL


BACKEND = RohmBackend()


def rohm_device_enum_items():
    return (
        ("AUTO", "Auto", "Prefer GPU acceleration, then fall back to CPU"),
        ("GPU", "GPU", "Require WebGPU or CoreML acceleration"),
        ("CPU", "CPU", "Use ONNX Runtime CPU inference"),
    )


def is_published() -> bool:
    return bool(MODEL_SPEC.get("repo") and MODEL_SPEC.get("manifest_path"))


def backend_root(*, create: bool = True) -> Path:
    return lifecycle.backend_root("rohm", create=create)


def _config_path(*, create: bool = True) -> Path:
    return backend_root(create=create) / "storage.json"


def configured_storage_parent() -> Path:
    configured = lifecycle.read_storage_parent(_config_path(create=False))
    if configured:
        return lifecycle.resolve_user_path(configured)
    return sam3d_backend.preferred_model_storage_parent().resolve(False)


def model_storage_root(parent: str = "") -> Path:
    selected = (
        lifecycle.resolve_user_path(parent) if parent else configured_storage_parent()
    )
    return STORAGE.root(selected)


def active_directory() -> Path:
    return model_storage_root() / "active"


def manifest_path() -> Path:
    return active_directory() / "manifest.json"


def backend_present() -> bool:
    return manifest_path().is_file()


def bundle_url() -> str:
    return f"{HF_ENDPOINT}/{MODEL_SPEC['repo']}"


def _hf_url(path: str, revision: str = "") -> str:
    return lifecycle.hf_resolve_url(
        str(MODEL_SPEC["repo"]),
        revision or str(MODEL_SPEC.get("revision") or "main"),
        path,
        label="RoHM download",
    )


def pinned_bundle() -> PinnedBundle:
    def list_files(staging: Path) -> list[BundleFile]:
        bundle = rohm_ort_runtime.load_bundle(staging, verify_files=False)
        return [
            BundleFile(str(PurePosixPath(item.path)), item.bytes, item.sha256)
            for item in bundle.assets.values()
        ]

    return PinnedBundle(
        label="RoHM",
        repository=str(MODEL_SPEC["repo"]),
        revision=str(MODEL_SPEC.get("revision") or ""),
        manifest_path=str(MODEL_SPEC["manifest_path"]),
        list_files=list_files,
        verify=rohm_ort_runtime.load_bundle,
    )


def install_status_snapshot() -> JobSnapshot:
    return BACKEND.install_status_snapshot()


def model_load_status_snapshot() -> JobSnapshot:
    return BACKEND.model_load_status_snapshot()


def invalidate_backend_snapshot() -> None:
    BACKEND.invalidate()


def start_install(
    storage_parent: str = "", *, online_access: bool = True
) -> BackgroundJob:
    """Start downloading the pinned RoHM bundle on a worker thread.

    ``online_access`` is ``bpy.app.online_access``.
    """
    lifecycle.require_online_access(online_access, "RoHM")
    if not is_published():
        raise ModelMissing("RoHM ONNX bundle is not published")
    root = model_storage_root(storage_parent)
    config = _config_path()

    def work(job: BackgroundJob) -> str:
        try:
            return lifecycle.install_pinned_bundle(
                job,
                pinned_bundle(),
                STORAGE,
                root,
                config,
                online_access=online_access,
            )
        finally:
            invalidate_backend_snapshot()

    return BACKEND.start_job(
        JobKind.INSTALL,
        OperationKind.ROHM_INSTALL,
        "rohm-install",
        lifecycle.run_job,
        (work,),
        label="Starting RoHM download",
        kwargs={
            "describe_error": lambda exc, _phase: f"RoHM installation failed: {exc}"
        },
    )


def cancel_install() -> bool:
    return BACKEND.cancel_job(JobKind.INSTALL, "Cancelling RoHM download")


def _build_snapshot(requested: str) -> BackendSnapshot:
    session = BACKEND.session
    if session is not None:
        diagnostic = session.diagnostics
        return BackendSnapshot(
            installed=True,
            state="LOADED",
            message="RoHM · Loaded",
            requested_mode=diagnostic.requested_mode,
            actual_ep=diagnostic.actual_ep,
            actual_device=diagnostic.actual_device,
            fallback=diagnostic.fallback,
            precision=diagnostic.precision,
            download_bytes=diagnostic.download_bytes,
            installed_bytes=diagnostic.installed_bytes,
            upstream_revision=diagnostic.upstream_revision,
            model_directory=str(active_directory()),
        )
    if not manifest_path().is_file():
        return BackendSnapshot(
            state="NOT_INSTALLED",
            message="RoHM is not installed",
            requested_mode=requested,
            precision=str(MODEL_SPEC["precision"]),
            download_bytes=int(MODEL_SPEC["download_bytes"]),
            model_directory=str(active_directory()),
        )
    try:
        bundle = rohm_ort_runtime.load_bundle(active_directory(), verify_files=False)
        ort = sam3d_ort_runtime.import_onnxruntime()
        available = sam3d_ort_runtime.available_execution_providers(ort)
        provider, actual_device, fallback = sam3d_ort_runtime.select_execution_provider(
            requested, available
        )
    except (MtpError, OSError, ValueError) as exc:
        return BackendSnapshot(
            state="ERROR",
            message=f"RoHM is invalid; use Repair ({exc})",
            requested_mode=requested,
            download_bytes=int(MODEL_SPEC["download_bytes"]),
            model_directory=str(active_directory()),
        )
    installed_bytes = sum(item.bytes for item in bundle.assets.values())
    return BackendSnapshot(
        installed=True,
        state="READY",
        message=(
            f"RoHM · {actual_device.upper()}"
            + (" · CPU fallback" if fallback else "")
            + " · Ready"
        ),
        requested_mode=requested,
        actual_ep=provider,
        actual_device=actual_device,
        fallback=fallback,
        precision=bundle.precision,
        download_bytes=bundle.download_bytes,
        installed_bytes=installed_bytes,
        upstream_revision=bundle.upstream_revision,
        model_directory=str(active_directory()),
    )


def backend_snapshot(requested_mode: object = "AUTO") -> BackendSnapshot:
    requested = sam3d_ort_runtime.normalize_mode(requested_mode)
    return BACKEND.cached_snapshot((requested,), lambda: _build_snapshot(requested))


def verify_bundle(bundle_directory: str = "") -> BackendSnapshot:
    invalidate_backend_snapshot()
    directory = Path(bundle_directory) if bundle_directory else active_directory()
    if not (directory / "manifest.json").is_file():
        raise ModelMissing("RoHM is not installed")
    rohm_ort_runtime.load_bundle(directory)
    invalidate_backend_snapshot()
    if bundle_directory:
        return BackendSnapshot(
            installed=True,
            state="READY",
            message="RoHM bundle verified",
            model_directory=str(directory),
        )
    return backend_snapshot("AUTO")


def loaded_session() -> rohm_ort_runtime.RohmSession:
    session = BACKEND.session
    if session is None:
        raise ModelMissing("RoHM is not loaded")
    return session


def is_model_loaded() -> bool:
    return BACKEND.is_model_loaded()


def ensure_model_loaded(
    device: object,
    bundle_directory: str = "",
) -> rohm_ort_runtime.RohmSession:
    requested = sam3d_ort_runtime.normalize_mode(device)
    current = BACKEND.session
    if current is not None and current.diagnostics.requested_mode == requested:
        return current
    BACKEND.take_session()
    directory = Path(bundle_directory) if bundle_directory else active_directory()
    verify_bundle(str(directory))
    bundle = rohm_ort_runtime.load_bundle(directory)
    session = BACKEND.set_session(rohm_ort_runtime.RohmSession(bundle, requested))
    BACKEND.last_diagnostic = "Bundle verification and runtime session creation passed"
    return session


def start_load(device: object) -> BackgroundJob:
    bundle_directory = str(active_directory())

    def work(job: BackgroundJob) -> str:
        job.update(0.15, "Verifying model files", "Verification")
        verify_bundle(bundle_directory)
        job.check_cancelled("RoHM load cancelled")
        job.update(0.45, "Creating ONNX Runtime sessions", "Loading")
        ensure_model_loaded(device, bundle_directory)
        if job.cancel.is_set():
            unload_model()
            raise Cancelled("RoHM load cancelled")
        job.update(1.0, "RoHM loaded", "Complete")
        return "RoHM loaded"

    def describe(exc: BaseException, _phase: str) -> str:
        BACKEND.last_diagnostic = str(exc)
        return f"RoHM loading failed: {exc}"

    return BACKEND.start_job(
        JobKind.MODEL,
        OperationKind.ROHM_MODEL_LOAD,
        "rohm-load",
        lifecycle.run_job,
        (work,),
        label="Loading RoHM",
        kwargs={"describe_error": describe},
    )


def unload_model() -> None:
    BACKEND.take_session()


def last_diagnostic() -> str:
    return BACKEND.last_diagnostic


def verify(device: object) -> str:
    """Check every file and create a runtime session, then restore load state."""

    def action() -> str:
        was_loaded = is_model_loaded()
        try:
            snapshot = verify_bundle()
            ensure_model_loaded(device)
            BACKEND.last_diagnostic = (
                "All declared files passed size/SHA-256 verification and "
                "runtime session creation passed"
            )
        except (MtpError, OSError, ValueError) as exc:
            BACKEND.last_diagnostic = str(exc)
            raise
        finally:
            if not was_loaded:
                unload_model()
        return snapshot.message

    return BACKEND.run_exclusive(
        OperationKind.ROHM_MODEL_LOAD, "rohm-verify", action, status="Verifying RoHM"
    )


def unload() -> str:
    def action() -> str:
        unload_model()
        return "RoHM unloaded"

    return BACKEND.run_exclusive(
        OperationKind.ROHM_MODEL_UNLOAD,
        "rohm-unload",
        action,
        status="Unloading RoHM",
    )


def uninstall() -> str:
    def action() -> str:
        root = model_storage_root().resolve(False)
        with lifecycle.model_installation_lock(root, "RoHM"):
            unload_model()
            try:
                STORAGE.remove(root)
            finally:
                invalidate_backend_snapshot()
        return "RoHM uninstalled"

    return BACKEND.run_exclusive(
        OperationKind.ROHM_UNINSTALL,
        "rohm-uninstall",
        action,
        status="Uninstalling RoHM",
    )


def shutdown() -> None:
    BACKEND.shutdown()
    unload_model()


__all__ = [
    "BACKEND",
    "MODEL_SPEC",
    "BackendSnapshot",
    "backend_present",
    "backend_snapshot",
    "bundle_url",
    "cancel_install",
    "ensure_model_loaded",
    "install_status_snapshot",
    "is_model_loaded",
    "is_published",
    "last_diagnostic",
    "loaded_session",
    "model_load_status_snapshot",
    "rohm_device_enum_items",
    "shutdown",
    "start_install",
    "start_load",
    "uninstall",
    "unload",
    "verify",
]
