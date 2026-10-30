"""Download, validate, and own the optional immutable AnyCalib ONNX bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
from . import runtime as anycalib_ort_runtime

HF_ENDPOINT = lifecycle.HF_ENDPOINT
MODEL_STORAGE_FOLDER = "media_to_pose_anycalib"
MODEL_STORAGE_MARKER = lifecycle.STORAGE_MARKER
# Filled with the immutable commit and exact bytes after the publication gate passes.
MODEL_SPEC = {
    "repo": "KitsuMate/anycalib-onnx",
    "revision": "89e2ff96320bc21d6400575916e291143214c696",
    "upstream_revision": anycalib_ort_runtime.UPSTREAM_REVISION,
    "variants": {
        "fp16": {
            "manifest_path": "onnx/fp16/manifest.json",
            "download_bytes": 640_916_741,
        },
        "fp32": {
            "manifest_path": "onnx/fp32/manifest.json",
            "download_bytes": 1_281_312_183,
        },
    },
}
STORAGE = ManagedStorage(MODEL_STORAGE_FOLDER, "anycalib_models", "AnyCalib")


@dataclass(frozen=True)
class BackendSnapshot:
    installed: bool = False
    state: str = "NOT_INSTALLED"
    message: str = "AnyCalib is not installed"
    requested_mode: str = "AUTO"
    actual_ep: str = ""
    actual_device: str = ""
    fallback: bool = False
    precision: str = ""
    download_bytes: int = 0
    installed_bytes: int = 0
    model_directory: str = ""


class AnyCalibBackend(ModelBackend):
    label = "AnyCalib"
    install_operation = OperationKind.ANYCALIB_INSTALL
    load_operation = OperationKind.ANYCALIB_MODEL_LOAD
    unload_operation = OperationKind.ANYCALIB_MODEL_UNLOAD
    uninstall_operation = OperationKind.ANYCALIB_UNINSTALL


BACKEND = AnyCalibBackend()


def is_published() -> bool:
    variants = MODEL_SPEC.get("variants") or {}
    return bool(
        MODEL_SPEC.get("repo")
        and any(spec.get("manifest_path") for spec in variants.values())
    )


def backend_root(*, create: bool = True) -> Path:
    return lifecycle.backend_root("anycalib", create=create)


def storage_config_path(*, create: bool = True) -> Path:
    return backend_root(create=create) / "storage.json"


def configured_storage_parent() -> Path:
    configured = lifecycle.read_storage_parent(storage_config_path(create=False))
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
        label="AnyCalib download",
    )


def _preferred_precision(requested_mode: object) -> str:
    requested = sam3d_ort_runtime.normalize_mode(requested_mode)
    if requested == "CPU":
        return "fp32"
    try:
        ort = sam3d_ort_runtime.import_onnxruntime()
        available = sam3d_ort_runtime.available_execution_providers(ort)
        _provider, device, fallback = sam3d_ort_runtime.select_execution_provider(
            requested, available
        )
    except (MtpError, OSError):
        return "fp32"
    return "fp16" if device == "webgpu" and not fallback else "fp32"


def pinned_bundle(precision: str) -> PinnedBundle:
    def list_files(staging: Path) -> list[BundleFile]:
        graph = anycalib_ort_runtime.load_bundle_manifest(staging).graph
        return [BundleFile(graph.path, graph.bytes, graph.sha256)]

    return PinnedBundle(
        label="AnyCalib",
        repository=str(MODEL_SPEC["repo"]),
        revision=str(MODEL_SPEC.get("revision") or ""),
        manifest_path=str(MODEL_SPEC["variants"][precision]["manifest_path"]),
        list_files=list_files,
        verify=anycalib_ort_runtime.load_bundle,
    )


def install_status_snapshot() -> JobSnapshot:
    return BACKEND.install_status_snapshot()


def model_load_status_snapshot() -> JobSnapshot:
    return BACKEND.model_load_status_snapshot()


def invalidate_backend_snapshot() -> None:
    BACKEND.invalidate()


def start_install(
    storage_parent: str = "",
    device: object = "AUTO",
    *,
    online_access: bool = True,
) -> BackgroundJob:
    """Start downloading the precision that suits ``device`` on a worker thread.

    ``online_access`` is ``bpy.app.online_access``.
    """
    lifecycle.require_online_access(online_access, "AnyCalib")
    if not is_published():
        raise ModelMissing("AnyCalib ONNX bundle is not published")
    root = model_storage_root(storage_parent)
    precision = _preferred_precision(device)
    config = storage_config_path()

    def work(job: BackgroundJob) -> str:
        try:
            lifecycle.install_pinned_bundle(
                job,
                pinned_bundle(precision),
                STORAGE,
                root,
                config,
                online_access=online_access,
            )
        finally:
            invalidate_backend_snapshot()
        job.update(1.0, f"AnyCalib {precision.upper()} installed", "Complete")
        return "AnyCalib installed and verified"

    return BACKEND.start_job(
        JobKind.INSTALL,
        OperationKind.ANYCALIB_INSTALL,
        "anycalib-install",
        lifecycle.run_job,
        (work,),
        label="Starting AnyCalib download",
        kwargs={
            "describe_error": lambda exc, _phase: f"AnyCalib installation failed: {exc}"
        },
    )


def cancel_install() -> bool:
    return BACKEND.cancel_job(JobKind.INSTALL, "Cancelling AnyCalib download")


def _build_snapshot(requested: str) -> BackendSnapshot:
    preferred = _preferred_precision(requested)
    if not is_published():
        return BackendSnapshot(
            state="UNPUBLISHED",
            message="AnyCalib bundle is awaiting publication",
            requested_mode=requested,
            precision=preferred,
            model_directory=str(active_directory()),
        )
    session = BACKEND.session
    if session is not None:
        diagnostics = session.diagnostics
        return BackendSnapshot(
            installed=True,
            state="LOADED",
            message=f"AnyCalib · {diagnostics.actual_device.upper()} · Loaded",
            requested_mode=diagnostics.requested_mode,
            actual_ep=diagnostics.actual_ep,
            actual_device=diagnostics.actual_device,
            fallback=diagnostics.fallback,
            precision=diagnostics.precision,
            download_bytes=diagnostics.download_bytes,
            installed_bytes=diagnostics.installed_bytes,
            model_directory=str(active_directory()),
        )
    if not manifest_path().is_file():
        return BackendSnapshot(
            state="NOT_INSTALLED",
            message="AnyCalib model is not installed",
            requested_mode=requested,
            precision=preferred,
            download_bytes=int(MODEL_SPEC["variants"][preferred]["download_bytes"]),
            model_directory=str(active_directory()),
        )
    try:
        bundle = anycalib_ort_runtime.load_bundle(
            active_directory(), verify_files=False
        )
        ort = sam3d_ort_runtime.import_onnxruntime()
        provider, device, fallback = sam3d_ort_runtime.select_execution_provider(
            requested, sam3d_ort_runtime.available_execution_providers(ort)
        )
    except (MtpError, OSError, ValueError) as exc:
        return BackendSnapshot(
            state="ERROR",
            message=f"AnyCalib is invalid; use Repair ({exc})",
            requested_mode=requested,
            model_directory=str(active_directory()),
        )
    return BackendSnapshot(
        installed=True,
        state="READY",
        message=f"AnyCalib · {bundle.precision.upper()} · Ready",
        requested_mode=requested,
        actual_ep=provider,
        actual_device=device,
        fallback=fallback,
        precision=bundle.precision,
        download_bytes=bundle.download_bytes,
        installed_bytes=bundle.graph.bytes,
        model_directory=str(active_directory()),
    )


def backend_snapshot(requested_mode: object = "AUTO") -> BackendSnapshot:
    requested = sam3d_ort_runtime.normalize_mode(requested_mode)
    return BACKEND.cached_snapshot((requested,), lambda: _build_snapshot(requested))


def ensure_model_loaded(
    device: object = "AUTO",
    bundle_directory: str = "",
) -> anycalib_ort_runtime.AnyCalibSession:
    requested = sam3d_ort_runtime.normalize_mode(device)
    current = BACKEND.session
    if current is not None and current.diagnostics.requested_mode == requested:
        return current
    previous = BACKEND.take_session()
    if previous is not None:
        previous.close()
    bundle = anycalib_ort_runtime.load_bundle(
        Path(bundle_directory) if bundle_directory else active_directory()
    )
    session = anycalib_ort_runtime.AnyCalibSession(bundle, requested)
    active = BACKEND.set_session(session)
    if active is not session:
        session.close()
    BACKEND.last_diagnostic = "Bundle verification and runtime session creation passed"
    return active


def is_model_loaded() -> bool:
    return BACKEND.is_model_loaded()


def unload_model() -> None:
    session = BACKEND.take_session()
    if session is not None:
        session.close()


def start_load(device: object = "AUTO") -> BackgroundJob:
    bundle_directory = str(active_directory())

    def work(job: BackgroundJob) -> str:
        job.update(0.2, "Verifying AnyCalib", "Verification")
        ensure_model_loaded(device, bundle_directory)
        if job.cancel.is_set():
            unload_model()
            raise Cancelled("AnyCalib load cancelled")
        job.update(1.0, "AnyCalib loaded", "Complete")
        return "AnyCalib loaded"

    def describe(exc: BaseException, _phase: str) -> str:
        BACKEND.last_diagnostic = str(exc)
        return f"AnyCalib loading failed: {exc}"

    return BACKEND.start_job(
        JobKind.MODEL,
        OperationKind.ANYCALIB_MODEL_LOAD,
        "anycalib-load",
        lifecycle.run_job,
        (work,),
        label="Loading AnyCalib",
        kwargs={"describe_error": describe},
    )


def unload() -> str:
    def action() -> str:
        unload_model()
        return "AnyCalib unloaded"

    return BACKEND.run_exclusive(
        OperationKind.ANYCALIB_MODEL_UNLOAD,
        "anycalib-unload",
        action,
        status="Unloading AnyCalib",
    )


def uninstall() -> str:
    def action() -> str:
        root = model_storage_root().resolve(False)
        with lifecycle.model_installation_lock(root, "AnyCalib"):
            unload_model()
            try:
                STORAGE.remove(root)
            finally:
                invalidate_backend_snapshot()
        return "AnyCalib uninstalled"

    return BACKEND.run_exclusive(
        OperationKind.ANYCALIB_UNINSTALL,
        "anycalib-uninstall",
        action,
        status="Uninstalling AnyCalib",
    )


def shutdown() -> None:
    BACKEND.shutdown()
    unload_model()


__all__ = [
    "BACKEND",
    "MODEL_SPEC",
    "BackendSnapshot",
    "active_directory",
    "backend_present",
    "backend_snapshot",
    "bundle_url",
    "cancel_install",
    "configured_storage_parent",
    "ensure_model_loaded",
    "install_status_snapshot",
    "is_model_loaded",
    "is_published",
    "model_load_status_snapshot",
    "model_storage_root",
    "shutdown",
    "start_install",
    "start_load",
    "uninstall",
    "unload",
    "unload_model",
]
