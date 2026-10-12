"""Optional SAM 3D Body model installation and runtime state."""

from __future__ import annotations

import platform
import re
import shutil
import sys
import time
import urllib.error
import uuid
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from ...constants import CAPTURE_CONFIDENCE, extension_version
from ...core import paths
from ...core.errors import BackendError, Cancelled, InvalidRequest, ModelMissing
from ...core.fs import atomic_json, read_json, remove_tree
from ...core.log import get_logger
from ...core.requests import SamSetupRequest
from ...operation import OperationKind
from .. import lifecycle
from ..lifecycle import (
    BackgroundJob,
    InstallationLock,
    JobKind,
    JobSnapshot,
    ManagedStorage,
    ModelBackend,
)
from . import ort_runtime as sam3d_ort_runtime

LOGGER = get_logger(__name__)

RUNTIME_MANIFEST_VERSION = 4
BACKEND_VERSION = RUNTIME_MANIFEST_VERSION
ADAPTER_VERSION = "2.0.0"
HF_ENDPOINT = lifecycle.HF_ENDPOINT
MIN_RUNTIME_FREE_BYTES = 256 * 1024 * 1024
MODEL_DOWNLOAD_HEADROOM_BYTES = 512 * 1024 * 1024
MODEL_STORAGE_FOLDER = "media_to_pose_sam3d"
MODEL_STORAGE_MARKER = lifecycle.STORAGE_MARKER
STALE_INSTALL_LOCK_SECONDS = 24 * 60 * 60
STORAGE = ManagedStorage(
    MODEL_STORAGE_FOLDER,
    "sam3d_models",
    "SAM 3D Body",
    marker_version=1,
    private=True,
)

MODEL_VARIANTS = {
    "VITH": {
        "label": "ViT-H",
        "repo": "KitsuMate/sam-3d-body-onnx",
        "revision": "1bcab38d403557b22523b228d630e20f7e8e62a0",
        # This identifies Meta's immutable source checkpoint.  It is separate
        # from ``revision`` above: a manifest cannot contain the hash of the
        # Hugging Face commit that will be created by uploading that manifest.
        "model_revision": "f7f4ba4f67739e297f291639afb7c59041745367",
        "manifest_path": "onnx/vith/manifest.json",
        "precision": "mixed-fp16-fp32",
        "download_bytes": 1_761_595_137,
    }
}

_DEVICE_ENUM_ITEMS: tuple[tuple[str, str, str], ...] = (
    (
        "AUTO",
        "Auto",
        "Prefer the platform GPU execution provider, then fall back to CPU",
    ),
    ("GPU", "GPU", "Require WebGPU/CoreML acceleration"),
    ("CPU", "CPU", "Use ONNX Runtime CPU inference"),
)


@dataclass(frozen=True)
class BackendSnapshot:
    installed: bool
    message: str
    variant: str
    device: str
    model_revision: str = ""
    model_directory: str = ""
    runtime_build: str = ""
    gpu_name: str = ""
    requested_mode: str = "AUTO"
    actual_ep: str = ""
    actual_device: str = ""
    fallback: bool = False
    accelerator_nodes_verified: bool = False
    precision: str = ""
    hardware_vendor_id: int = 0
    hardware_device_id: int = 0


InstallStatusSnapshot = JobSnapshot


@dataclass(frozen=True)
class _InstallLayout:
    root: Path
    staging: Path
    backup: Path
    storage_parent: Path
    storage_root: Path
    model_staging: Path
    model_current: Path
    model_backup: Path
    previous_storage_root: Path | None
    storage_config: Path | None = None


class SamBackend(ModelBackend):
    label = "SAM 3D Body"
    install_operation = OperationKind.SAM3D_INSTALL
    load_operation = OperationKind.SAM3D_MODEL_LOAD
    unload_operation = OperationKind.SAM3D_MODEL_UNLOAD
    uninstall_operation = OperationKind.SAM3D_UNINSTALL

    def is_model_loaded(self) -> bool:
        # The ONNX estimator lives in the provider module, not in the
        # generic session slot.
        return is_model_loaded()


BACKEND = SamBackend()

_redact = lifecycle.redact
_safe_metadata = lifecycle.safe_metadata


def backend_root(*, create: bool = True) -> Path:
    return lifecycle.backend_root("sam3d", create=create)


def _verified_backend_root() -> Path:
    root = backend_root(create=False).resolve()
    user_root = paths.user_directory("", create=False).resolve()
    if user_root not in root.parents or root.name != "sam3d":
        raise BackendError("Refusing to remove an unexpected backend path")
    return root


def active_directory(*, create_root: bool = True) -> Path:
    return backend_root(create=create_root) / "active"


def install_log_path(*, create_root: bool = True) -> Path:
    return backend_root(create=create_root) / "install.log"


def manifest_path() -> Path:
    return active_directory(create_root=False) / "manifest.json"


def backend_present() -> bool:
    return manifest_path().is_file()


def model_spec(variant: str) -> dict:
    try:
        return MODEL_VARIANTS[str(variant)]
    except KeyError as exc:
        raise InvalidRequest(f"Unknown SAM 3D Body model variant: {variant!r}") from exc


def _read_json(path: Path) -> dict | None:
    value = read_json(path)
    return value if isinstance(value, dict) else None


def model_storage_config_path(*, create_root: bool = True) -> Path:
    return backend_root(create=create_root) / "model_storage.json"


def _normalize_storage_parent(value: str = "") -> Path:
    selected = (
        lifecycle.resolve_user_path(value)
        if value
        else backend_root(create=True).resolve(strict=False)
    )
    return STORAGE.root(selected).parent


def configured_model_storage_parent() -> Path:
    config = _read_json(model_storage_config_path(create_root=False)) or {}
    return _normalize_storage_parent(str(config.get("storage_parent") or ""))


def preferred_model_storage_parent() -> Path:
    """Return the storage parent presented by the SAM 3D install dialog."""
    manifest = _read_json(manifest_path()) or {}
    stored_parent = str(manifest.get("model_storage_parent") or "")
    return (
        _normalize_storage_parent(stored_parent)
        if stored_parent
        else configured_model_storage_parent()
    )


def model_storage_root(selection: str = "") -> Path:
    parent = (
        _normalize_storage_parent(selection)
        if selection
        else configured_model_storage_parent()
    )
    return parent / MODEL_STORAGE_FOLDER


def installed_model_directory(manifest: dict | None = None) -> Path:
    manifest = manifest if manifest is not None else (_read_json(manifest_path()) or {})
    configured = str(manifest.get("model_directory") or "")
    if configured:
        return Path(configured).resolve(strict=False)
    return active_directory(create_root=False) / "model"


def installed_model_storage_root(
    manifest: dict | None = None,
) -> Path | None:
    model = installed_model_directory(manifest)
    parent = model.parent
    if model.name == "active" and STORAGE.owns(parent):
        return parent
    return None


def _closest_existing_directory(path: Path) -> Path:
    candidate = path.resolve(strict=False)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise BackendError(f"No accessible parent directory exists for {path}")
    return candidate


def install_running() -> bool:
    job = BACKEND.job(JobKind.INSTALL)
    return bool(job and job.alive())


def install_status_snapshot() -> JobSnapshot:
    return BACKEND.install_status_snapshot()


def model_load_running() -> bool:
    return BACKEND.job_running(JobKind.MODEL)


def model_load_status_snapshot() -> JobSnapshot:
    return BACKEND.model_load_status_snapshot()


def sam_backend_ready(settings) -> bool:
    """Return whether the selected SAM runtime is complete and self-tested."""
    return backend_snapshot(
        str(settings.sam3d_model_variant),
        str(settings.sam3d_device),
    ).installed


def sam3d_device_enum_items() -> tuple[tuple[str, str, str], ...]:
    """Return the stable backend-agnostic AUTO/GPU/CPU choices."""
    return _DEVICE_ENUM_ITEMS


def backend_snapshot(variant: str, device: str) -> BackendSnapshot:
    """Return cached readiness so Blender panel redraws never rescan ONNX files."""
    return BACKEND.cached_snapshot(
        (str(variant), str(device)), lambda: _build_snapshot(variant, device)
    )


def _build_snapshot(variant: str, device: str) -> BackendSnapshot:
    requested_mode = sam3d_ort_runtime.normalize_mode(device)
    manifest = _read_json(manifest_path())
    if not manifest:
        return BackendSnapshot(
            False,
            "SAM 3D Body is not installed",
            variant,
            requested_mode,
            requested_mode=requested_mode,
        )
    selected = model_spec(variant)
    if (
        manifest.get("runtime_manifest_version") != RUNTIME_MANIFEST_VERSION
        or manifest.get("backend_version") != BACKEND_VERSION
        or manifest.get("backend_kind") != "onnxruntime"
    ):
        return BackendSnapshot(
            False,
            "Installed backend metadata is invalid; reinstall SAM 3D Body",
            variant,
            requested_mode,
            model_directory=str(installed_model_directory(manifest)),
            requested_mode=requested_mode,
        )
    if manifest.get("variant") != variant:
        return BackendSnapshot(
            False,
            f"{selected['label']} is not installed",
            variant,
            requested_mode,
            requested_mode=requested_mode,
        )
    model = installed_model_directory(manifest)
    bundle_relative = str(manifest.get("bundle_manifest") or "bundle.json")
    bundle_path = model / bundle_relative
    try:
        bundle = sam3d_ort_runtime.load_bundle_manifest(
            bundle_path,
            expected_variant=variant,
            expected_model_revision=str(manifest.get("model_revision") or ""),
            verify_files=False,
        )
        graph_complete = all(
            _bundle_destination(bundle.root, item.path).is_file()
            and _bundle_destination(bundle.root, item.path).stat().st_size == item.bytes
            for item in bundle.files
        )
    except (OSError, BackendError) as exc:
        graph_complete = False
        bundle_error = str(exc)
    else:
        bundle_error = ""
    if not graph_complete:
        return BackendSnapshot(
            False,
            "Installed ONNX graph bundle is incomplete; run Repair"
            + (f" ({bundle_error})" if bundle_error else ""),
            variant,
            requested_mode,
            model_revision=str(manifest.get("model_revision", "")),
            model_directory=str(model),
            requested_mode=requested_mode,
        )
    available = tuple(str(item) for item in manifest.get("available_providers", ()))
    if requested_mode == str(manifest.get("requested_mode") or ""):
        actual_ep = str(manifest.get("actual_ep") or "")
        actual_device = str(manifest.get("actual_device") or "")
        fallback = bool(manifest.get("fallback"))
    else:
        try:
            actual_ep, actual_device, fallback = (
                sam3d_ort_runtime.select_execution_provider(
                    requested_mode,
                    available,
                )
            )
        except sam3d_ort_runtime.OrtRuntimeError as exc:
            return BackendSnapshot(
                False,
                str(exc),
                variant,
                requested_mode,
                model_revision=bundle.model_revision,
                model_directory=str(model),
                requested_mode=requested_mode,
                precision=bundle.precision,
            )
    if not actual_ep:
        return BackendSnapshot(
            False,
            "The installed ONNX Runtime exposes no usable execution provider",
            variant,
            requested_mode,
            model_revision=bundle.model_revision,
            model_directory=str(model),
            requested_mode=requested_mode,
            precision=bundle.precision,
        )
    if actual_device in {"webgpu", "coreml"} and not bool(
        manifest.get("accelerator_nodes_verified")
    ):
        return BackendSnapshot(
            False,
            "The installed GPU backend has not passed an accelerator node-execution test; run Repair",
            variant,
            requested_mode,
            model_revision=bundle.model_revision,
            model_directory=str(model),
            requested_mode=requested_mode,
            actual_ep=actual_ep,
            actual_device=actual_device,
            precision=bundle.precision,
        )
    compute_label = {
        "webgpu": "WebGPU",
        "coreml": "CoreML",
        "cpu": "CPU",
    }.get(actual_device, actual_device or actual_ep)
    hardware_vendor_id = (
        int(manifest.get("hardware_vendor_id") or 0) if actual_device == "webgpu" else 0
    )
    hardware_device_id = (
        int(manifest.get("hardware_device_id") or 0) if actual_device == "webgpu" else 0
    )
    hardware_label = (
        f"WebGPU adapter {hardware_vendor_id:#06x}:{hardware_device_id:#06x}"
        if hardware_vendor_id
        else ""
    )
    fallback_text = " · CPU fallback" if fallback else ""
    return BackendSnapshot(
        installed=True,
        message=f"{selected['label']} · {compute_label}{fallback_text} · Ready",
        variant=variant,
        device=requested_mode,
        model_revision=bundle.model_revision,
        model_directory=str(model),
        runtime_build=actual_ep,
        gpu_name=hardware_label,
        requested_mode=requested_mode,
        actual_ep=actual_ep,
        actual_device=actual_device,
        fallback=fallback,
        precision=bundle.precision,
        hardware_vendor_id=hardware_vendor_id,
        hardware_device_id=hardware_device_id,
        accelerator_nodes_verified=bool(manifest.get("accelerator_nodes_verified")),
    )


def invalidate_backend_snapshot() -> None:
    """Discard readiness cached before an install-state mutation."""
    BACKEND.invalidate()


def runtime_paths() -> dict[str, str]:
    active = active_directory(create_root=False)
    manifest = _read_json(manifest_path()) or {}
    model = installed_model_directory(manifest)
    return {
        "backend_directory": str(active),
        "model_directory": str(model),
        "bundle_manifest_path": str(
            model / str(manifest.get("bundle_manifest") or "bundle.json")
        ),
        "bundle_revision": str(manifest.get("bundle_revision") or ""),
        "model_revision": str(manifest.get("model_revision") or ""),
    }


def resolve_device_name(requested: str) -> str:
    mode = sam3d_ort_runtime.normalize_mode(requested)
    manifest = _read_json(manifest_path()) or {}
    available = tuple(str(item) for item in manifest.get("available_providers", ()))
    try:
        _provider, device, _fallback = sam3d_ort_runtime.select_execution_provider(
            mode,
            available,
        )
    except sam3d_ort_runtime.OrtRuntimeError:
        return "cpu" if mode == "CPU" else "gpu"
    return device


def provider_options_from_settings(settings) -> dict:
    options = {
        "model_directory": str(paths.BUNDLED_MODELS),
        "use_body": bool(settings.use_body),
        "use_hands": bool(settings.use_hands),
        "use_face": False,
        "mirror": bool(settings.mirror_input),
        "confidence": CAPTURE_CONFIDENCE,
        "smoothing": float(settings.smoothing),
        "variant": str(settings.sam3d_model_variant),
        "device": sam3d_ort_runtime.normalize_mode(settings.sam3d_device),
    }
    options.update(runtime_paths())
    return options


def runtime_diagnostics(variant: str, device: str) -> dict:
    """Return stable requested/actual runtime facts for UI and bug reports."""
    try:
        from ...providers.sam3d.state import runtime_diagnostics as loaded_state

        loaded = loaded_state()
    except (ImportError, AttributeError, RuntimeError):
        loaded = None
    if loaded is not None:
        return loaded.as_dict()
    snapshot = backend_snapshot(variant, device)
    return {
        "requested_mode": snapshot.requested_mode,
        "actual_ep": snapshot.actual_ep,
        "actual_device": snapshot.actual_device,
        "fallback": snapshot.fallback,
        "model_revision": snapshot.model_revision,
        "precision": snapshot.precision,
        "hardware_vendor_id": snapshot.hardware_vendor_id,
        "hardware_device_id": snapshot.hardware_device_id,
    }


def _download_file(
    url: str,
    destination: Path,
    job: BackgroundJob,
    start: float,
    end: float,
    label: str,
    phase: str,
    *,
    expected_sha256: str = "",
    expected_bytes: int = 0,
    maximum_bytes: int = 0,
) -> None:
    job.update(start, label, phase)
    lifecycle.download_file(
        url,
        destination,
        cancel=job.cancel,
        label=label,
        progress=job.download_progress(start, end, label, phase),
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
        maximum_bytes=maximum_bytes,
        user_agent=f"Media-to-Pose/{extension_version()}",
        cancel_message="Installation cancelled",
    )


def _InstallationLock(root: Path) -> InstallationLock:  # noqa: N802 - kept name
    return InstallationLock(
        Path(root) / ".install.lock",
        "SAM 3D Body",
        max_age_seconds=STALE_INSTALL_LOCK_SECONDS,
    )


def _cleanup_abandoned_staging(parent: Path) -> None:
    try:
        candidates = tuple(parent.glob(".staging-*"))
    except OSError:
        return
    resolved_parent = parent.resolve(strict=False)
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved.parent != resolved_parent or not resolved.name.startswith(
            ".staging-"
        ):
            continue
        if resolved.is_dir():
            remove_tree(resolved)
        elif resolved.exists():
            with suppress(OSError):
                resolved.unlink()


def _activate_staged_install(
    runtime_staging: Path,
    runtime_current: Path,
    runtime_backup: Path,
    model_staging: Path,
    model_current: Path,
    model_backup: Path,
) -> None:
    runtime_parent = runtime_staging.parent.resolve()
    if any(
        path.parent.resolve() != runtime_parent
        for path in (runtime_current, runtime_backup)
    ):
        raise BackendError("Unexpected runtime activation path")
    model_parent = model_staging.parent.resolve()
    if any(
        path.parent.resolve() != model_parent for path in (model_current, model_backup)
    ):
        raise BackendError("Unexpected model activation path")

    for backup in (runtime_backup, model_backup):
        if backup.is_dir():
            shutil.rmtree(backup)
        elif backup.exists():
            backup.unlink()
        if backup.exists():
            raise BackendError(f"Could not clear activation backup {backup}")

    runtime_had_current = runtime_current.exists()
    model_had_current = model_current.exists()
    runtime_backed_up = False
    model_backed_up = False
    runtime_activated = False
    model_activated = False
    try:
        if runtime_had_current:
            Path(runtime_current).replace(runtime_backup)
            runtime_backed_up = True
        if model_had_current:
            Path(model_current).replace(model_backup)
            model_backed_up = True
        Path(runtime_staging).replace(runtime_current)
        runtime_activated = True
        # Move the large model bundle last. Every operation after this point is
        # non-failing cleanup, so an earlier Windows file-lock error leaves the
        # verified download cache in model_staging for the next retry.
        Path(model_staging).replace(model_current)
        model_activated = True
    except OSError as activation_error:
        rollback_errors = []

        def rollback(action, label: str) -> None:
            try:
                action()
            except OSError as exc:
                rollback_errors.append(f"{label}: {_safe_metadata(exc)}")

        if runtime_activated and runtime_current.exists():
            rollback(
                lambda: shutil.rmtree(runtime_current),
                "remove incomplete runtime",
            )
        if runtime_backed_up and runtime_backup.exists():
            rollback(
                lambda: Path(runtime_backup).replace(runtime_current),
                "restore previous runtime",
            )
        if model_activated and model_current.exists():
            rollback(
                lambda: shutil.rmtree(model_current),
                "remove incomplete model",
            )
        if model_backed_up and model_backup.exists():
            rollback(
                lambda: Path(model_backup).replace(model_current),
                "restore previous model",
            )
        if rollback_errors:
            raise BackendError(
                "Activation failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from activation_error
        raise
    remove_tree(runtime_backup)
    remove_tree(model_backup)


def _friendly_install_error(exc: BaseException, phase: str) -> str:
    detail = _redact(exc)
    missing = re.search(r"No module named ['\"]([^'\"]+)", detail)
    if missing:
        detail = (
            f"Required Python module '{missing.group(1)}' was unavailable. "
            "The install log contains the failing import context."
        )
    elif isinstance(exc, PermissionError):
        if phase.startswith("5/5 Activation"):
            detail = (
                "Could not replace the active installation. Close other Blender "
                "instances using this provider, then retry. Files already "
                f"downloaded are retained. Details: {detail}"
            )
        else:
            detail = f"Permission or model-access check failed: {detail}"
    elif isinstance(exc, OSError):
        detail = f"File, platform, or storage error: {detail}"
    return f"{phase or 'Installation'} failed: {detail}"


def _check_install_cancel(job: BackgroundJob) -> None:
    job.check_cancelled("Installation cancelled")


def _new_install_layout(model_storage_selection: str) -> _InstallLayout:
    root = backend_root(create=True)
    install_id = f"{uuid.uuid4().hex}"
    storage_parent = _normalize_storage_parent(model_storage_selection)
    storage_root = storage_parent / MODEL_STORAGE_FOLDER
    previous_manifest = _read_json(manifest_path()) or {}
    return _InstallLayout(
        root=root,
        storage_config=root / "model_storage.json",
        staging=root / f".staging-{install_id}",
        backup=root / ".previous",
        storage_parent=storage_parent,
        storage_root=storage_root,
        model_staging=storage_root / ".download",
        model_current=storage_root / "active",
        model_backup=storage_root / ".previous",
        previous_storage_root=installed_model_storage_root(previous_manifest),
    )


def _validate_install_platform() -> None:
    machine = platform.machine().casefold()
    supported = (
        (sys.platform == "win32" and machine in {"amd64", "x86_64"})
        or (sys.platform.startswith("linux") and machine in {"amd64", "x86_64"})
        or (sys.platform == "darwin" and machine in {"arm64", "aarch64"})
    )
    if not supported:
        raise BackendError(f"SAM 3D Body is unsupported on {sys.platform}/{machine}")
    if sys.version_info[:2] != (3, 13):
        raise BackendError("SAM 3D Body requires Blender's CPython 3.13 runtime")


def _activate_install(job: BackgroundJob, layout: _InstallLayout) -> str:
    _check_install_cancel(job)
    job.update(0.98, "Activating validated model bundle", "5/5 Activation")
    invalidate_backend_snapshot()
    try:
        _activate_staged_install(
            layout.staging,
            layout.root / "active",
            layout.backup,
            layout.model_staging,
            layout.model_current,
            layout.model_backup,
        )
    finally:
        invalidate_backend_snapshot()
    try:
        lifecycle.write_storage_parent(
            layout.storage_config or (layout.root / "model_storage.json"),
            layout.storage_parent,
            private=True,
        )
    except OSError as exc:
        LOGGER.warning(
            "Model storage preference was not saved: %s", _safe_metadata(exc)
        )
    previous = layout.previous_storage_root
    if previous and previous.resolve() != layout.storage_root.resolve():
        try:
            STORAGE.remove(previous)
        except (OSError, BackendError) as exc:
            LOGGER.warning(
                "Previous model storage was retained: %s", _safe_metadata(exc)
            )
    job.update(1.0, "SAM 3D Body installed and ready", "5/5 Activation")
    return "SAM 3D Body installed and ready"


def _hf_bundle_url(repository: str, revision: str, path: str) -> str:
    return lifecycle.hf_resolve_url(
        repository, revision, path, label="Hugging Face bundle"
    )


def _bundle_destination(root: Path, relative: str) -> Path:
    return lifecycle.contained_path(root, relative, label="ONNX bundle")


def _prepare_ort_staging(layout: _InstallLayout) -> None:
    if layout.staging.exists():
        shutil.rmtree(layout.staging)
    layout.staging.mkdir(parents=True)
    layout.model_staging.mkdir(parents=True, exist_ok=True)


def _preferred_bundle_revision(specification: dict) -> str:
    revision = str(specification.get("revision") or "")
    return revision if sam3d_ort_runtime.is_immutable_revision(revision) else "main"


def _check_bundle_storage(
    layout: _InstallLayout,
    bundle,
    cached_paths: set[str],
) -> None:
    runtime_disk = _closest_existing_directory(layout.root)
    model_disk = _closest_existing_directory(layout.storage_root)
    model_required = (
        sum(int(item.bytes) for item in bundle.files if item.path not in cached_paths)
        + MODEL_DOWNLOAD_HEADROOM_BYTES
    )
    same_volume = Path(runtime_disk).stat().st_dev == Path(model_disk).stat().st_dev
    runtime_free = shutil.disk_usage(runtime_disk).free
    model_free = shutil.disk_usage(model_disk).free
    if same_volume:
        enough = runtime_free >= MIN_RUNTIME_FREE_BYTES + model_required
    else:
        enough = runtime_free >= MIN_RUNTIME_FREE_BYTES and model_free >= model_required
    if not enough:
        raise BackendError("Insufficient free space for the SAM 3D Body ONNX bundle")


def _download_ort_bundle(
    job: BackgroundJob,
    layout: _InstallLayout,
    variant: str,
    specification: dict,
):
    repository = str(specification["repo"])
    revision = _preferred_bundle_revision(specification)
    manifest_member = str(specification.get("manifest_path") or "")
    local_manifest = layout.model_staging / "bundle.json"
    try:
        _download_file(
            _hf_bundle_url(repository, revision, manifest_member),
            local_manifest,
            job,
            0.10,
            0.14,
            "Downloading preferred bundle manifest",
            "2/5 Model Bundle",
            maximum_bytes=4 * 1024 * 1024,
        )
    except urllib.error.HTTPError as exc:
        if revision == "main" or exc.code not in {404, 410}:
            raise
        revision = "main"
        job.log(
            "WARNING",
            "Preferred SAM 3D Body bundle is unavailable; using the latest "
            "published bundle.",
        )
        _download_file(
            _hf_bundle_url(repository, revision, manifest_member),
            local_manifest,
            job,
            0.10,
            0.14,
            "Downloading latest bundle manifest",
            "2/5 Model Bundle",
            maximum_bytes=4 * 1024 * 1024,
        )
    specification["resolved_revision"] = revision
    bundle = sam3d_ort_runtime.load_bundle_manifest(
        local_manifest,
        expected_variant=variant,
        verify_files=False,
    )
    cached_paths = set()
    for bundle_file in bundle.files:
        destination = _bundle_destination(layout.model_staging, bundle_file.path)
        if (
            destination.is_file()
            and destination.stat().st_size == bundle_file.bytes
            and lifecycle.sha256_file(destination).casefold()
            == bundle_file.sha256.casefold()
        ):
            cached_paths.add(bundle_file.path)
        elif destination.is_dir():
            shutil.rmtree(destination)
        elif destination.exists():
            destination.unlink()
    _check_bundle_storage(layout, bundle, cached_paths)
    manifest_parent = PurePosixPath(manifest_member).parent
    total_bytes = sum(int(item.bytes) for item in bundle.files)
    completed = 0
    for index, bundle_file in enumerate(bundle.files, start=1):
        _check_install_cancel(job)
        start = 0.14 + 0.64 * (completed / total_bytes)
        completed += int(bundle_file.bytes)
        end = 0.14 + 0.64 * (completed / total_bytes)
        if bundle_file.path in cached_paths:
            job.update(
                end,
                f"Using verified cached model file {index}/{len(bundle.files)}",
                "2/5 Model Bundle",
            )
            continue
        remote_member = str(manifest_parent / bundle_file.path)
        _download_file(
            _hf_bundle_url(repository, revision, remote_member),
            _bundle_destination(layout.model_staging, bundle_file.path),
            job,
            start,
            end,
            f"Downloading model file {index}/{len(bundle.files)}",
            "2/5 Model Bundle",
            expected_sha256=bundle_file.sha256,
            expected_bytes=bundle_file.bytes,
        )
    return sam3d_ort_runtime.load_bundle_manifest(
        local_manifest,
        expected_variant=variant,
        verify_files=True,
    )


def _self_test_ort_bundle(job: BackgroundJob, bundle, requested_device: str):
    job.update(0.82, "Creating ONNX Runtime session", "3/5 Self-Test")
    requested_mode = sam3d_ort_runtime.normalize_mode(requested_device)
    import numpy as np

    def run(mode: str, *, verify_accelerator: bool):
        estimator = sam3d_ort_runtime.create_estimator(
            bundle,
            mode,
            verify_accelerator_execution=verify_accelerator,
        )
        try:
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            boxes = np.asarray([[0.0, 0.0, 63.0, 63.0]], dtype=np.float32)
            output = estimator.process_one_image(
                image,
                bboxes=boxes,
                inference_type="full",
            )
            if len(output) != 1 or any(
                key not in output[0] for key in sam3d_ort_runtime.REQUIRED_OUTPUT_KEYS
            ):
                raise BackendError("ONNX bundle self-test returned an invalid contract")
            return estimator.diagnostics
        finally:
            estimator.close()

    try:
        return run(
            requested_mode,
            verify_accelerator=requested_mode != "CPU",
        )
    except sam3d_ort_runtime.OrtRuntimeError as accelerator_error:
        if requested_mode != "AUTO":
            raise
        job.log(
            "WARNING",
            "Accelerated self-test failed; validating the explicit CPU fallback: "
            f"{accelerator_error}",
        )
        return replace(
            run("CPU", verify_accelerator=False),
            requested_mode="AUTO",
            fallback=True,
        )


def _write_ort_install_manifest(
    layout: _InstallLayout,
    variant: str,
    specification: dict,
    diagnostics,
) -> None:
    atomic_json(
        layout.staging / "manifest.json",
        {
            "runtime_manifest_version": RUNTIME_MANIFEST_VERSION,
            "backend_version": BACKEND_VERSION,
            "backend_kind": "onnxruntime",
            "adapter_version": ADAPTER_VERSION,
            "bundle_contract": sam3d_ort_runtime.BUNDLE_CONTRACT,
            "bundle_manifest": "bundle.json",
            "variant": variant,
            "model_repo": specification["repo"],
            "bundle_revision": specification.get("resolved_revision")
            or specification.get("revision")
            or "main",
            "model_revision": diagnostics.model_revision,
            "precision": diagnostics.precision,
            "requested_mode": diagnostics.requested_mode,
            "actual_ep": diagnostics.actual_ep,
            "actual_device": diagnostics.actual_device,
            "fallback": diagnostics.fallback,
            "available_providers": list(diagnostics.available_providers),
            "hardware_vendor_id": diagnostics.hardware_vendor_id,
            "hardware_device_id": diagnostics.hardware_device_id,
            "accelerator_nodes_verified": diagnostics.accelerator_nodes_verified,
            "accelerator_node_count": diagnostics.accelerator_node_count,
            "profiled_node_count": diagnostics.profiled_node_count,
            "installed_at": int(time.time()),
            "model_directory": str(layout.model_current.resolve(strict=False)),
            "model_storage_parent": str(layout.storage_parent),
        },
        private=True,
    )


def _install(
    job: BackgroundJob,
    variant: str,
    requested_device: str,
    layout: _InstallLayout,
) -> str:
    """Install model files only; ORT itself must be shipped by the extension."""
    install_lock = _InstallationLock(layout.root)
    try:
        install_lock.acquire()
        _cleanup_abandoned_staging(layout.root)
        layout.storage_root.mkdir(parents=True, exist_ok=True)
        _cleanup_abandoned_staging(layout.storage_root)
        specification = dict(model_spec(variant))
        job.update(0.01, "Checking model publication", "1/5 Preflight")
        _validate_install_platform()
        runtime = sam3d_ort_runtime.probe_runtime(requested_device)
        job.log(
            "INFO",
            f"ORT requested={runtime.requested_mode}; selected={runtime.actual_ep}; "
            f"fallback={runtime.fallback}",
        )
        STORAGE.ensure(layout.storage_root)
        _prepare_ort_staging(layout)
        bundle = _download_ort_bundle(job, layout, variant, specification)
        diagnostics = _self_test_ort_bundle(job, bundle, requested_device)
        _check_install_cancel(job)
        job.update(0.94, "Writing validated backend manifest", "4/5 Manifest")
        _write_ort_install_manifest(layout, variant, specification, diagnostics)
        job.update(0.97, "Activating graph bundle", "5/5 Activation")
        return _activate_install(job, layout)
    except Cancelled as exc:
        raise Cancelled(f"{job.phase or 'Installation'} cancelled") from exc
    finally:
        remove_tree(layout.staging)
        if layout.model_staging.exists():
            job.log("DETAIL", "Completed model downloads retained for the next retry")
        job.log("DETAIL", "Temporary runtime staging cleaned")
        install_lock.release()


def start_install(
    variant: str,
    device: str,
    storage_selection: str = "",
    *,
    scene_name: str = "",
    scene_pointer: int = 0,
) -> BackgroundJob:
    """Start installing ``variant`` into the selected storage parent."""
    if is_model_loaded():
        raise InvalidRequest("Unload SAM 3D Body before repairing it")
    storage_parent = _normalize_storage_parent(storage_selection)
    layout = _new_install_layout(str(storage_parent))

    def prepare(job: BackgroundJob) -> None:
        job.log(
            "INFO",
            f"Starting SAM 3D Body installation (model={variant}, device={device})",
        )
        job.log("INFO", f"Selected model storage: {storage_parent}")

    return BACKEND.start_job(
        JobKind.INSTALL,
        OperationKind.SAM3D_INSTALL,
        "sam3d-install",
        lifecycle.run_job,
        (lambda job: _install(job, str(variant), str(device), layout),),
        label="Installing SAM 3D Body",
        status="Starting SAM 3D Body installation",
        scene_name=scene_name,
        scene_pointer=scene_pointer,
        request=SamSetupRequest(
            operation="install",
            model_variant=str(variant),
            device=str(device),
            model_storage_directory=str(storage_parent),
        ),
        log_path=install_log_path(create_root=True),
        prepare=prepare,
        kwargs={"describe_error": _friendly_install_error},
    )


def cancel_install() -> bool:
    return BACKEND.cancel_job(JobKind.INSTALL, "Cancelling installation safely")


def dismiss_install_result() -> bool:
    if install_running():
        return False
    BACKEND.clear_job(JobKind.INSTALL)
    return True


def is_model_loaded() -> bool:
    try:
        from ...providers.sam3d.state import model_loaded

        return model_loaded()
    except (ImportError, AttributeError, RuntimeError):
        return False


def _load(
    job: BackgroundJob,
    variant: str,
    requested_device: str,
    resolved_runtime_paths: Mapping[str, str],
) -> str:
    job.update(0.05, "Opening ONNX Runtime graph bundle", "Model Load")
    from ...providers.sam3d.state import load_model_cache

    job.update(0.20, "Creating the requested execution-provider session", "Model Load")
    try:
        device = load_model_cache(
            variant, requested_device, dict(resolved_runtime_paths)
        )
        if job.cancel.is_set():
            unload_model()
            raise Cancelled("Model loading cancelled")
    except Cancelled as exc:
        raise Cancelled("SAM 3D Body model loading cancelled") from exc
    job.update(1.0, f"Model loaded on {device.upper()}", "Model Load")
    return f"SAM 3D Body model loaded on {device.upper()}"


def start_load(
    variant: str,
    device: str,
    *,
    scene_name: str = "",
    scene_pointer: int = 0,
) -> BackgroundJob:
    snapshot = backend_snapshot(str(variant), str(device))
    if not snapshot.installed:
        raise ModelMissing("Repair the SAM 3D Body backend first")
    resolved_runtime_paths = runtime_paths()
    return BACKEND.start_job(
        JobKind.MODEL,
        OperationKind.SAM3D_MODEL_LOAD,
        "sam3d-model-load",
        lifecycle.run_job,
        (lambda job: _load(job, str(variant), str(device), resolved_runtime_paths),),
        label="Loading SAM 3D Body model",
        scene_name=scene_name,
        scene_pointer=scene_pointer,
        request=SamSetupRequest(
            operation="load", model_variant=str(variant), device=str(device)
        ),
        kwargs={
            "describe_error": lambda exc, _phase: _friendly_install_error(
                exc, "Model Load"
            )
        },
    )


def unload_model() -> None:
    from ...providers.sam3d.state import unload_model_cache

    unload_model_cache()


def unload() -> str:
    def action() -> str:
        unload_model()
        return "SAM 3D Body model unloaded"

    return BACKEND.run_exclusive(
        OperationKind.SAM3D_MODEL_UNLOAD,
        "sam3d-model-unload",
        action,
        status="Unloading SAM 3D Body model",
    )


def uninstall() -> str:
    def action() -> str:
        if is_model_loaded():
            raise InvalidRequest("Unload SAM 3D Body before uninstalling")
        invalidate_backend_snapshot()
        try:
            manifest = _read_json(manifest_path()) or {}
            storage_root = installed_model_storage_root(manifest)
            root = _verified_backend_root()
            if (
                storage_root
                and root not in storage_root.resolve().parents
                and storage_root.exists()
            ):
                STORAGE.remove(storage_root)
            if root.exists():
                shutil.rmtree(root)
        except OSError as exc:
            raise BackendError(
                f"Could not remove SAM 3D Body: {_safe_metadata(exc)}"
            ) from exc
        finally:
            invalidate_backend_snapshot()
        return "SAM 3D Body backend removed"

    return BACKEND.run_exclusive(
        OperationKind.SAM3D_UNINSTALL,
        "sam3d-uninstall",
        action,
        status="Uninstalling SAM 3D Body",
    )


def install_diagnostics_text() -> str:
    snapshot = install_status_snapshot()
    manifest = _read_json(manifest_path()) or {}
    lines = [
        "Media to Pose SAM 3D Body installer",
        f"Extension: {extension_version()}",
        f"Platform: {sys.platform} / {platform.machine()}",
        f"Python: {sys.version.split()[0]}",
        f"Phase: {snapshot.phase or 'Unknown'}",
        f"Outcome: {snapshot.outcome or 'Running'}",
        f"Progress: {snapshot.progress * 100.0:.1f}%",
        f"Status: {snapshot.status or 'No status'}",
    ]
    if manifest.get("backend_kind") == "onnxruntime":
        lines.extend(
            (
                f"Requested mode: {manifest.get('requested_mode') or 'unknown'}",
                f"Actual EP: {manifest.get('actual_ep') or 'unknown'}",
                f"Actual device: {manifest.get('actual_device') or 'unknown'}",
                f"Fallback: {bool(manifest.get('fallback'))}",
                f"Model revision: {manifest.get('model_revision') or 'unknown'}",
                f"Precision: {manifest.get('precision') or 'unknown'}",
            )
        )
    if snapshot.log_path:
        lines.append(f"Log: {snapshot.log_path}")
    if snapshot.logs:
        lines.extend(("", "Recent log:", *snapshot.logs[-30:]))
    return _redact("\n".join(lines))


def install_log_file() -> Path:
    snapshot = install_status_snapshot()
    return (
        Path(snapshot.log_path)
        if snapshot.log_path
        else install_log_path(create_root=False)
    )


def shutdown() -> None:
    BACKEND.shutdown(join_timeout=0.5)
    if not model_load_running():
        unload_model()


__all__ = [
    "BACKEND",
    "MODEL_VARIANTS",
    "BackendSnapshot",
    "BackgroundJob",
    "backend_present",
    "backend_snapshot",
    "cancel_install",
    "dismiss_install_result",
    "install_diagnostics_text",
    "install_log_file",
    "install_running",
    "install_status_snapshot",
    "is_model_loaded",
    "model_load_running",
    "model_load_status_snapshot",
    "model_spec",
    "preferred_model_storage_parent",
    "runtime_diagnostics",
    "runtime_paths",
    "sam3d_device_enum_items",
    "sam_backend_ready",
    "shutdown",
    "start_install",
    "start_load",
    "uninstall",
    "unload",
]
