"""Optional SAM 3D Body model installation and runtime state."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import queue
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from contextlib import suppress
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Dict, Mapping, Optional, Tuple

import bpy
from bpy.props import StringProperty
from bpy.types import Operator

from . import sam3d_ort_runtime
from .constants import CAPTURE_CONFIDENCE, extension_version
from .operation import (
    COORDINATOR,
    OperationKind,
    OperationState,
    OperationToken,
    SamSetupRequest,
)
from .registration import register_classes, unregister_classes
from .sam3d_state import SAM_JOBS, SamJobSlot

RUNTIME_MANIFEST_VERSION = 4
BACKEND_VERSION = RUNTIME_MANIFEST_VERSION
ADAPTER_VERSION = "2.0.0"
HF_ENDPOINT = "https://huggingface.co"
MIN_RUNTIME_FREE_BYTES = 256 * 1024 * 1024
MODEL_DOWNLOAD_HEADROOM_BYTES = 512 * 1024 * 1024
MODEL_STORAGE_FOLDER = "media_to_pose_sam3d"
MODEL_STORAGE_MARKER = ".media_to_pose_storage.json"

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

_DEVICE_ENUM_ITEMS: Tuple[Tuple[str, str, str], ...] = (
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


@dataclass(frozen=True)
class InstallStatusSnapshot:
    running: bool = False
    progress: float = 0.0
    phase: str = ""
    status: str = ""
    outcome: str = ""
    error: str = ""
    logs: Tuple[str, ...] = ()
    log_path: str = ""


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
    previous_storage_root: Optional[Path]
    storage_config: Optional[Path] = None


class BackgroundJob:
    def __init__(
        self,
        label: str,
        *,
        token: Optional[OperationToken] = None,
        request: Optional[SamSetupRequest] = None,
    ):
        self.label = label
        self.token = token
        self.request = request
        self.cancel = token.cancel_event if token else threading.Event()
        self.messages = queue.Queue()
        self.progress = 0.0
        self.status = label
        self.phase = ""
        self.outcome = ""
        self.error = ""
        self.logs = []
        self.log_path = ""
        self.thread = None
        self.process = None
        self.started = False
        self.terminal = False
        self._lock = threading.RLock()

    def configure_log(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        previous = path.with_name("install.previous.log")
        try:
            if previous.exists():
                previous.unlink()
            if path.exists():
                os.replace(path, previous)
            path.write_text("", encoding="utf8")
            self.log_path = str(path)
        except OSError as exc:
            self.log_path = ""
            print(
                "[Media to Pose] Could not initialize install log: "
                f"{_safe_metadata(exc)}"
            )

    def log(self, level: str, message: object) -> None:
        cleaned = re.sub(
            r"[\x00-\x1f\x7f]+",
            " ",
            _redact(message),
        ).strip()
        if not cleaned:
            return
        cleaned = cleaned[:2000]
        timestamp = time.strftime("%H:%M:%S")
        entry = f"{timestamp} [{level.upper()}] {cleaned}"
        with self._lock:
            self.logs.append(entry)
            if len(self.logs) > 200:
                del self.logs[:-200]
            log_path = self.log_path
        print(f"[Media to Pose] {entry}")
        if log_path:
            try:
                with Path(log_path).open("a", encoding="utf8") as handle:
                    handle.write(entry + "\n")
            except OSError as exc:
                print(
                    "[Media to Pose] Could not append SAM installation log: "
                    f"{_safe_metadata(exc)}"
                )

    def update(
        self,
        progress: float,
        status: str,
        *,
        phase: str = "",
        log_change: bool = True,
    ) -> None:
        if self.token is not None and not COORDINATOR.update(
            self.token,
            status=status,
            phase=phase or None,
            progress=progress,
        ):
            return
        with self._lock:
            changed = status != self.status or (phase and phase != self.phase)
            self.progress = min(max(float(progress), 0.0), 1.0)
            self.status = status
            if phase:
                self.phase = phase
        if changed and log_change:
            prefix = f"{self.phase}: " if self.phase else ""
            self.log("INFO", f"{prefix}{status}")

    def finish(self, outcome: str, message: str) -> None:
        safe_message = _redact(message)
        with self._lock:
            self.outcome = outcome
            self.error = safe_message if outcome == "ERROR" else ""
            self.status = safe_message
            self.terminal = True
        level = "ERROR" if outcome == "ERROR" else "INFO"
        self.log(level, safe_message)
        if self.token is not None:
            state = (
                OperationState.SUCCEEDED
                if outcome in {"DONE", "SUCCESS"}
                else OperationState.CANCELLED
                if outcome == "CANCELLED"
                else OperationState.FAILED
            )
            COORDINATOR.finish(self.token, state, safe_message)

    def snapshot(self) -> InstallStatusSnapshot:
        with self._lock:
            return InstallStatusSnapshot(
                running=bool(
                    self.thread and self.thread.is_alive() and not self.terminal
                ),
                progress=self.progress,
                phase=self.phase,
                status=self.status,
                outcome=self.outcome,
                error=self.error,
                logs=tuple(self.logs),
                log_path=self.log_path,
            )


def _redact(value: object) -> str:
    text = str(value or "")
    text = re.sub(
        r"(?i)\b(?:hf|api_org)_[A-Za-z0-9._-]{6,}\b",
        "[redacted]",
        text,
    )
    return text


def _safe_metadata(value: object, *, limit: int = 160) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", _redact(value))
    return text.strip()[:limit]


def backend_root(*, create: bool = True) -> Path:
    value = bpy.utils.user_resource(
        "SCRIPTS",
        path="media_to_pose/backends/sam3d",
        create=create,
    )
    if not value:
        scripts = bpy.utils.user_resource("SCRIPTS", create=create)
        if not scripts:
            raise RuntimeError("Blender user scripts directory is unavailable")
        value = str(Path(scripts) / "media_to_pose" / "backends" / "sam3d")
    return Path(value)


def _verified_backend_root() -> Path:
    root = backend_root(create=False).resolve()
    scripts_value = bpy.utils.user_resource("SCRIPTS", create=False)
    if not scripts_value:
        raise RuntimeError("Blender user scripts directory is unavailable")
    scripts = Path(scripts_value).resolve()
    if scripts not in root.parents or root.name != "sam3d":
        raise RuntimeError("Refusing to remove an unexpected backend path")
    return root


def active_directory(*, create_root: bool = True) -> Path:
    return backend_root(create=create_root) / "active"


def install_log_path(*, create_root: bool = True) -> Path:
    return backend_root(create=create_root) / "install.log"


def manifest_path() -> Path:
    return active_directory(create_root=False) / "manifest.json"


def backend_present() -> bool:
    return manifest_path().is_file()


def model_spec(variant: str) -> Dict:
    try:
        return MODEL_VARIANTS[str(variant)]
    except KeyError as exc:
        raise ValueError(f"Unknown SAM 3D Body model variant: {variant!r}") from exc


def _read_json(path: Path) -> Optional[Dict]:
    try:
        value = json.loads(path.read_text(encoding="utf8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def _atomic_json(path: Path, value: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf8",
        ) as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
        with suppress(OSError):
            os.chmod(temporary, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(temporary, path)
    except Exception:
        with suppress(OSError):
            os.unlink(temporary)
        raise


def model_storage_config_path(*, create_root: bool = True) -> Path:
    return backend_root(create=create_root) / "model_storage.json"


def _normalize_storage_parent(value: str = "") -> Path:
    if value:
        expanded = os.path.expandvars(os.path.expanduser(value.strip()))
        selected = Path(bpy.path.abspath(expanded))
    else:
        selected = backend_root(create=True)
    selected = selected.resolve(strict=False)
    if selected.name.casefold() == MODEL_STORAGE_FOLDER.casefold():
        selected = selected.parent
    return selected


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


def _storage_marker_path(root: Path) -> Path:
    return root / MODEL_STORAGE_MARKER


def _ensure_model_storage(root: Path) -> None:
    marker = _storage_marker_path(root)
    if root.exists() and not marker.is_file():
        try:
            has_contents = next(root.iterdir(), None) is not None
        except OSError as exc:
            raise OSError(f"Cannot inspect model storage directory: {root}") from exc
        if has_contents:
            raise OSError(
                "The selected model location already contains an unmanaged "
                f"{MODEL_STORAGE_FOLDER} directory"
            )
    root.mkdir(parents=True, exist_ok=True)
    existing = _read_json(marker)
    if existing and (
        existing.get("owner") != "media_to_pose"
        or existing.get("kind") != "sam3d_models"
    ):
        raise OSError("The selected model location is owned by another application")
    _atomic_json(
        marker,
        {
            "owner": "media_to_pose",
            "kind": "sam3d_models",
            "version": 1,
        },
    )


def _verified_model_storage_root(root: Path) -> Path:
    resolved = root.resolve(strict=False)
    marker = _read_json(_storage_marker_path(resolved)) or {}
    if (
        resolved.name.casefold() != MODEL_STORAGE_FOLDER.casefold()
        or marker.get("owner") != "media_to_pose"
        or marker.get("kind") != "sam3d_models"
    ):
        raise OSError("Refusing to remove an unrecognized model directory")
    return resolved


def installed_model_directory(manifest: Optional[Dict] = None) -> Path:
    manifest = manifest if manifest is not None else (_read_json(manifest_path()) or {})
    configured = str(manifest.get("model_directory") or "")
    if configured:
        return Path(configured).resolve(strict=False)
    return active_directory(create_root=False) / "model"


def installed_model_storage_root(
    manifest: Optional[Dict] = None,
) -> Optional[Path]:
    model = installed_model_directory(manifest)
    parent = model.parent
    if (
        model.name == "active"
        and parent.name.casefold() == MODEL_STORAGE_FOLDER.casefold()
        and _storage_marker_path(parent).is_file()
    ):
        return parent
    return None


def _closest_existing_directory(path: Path) -> Path:
    candidate = path.resolve(strict=False)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise OSError(f"No accessible parent directory exists for {path}")
    return candidate


def install_running() -> bool:
    job = SAM_JOBS.current(SamJobSlot.INSTALL)
    return bool(job and job.thread and job.thread.is_alive())


def install_status_snapshot() -> InstallStatusSnapshot:
    job = SAM_JOBS.current(SamJobSlot.INSTALL)
    if not job:
        return InstallStatusSnapshot()
    return job.snapshot()


def model_load_running() -> bool:
    job = SAM_JOBS.current(SamJobSlot.MODEL)
    return bool(job and job.thread and job.thread.is_alive() and not job.terminal)


def model_load_status_snapshot() -> InstallStatusSnapshot:
    job = SAM_JOBS.current(SamJobSlot.MODEL)
    if not job:
        return InstallStatusSnapshot()
    return job.snapshot()


def sam_backend_ready(settings) -> bool:
    """Return whether the selected SAM runtime is complete and self-tested."""
    return backend_snapshot(
        str(settings.sam3d_model_variant),
        str(settings.sam3d_device),
    ).installed


def sam3d_device_enum_items() -> Tuple[Tuple[str, str, str], ...]:
    """Return the stable backend-agnostic AUTO/GPU/CPU choices."""
    return _DEVICE_ENUM_ITEMS


@lru_cache(maxsize=12)
def backend_snapshot(variant: str, device: str) -> BackendSnapshot:
    """Return cached readiness so Blender panel redraws never rescan ONNX files."""
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
    except (OSError, sam3d_ort_runtime.OrtRuntimeError) as exc:
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
    backend_snapshot.cache_clear()


def runtime_paths() -> Dict[str, str]:
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
        return device
    except sam3d_ort_runtime.OrtRuntimeError:
        return "cpu" if mode == "CPU" else "gpu"


def provider_options_from_settings(settings) -> Dict:
    options = {
        "model_directory": str(Path(__file__).parent / "assets" / "models"),
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


def runtime_diagnostics(variant: str, device: str) -> Dict:
    """Return stable requested/actual runtime facts for UI and bug reports."""
    try:
        from .providers.sam3d_body_provider import runtime_diagnostics as loaded_state

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    job.update(start, label, phase=phase)
    headers = {"User-Agent": f"Media-to-Pose/{extension_version()}"}
    request = urllib.request.Request(
        url,
        headers=headers,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            total = int(response.headers.get("Content-Length") or 0)
            if expected_bytes and total and total != expected_bytes:
                raise RuntimeError(f"{label} has an unexpected download size")
            if maximum_bytes and total and total > maximum_bytes:
                raise RuntimeError(f"{label} exceeds the safe download limit")
            received = 0
            with partial.open("xb") as handle:
                while True:
                    if job.cancel.is_set():
                        raise InterruptedError("Installation cancelled")
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                    if expected_bytes and received > expected_bytes:
                        raise RuntimeError(f"{label} exceeds its declared size")
                    if maximum_bytes and received > maximum_bytes:
                        raise RuntimeError(f"{label} exceeds the safe download limit")
                    ratio = received / total if total else 0.0
                    size = f"{received / (1024**2):.0f} MiB"
                    if total:
                        size += f" / {total / (1024**2):.0f} MiB"
                    job.update(
                        start + (end - start) * ratio,
                        f"{label} · {size}",
                        phase=phase,
                        log_change=False,
                    )
        if expected_bytes and received != expected_bytes:
            raise RuntimeError(f"{label} has an unexpected download size")
        if (
            expected_sha256
            and digest.hexdigest().casefold() != expected_sha256.casefold()
        ):
            raise RuntimeError(f"{label} failed SHA-256 verification")
        os.replace(partial, destination)
    except Exception:
        with suppress(OSError):
            partial.unlink()
        raise


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class _InstallationLock:
    """Small cross-process lock that survives Blender reloads safely."""

    def __init__(self, root: Path):
        self.path = root / ".install.lock"
        self.identifier = uuid.uuid4().hex
        self.acquired = False

    def acquire(self) -> None:
        payload = json.dumps(
            {
                "pid": os.getpid(),
                "created_at": int(time.time()),
                "identifier": self.identifier,
            }
        ).encode("utf8")
        for _attempt in range(2):
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                existing = _read_json(self.path) or {}
                pid = int(existing.get("pid") or 0)
                created_at = int(existing.get("created_at") or 0)
                stale = not _pid_running(pid) or (
                    created_at and time.time() - created_at > 24 * 60 * 60
                )
                if not stale:
                    raise RuntimeError(
                        "Another Blender process is installing SAM 3D Body"
                    ) from None
                try:
                    self.path.unlink()
                except OSError as exc:
                    raise RuntimeError(
                        "A stale SAM installation lock could not be removed"
                    ) from exc
                continue
            try:
                os.write(descriptor, payload)
            finally:
                os.close(descriptor)
            self.acquired = True
            return
        raise RuntimeError("Could not acquire the SAM installation lock")

    def release(self) -> None:
        if not self.acquired:
            return
        current = _read_json(self.path) or {}
        if current.get("identifier") == self.identifier:
            with suppress(FileNotFoundError):
                self.path.unlink()
        self.acquired = False


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
            shutil.rmtree(resolved, ignore_errors=True)
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
        raise OSError("Unexpected runtime activation path")
    model_parent = model_staging.parent.resolve()
    if any(
        path.parent.resolve() != model_parent for path in (model_current, model_backup)
    ):
        raise OSError("Unexpected model activation path")

    for backup in (runtime_backup, model_backup):
        if backup.is_dir():
            shutil.rmtree(backup)
        elif backup.exists():
            backup.unlink()
        if backup.exists():
            raise OSError(f"Could not clear activation backup {backup}")

    runtime_had_current = runtime_current.exists()
    model_had_current = model_current.exists()
    runtime_backed_up = False
    model_backed_up = False
    runtime_activated = False
    model_activated = False
    try:
        if runtime_had_current:
            os.replace(runtime_current, runtime_backup)
            runtime_backed_up = True
        if model_had_current:
            os.replace(model_current, model_backup)
            model_backed_up = True
        os.replace(runtime_staging, runtime_current)
        runtime_activated = True
        # Move the large model bundle last. Every operation after this point is
        # non-failing cleanup, so an earlier Windows file-lock error leaves the
        # verified download cache in model_staging for the next retry.
        os.replace(model_staging, model_current)
        model_activated = True
    except Exception as activation_error:
        rollback_errors = []

        def rollback(action, label: str) -> None:
            try:
                action()
            except Exception as exc:
                rollback_errors.append(f"{label}: {_safe_metadata(exc)}")

        if runtime_activated and runtime_current.exists():
            rollback(
                lambda: shutil.rmtree(runtime_current),
                "remove incomplete runtime",
            )
        if runtime_backed_up and runtime_backup.exists():
            rollback(
                lambda: os.replace(runtime_backup, runtime_current),
                "restore previous runtime",
            )
        if model_activated and model_current.exists():
            rollback(
                lambda: shutil.rmtree(model_current),
                "remove incomplete model",
            )
        if model_backed_up and model_backup.exists():
            rollback(
                lambda: os.replace(model_backup, model_current),
                "restore previous model",
            )
        if rollback_errors:
            raise RuntimeError(
                "Activation failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from activation_error
        raise
    shutil.rmtree(runtime_backup, ignore_errors=True)
    shutil.rmtree(model_backup, ignore_errors=True)


def _friendly_install_error(exc: Exception, phase: str) -> str:
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
    if job.cancel.is_set():
        raise InterruptedError("Installation cancelled")


def _new_install_layout(model_storage_selection: str) -> _InstallLayout:
    root = backend_root(create=True)
    install_id = f"{os.getpid()}-{uuid.uuid4().hex}"
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
        raise OSError(f"SAM 3D Body is unsupported on {sys.platform}/{machine}")
    if sys.version_info[:2] != (3, 13):
        raise OSError("SAM 3D Body requires Blender's CPython 3.13 runtime")


def _activate_install(job: BackgroundJob, layout: _InstallLayout) -> None:
    _check_install_cancel(job)
    job.update(
        0.98,
        "Activating validated model bundle",
        phase="5/5 Activation",
    )
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
        _atomic_json(
            layout.storage_config or (layout.root / "model_storage.json"),
            {"storage_parent": str(layout.storage_parent.resolve(strict=False))},
        )
    except OSError as exc:
        print(
            "[Media to Pose] Model storage preference was not saved: "
            f"{_safe_metadata(exc)}"
        )
    previous = layout.previous_storage_root
    if previous and previous.resolve() != layout.storage_root.resolve():
        try:
            shutil.rmtree(_verified_model_storage_root(previous))
        except OSError as exc:
            print(
                "[Media to Pose] Previous model storage was retained: "
                f"{_safe_metadata(exc)}"
            )
    job.update(
        1.0,
        "SAM 3D Body installed and ready",
        phase="5/5 Activation",
    )
    job.finish("DONE", "SAM 3D Body installed and ready")
    job.messages.put(("DONE", "SAM 3D Body installed"))


def _hf_bundle_url(repository: str, revision: str, path: str) -> str:
    raw = str(path)
    relative = PurePosixPath(raw)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or "\\" in raw
        or "\x00" in raw
        or any(":" in part for part in relative.parts)
    ):
        raise RuntimeError(f"Unsafe Hugging Face bundle path: {path!r}")
    encoded_repository = "/".join(
        urllib.parse.quote(part, safe="") for part in repository.split("/")
    )
    encoded_path = "/".join(
        urllib.parse.quote(part, safe="") for part in relative.parts
    )
    return f"{HF_ENDPOINT}/{encoded_repository}/resolve/{revision}/{encoded_path}"


def _bundle_destination(root: Path, relative: str) -> Path:
    raw = str(relative)
    parts = PurePosixPath(raw)
    if (
        parts.is_absolute()
        or ".." in parts.parts
        or not parts.parts
        or "\\" in raw
        or "\x00" in raw
        or any(":" in part for part in parts.parts)
    ):
        raise RuntimeError(f"Unsafe ONNX bundle path: {relative!r}")
    destination = root.joinpath(*parts.parts).resolve(strict=False)
    resolved_root = root.resolve(strict=False)
    if destination != resolved_root and resolved_root not in destination.parents:
        raise RuntimeError(f"ONNX bundle path escapes staging: {relative!r}")
    return destination


def _prepare_ort_staging(layout: _InstallLayout) -> None:
    if layout.staging.exists():
        shutil.rmtree(layout.staging)
    layout.staging.mkdir(parents=True)
    layout.model_staging.mkdir(parents=True, exist_ok=True)


def _preferred_bundle_revision(specification: Dict) -> str:
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
    same_volume = os.stat(runtime_disk).st_dev == os.stat(model_disk).st_dev
    runtime_free = shutil.disk_usage(runtime_disk).free
    model_free = shutil.disk_usage(model_disk).free
    if same_volume:
        enough = runtime_free >= MIN_RUNTIME_FREE_BYTES + model_required
    else:
        enough = runtime_free >= MIN_RUNTIME_FREE_BYTES and model_free >= model_required
    if not enough:
        raise OSError("Insufficient free space for the SAM 3D Body ONNX bundle")


def _download_ort_bundle(
    job: BackgroundJob,
    layout: _InstallLayout,
    variant: str,
    specification: Dict,
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
            and _sha256(destination).casefold() == bundle_file.sha256.casefold()
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
                phase="2/5 Model Bundle",
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
    job.update(0.82, "Creating ONNX Runtime session", phase="3/5 Self-Test")
    requested_mode = sam3d_ort_runtime.normalize_mode(requested_device)

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
                raise RuntimeError("ONNX bundle self-test returned an invalid contract")
            return estimator.diagnostics
        finally:
            estimator.close()

    try:
        import numpy as np

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
    specification: Dict,
    diagnostics,
) -> None:
    _atomic_json(
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
    )


def _install_worker(
    job: BackgroundJob,
    variant: str,
    requested_device: str,
    layout: _InstallLayout,
) -> None:
    """Install model files only; ORT itself must be shipped by the extension."""
    install_lock = _InstallationLock(layout.root)
    try:
        install_lock.acquire()
        _cleanup_abandoned_staging(layout.root)
        layout.storage_root.mkdir(parents=True, exist_ok=True)
        _cleanup_abandoned_staging(layout.storage_root)
        specification = dict(model_spec(variant))
        job.update(0.01, "Checking model publication", phase="1/5 Preflight")
        _validate_install_platform()
        runtime = sam3d_ort_runtime.probe_runtime(requested_device)
        job.log(
            "INFO",
            f"ORT requested={runtime.requested_mode}; selected={runtime.actual_ep}; "
            f"fallback={runtime.fallback}",
        )
        _ensure_model_storage(layout.storage_root)
        _prepare_ort_staging(layout)
        bundle = _download_ort_bundle(
            job,
            layout,
            variant,
            specification,
        )
        diagnostics = _self_test_ort_bundle(job, bundle, requested_device)
        _check_install_cancel(job)
        job.update(0.94, "Writing validated backend manifest", phase="4/5 Manifest")
        _write_ort_install_manifest(
            layout,
            variant,
            specification,
            diagnostics,
        )
        job.update(0.97, "Activating graph bundle", phase="5/5 Activation")
        _activate_install(job, layout)
    except InterruptedError:
        message = f"{job.phase or 'Installation'} cancelled"
        job.finish("CANCELLED", message)
        job.messages.put(("CANCELLED", message))
    except Exception as exc:
        job.log(
            "ERROR",
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
        message = _friendly_install_error(exc, job.phase)
        job.finish("ERROR", message)
        job.messages.put(("ERROR", message))
    finally:
        job.process = None
        if layout.staging.exists():
            shutil.rmtree(layout.staging, ignore_errors=True)
        if layout.model_staging.exists():
            job.log(
                "DETAIL",
                "Completed model downloads retained for the next retry",
            )
        job.log("DETAIL", "Temporary runtime staging cleaned")
        install_lock.release()


def is_model_loaded() -> bool:
    try:
        from .providers.sam3d_body_provider import model_loaded

        return model_loaded()
    except (ImportError, AttributeError, RuntimeError):
        return False


def _model_load_worker(
    job: BackgroundJob,
    variant: str,
    requested_device: str,
    resolved_runtime_paths: Mapping[str, str],
) -> None:
    try:
        job.update(
            0.05,
            "Opening ONNX Runtime graph bundle",
            phase="Model Load",
        )
        from .providers.sam3d_body_provider import load_model_cache

        job.update(
            0.20,
            "Creating the requested execution-provider session",
            phase="Model Load",
        )
        device = load_model_cache(
            variant,
            requested_device,
            dict(resolved_runtime_paths),
        )
        if job.cancel.is_set():
            unload_model()
            raise InterruptedError("Model loading cancelled")
        job.update(
            1.0,
            f"Model loaded on {device.upper()}",
            phase="Model Load",
        )
        job.finish("DONE", f"SAM 3D Body model loaded on {device.upper()}")
        job.messages.put(("DONE", f"SAM 3D Body model loaded on {device.upper()}"))
    except InterruptedError:
        message = "SAM 3D Body model loading cancelled"
        job.finish("CANCELLED", message)
        job.messages.put(("CANCELLED", message))
    except Exception as exc:
        job.log(
            "ERROR",
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
        message = _friendly_install_error(exc, "Model Load")
        job.finish("ERROR", message)
        job.messages.put(("ERROR", message))


def unload_model() -> None:
    from .providers.sam3d_body_provider import unload_model_cache

    unload_model_cache()


class _ModalJobOperator:
    _timer = None
    _job = None

    def _begin_modal(self, context, job):
        self._job = job
        job.started = True
        if job.token is not None:
            COORDINATOR.attach_worker(job.token, thread=job.thread)
        try:
            job.thread.start()
        except Exception as exc:
            job.finish("ERROR", f"Could not start background operation: {exc}")
            if job.token is not None:
                COORDINATOR.release(job.token)
            self.report({"ERROR"}, f"Could not start background operation: {exc}")
            return {"CANCELLED"}
        self._timer = context.window_manager.event_timer_add(
            0.15,
            window=context.window,
        )
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _finish_modal(self, context):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

    @staticmethod
    def _tag_redraw(context) -> None:
        for area in context.screen.areas if context.screen else ():
            area.tag_redraw()

    def modal(self, context, event):
        if event.type == "ESC":
            if self._job.token is not None:
                COORDINATOR.request_cancel(self._job.token)
            else:
                self._job.cancel.set()
            if self._job.process and self._job.process.poll() is None:
                self._job.process.terminate()
            return {"RUNNING_MODAL"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        try:
            kind, payload = self._job.messages.get_nowait()
        except queue.Empty:
            self._tag_redraw(context)
            return {"RUNNING_MODAL"}
        self._finish_modal(context)
        if self._job.token is not None:
            COORDINATOR.release(self._job.token)
        self._tag_redraw(context)
        if kind == "DONE":
            self.report({"INFO"}, payload)
            return {"FINISHED"}
        if kind == "CANCELLED":
            self.report({"WARNING"}, payload)
            return {"CANCELLED"}
        self.report({"ERROR"}, payload)
        return {"CANCELLED"}

    def cancel(self, context):
        if self._job:
            if self._job.token is not None:
                COORDINATOR.request_cancel(self._job.token)
            else:
                self._job.cancel.set()
        self._finish_modal(context)


class MTP_OT_sam3d_setup(Operator):
    bl_idname = "mtp.sam3d_setup"
    bl_label = "SAM 3D Body Setup"
    bl_description = "Install the public SAM 3D Body ONNX model bundle"

    def invoke(self, _context, _event):
        return bpy.ops.mtp.sam3d_install("INVOKE_DEFAULT")

    def execute(self, _context):
        return bpy.ops.mtp.sam3d_install("INVOKE_DEFAULT")


class MTP_OT_sam3d_install(_ModalJobOperator, Operator):
    bl_idname = "mtp.sam3d_install"
    bl_label = "Install SAM 3D Body"
    bl_description = "Download and install the optional SAM 3D Body backend"

    model_storage_directory: StringProperty(
        name="Model Storage Location",
        description=(
            "Parent directory where Media to Pose stores the downloaded "
            "SAM 3D Body ONNX graph bundle"
        ),
        subtype="DIR_PATH",
        options={"SKIP_SAVE"},
    )

    def invoke(self, context, _event):
        if COORDINATOR.is_busy():
            self.report({"INFO"}, "Media to Pose is already busy")
            return {"CANCELLED"}
        self.model_storage_directory = str(preferred_model_storage_parent())
        return context.window_manager.invoke_props_dialog(
            self,
            width=620,
            title=(
                "Repair SAM 3D Body" if backend_present() else "Install SAM 3D Body"
            ),
            confirm_text=("Repair" if backend_present() else "Install"),
        )

    def draw(self, context):
        layout = self.layout
        settings = context.scene.mtp_settings
        spec = model_spec(settings.sam3d_model_variant)
        size = int(spec.get("download_bytes") or 0)
        label = f"Download {spec['label']} ONNX graph bundle"
        if size:
            label += f" ({size / (1024**3):.1f} GiB)"
        layout.label(text=label, icon="IMPORT")
        layout.label(
            text=f"Source: {spec['repo']} @ {spec.get('revision') or 'not published'}",
            icon="PACKAGE",
        )
        layout.prop(self, "model_storage_directory")
        try:
            managed = model_storage_root(self.model_storage_directory)
            layout.label(
                text=f"Managed folder: {managed}",
                icon="FILE_FOLDER",
            )
        except (OSError, RuntimeError, ValueError):
            layout.label(
                text="Choose an accessible model storage location",
                icon="ERROR",
            )
        layout.label(
            text="ONNX Runtime is shipped with the extension; setup installs models only.",
            icon="INFO",
        )

    def execute(self, context):
        if COORDINATOR.is_busy():
            return {"CANCELLED"}
        settings = context.scene.mtp_settings
        if is_model_loaded():
            self.report({"ERROR"}, "Unload SAM 3D Body before repairing it")
            return {"CANCELLED"}
        try:
            storage_parent = _normalize_storage_parent(self.model_storage_directory)
        except (OSError, RuntimeError, ValueError) as exc:
            self.report(
                {"ERROR"},
                f"Invalid model storage location: {_safe_metadata(exc)}",
            )
            return {"CANCELLED"}
        install_layout = _new_install_layout(str(storage_parent))
        token = COORDINATOR.acquire(
            OperationKind.SAM3D_INSTALL,
            "sam3d-install",
            scene_name=context.scene.name,
            scene_pointer=context.scene.as_pointer(),
            status="Starting SAM 3D Body installation",
        )
        if token is None:
            self.report({"ERROR"}, "Media to Pose is already busy")
            return {"CANCELLED"}
        job = BackgroundJob(
            "Installing SAM 3D Body",
            token=token,
            request=SamSetupRequest(
                operation="install",
                model_variant=str(settings.sam3d_model_variant),
                device=str(settings.sam3d_device),
                model_storage_directory=str(storage_parent),
            ),
        )
        job.configure_log(install_log_path(create_root=True))
        job.log(
            "INFO",
            (
                "Starting SAM 3D Body installation "
                f"(model={settings.sam3d_model_variant}, "
                f"device={settings.sam3d_device})"
            ),
        )
        job.log("INFO", f"Selected model storage: {storage_parent}")
        job.thread = threading.Thread(
            target=_install_worker,
            args=(
                job,
                settings.sam3d_model_variant,
                settings.sam3d_device,
                install_layout,
            ),
            daemon=True,
            name="MTP-SAM3D-Installer",
        )
        if not SAM_JOBS.assign(SamJobSlot.INSTALL, job):
            COORDINATOR.finish(
                token,
                OperationState.FAILED,
                "Installation ownership was superseded",
            )
            COORDINATOR.release(token)
            self.report({"ERROR"}, "Could not acquire installation ownership")
            return {"CANCELLED"}
        settings.progress = 0.0
        settings.status = "Starting SAM 3D Body installation"
        settings.status_kind = "PROGRESS"
        result = self._begin_modal(context, job)
        return result

    def modal(self, context, event):
        scene = (
            bpy.data.scenes.get(self._job.token.scene_name)
            if self._job and self._job.token
            else context.scene
        )
        if scene is None:
            if self._job and self._job.token:
                COORDINATOR.request_cancel(
                    self._job.token, status="Originating scene was removed"
                )
            return super().modal(context, event)
        settings = scene.mtp_settings
        job = self._job
        if job and SAM_JOBS.current(SamJobSlot.INSTALL) is job:
            settings.progress = job.progress * 100.0
            settings.status = job.status
            settings.status_kind = (
                "PROGRESS"
                if not job.terminal
                else "READY"
                if job.outcome in {"DONE", "SUCCESS"}
                else "WARNING"
                if job.outcome == "CANCELLED"
                else "ERROR"
            )
        result = super().modal(context, event)
        if (
            result not in ({"RUNNING_MODAL"}, {"PASS_THROUGH"})
            and job
            and SAM_JOBS.current(SamJobSlot.INSTALL) is job
        ):
            settings.progress = job.progress * 100.0
            settings.status = job.status
            settings.status_kind = (
                "READY"
                if job.outcome in {"DONE", "SUCCESS"}
                else "WARNING"
                if job.outcome == "CANCELLED"
                else "ERROR"
            )
        return result


class MTP_OT_sam3d_cancel_install(Operator):
    bl_idname = "mtp.sam3d_cancel_install"
    bl_label = "Cancel SAM 3D Body Installation"

    def execute(self, _context):
        job = SAM_JOBS.current(SamJobSlot.INSTALL)
        if not job:
            return {"CANCELLED"}
        if job.token is not None:
            COORDINATOR.request_cancel(
                job.token,
                status="Cancelling installation safely",
            )
        else:
            job.cancel.set()
        job.update(
            job.progress,
            "Cancelling installation safely",
            phase=job.phase,
        )
        return {"FINISHED"}


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


class MTP_OT_sam3d_copy_install_diagnostics(Operator):
    bl_idname = "mtp.sam3d_copy_install_diagnostics"
    bl_label = "Copy SAM Installation Diagnostics"
    bl_description = "Copy redacted SAM installation details and recent logs"

    def execute(self, context):
        context.window_manager.clipboard = install_diagnostics_text()
        self.report({"INFO"}, "Redacted installation diagnostics copied")
        return {"FINISHED"}


class MTP_OT_sam3d_open_install_log(Operator):
    bl_idname = "mtp.sam3d_open_install_log"
    bl_label = "Open SAM Installation Log"
    bl_description = "Open the retained SAM installation log"

    def execute(self, _context):
        snapshot = install_status_snapshot()
        path = (
            Path(snapshot.log_path)
            if snapshot.log_path
            else (install_log_path(create_root=False))
        )
        if not path.is_file():
            self.report({"WARNING"}, "No SAM installation log is available")
            return {"CANCELLED"}
        try:
            bpy.ops.wm.path_open(filepath=str(path))
        except RuntimeError:
            try:
                if os.name == "nt":
                    os.startfile(str(path))
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(path)])
                else:
                    subprocess.Popen(["xdg-open", str(path)])
            except OSError as exc:
                self.report(
                    {"ERROR"},
                    f"Could not open install log: {_safe_metadata(exc)}",
                )
                return {"CANCELLED"}
        return {"FINISHED"}


class MTP_OT_sam3d_dismiss_install_result(Operator):
    bl_idname = "mtp.sam3d_dismiss_install_result"
    bl_label = "Dismiss SAM Installation Result"

    def execute(self, context):
        if install_running():
            return {"CANCELLED"}
        SAM_JOBS.clear(SamJobSlot.INSTALL)
        settings = context.scene.mtp_settings
        settings.progress = 0.0
        settings.status = "Ready"
        settings.status_kind = "READY"
        return {"FINISHED"}


class MTP_OT_sam3d_load(_ModalJobOperator, Operator):
    bl_idname = "mtp.sam3d_load"
    bl_label = "Load SAM 3D Body Model"
    bl_description = "Load the installed model into RAM or VRAM for capture"

    def execute(self, context):
        settings = context.scene.mtp_settings
        if COORDINATOR.is_busy():
            self.report({"WARNING"}, "Media to Pose is already busy")
            return {"CANCELLED"}
        if not sam_backend_ready(settings):
            self.report({"ERROR"}, "Repair the SAM 3D Body backend first")
            return {"CANCELLED"}
        if is_model_loaded():
            self.report({"INFO"}, "SAM 3D Body model is already loaded")
            return {"FINISHED"}
        resolved_runtime_paths = runtime_paths()
        token = COORDINATOR.acquire(
            OperationKind.SAM3D_MODEL_LOAD,
            "sam3d-model-load",
            scene_name=context.scene.name,
            scene_pointer=context.scene.as_pointer(),
            status="Loading SAM 3D Body model",
        )
        if token is None:
            self.report({"WARNING"}, "Media to Pose is already busy")
            return {"CANCELLED"}
        job = BackgroundJob(
            "Loading SAM 3D Body model",
            token=token,
            request=SamSetupRequest(
                operation="load",
                model_variant=str(settings.sam3d_model_variant),
                device=str(settings.sam3d_device),
            ),
        )
        if not SAM_JOBS.assign(SamJobSlot.MODEL, job):
            COORDINATOR.finish(
                token,
                OperationState.FAILED,
                "Model-load ownership was superseded",
            )
            COORDINATOR.release(token)
            self.report({"ERROR"}, "Could not acquire model-load ownership")
            return {"CANCELLED"}
        job.thread = threading.Thread(
            target=_model_load_worker,
            args=(
                job,
                str(settings.sam3d_model_variant),
                str(settings.sam3d_device),
                resolved_runtime_paths,
            ),
            daemon=True,
            name="MTP-SAM3D-ModelLoad",
        )
        settings.progress = 0.0
        settings.status = "Loading SAM 3D Body model"
        settings.status_kind = "PROGRESS"
        return self._begin_modal(context, job)

    def modal(self, context, event):
        scene = (
            bpy.data.scenes.get(self._job.token.scene_name)
            if self._job and self._job.token
            else context.scene
        )
        if scene is None:
            if self._job and self._job.token:
                COORDINATOR.request_cancel(
                    self._job.token,
                    status="Originating scene was removed",
                )
            return super().modal(context, event)
        settings = scene.mtp_settings
        job = self._job
        if job and SAM_JOBS.current(SamJobSlot.MODEL) is job:
            settings.progress = job.progress * 100.0
            settings.status = job.status
            settings.status_kind = (
                "PROGRESS"
                if not job.terminal
                else "READY"
                if job.outcome in {"DONE", "SUCCESS"}
                else "WARNING"
                if job.outcome == "CANCELLED"
                else "ERROR"
            )
        result = super().modal(context, event)
        if (
            result not in ({"RUNNING_MODAL"}, {"PASS_THROUGH"})
            and job
            and SAM_JOBS.current(SamJobSlot.MODEL) is job
        ):
            settings.progress = job.progress * 100.0
            settings.status = job.status
        return result


class MTP_OT_sam3d_unload(Operator):
    bl_idname = "mtp.sam3d_unload"
    bl_label = "Unload SAM 3D Body"
    bl_description = "Release the loaded SAM 3D Body model and GPU memory"

    def execute(self, _context):
        token = COORDINATOR.acquire(
            OperationKind.SAM3D_MODEL_UNLOAD,
            "sam3d-model-unload",
            status="Unloading SAM 3D Body model",
        )
        if token is None:
            self.report({"WARNING"}, "Media to Pose is already busy")
            return {"CANCELLED"}
        try:
            unload_model()
        except Exception as exc:
            COORDINATOR.finish(token, OperationState.FAILED, str(exc))
            COORDINATOR.release(token)
            self.report({"ERROR"}, _safe_metadata(exc))
            return {"CANCELLED"}
        COORDINATOR.finish(token, OperationState.SUCCEEDED, "Model unloaded")
        COORDINATOR.release(token)
        self.report({"INFO"}, "SAM 3D Body model unloaded")
        return {"FINISHED"}


class MTP_OT_sam3d_uninstall(Operator):
    bl_idname = "mtp.sam3d_uninstall"
    bl_label = "Uninstall SAM 3D Body"

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, _context):
        token = COORDINATOR.acquire(
            OperationKind.SAM3D_UNINSTALL,
            "sam3d-uninstall",
            status="Uninstalling SAM 3D Body",
        )
        if token is None:
            self.report({"ERROR"}, "Media to Pose is already busy")
            return {"CANCELLED"}
        if is_model_loaded():
            COORDINATOR.finish(
                token,
                OperationState.FAILED,
                "Unload SAM 3D Body before uninstalling",
            )
            COORDINATOR.release(token)
            self.report({"ERROR"}, "Unload SAM 3D Body before uninstalling")
            return {"CANCELLED"}
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
                shutil.rmtree(_verified_model_storage_root(storage_root))
            if root.exists():
                shutil.rmtree(root)
        except (OSError, RuntimeError, ValueError) as exc:
            message = f"Could not remove SAM 3D Body: {_safe_metadata(exc)}"
            self.report({"ERROR"}, message)
            COORDINATOR.finish(token, OperationState.FAILED, message)
            COORDINATOR.release(token)
            return {"CANCELLED"}
        finally:
            invalidate_backend_snapshot()
        COORDINATOR.finish(token, OperationState.SUCCEEDED, "SAM 3D Body uninstalled")
        COORDINATOR.release(token)
        self.report({"INFO"}, "SAM 3D Body backend removed")
        return {"FINISHED"}


CLASSES = (
    MTP_OT_sam3d_setup,
    MTP_OT_sam3d_install,
    MTP_OT_sam3d_cancel_install,
    MTP_OT_sam3d_copy_install_diagnostics,
    MTP_OT_sam3d_open_install_log,
    MTP_OT_sam3d_dismiss_install_result,
    MTP_OT_sam3d_load,
    MTP_OT_sam3d_unload,
    MTP_OT_sam3d_uninstall,
)


def register() -> None:
    invalidate_backend_snapshot()
    register_classes(CLASSES)


def unregister() -> None:
    invalidate_backend_snapshot()
    jobs = SAM_JOBS.jobs()
    for job in jobs:
        if job:
            if job.token is not None:
                COORDINATOR.request_cancel(
                    job.token,
                    status="Cancelling during add-on shutdown",
                )
            else:
                job.cancel.set()
            if job.process and job.process.poll() is None:
                job.process.terminate()
    for job in jobs:
        if (
            job
            and job.thread
            and job.thread.is_alive()
            and job.thread is not threading.current_thread()
        ):
            job.thread.join(timeout=0.5)
    for slot in SamJobSlot:
        job = SAM_JOBS.current(slot)
        if not (job and job.thread and job.thread.is_alive()):
            SAM_JOBS.clear(slot, job)
    if not model_load_running():
        unload_model()
    unregister_classes(CLASSES)
