"""Download, validate, and own the optional immutable RoHM ONNX bundle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Optional

import bpy
from bpy.props import StringProperty
from bpy.types import Operator

from . import rohm_ort_runtime, sam3d_backend, sam3d_ort_runtime
from .model_lifecycle import model_installation_lock
from .operation import COORDINATOR, OperationKind, OperationState
from .registration import register_classes, unregister_classes

HF_ENDPOINT = "https://huggingface.co"
MODEL_STORAGE_FOLDER = "media_to_pose_rohm"
MODEL_STORAGE_MARKER = ".media_to_pose_storage.json"
MODEL_SPEC = {
    "repo": "KitsuMate/rohm-onnx",
    "revision": "21bb4c53d1265c522e3962a466c4bb9e41169a36",
    "upstream_revision": "57ba22cde1595285276ddb88c8c82b25bf0d9722",
    "manifest_path": "bundles/fp32/manifest.json",
    "precision": "fp32",
    "download_bytes": 222_030_103,
}


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


@dataclass(frozen=True)
class JobSnapshot:
    running: bool = False
    progress: float = 0.0
    phase: str = ""
    status: str = ""
    outcome: str = ""
    error: str = ""


class _Job:
    def __init__(self, token, label: str):
        self.token = token
        self.cancel = token.cancel_event
        self.progress = 0.0
        self.phase = ""
        self.status = label
        self.outcome = ""
        self.error = ""
        self.thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

    def update(self, progress: float, status: str, phase: str) -> None:
        with self._lock:
            self.progress = min(max(float(progress), 0.0), 1.0)
            self.status = str(status)
            self.phase = str(phase)
        COORDINATOR.update(
            self.token,
            progress=self.progress,
            status=self.status,
            phase=self.phase,
        )

    def finish(self, outcome: str, message: str) -> None:
        with self._lock:
            self.outcome = outcome
            self.status = str(message)
            self.error = str(message) if outcome == "ERROR" else ""
        state = {
            "DONE": OperationState.SUCCEEDED,
            "CANCELLED": OperationState.CANCELLED,
        }.get(outcome, OperationState.FAILED)
        COORDINATOR.finish(self.token, state, str(message))

    def snapshot(self) -> JobSnapshot:
        with self._lock:
            return JobSnapshot(
                running=bool(
                    self.thread and self.thread.is_alive() and not self.outcome
                ),
                progress=self.progress,
                phase=self.phase,
                status=self.status,
                outcome=self.outcome,
                error=self.error,
            )


_STATE_LOCK = threading.RLock()
_INSTALL_JOB: Optional[_Job] = None
_LOAD_JOB: Optional[_Job] = None
_SESSION: Optional[rohm_ort_runtime.RohmSession] = None
_LAST_DIAGNOSTIC = ""


def rohm_device_enum_items():
    return (
        ("AUTO", "Auto", "Prefer GPU acceleration, then fall back to CPU"),
        ("GPU", "GPU", "Require WebGPU or CoreML acceleration"),
        ("CPU", "CPU", "Use ONNX Runtime CPU inference"),
    )


def is_published() -> bool:
    return bool(MODEL_SPEC.get("repo") and MODEL_SPEC.get("manifest_path"))


def backend_root(*, create: bool = True) -> Path:
    value = bpy.utils.user_resource(
        "SCRIPTS", path="media_to_pose/backends/rohm", create=create
    )
    if not value:
        scripts = bpy.utils.user_resource("SCRIPTS", create=create)
        if not scripts:
            raise RuntimeError("Blender user scripts directory is unavailable")
        value = str(Path(scripts) / "media_to_pose" / "backends" / "rohm")
    return Path(value)


def _config_path(*, create: bool = True) -> Path:
    return backend_root(create=create) / "storage.json"


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf8")
    os.replace(temporary, path)


def configured_storage_parent() -> Path:
    configured = str(_read_json(_config_path(create=False)).get("storage_parent") or "")
    if configured:
        return Path(bpy.path.abspath(os.path.expandvars(configured))).resolve(False)
    return sam3d_backend.preferred_model_storage_parent().resolve(False)


def model_storage_root(parent: str = "") -> Path:
    selected = (
        Path(bpy.path.abspath(os.path.expandvars(parent))).resolve(False)
        if parent
        else configured_storage_parent()
    )
    if selected.name.casefold() == MODEL_STORAGE_FOLDER.casefold():
        selected = selected.parent
    return selected / MODEL_STORAGE_FOLDER


def _ensure_storage(root: Path) -> None:
    marker = root / MODEL_STORAGE_MARKER
    if root.exists() and any(root.iterdir()) and not marker.is_file():
        raise OSError(f"Refusing unmanaged RoHM storage directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    expected = {"kind": "rohm_models", "owner": "media_to_pose"}
    existing = _read_json(marker)
    if existing and existing != expected:
        raise OSError("RoHM storage is owned by another application")
    _write_json(marker, expected)


def active_directory() -> Path:
    return model_storage_root() / "active"


def manifest_path() -> Path:
    return active_directory() / "manifest.json"


def backend_present() -> bool:
    return manifest_path().is_file()


def bundle_url() -> str:
    return f"{HF_ENDPOINT}/{MODEL_SPEC['repo']}"


def _hf_url(path: str, revision: str = "") -> str:
    relative = PurePosixPath(str(path))
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(path):
        raise RuntimeError(f"Unsafe RoHM download path: {path!r}")
    repository = "/".join(
        urllib.parse.quote(part, safe="") for part in str(MODEL_SPEC["repo"]).split("/")
    )
    encoded = "/".join(urllib.parse.quote(part, safe="") for part in relative.parts)
    selected_revision = revision or str(MODEL_SPEC.get("revision") or "main")
    return f"{HF_ENDPOINT}/{repository}/resolve/{selected_revision}/{encoded}"


def _download(
    job: _Job,
    remote_path: str,
    destination: Path,
    *,
    revision: str = "",
    expected_bytes: int = 0,
    expected_sha256: str = "",
    progress_start: float,
    progress_end: float,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    request = urllib.request.Request(
        _hf_url(remote_path, revision), headers={"User-Agent": "Media-to-Pose/RoHM"}
    )
    digest = hashlib.sha256()
    received = 0
    try:
        with (
            urllib.request.urlopen(request, timeout=15) as response,
            partial.open("xb") as output,
        ):
            total = int(response.headers.get("Content-Length") or expected_bytes or 0)
            while True:
                if job.cancel.is_set():
                    raise InterruptedError("RoHM download cancelled")
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                ratio = received / total if total else 0.0
                job.update(
                    progress_start + (progress_end - progress_start) * ratio,
                    f"Downloading {destination.name} · {received / (1024**2):.0f} MiB",
                    "Download",
                )
        if expected_bytes and received != expected_bytes:
            raise RuntimeError(f"{destination.name} has an unexpected byte size")
        if (
            expected_sha256
            and digest.hexdigest().casefold() != expected_sha256.casefold()
        ):
            raise RuntimeError(f"{destination.name} failed SHA-256 verification")
        os.replace(partial, destination)
    except Exception:
        with suppress(OSError):
            partial.unlink()
        raise


def _install_worker(
    job: _Job,
    storage_root: str,
    storage_config_path: str,
) -> None:
    root = Path(storage_root)
    staging = root / f".staging-{os.getpid()}-{uuid.uuid4().hex}"
    backup = root / ".previous"
    current = root / "active"
    install_lock = model_installation_lock(root, "RoHM")
    try:
        install_lock.acquire()
        _ensure_storage(root)
        staging.mkdir(parents=True)
        manifest_remote = str(MODEL_SPEC["manifest_path"])
        manifest_local = staging / "manifest.json"
        revision = str(MODEL_SPEC.get("revision") or "")
        if not sam3d_ort_runtime.is_immutable_revision(revision):
            revision = "main"
        job.update(0.01, "Downloading RoHM manifest", "Manifest")
        try:
            _download(
                job,
                manifest_remote,
                manifest_local,
                revision=revision,
                progress_start=0.01,
                progress_end=0.05,
            )
        except urllib.error.HTTPError as exc:
            if revision == "main" or exc.code not in {404, 410}:
                raise
            revision = "main"
            job.update(
                0.01,
                "Preferred RoHM bundle unavailable; using latest",
                "Manifest",
            )
            _download(
                job,
                manifest_remote,
                manifest_local,
                revision=revision,
                progress_start=0.01,
                progress_end=0.05,
            )
        if job.cancel.is_set():
            raise InterruptedError("RoHM installation cancelled")
        bundle = rohm_ort_runtime.load_bundle(
            staging,
            verify_files=False,
        )
        remote_root = str(PurePosixPath(manifest_remote).parent)
        assets = sorted(bundle.assets.values(), key=lambda item: item.path)
        total_bytes = sum(item.bytes for item in assets)
        completed = 0
        for item in assets:
            start = 0.05 + 0.80 * completed / max(total_bytes, 1)
            completed += item.bytes
            end = 0.05 + 0.80 * completed / max(total_bytes, 1)
            _download(
                job,
                str(PurePosixPath(remote_root) / item.path),
                staging.joinpath(*PurePosixPath(item.path).parts),
                expected_bytes=item.bytes,
                expected_sha256=item.sha256,
                revision=revision,
                progress_start=start,
                progress_end=end,
            )
        job.update(0.88, "Verifying RoHM bundle", "Verification")
        rohm_ort_runtime.load_bundle(staging)
        if job.cancel.is_set():
            raise InterruptedError("RoHM installation cancelled")
        job.update(0.96, "Activating verified bundle", "Activation")
        if backup.exists():
            shutil.rmtree(backup)
        if current.exists():
            os.replace(current, backup)
        try:
            os.replace(staging, current)
        except Exception:
            if backup.exists() and not current.exists():
                os.replace(backup, current)
            raise
        shutil.rmtree(backup, ignore_errors=True)
        _write_json(Path(storage_config_path), {"storage_parent": str(root.parent)})
        invalidate_backend_snapshot()
        job.update(1.0, "RoHM installed and verified", "Complete")
        job.finish("DONE", "RoHM installed and verified")
    except InterruptedError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        job.finish("CANCELLED", str(exc))
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        job.finish("ERROR", f"RoHM installation failed: {exc}")
    finally:
        install_lock.release()


def install_status_snapshot() -> JobSnapshot:
    with _STATE_LOCK:
        return _INSTALL_JOB.snapshot() if _INSTALL_JOB else JobSnapshot()


def model_load_status_snapshot() -> JobSnapshot:
    with _STATE_LOCK:
        return _LOAD_JOB.snapshot() if _LOAD_JOB else JobSnapshot()


def invalidate_backend_snapshot() -> None:
    backend_snapshot.cache_clear()


@lru_cache(maxsize=6)
def backend_snapshot(requested_mode: object = "AUTO") -> BackendSnapshot:
    requested = sam3d_ort_runtime.normalize_mode(requested_mode)
    with _STATE_LOCK:
        session = _SESSION
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
    path = manifest_path()
    if not path.is_file():
        return BackendSnapshot(
            state="NOT_INSTALLED",
            message="RoHM is not installed",
            requested_mode=requested,
            precision=str(MODEL_SPEC["precision"]),
            download_bytes=int(MODEL_SPEC["download_bytes"]),
            model_directory=str(active_directory()),
        )
    try:
        bundle = rohm_ort_runtime.load_bundle(
            active_directory(),
            verify_files=False,
        )
        ort = sam3d_ort_runtime.import_onnxruntime()
        available = sam3d_ort_runtime.available_execution_providers(ort)
        provider, actual_device, fallback = sam3d_ort_runtime.select_execution_provider(
            requested, available
        )
    except Exception as exc:
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


def verify_bundle(bundle_directory: str = "") -> BackendSnapshot:
    invalidate_backend_snapshot()
    directory = Path(bundle_directory) if bundle_directory else active_directory()
    if not (directory / "manifest.json").is_file():
        raise RuntimeError("RoHM is not installed")
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
    with _STATE_LOCK:
        if _SESSION is None:
            raise RuntimeError("RoHM is not loaded")
        return _SESSION


def is_model_loaded() -> bool:
    with _STATE_LOCK:
        return _SESSION is not None


def ensure_model_loaded(
    device: object,
    bundle_directory: str = "",
) -> rohm_ort_runtime.RohmSession:
    global _LAST_DIAGNOSTIC, _SESSION
    requested = sam3d_ort_runtime.normalize_mode(device)
    with _STATE_LOCK:
        if _SESSION is not None and _SESSION.diagnostics.requested_mode == requested:
            return _SESSION
        _SESSION = None
    directory = Path(bundle_directory) if bundle_directory else active_directory()
    verify_bundle(str(directory))
    bundle = rohm_ort_runtime.load_bundle(directory)
    session = rohm_ort_runtime.RohmSession(bundle, requested)
    with _STATE_LOCK:
        if _SESSION is None:
            _SESSION = session
        _LAST_DIAGNOSTIC = "Bundle verification and runtime session creation passed"
        result = _SESSION
    invalidate_backend_snapshot()
    return result


def _load_worker(job: _Job, device: object, bundle_directory: str) -> None:
    global _LAST_DIAGNOSTIC
    try:
        job.update(0.15, "Verifying model files", "Verification")
        verify_bundle(bundle_directory)
        if job.cancel.is_set():
            raise InterruptedError("RoHM load cancelled")
        job.update(0.45, "Creating ONNX Runtime sessions", "Loading")
        ensure_model_loaded(device, bundle_directory)
        if job.cancel.is_set():
            unload_model()
            raise InterruptedError("RoHM load cancelled")
        job.update(1.0, "RoHM loaded", "Complete")
        job.finish("DONE", "RoHM loaded")
    except InterruptedError as exc:
        job.finish("CANCELLED", str(exc))
    except Exception as exc:
        with _STATE_LOCK:
            _LAST_DIAGNOSTIC = str(exc)
        job.finish("ERROR", f"RoHM loading failed: {exc}")


def unload_model() -> None:
    global _SESSION
    with _STATE_LOCK:
        _SESSION = None
    invalidate_backend_snapshot()


def last_diagnostic() -> str:
    with _STATE_LOCK:
        return _LAST_DIAGNOSTIC


class MTP_OT_rohm_install(Operator):
    bl_idname = "mtp.rohm_install"
    bl_label = "Install AI Motion Repair"
    bl_description = "Download, verify, repair, or move the RoHM model"

    model_storage_directory: StringProperty(
        name="Model Storage Parent",
        subtype="DIR_PATH",
        default="",
    )
    _timer = None
    _scene_name = ""

    @classmethod
    def poll(cls, _context):
        if install_status_snapshot().running:
            cls.poll_message_set("AI motion model installation is already running")
            return False
        if COORDINATOR.is_busy():
            cls.poll_message_set("Media to Pose is busy with another operation")
            return False
        return True

    def invoke(self, context, _event):
        if install_status_snapshot().running or COORDINATOR.is_busy():
            message = (
                "AI motion model installation is already running"
                if install_status_snapshot().running
                else "Media to Pose is busy with another operation"
            )
            settings = getattr(context.scene, "mtp_settings", None)
            if settings is not None:
                settings.status = message
                settings.status_kind = "WARNING"
            self.report({"WARNING"}, message)
            return {"CANCELLED"}
        self.model_storage_directory = str(configured_storage_parent())
        return context.window_manager.invoke_props_dialog(self, width=540)

    def draw(self, _context):
        layout = self.layout
        size = int(MODEL_SPEC["download_bytes"])
        layout.label(
            text=f"Download: {size / (1024**2):.0f} MiB · processing remains local",
            icon="IMPORT",
        )
        layout.label(text=f"Source: {MODEL_SPEC['repo']}", icon="WORLD")
        layout.label(
            text=f"Preferred revision: {MODEL_SPEC['revision']}", icon="LOCKED"
        )
        layout.prop(self, "model_storage_directory")

    def execute(self, _context):
        global _INSTALL_JOB
        if install_status_snapshot().running or COORDINATOR.is_busy():
            self.report({"WARNING"}, "Media to Pose is busy")
            return {"CANCELLED"}
        if not is_published():
            self.report({"ERROR"}, "RoHM ONNX bundle is not published")
            return {"CANCELLED"}
        storage_root = str(model_storage_root(self.model_storage_directory))
        storage_config_path = str(_config_path())
        token = COORDINATOR.acquire(
            OperationKind.ROHM_INSTALL,
            "rohm-install",
            status="Starting RoHM download",
        )
        if token is None:
            self.report({"ERROR"}, "Media to Pose is busy")
            return {"CANCELLED"}
        job = _Job(token, "Starting RoHM download")
        job.thread = threading.Thread(
            target=_install_worker,
            args=(job, storage_root, storage_config_path),
            daemon=True,
            name="MTP-RoHM-Install",
        )
        with _STATE_LOCK:
            _INSTALL_JOB = job
        COORDINATOR.attach_worker(token, thread=job.thread)
        try:
            job.thread.start()
        except Exception as exc:
            job.finish("ERROR", f"Could not start AI motion model download: {exc}")
            COORDINATOR.release(token)
            self.report({"ERROR"}, f"Could not start AI motion model download: {exc}")
            return {"CANCELLED"}
        self._scene_name = _context.scene.name
        _context.scene.mtp_settings.progress = 0.0
        _context.scene.mtp_settings.status = "Starting AI motion model download"
        _context.scene.mtp_settings.status_kind = "PROGRESS"
        self._timer = _context.window_manager.event_timer_add(
            0.15, window=_context.window
        )
        _context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "ESC":
            with _STATE_LOCK:
                job = _INSTALL_JOB
            if job is not None:
                COORDINATOR.request_cancel(
                    job.token, status="Cancelling AI motion model download"
                )
            return {"RUNNING_MODAL"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        with _STATE_LOCK:
            job = _INSTALL_JOB
        snapshot = job.snapshot() if job is not None else JobSnapshot()
        scene = bpy.data.scenes.get(self._scene_name)
        if scene is not None:
            scene.mtp_settings.progress = snapshot.progress * 100.0
            scene.mtp_settings.status = snapshot.status
            scene.mtp_settings.status_kind = (
                "PROGRESS"
                if snapshot.running
                else "READY"
                if snapshot.outcome == "DONE"
                else "WARNING"
                if snapshot.outcome == "CANCELLED"
                else "ERROR"
            )
        for area in context.screen.areas if context.screen else ():
            area.tag_redraw()
        if job is not None and (snapshot.running or job.thread.is_alive()):
            return {"RUNNING_MODAL"}
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        if job is not None:
            COORDINATOR.release(job.token)
        if snapshot.outcome == "DONE":
            self.report({"INFO"}, snapshot.status)
            return {"FINISHED"}
        self.report(
            {"WARNING" if snapshot.outcome == "CANCELLED" else "ERROR"},
            snapshot.status or "AI motion model installation stopped",
        )
        return {"CANCELLED"}

    def cancel(self, context):
        with _STATE_LOCK:
            job = _INSTALL_JOB
        if job is not None:
            COORDINATOR.request_cancel(
                job.token, status="Cancelling AI motion model download"
            )
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None


class MTP_OT_rohm_cancel_install(Operator):
    bl_idname = "mtp.rohm_cancel_install"
    bl_label = "Cancel RoHM Download"

    def execute(self, _context):
        with _STATE_LOCK:
            job = _INSTALL_JOB
        if job and not job.cancel.is_set():
            COORDINATOR.request_cancel(job.token, status="Cancelling RoHM download")
            job.update(job.progress, "Cancelling safely", job.phase)
            return {"FINISHED"}
        return {"CANCELLED"}


class MTP_OT_rohm_load(Operator):
    bl_idname = "mtp.rohm_load"
    bl_label = "Load RoHM"

    def execute(self, context):
        global _LOAD_JOB
        bundle_directory = str(active_directory())
        token = COORDINATOR.acquire(
            OperationKind.ROHM_MODEL_LOAD,
            "rohm-load",
            status="Loading RoHM",
        )
        if token is None:
            self.report({"ERROR"}, "Media to Pose is busy")
            return {"CANCELLED"}
        job = _Job(token, "Loading RoHM")
        job.thread = threading.Thread(
            target=_load_worker,
            args=(
                job,
                context.scene.mtp_settings.rohm_device,
                bundle_directory,
            ),
            daemon=True,
            name="MTP-RoHM-Load",
        )
        with _STATE_LOCK:
            _LOAD_JOB = job
        COORDINATOR.attach_worker(token, thread=job.thread)
        try:
            job.thread.start()
        except Exception as exc:
            job.finish("ERROR", f"Could not start AI motion model load: {exc}")
            COORDINATOR.release(token)
            self.report({"ERROR"}, f"Could not start AI motion model load: {exc}")
            return {"CANCELLED"}
        return {"FINISHED"}


class MTP_OT_rohm_unload(Operator):
    bl_idname = "mtp.rohm_unload"
    bl_label = "Unload RoHM"

    def execute(self, _context):
        token = COORDINATOR.acquire(
            OperationKind.ROHM_MODEL_UNLOAD,
            "rohm-unload",
            status="Unloading RoHM",
        )
        if token is None:
            self.report({"ERROR"}, "Media to Pose is busy")
            return {"CANCELLED"}
        unload_model()
        COORDINATOR.finish(token, OperationState.SUCCEEDED, "RoHM unloaded")
        COORDINATOR.release(token)
        return {"FINISHED"}


class MTP_OT_rohm_verify(Operator):
    bl_idname = "mtp.rohm_verify"
    bl_label = "Verify RoHM"

    def execute(self, context):
        global _LAST_DIAGNOSTIC
        token = COORDINATOR.acquire(
            OperationKind.ROHM_MODEL_LOAD,
            "rohm-verify",
            status="Verifying RoHM",
        )
        if token is None:
            self.report({"WARNING"}, "Media to Pose is busy")
            return {"CANCELLED"}
        was_loaded = is_model_loaded()
        try:
            snapshot = verify_bundle()
            ensure_model_loaded(context.scene.mtp_settings.rohm_device)
            _LAST_DIAGNOSTIC = (
                "All declared files passed size/SHA-256 verification and "
                "runtime session creation passed"
            )
            if not was_loaded:
                unload_model()
            COORDINATOR.finish(token, OperationState.SUCCEEDED, "RoHM verified")
            COORDINATOR.release(token)
            self.report({"INFO"}, snapshot.message)
            return {"FINISHED"}
        except Exception as exc:
            if not was_loaded:
                unload_model()
            _LAST_DIAGNOSTIC = str(exc)
            COORDINATOR.finish(token, OperationState.FAILED, str(exc))
            COORDINATOR.release(token)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


class MTP_OT_rohm_uninstall(Operator):
    bl_idname = "mtp.rohm_uninstall"
    bl_label = "Uninstall RoHM"
    bl_options = {"INTERNAL"}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, _context):
        token = COORDINATOR.acquire(
            OperationKind.ROHM_UNINSTALL,
            "rohm-uninstall",
            status="Uninstalling RoHM",
        )
        if token is None:
            self.report({"ERROR"}, "Media to Pose is busy")
            return {"CANCELLED"}
        install_lock = None
        try:
            root = model_storage_root().resolve(False)
            install_lock = model_installation_lock(root, "RoHM")
            install_lock.acquire()
            unload_model()
            marker = _read_json(root / MODEL_STORAGE_MARKER)
            if root.name.casefold() != MODEL_STORAGE_FOLDER.casefold() or marker != {
                "kind": "rohm_models",
                "owner": "media_to_pose",
            }:
                raise OSError("Refusing to remove unrecognized RoHM storage")
            if root.exists():
                shutil.rmtree(root)
            invalidate_backend_snapshot()
        except Exception as exc:
            COORDINATOR.finish(token, OperationState.FAILED, str(exc))
            COORDINATOR.release(token)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        finally:
            if install_lock is not None:
                install_lock.release()
        COORDINATOR.finish(token, OperationState.SUCCEEDED, "RoHM uninstalled")
        COORDINATOR.release(token)
        return {"FINISHED"}


CLASSES = (
    MTP_OT_rohm_install,
    MTP_OT_rohm_cancel_install,
    MTP_OT_rohm_load,
    MTP_OT_rohm_unload,
    MTP_OT_rohm_verify,
    MTP_OT_rohm_uninstall,
)


def register() -> None:
    invalidate_backend_snapshot()
    register_classes(CLASSES)


def unregister() -> None:
    with _STATE_LOCK:
        jobs = tuple(job for job in (_INSTALL_JOB, _LOAD_JOB) if job)
    for job in jobs:
        COORDINATOR.request_cancel(job.token, status="Cancelling RoHM operation")
        if job.thread:
            job.thread.join(timeout=2.0)
    unload_model()
    unregister_classes(CLASSES)


__all__ = [
    "MODEL_SPEC",
    "backend_present",
    "backend_snapshot",
    "bundle_url",
    "ensure_model_loaded",
    "install_status_snapshot",
    "is_model_loaded",
    "is_published",
    "last_diagnostic",
    "loaded_session",
    "model_load_status_snapshot",
    "rohm_device_enum_items",
]
