"""Download, validate, and own the optional immutable AnyCalib ONNX bundle."""

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

import bpy
from bpy.props import StringProperty
from bpy.types import Operator

from . import anycalib_ort_runtime, sam3d_backend, sam3d_ort_runtime
from .model_lifecycle import model_installation_lock
from .operation import COORDINATOR, OperationKind, OperationState
from .registration import register_classes, unregister_classes

HF_ENDPOINT = "https://huggingface.co"
MODEL_STORAGE_FOLDER = "media_to_pose_anycalib"
MODEL_STORAGE_MARKER = ".media_to_pose_storage.json"
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
        self.thread: threading.Thread | None = None
        self._lock = threading.RLock()

    def update(self, value: float, message: str, phase: str = "") -> None:
        with self._lock:
            self.progress = min(max(float(value), 0.0), 1.0)
            self.status = str(message)
            self.phase = str(phase)
        COORDINATOR.update(
            self.token, progress=self.progress, status=self.status, phase=phase
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


_LOCK = threading.RLock()
_INSTALL_JOB: _Job | None = None
_LOAD_JOB: _Job | None = None
_SESSION: anycalib_ort_runtime.AnyCalibSession | None = None
_LAST_DIAGNOSTIC = ""


def is_published() -> bool:
    variants = MODEL_SPEC.get("variants") or {}
    return bool(
        MODEL_SPEC.get("repo")
        and any(spec.get("manifest_path") for spec in variants.values())
    )


def backend_root(*, create: bool = True) -> Path:
    value = bpy.utils.user_resource(
        "SCRIPTS", path="media_to_pose/backends/anycalib", create=create
    )
    if not value:
        scripts = bpy.utils.user_resource("SCRIPTS", create=create)
        if not scripts:
            raise RuntimeError("Blender user scripts directory is unavailable")
        value = str(Path(scripts) / "media_to_pose" / "backends" / "anycalib")
    return Path(value)


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
    Path(temporary).replace(path)


def configured_storage_parent() -> Path:
    configured = str(
        _read_json(backend_root(create=False) / "storage.json").get("storage_parent")
        or ""
    )
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


def active_directory() -> Path:
    return model_storage_root() / "active"


def manifest_path() -> Path:
    return active_directory() / "manifest.json"


def backend_present() -> bool:
    return manifest_path().is_file()


def bundle_url() -> str:
    return f"{HF_ENDPOINT}/{MODEL_SPEC['repo']}"


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
        return "fp16" if device == "webgpu" and not fallback else "fp32"
    except Exception:
        return "fp32"


def _hf_url(path: str, revision: str = "") -> str:
    relative = PurePosixPath(str(path))
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(path):
        raise RuntimeError(f"Unsafe AnyCalib download path: {path!r}")
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
    expected_bytes: int = 0,
    expected_sha256: str = "",
    revision: str = "",
    start: float,
    end: float,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    digest = hashlib.sha256()
    received = 0
    request = urllib.request.Request(
        _hf_url(remote_path, revision),
        headers={"User-Agent": "Media-to-Pose/AnyCalib"},
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=15) as response,
            partial.open("xb") as output,
        ):
            total = int(response.headers.get("Content-Length") or expected_bytes or 0)
            while True:
                if job.cancel.is_set():
                    raise InterruptedError("AnyCalib download cancelled")
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                ratio = received / total if total else 0.0
                job.update(
                    start + (end - start) * ratio,
                    f"Downloading {destination.name}",
                    "Download",
                )
        if expected_bytes and received != expected_bytes:
            raise RuntimeError(f"{destination.name} has an unexpected byte size")
        if (
            expected_sha256
            and digest.hexdigest().casefold() != expected_sha256.casefold()
        ):
            raise RuntimeError(f"{destination.name} failed SHA-256 verification")
        Path(partial).replace(destination)
    except Exception:
        with suppress(OSError):
            partial.unlink()
        raise


def _ensure_storage(root: Path) -> None:
    marker = root / MODEL_STORAGE_MARKER
    if root.exists() and any(root.iterdir()) and not marker.is_file():
        raise OSError(f"Refusing unmanaged AnyCalib storage directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    expected = {"kind": "anycalib_models", "owner": "media_to_pose"}
    existing = _read_json(marker)
    if existing and existing != expected:
        raise OSError("AnyCalib storage is owned by another application")
    _write_json(marker, expected)


def _install_worker(
    job: _Job,
    storage_root: str,
    precision: str,
    storage_config_path: str,
) -> None:
    spec = MODEL_SPEC["variants"][precision]
    root = Path(storage_root)
    staging = root / f".staging-{os.getpid()}-{uuid.uuid4().hex}"
    current, backup = root / "active", root / ".previous"
    install_lock = model_installation_lock(root, "AnyCalib")
    try:
        install_lock.acquire()
        _ensure_storage(root)
        staging.mkdir(parents=True)
        remote_manifest = str(spec["manifest_path"])
        revision = str(MODEL_SPEC.get("revision") or "")
        if not sam3d_ort_runtime.is_immutable_revision(revision):
            revision = "main"
        job.update(0.01, f"Downloading {precision.upper()} manifest", "Manifest")
        try:
            _download(
                job,
                remote_manifest,
                staging / "manifest.json",
                revision=revision,
                start=0.01,
                end=0.04,
            )
        except urllib.error.HTTPError as exc:
            if revision == "main" or exc.code not in {404, 410}:
                raise
            revision = "main"
            job.update(
                0.01,
                "Preferred AnyCalib bundle unavailable; using latest",
                "Manifest",
            )
            _download(
                job,
                remote_manifest,
                staging / "manifest.json",
                revision=revision,
                start=0.01,
                end=0.04,
            )
        if job.cancel.is_set():
            raise InterruptedError("AnyCalib installation cancelled")
        bundle = anycalib_ort_runtime.load_bundle_manifest(staging)
        remote_root = str(PurePosixPath(remote_manifest).parent)
        _download(
            job,
            str(PurePosixPath(remote_root) / bundle.graph.path),
            staging / bundle.graph.path,
            expected_bytes=bundle.graph.bytes,
            expected_sha256=bundle.graph.sha256,
            revision=revision,
            start=0.04,
            end=0.90,
        )
        job.update(0.92, "Verifying AnyCalib bundle", "Verification")
        anycalib_ort_runtime.load_bundle(staging)
        if job.cancel.is_set():
            raise InterruptedError("AnyCalib installation cancelled")
        if backup.exists():
            shutil.rmtree(backup)
        if current.exists():
            Path(current).replace(backup)
        try:
            Path(staging).replace(current)
        except Exception:
            if backup.exists() and not current.exists():
                Path(backup).replace(current)
            raise
        shutil.rmtree(backup, ignore_errors=True)
        _write_json(Path(storage_config_path), {"storage_parent": str(root.parent)})
        invalidate_backend_snapshot()
        job.update(1.0, f"AnyCalib {precision.upper()} installed", "Complete")
        job.finish("DONE", "AnyCalib installed and verified")
    except InterruptedError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        job.finish("CANCELLED", str(exc))
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        job.finish("ERROR", f"AnyCalib installation failed: {exc}")
    finally:
        install_lock.release()


def install_status_snapshot() -> JobSnapshot:
    with _LOCK:
        return _INSTALL_JOB.snapshot() if _INSTALL_JOB else JobSnapshot()


def model_load_status_snapshot() -> JobSnapshot:
    with _LOCK:
        return _LOAD_JOB.snapshot() if _LOAD_JOB else JobSnapshot()


def invalidate_backend_snapshot() -> None:
    backend_snapshot.cache_clear()


@lru_cache(maxsize=4)
def backend_snapshot(requested_mode: object = "AUTO") -> BackendSnapshot:
    requested = sam3d_ort_runtime.normalize_mode(requested_mode)
    preferred = _preferred_precision(requested)
    if not is_published():
        return BackendSnapshot(
            state="UNPUBLISHED",
            message="AnyCalib bundle is awaiting publication",
            requested_mode=requested,
            precision=preferred,
            model_directory=str(active_directory()),
        )
    with _LOCK:
        session = _SESSION
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
    except Exception as exc:
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


def ensure_model_loaded(
    device: object = "AUTO",
    bundle_directory: str = "",
) -> anycalib_ort_runtime.AnyCalibSession:
    global _SESSION, _LAST_DIAGNOSTIC
    requested = sam3d_ort_runtime.normalize_mode(device)
    previous = None
    with _LOCK:
        if _SESSION is not None and _SESSION.diagnostics.requested_mode == requested:
            return _SESSION
        previous, _SESSION = _SESSION, None
    if previous is not None:
        previous.close()
    bundle = anycalib_ort_runtime.load_bundle(
        Path(bundle_directory) if bundle_directory else active_directory()
    )
    session = anycalib_ort_runtime.AnyCalibSession(bundle, requested)
    with _LOCK:
        _SESSION = session
        _LAST_DIAGNOSTIC = "Bundle verification and runtime session creation passed"
    invalidate_backend_snapshot()
    return session


def is_model_loaded() -> bool:
    with _LOCK:
        return _SESSION is not None


def unload_model() -> None:
    global _SESSION
    with _LOCK:
        session, _SESSION = _SESSION, None
    if session is not None:
        session.close()
    invalidate_backend_snapshot()


def _load_worker(job: _Job, device: object, bundle_directory: str) -> None:
    global _LAST_DIAGNOSTIC
    try:
        job.update(0.2, "Verifying AnyCalib", "Verification")
        ensure_model_loaded(device, bundle_directory)
        if job.cancel.is_set():
            unload_model()
            raise InterruptedError("AnyCalib load cancelled")
        job.update(1.0, "AnyCalib loaded", "Complete")
        job.finish("DONE", "AnyCalib loaded")
    except InterruptedError as exc:
        job.finish("CANCELLED", str(exc))
    except Exception as exc:
        _LAST_DIAGNOSTIC = str(exc)
        job.finish("ERROR", f"AnyCalib loading failed: {exc}")


class MTP_OT_anycalib_install(Operator):
    bl_idname = "mtp.anycalib_install"
    bl_label = "Install Camera Calibration Model"
    bl_description = "Download, verify, repair, or move the AnyCalib model"

    model_storage_directory: StringProperty(
        name="Model Storage Parent", subtype="DIR_PATH", default=""
    )
    _timer = None
    _scene_name = ""

    @classmethod
    def poll(cls, _context):
        snapshot = install_status_snapshot()
        if snapshot.running:
            cls.poll_message_set("Camera model installation is already running")
            return False
        if COORDINATOR.is_busy():
            cls.poll_message_set("Media to Pose is busy with another operation")
            return False
        return True

    def _report_blocked(self, context) -> None:
        snapshot = install_status_snapshot()
        message = (
            "Camera model installation is already running"
            if snapshot.running
            else "Media to Pose is busy with another operation"
        )
        settings = getattr(getattr(context, "scene", None), "mtp_settings", None)
        if settings is not None:
            settings.status = message
            settings.status_kind = "WARNING"
        self.report({"WARNING"}, message)

    def invoke(self, context, _event):
        if install_status_snapshot().running or COORDINATOR.is_busy():
            self._report_blocked(context)
            return {"CANCELLED"}
        self.model_storage_directory = str(configured_storage_parent())
        return context.window_manager.invoke_props_dialog(self, width=540)

    def draw(self, _context):
        requested = _preferred_precision("AUTO")
        size = int(MODEL_SPEC["variants"][requested]["download_bytes"])
        self.layout.label(
            text=f"Download: {size / (1024**2):.0f} MiB · processing remains local",
            icon="IMPORT",
        )
        self.layout.label(text=f"Source: {MODEL_SPEC['repo']}", icon="WORLD")
        self.layout.label(
            text=f"Preferred revision: {MODEL_SPEC['revision']}", icon="LOCKED"
        )
        self.layout.prop(self, "model_storage_directory")

    def execute(self, context):
        global _INSTALL_JOB
        if install_status_snapshot().running or COORDINATOR.is_busy():
            self._report_blocked(context)
            return {"CANCELLED"}
        if not is_published():
            self.report({"ERROR"}, "AnyCalib ONNX bundle is not published")
            return {"CANCELLED"}
        storage_root = str(model_storage_root(self.model_storage_directory))
        precision = _preferred_precision(context.scene.mtp_settings.anycalib_device)
        storage_config_path = str(backend_root() / "storage.json")
        token = COORDINATOR.acquire(
            OperationKind.ANYCALIB_INSTALL,
            "anycalib-install",
            status="Starting AnyCalib download",
        )
        if token is None:
            self._report_blocked(context)
            return {"CANCELLED"}
        job = _Job(token, "Starting AnyCalib download")
        job.thread = threading.Thread(
            target=_install_worker,
            args=(job, storage_root, precision, storage_config_path),
            daemon=True,
            name="MTP-AnyCalib-Install",
        )
        with _LOCK:
            _INSTALL_JOB = job
        COORDINATOR.attach_worker(token, thread=job.thread)
        try:
            job.thread.start()
        except Exception as exc:
            job.finish("ERROR", f"Could not start camera model download: {exc}")
            COORDINATOR.release(token)
            self.report({"ERROR"}, f"Could not start camera model download: {exc}")
            return {"CANCELLED"}
        self._scene_name = context.scene.name
        context.scene.mtp_settings.progress = 0.0
        context.scene.mtp_settings.status = "Starting camera model download"
        context.scene.mtp_settings.status_kind = "PROGRESS"
        self._timer = context.window_manager.event_timer_add(
            0.15, window=context.window
        )
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "ESC":
            with _LOCK:
                job = _INSTALL_JOB
            if job is not None:
                COORDINATOR.request_cancel(
                    job.token, status="Cancelling camera model download"
                )
            return {"RUNNING_MODAL"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        with _LOCK:
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
            snapshot.status or "Camera model installation stopped",
        )
        return {"CANCELLED"}

    def cancel(self, context):
        with _LOCK:
            job = _INSTALL_JOB
        if job is not None:
            COORDINATOR.request_cancel(
                job.token, status="Cancelling camera model download"
            )
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None


class MTP_OT_anycalib_load(Operator):
    bl_idname = "mtp.anycalib_load"
    bl_label = "Load AnyCalib"

    def execute(self, context):
        global _LOAD_JOB
        bundle_directory = str(active_directory())
        token = COORDINATOR.acquire(
            OperationKind.ANYCALIB_MODEL_LOAD,
            "anycalib-load",
            status="Loading AnyCalib",
        )
        if token is None:
            self.report({"ERROR"}, "Media to Pose is busy")
            return {"CANCELLED"}
        job = _Job(token, "Loading AnyCalib")
        job.thread = threading.Thread(
            target=_load_worker,
            args=(
                job,
                context.scene.mtp_settings.anycalib_device,
                bundle_directory,
            ),
            daemon=True,
            name="MTP-AnyCalib-Load",
        )
        with _LOCK:
            _LOAD_JOB = job
        COORDINATOR.attach_worker(token, thread=job.thread)
        try:
            job.thread.start()
        except Exception as exc:
            job.finish("ERROR", f"Could not start camera model load: {exc}")
            COORDINATOR.release(token)
            self.report({"ERROR"}, f"Could not start camera model load: {exc}")
            return {"CANCELLED"}
        return {"FINISHED"}


class MTP_OT_anycalib_cancel_install(Operator):
    bl_idname = "mtp.anycalib_cancel_install"
    bl_label = "Cancel AnyCalib Download"

    def execute(self, _context):
        with _LOCK:
            job = _INSTALL_JOB
        if job is not None and not job.cancel.is_set():
            COORDINATOR.request_cancel(job.token, status="Cancelling AnyCalib download")
            job.update(job.progress, "Cancelling safely", job.phase)
            return {"FINISHED"}
        return {"CANCELLED"}


class MTP_OT_anycalib_unload(Operator):
    bl_idname = "mtp.anycalib_unload"
    bl_label = "Unload AnyCalib"

    def execute(self, _context):
        token = COORDINATOR.acquire(
            OperationKind.ANYCALIB_MODEL_UNLOAD,
            "anycalib-unload",
            status="Unloading AnyCalib",
        )
        if token is None:
            self.report({"ERROR"}, "Media to Pose is busy")
            return {"CANCELLED"}
        try:
            unload_model()
        except Exception as exc:
            COORDINATOR.finish(token, OperationState.FAILED, str(exc))
            COORDINATOR.release(token)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        COORDINATOR.finish(token, OperationState.SUCCEEDED, "AnyCalib unloaded")
        COORDINATOR.release(token)
        return {"FINISHED"}


class MTP_OT_anycalib_uninstall(Operator):
    bl_idname = "mtp.anycalib_uninstall"
    bl_label = "Uninstall AnyCalib"
    bl_options = {"INTERNAL"}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, _context):
        token = COORDINATOR.acquire(
            OperationKind.ANYCALIB_UNINSTALL,
            "anycalib-uninstall",
            status="Uninstalling AnyCalib",
        )
        if token is None:
            self.report({"ERROR"}, "Media to Pose is busy")
            return {"CANCELLED"}
        install_lock = None
        try:
            root = model_storage_root().resolve(False)
            install_lock = model_installation_lock(root, "AnyCalib")
            install_lock.acquire()
            unload_model()
            expected = {"kind": "anycalib_models", "owner": "media_to_pose"}
            if (
                root.name.casefold() != MODEL_STORAGE_FOLDER.casefold()
                or _read_json(root / MODEL_STORAGE_MARKER) != expected
            ):
                raise OSError("Refusing to remove unrecognized AnyCalib storage")
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
        COORDINATOR.finish(token, OperationState.SUCCEEDED, "AnyCalib uninstalled")
        COORDINATOR.release(token)
        return {"FINISHED"}


CLASSES = (
    MTP_OT_anycalib_install,
    MTP_OT_anycalib_cancel_install,
    MTP_OT_anycalib_load,
    MTP_OT_anycalib_unload,
    MTP_OT_anycalib_uninstall,
)


def register() -> None:
    invalidate_backend_snapshot()
    register_classes(CLASSES)


def unregister() -> None:
    with _LOCK:
        jobs = tuple(job for job in (_INSTALL_JOB, _LOAD_JOB) if job)
    for job in jobs:
        COORDINATOR.request_cancel(job.token, status="Cancelling AnyCalib operation")
        if job.thread:
            job.thread.join(timeout=2.0)
    unload_model()
    unregister_classes(CLASSES)
