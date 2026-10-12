"""Shared install and load lifecycle for downloadable model backends.

AnyCalib, RoHM and SAM 3D Body all follow the same steps: download pinned,
SHA-256 verified files into a managed storage folder, guard installation
with a cross-process lock, run long work on a background job that reports
through the operation coordinator, and keep one loaded runtime session.
This module holds those steps once; each backend only describes its bundle.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import re
import threading
import time
import traceback
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

from ..core import paths
from ..core.coordinator import (
    COORDINATOR,
    OperationKind,
    OperationState,
    OperationToken,
)
from ..core.errors import BackendError, Cancelled, OperationBusy
from ..core.fs import atomic_json, read_json, remove_tree, replace_directory
from ..core.log import get_logger

LOGGER = get_logger(__name__)

HF_ENDPOINT = "https://huggingface.co"
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 15
STORAGE_OWNER = "media_to_pose"
STORAGE_MARKER = ".media_to_pose_storage.json"
_LOG_LIMIT = 200
_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "DETAIL": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
}


# --------------------------------------------------------------------------
# Text hygiene


def redact(value: object) -> str:
    """Remove Hugging Face tokens from messages before they are shown or logged."""
    return re.sub(
        r"(?i)\b(?:hf|api_org)_[A-Za-z0-9._-]{6,}\b",
        "[redacted]",
        str(value or ""),
    )


def safe_metadata(value: object, *, limit: int = 160) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", redact(value))
    return text.strip()[:limit]


# --------------------------------------------------------------------------
# Paths


def resolve_user_path(value: str) -> Path:
    """Expand ``~``, environment variables and Blender ``//`` relative paths."""
    return paths.expand_user_path(value).resolve(strict=False)


def backend_root(name: str, *, create: bool = True) -> Path:
    """Return ``<user data>/media_to_pose/backends/<name>``."""
    return paths.user_directory(f"backends/{name}", create=create)


def safe_relative_path(value: str, *, label: str = "bundle") -> PurePosixPath:
    """Validate a manifest-relative POSIX path that must stay inside its root."""
    raw = str(value)
    relative = PurePosixPath(raw)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or "\\" in raw
        or "\x00" in raw
        or any(":" in part for part in relative.parts)
    ):
        raise BackendError(f"Unsafe {label} path: {value!r}")
    return relative


def contained_path(root: Path, relative: str, *, label: str = "bundle") -> Path:
    """Join a validated relative path to ``root`` and refuse escapes."""
    parts = safe_relative_path(relative, label=label).parts
    destination = root.joinpath(*parts).resolve(strict=False)
    resolved_root = root.resolve(strict=False)
    if destination != resolved_root and resolved_root not in destination.parents:
        raise BackendError(f"{label} path escapes its directory: {relative!r}")
    return destination


def hf_resolve_url(
    repository: str,
    revision: str,
    path: str,
    *,
    label: str = "download",
    endpoint: str = HF_ENDPOINT,
) -> str:
    """Return the Hugging Face ``resolve`` URL for one repository file."""
    relative = safe_relative_path(path, label=label)
    encoded_repository = "/".join(
        urllib.parse.quote(part, safe="") for part in str(repository).split("/")
    )
    encoded_path = "/".join(
        urllib.parse.quote(part, safe="") for part in relative.parts
    )
    selected = str(revision or "main")
    return f"{endpoint}/{encoded_repository}/resolve/{selected}/{encoded_path}"


def is_immutable_revision(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{40}", str(value or "")))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Pinned download


ProgressCallback = Callable[[int, int], None]


def download_file(
    url: str,
    destination: Path,
    *,
    cancel: threading.Event,
    label: str = "",
    progress: ProgressCallback | None = None,
    expected_bytes: int = 0,
    expected_sha256: str = "",
    maximum_bytes: int = 0,
    user_agent: str = "Media-to-Pose",
    timeout: float = DOWNLOAD_TIMEOUT_SECONDS,
    cancel_message: str = "",
) -> None:
    """Stream ``url`` to ``destination`` with size and SHA-256 checks.

    The file is written to a unique ``.part`` sibling and moved into place
    only after every check passes, so a failure or cancellation never leaves
    a partial file behind.
    """
    label = label or Path(destination).name
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    digest = hashlib.sha256()
    received = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            total = int(response.headers.get("Content-Length") or 0)
            if expected_bytes and total and total != expected_bytes:
                raise BackendError(f"{label} has an unexpected download size")
            if maximum_bytes and total and total > maximum_bytes:
                raise BackendError(f"{label} exceeds the safe download limit")
            total = total or int(expected_bytes or 0)
            with partial.open("xb") as handle:
                while True:
                    if cancel.is_set():
                        raise Cancelled(cancel_message or f"{label} download cancelled")
                    chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                    if expected_bytes and received > expected_bytes:
                        raise BackendError(f"{label} exceeds its declared size")
                    if maximum_bytes and received > maximum_bytes:
                        raise BackendError(f"{label} exceeds the safe download limit")
                    if progress is not None:
                        progress(received, total)
        if expected_bytes and received != expected_bytes:
            raise BackendError(f"{label} has an unexpected byte size")
        if (
            expected_sha256
            and digest.hexdigest().casefold() != expected_sha256.casefold()
        ):
            raise BackendError(f"{label} failed SHA-256 verification")
        partial.replace(destination)
    except BaseException:
        with suppress(OSError):
            partial.unlink()
        raise


# --------------------------------------------------------------------------
# Cross-process installation lock


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


class InstallationLock:
    """Exclusive, stale-safe installation lease shared by Blender processes.

    The lease is a small JSON file created with ``O_EXCL``. A lease whose
    process is gone (or older than ``max_age_seconds``) is replaced.
    """

    def __init__(
        self,
        path: Path,
        model_label: str,
        *,
        max_age_seconds: float | None = None,
    ):
        self.path = Path(path)
        self.model_label = str(model_label)
        self.max_age_seconds = max_age_seconds
        self.identifier = uuid.uuid4().hex
        self.acquired = False

    def _stale(self) -> bool:
        existing = read_json(self.path)
        existing = existing if isinstance(existing, dict) else {}
        pid = int(existing.get("pid") or 0)
        created_at = int(existing.get("created_at") or 0)
        expired = bool(
            self.max_age_seconds
            and created_at
            and time.time() - created_at > self.max_age_seconds
        )
        return not _pid_running(pid) or expired

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "pid": os.getpid(),
                "created_at": int(time.time()),
                "identifier": self.identifier,
            }
        ).encode("utf8")
        for _attempt in range(2):
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if not self._stale():
                    raise OperationBusy(
                        f"Another Blender process is managing {self.model_label}"
                    ) from None
                try:
                    self.path.unlink()
                except OSError as exc:
                    raise BackendError(
                        f"A stale {self.model_label} installation lock could not "
                        "be removed"
                    ) from exc
                continue
            try:
                os.write(descriptor, payload)
            finally:
                os.close(descriptor)
            self.acquired = True
            return
        raise OperationBusy(
            f"Could not acquire the {self.model_label} installation lock"
        )

    def release(self) -> None:
        if not self.acquired:
            return
        current = read_json(self.path)
        if isinstance(current, dict) and current.get("identifier") == self.identifier:
            with suppress(FileNotFoundError):
                self.path.unlink()
        self.acquired = False

    def __enter__(self) -> InstallationLock:
        self.acquire()
        return self

    def __exit__(self, *_exc_info) -> None:
        self.release()


def model_installation_lock(root: Path, label: str) -> InstallationLock:
    """The lease beside a storage root: ``<parent>/.<folder>.install.lock``."""
    root = Path(root)
    return InstallationLock(root.parent / f".{root.name}.install.lock", label)


# --------------------------------------------------------------------------
# Managed storage


@dataclass(frozen=True)
class ManagedStorage:
    """A model folder that Media to Pose created and may replace or delete.

    The folder carries a marker file naming the owner and kind; nothing is
    written into, or removed from, a folder without a matching marker.
    """

    folder: str
    kind: str
    label: str
    marker_version: int | None = None
    private: bool = False

    def root(self, parent: Path) -> Path:
        """``<parent>/<folder>``, accepting the folder itself as ``parent``."""
        parent = Path(parent)
        if parent.name.casefold() == self.folder.casefold():
            parent = parent.parent
        return parent / self.folder

    def _marker(self) -> dict[str, Any]:
        marker: dict[str, Any] = {"kind": self.kind, "owner": STORAGE_OWNER}
        if self.marker_version is not None:
            marker["version"] = self.marker_version
        return marker

    def owns(self, root: Path) -> bool:
        marker = read_json(Path(root) / STORAGE_MARKER)
        return bool(
            Path(root).name.casefold() == self.folder.casefold()
            and isinstance(marker, dict)
            and marker.get("owner") == STORAGE_OWNER
            and marker.get("kind") == self.kind
        )

    def ensure(self, root: Path) -> None:
        """Create or adopt ``root``; refuse unmanaged or foreign folders."""
        root = Path(root)
        marker_path = root / STORAGE_MARKER
        if root.exists() and not marker_path.is_file():
            try:
                has_contents = next(root.iterdir(), None) is not None
            except OSError as exc:
                raise BackendError(
                    f"Cannot inspect model storage directory: {root}"
                ) from exc
            if has_contents:
                raise BackendError(
                    f"Refusing unmanaged {self.label} storage directory: {root}"
                )
        root.mkdir(parents=True, exist_ok=True)
        existing = read_json(marker_path)
        if (
            isinstance(existing, dict)
            and existing
            and (
                existing.get("owner") != STORAGE_OWNER
                or existing.get("kind") != self.kind
            )
        ):
            raise BackendError(f"{self.label} storage is owned by another application")
        atomic_json(marker_path, self._marker(), private=self.private)

    def verified(self, root: Path) -> Path:
        resolved = Path(root).resolve(strict=False)
        if not self.owns(resolved):
            raise BackendError(f"Refusing to remove unrecognized {self.label} storage")
        return resolved

    def remove(self, root: Path) -> None:
        """Delete an owned storage folder (no-op when it does not exist)."""
        if not Path(root).exists():
            return
        remove_tree(self.verified(root))


def read_storage_parent(config_path: Path) -> str:
    value = read_json(config_path)
    return str(value.get("storage_parent") or "") if isinstance(value, dict) else ""


def write_storage_parent(
    config_path: Path, parent: Path, *, private: bool = False
) -> None:
    atomic_json(
        config_path,
        {"storage_parent": str(Path(parent).resolve(strict=False))},
        private=private,
    )


# --------------------------------------------------------------------------
# Background jobs


class JobOutcome(str, Enum):
    NONE = ""
    DONE = "DONE"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


class JobKind(str, Enum):
    INSTALL = "installation"
    MODEL = "model"


@dataclass(frozen=True)
class JobSnapshot:
    running: bool = False
    progress: float = 0.0
    phase: str = ""
    status: str = ""
    outcome: str = ""
    error: str = ""
    logs: tuple[str, ...] = ()
    log_path: str = ""


class BackgroundJob:
    """Progress, log and outcome of one install or load running on a thread.

    Updates are forwarded to the operation coordinator while the job's token
    still owns it; a superseded job can no longer change shared progress.
    """

    def __init__(
        self,
        label: str,
        *,
        token: OperationToken | None = None,
        request: object = None,
    ):
        self.label = str(label)
        self.token = token
        self.request = request
        self.cancel = token.cancel_event if token else threading.Event()
        self.messages: queue.Queue = queue.Queue()
        self.progress = 0.0
        self.status = self.label
        self.phase = ""
        self.outcome = JobOutcome.NONE
        self.error = ""
        self.logs: list[str] = []
        self.log_path = ""
        self.thread: threading.Thread | None = None
        self.started = False
        self.terminal = False
        self._lock = threading.RLock()

    def configure_log(self, path: Path) -> None:
        """Keep a plain-text log at ``path`` (the previous one is retained)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        previous = path.with_name(f"{path.stem}.previous{path.suffix}")
        try:
            if previous.exists():
                previous.unlink()
            if path.exists():
                path.replace(previous)
            path.write_text("", encoding="utf8")
            self.log_path = str(path)
        except OSError as exc:
            self.log_path = ""
            LOGGER.warning("Could not initialize install log: %s", safe_metadata(exc))

    def log(self, level: str, message: object, *, echo: bool = True) -> None:
        """Append to the job log and file; ``echo`` also sends it to LOGGER."""
        cleaned = re.sub(r"[\x00-\x1f\x7f]+", " ", redact(message)).strip()
        if not cleaned:
            return
        cleaned = cleaned[:2000]
        entry = f"{time.strftime('%H:%M:%S')} [{level.upper()}] {cleaned}"
        with self._lock:
            self.logs.append(entry)
            if len(self.logs) > _LOG_LIMIT:
                del self.logs[:-_LOG_LIMIT]
            log_path = self.log_path
        if echo:
            LOGGER.log(_LOG_LEVELS.get(level.upper(), logging.INFO), "%s", cleaned)
        if log_path:
            try:
                with Path(log_path).open("a", encoding="utf8") as handle:
                    handle.write(entry + "\n")
            except OSError as exc:
                LOGGER.warning("Could not append install log: %s", safe_metadata(exc))

    def update(
        self,
        progress: float,
        status: str,
        phase: str = "",
        *,
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
            changed = status != self.status or bool(phase and phase != self.phase)
            self.progress = min(max(float(progress), 0.0), 1.0)
            self.status = str(status)
            if phase:
                self.phase = str(phase)
        if changed and log_change:
            prefix = f"{self.phase}: " if self.phase else ""
            self.log("INFO", f"{prefix}{status}")

    def check_cancelled(self, message: str = "Installation cancelled") -> None:
        if self.cancel.is_set():
            raise Cancelled(message)

    def download_progress(
        self, start: float, end: float, label: str, phase: str
    ) -> ProgressCallback:
        """Return a ``download_file`` callback that maps bytes onto a range."""

        def report(received: int, total: int) -> None:
            ratio = received / total if total else 0.0
            size = f"{received / (1024**2):.0f} MiB"
            if total:
                size += f" / {total / (1024**2):.0f} MiB"
            self.update(
                start + (end - start) * ratio,
                f"{label} · {size}",
                phase,
                log_change=False,
            )

        return report

    def finish(self, outcome: JobOutcome | str, message: str) -> None:
        outcome = JobOutcome(outcome)
        safe_message = redact(message)
        with self._lock:
            self.outcome = outcome
            self.error = safe_message if outcome is JobOutcome.ERROR else ""
            self.status = safe_message
            self.terminal = True
        self.log("ERROR" if outcome is JobOutcome.ERROR else "INFO", safe_message)
        self.messages.put((outcome.value, safe_message))
        if self.token is not None:
            state = {
                JobOutcome.DONE: OperationState.SUCCEEDED,
                JobOutcome.CANCELLED: OperationState.CANCELLED,
            }.get(outcome, OperationState.FAILED)
            COORDINATOR.finish(self.token, state, safe_message)

    def alive(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

    def snapshot(self) -> JobSnapshot:
        with self._lock:
            return JobSnapshot(
                running=bool(self.alive() and not self.terminal),
                progress=self.progress,
                phase=self.phase,
                status=self.status,
                outcome=self.outcome.value,
                error=self.error,
                logs=tuple(self.logs),
                log_path=self.log_path,
            )


def run_job(
    job: BackgroundJob,
    work: Callable[[BackgroundJob], str],
    *,
    describe_error: Callable[[BaseException, str], str] | None = None,
    cleanup: Callable[[], None] | None = None,
) -> None:
    """Thread entry point: run ``work`` and record exactly one outcome.

    ``work`` returns the success message. Cancellation and failures are
    turned into outcomes here, with the traceback written to the log.
    """
    try:
        message = work(job)
    except Cancelled as exc:
        job.finish(JobOutcome.CANCELLED, str(exc) or f"{job.label} cancelled")
    except Exception as exc:  # Thread entry point.
        LOGGER.exception("%s failed", job.label)
        job.log(
            "ERROR",
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            echo=False,
        )
        message = (
            describe_error(exc, job.phase)
            if describe_error is not None
            else f"{job.label} failed: {exc}"
        )
        job.finish(JobOutcome.ERROR, message)
    else:
        job.finish(JobOutcome.DONE, message)
    finally:
        if cleanup is not None:
            try:
                cleanup()
            except OSError:
                LOGGER.exception("%s cleanup failed", job.label)


# --------------------------------------------------------------------------
# Pinned ONNX bundles


@dataclass(frozen=True)
class BundleFile:
    """One file listed by a bundle manifest, relative to the manifest folder."""

    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class PinnedBundle:
    """A manifest-described file set on Hugging Face at a pinned revision.

    ``list_files`` reads the downloaded manifest from the staging directory;
    ``verify`` loads the complete staging directory and raises when any file
    is missing or altered.
    """

    label: str
    repository: str
    revision: str
    manifest_path: str
    list_files: Callable[[Path], list[BundleFile]]
    verify: Callable[[Path], object]


def install_pinned_bundle(
    job: BackgroundJob,
    bundle: PinnedBundle,
    storage: ManagedStorage,
    root: Path,
    storage_config: Path,
) -> str:
    """Download, verify and activate ``bundle`` under the storage ``root``.

    Files land in a private staging folder; only a fully verified folder
    replaces ``root/active``. The preferred revision falls back to ``main``
    only when it has been removed from the Hub (HTTP 404/410).
    """
    import urllib.error

    label = bundle.label
    root = Path(root)
    staging = root / f".staging-{os.getpid()}-{uuid.uuid4().hex}"
    lock = model_installation_lock(root, label)
    try:
        lock.acquire()
        storage.ensure(root)
        staging.mkdir(parents=True)
        revision = bundle.revision if is_immutable_revision(bundle.revision) else "main"
        cancelled = f"{label} download cancelled"

        def fetch(remote: str, destination: Path, start: float, end: float, **checks):
            download_file(
                hf_resolve_url(
                    bundle.repository, revision, remote, label=f"{label} download"
                ),
                destination,
                cancel=job.cancel,
                label=destination.name,
                progress=job.download_progress(
                    start, end, f"Downloading {destination.name}", "Download"
                ),
                user_agent=f"Media-to-Pose/{label}",
                cancel_message=cancelled,
                **checks,
            )

        job.update(0.01, f"Downloading {label} manifest", "Manifest")
        manifest = staging / "manifest.json"
        try:
            fetch(bundle.manifest_path, manifest, 0.01, 0.05)
        except urllib.error.HTTPError as exc:
            if revision == "main" or exc.code not in {404, 410}:
                raise
            revision = "main"
            job.update(
                0.01, f"Preferred {label} bundle unavailable; using latest", "Manifest"
            )
            fetch(bundle.manifest_path, manifest, 0.01, 0.05)
        job.check_cancelled(f"{label} installation cancelled")
        files = sorted(bundle.list_files(staging), key=lambda item: item.path)
        remote_root = PurePosixPath(bundle.manifest_path).parent
        total = sum(item.bytes for item in files)
        completed = 0
        for item in files:
            start = 0.05 + 0.80 * completed / max(total, 1)
            completed += item.bytes
            end = 0.05 + 0.80 * completed / max(total, 1)
            fetch(
                str(remote_root / item.path),
                contained_path(staging, item.path, label=f"{label} bundle"),
                start,
                end,
                expected_bytes=item.bytes,
                expected_sha256=item.sha256,
            )
        job.update(0.88, f"Verifying {label} bundle", "Verification")
        bundle.verify(staging)
        job.check_cancelled(f"{label} installation cancelled")
        job.update(0.96, "Activating verified bundle", "Activation")
        replace_directory(staging, root / "active", root / ".previous")
        write_storage_parent(storage_config, root.parent)
        job.update(1.0, f"{label} installed and verified", "Complete")
        return f"{label} installed and verified"
    finally:
        remove_tree(staging)
        lock.release()


# --------------------------------------------------------------------------
# Backend state


class ModelBackend:
    """Process-wide state of one optional model: jobs, session, snapshots.

    Subclasses describe the bundle (download and verification) and the
    runtime session; this class owns the generation-safe job slots, the
    loaded session and the readiness cache that Blender panels read.
    """

    label = "Model"
    install_operation = OperationKind.CAPTURE
    load_operation = OperationKind.CAPTURE
    unload_operation = OperationKind.CAPTURE
    uninstall_operation = OperationKind.CAPTURE

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[JobKind, BackgroundJob] = {}
        self._session: Any = None
        self.last_diagnostic = ""
        self._snapshot_cache: dict[tuple, Any] = {}

    # Jobs ---------------------------------------------------------------

    def assign_job(self, kind: JobKind, job: BackgroundJob) -> bool:
        """Record ``job`` only while its token owns the coordinator."""
        if job.token is None or not COORDINATOR.owns(job.token):
            return False
        with self._lock:
            current = self._jobs.get(kind)
            if current is not None and current is not job and current.alive():
                return False
            self._jobs[kind] = job
            return True

    def job(self, kind: JobKind) -> BackgroundJob | None:
        with self._lock:
            return self._jobs.get(kind)

    def clear_job(self, kind: JobKind, expected: BackgroundJob | None = None) -> bool:
        with self._lock:
            current = self._jobs.get(kind)
            if current is None or (expected is not None and current is not expected):
                return False
            del self._jobs[kind]
            return True

    def jobs(self) -> tuple[BackgroundJob, ...]:
        with self._lock:
            return tuple(self._jobs.values())

    def job_snapshot(self, kind: JobKind) -> JobSnapshot:
        job = self.job(kind)
        return job.snapshot() if job is not None else JobSnapshot()

    def install_status_snapshot(self) -> JobSnapshot:
        return self.job_snapshot(JobKind.INSTALL)

    def model_load_status_snapshot(self) -> JobSnapshot:
        return self.job_snapshot(JobKind.MODEL)

    def job_running(self, kind: JobKind) -> bool:
        job = self.job(kind)
        return bool(job and job.alive() and not job.terminal)

    def start_job(
        self,
        kind: JobKind,
        operation: OperationKind,
        owner: str,
        target: Callable[..., None],
        args: tuple = (),
        *,
        label: str,
        kwargs: dict[str, Any] | None = None,
        status: str = "",
        scene_name: str = "",
        scene_pointer: int = 0,
        request: object = None,
        log_path: Path | None = None,
        prepare: Callable[[BackgroundJob], None] | None = None,
    ) -> BackgroundJob:
        """Acquire the coordinator, then run ``target(job, *args)`` on a thread.

        Raises ``OperationBusy`` when another operation owns Media to Pose.
        """
        token = COORDINATOR.acquire(
            operation,
            owner,
            scene_name=scene_name,
            scene_pointer=scene_pointer,
            status=status or label,
        )
        if token is None:
            raise OperationBusy("Media to Pose is already busy")
        job = BackgroundJob(label, token=token, request=request)
        if log_path is not None:
            job.configure_log(log_path)
        if prepare is not None:
            prepare(job)
        if not self.assign_job(kind, job):
            COORDINATOR.finish(token, OperationState.FAILED, f"{label} was superseded")
            COORDINATOR.release(token)
            raise OperationBusy(
                f"Could not acquire {self.label} {kind.value} ownership"
            )
        job.thread = threading.Thread(
            target=target,
            args=(job, *args),
            kwargs=dict(kwargs or {}),
            daemon=True,
            name=f"MTP-{self.label.replace(' ', '')}-{kind.value}",
        )
        COORDINATOR.attach_worker(token, thread=job.thread)
        job.started = True
        try:
            job.thread.start()
        except RuntimeError as exc:
            job.finish(JobOutcome.ERROR, f"Could not start {label.lower()}: {exc}")
            COORDINATOR.release(token)
            raise BackendError(f"Could not start {label.lower()}: {exc}") from exc
        return job

    def cancel_job(self, kind: JobKind, status: str = "Cancelling safely") -> bool:
        job = self.job(kind)
        if job is None or job.terminal or job.cancel.is_set():
            return False
        if job.token is not None:
            COORDINATOR.request_cancel(job.token, status=status)
        else:
            job.cancel.set()
        job.update(job.progress, status, job.phase, log_change=False)
        return True

    def run_exclusive(
        self,
        operation: OperationKind,
        owner: str,
        action: Callable[[], str],
        *,
        status: str,
    ) -> str:
        """Run a short main-thread action while owning the coordinator."""
        token = COORDINATOR.acquire(operation, owner, status=status)
        if token is None:
            raise OperationBusy("Media to Pose is busy")
        try:
            message = action()
        except Exception as exc:
            COORDINATOR.finish(token, OperationState.FAILED, str(exc))
            COORDINATOR.release(token)
            raise
        COORDINATOR.finish(token, OperationState.SUCCEEDED, message)
        COORDINATOR.release(token)
        return message

    # Session ------------------------------------------------------------

    @property
    def session(self) -> Any:
        with self._lock:
            return self._session

    def set_session(self, session: Any) -> Any:
        """Install ``session`` unless another one won the race; return the active one."""
        with self._lock:
            if self._session is None:
                self._session = session
            active = self._session
        self.invalidate()
        return active

    def take_session(self) -> Any:
        with self._lock:
            session, self._session = self._session, None
        self.invalidate()
        return session

    def is_model_loaded(self) -> bool:
        return self.session is not None

    # Readiness cache ----------------------------------------------------

    def cached_snapshot(self, key: tuple, build: Callable[[], Any]) -> Any:
        """Return a cached snapshot so panel redraws never rescan model files."""
        with self._lock:
            if key in self._snapshot_cache:
                return self._snapshot_cache[key]
        value = build()
        with self._lock:
            self._snapshot_cache[key] = value
        return value

    def invalidate(self) -> None:
        with self._lock:
            self._snapshot_cache.clear()

    # Shutdown -----------------------------------------------------------

    def shutdown(self, *, join_timeout: float = 2.0) -> None:
        """Cancel and join background jobs during add-on unregistration."""
        # Only live jobs: cancelling a finished job's token would move a
        # terminal operation back to CANCELLING and keep the lease forever.
        jobs = tuple(job for job in self.jobs() if job.alive())
        for job in jobs:
            if job.token is not None:
                COORDINATOR.request_cancel(
                    job.token, status=f"Cancelling {self.label} operation"
                )
            else:
                job.cancel.set()
        for job in jobs:
            if job.thread is not None and job.thread is not threading.current_thread():
                job.thread.join(timeout=join_timeout)
        for kind in tuple(JobKind):
            job = self.job(kind)
            if job is not None and not job.alive():
                self.clear_job(kind, job)
        self.invalidate()


__all__ = [
    "HF_ENDPOINT",
    "STORAGE_MARKER",
    "BackgroundJob",
    "BundleFile",
    "PinnedBundle",
    "InstallationLock",
    "JobKind",
    "JobOutcome",
    "JobSnapshot",
    "ManagedStorage",
    "ModelBackend",
    "backend_root",
    "contained_path",
    "download_file",
    "hf_resolve_url",
    "install_pinned_bundle",
    "is_immutable_revision",
    "model_installation_lock",
    "read_storage_parent",
    "redact",
    "resolve_user_path",
    "run_job",
    "safe_metadata",
    "safe_relative_path",
    "sha256_file",
    "write_storage_parent",
]
