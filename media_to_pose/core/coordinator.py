"""Process-wide ownership for long-running Media to Pose operations.

Blender operator ``poll`` methods are only presentation hints.  Every
operation must acquire a token here before it starts work, and every callback
must prove that it still owns the current token before mutating Blender state.
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .errors import OperationBusy
from .log import get_logger

LOGGER = get_logger(__name__)

PROCESS_TERMINATE_GRACE_SECONDS = 2.0


class OperationKind(str, Enum):
    CAPTURE = "capture"
    CACHE_APPLY = "cache_apply"
    CACHE_MAINTENANCE = "cache_maintenance"
    POSTPROCESS_PREVIEW = "postprocess_preview"
    POSTPROCESS_APPLY = "postprocess_apply"
    SAM3D_INSTALL = "sam3d_install"
    SAM3D_MODEL_LOAD = "sam3d_model_load"
    SAM3D_MODEL_UNLOAD = "sam3d_model_unload"
    SAM3D_UNINSTALL = "sam3d_uninstall"
    ROHM_INSTALL = "rohm_install"
    ROHM_MODEL_LOAD = "rohm_model_load"
    ROHM_MODEL_UNLOAD = "rohm_model_unload"
    ROHM_UNINSTALL = "rohm_uninstall"
    ANYCALIB_INSTALL = "anycalib_install"
    ANYCALIB_MODEL_LOAD = "anycalib_model_load"
    ANYCALIB_MODEL_UNLOAD = "anycalib_model_unload"
    ANYCALIB_UNINSTALL = "anycalib_uninstall"


class OperationState(str, Enum):
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    SUCCEEDED = "SUCCEEDED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class OperationToken:
    generation: int
    kind: OperationKind
    owner: str
    created_monotonic: float
    scene_name: str = ""
    scene_pointer: int = 0
    target_name: str = ""
    target_pointer: int = 0
    cancel_event: threading.Event = field(
        default_factory=threading.Event,
        compare=False,
        repr=False,
    )


@dataclass(frozen=True)
class OperationSnapshot:
    active: bool = False
    generation: int = 0
    kind: str = ""
    owner: str = ""
    state: str = ""
    phase: str = ""
    status: str = ""
    progress: float = 0.0
    frame_current: int = 0
    frame_total: int = 0
    step_current: int = 0
    step_total: int = 0
    step_progress: float = 0.0
    step_eta_seconds: float = -1.0
    scene_name: str = ""
    target_name: str = ""
    cancel_requested: bool = False


class OperationCoordinator:
    """Own exactly one mutating operation for the current Blender process."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._generation = 0
        self._token: OperationToken | None = None
        self._state = OperationState.CANCELLED
        self._phase = ""
        self._status = ""
        self._progress = 0.0
        self._frame_current = 0
        self._frame_total = 0
        self._step_current = 0
        self._step_total = 0
        self._step_progress = 0.0
        self._step_started_monotonic = 0.0
        self._worker: threading.Thread | None = None
        self._process = None
        self._cancel_escalation_started = False

    def _owns_locked(self, token: OperationToken) -> bool:
        current = self._token
        return bool(
            current is not None
            and current.generation == token.generation
            and current.kind == token.kind
            and current.owner == token.owner
        )

    def _worker_stopped_locked(self) -> bool:
        thread_stopped = self._worker is None or not self._worker.is_alive()
        process_stopped = (
            self._process is None
            or getattr(self._process, "poll", lambda: 0)() is not None
        )
        return thread_stopped and process_stopped

    def _clear_locked(self) -> None:
        self._token = None
        self._worker = None
        self._process = None
        self._cancel_escalation_started = False
        self._phase = ""
        self._status = ""
        self._progress = 0.0
        self._frame_current = 0
        self._frame_total = 0
        self._step_current = 0
        self._step_total = 0
        self._step_progress = 0.0
        self._step_started_monotonic = 0.0

    def _reap_locked(self) -> None:
        if (
            self._token is not None
            and self._state
            in {
                OperationState.SUCCEEDED,
                OperationState.CANCELLED,
                OperationState.FAILED,
            }
            and self._worker_stopped_locked()
        ):
            self._clear_locked()

    def acquire(
        self,
        kind: OperationKind,
        owner: str,
        *,
        scene_name: str = "",
        scene_pointer: int = 0,
        target_name: str = "",
        target_pointer: int = 0,
        status: str = "Starting",
    ) -> OperationToken | None:
        with self._lock:
            self._reap_locked()
            if self._token is not None:
                return None
            self._generation += 1
            token = OperationToken(
                generation=self._generation,
                kind=kind,
                owner=str(owner),
                created_monotonic=time.monotonic(),
                scene_name=str(scene_name),
                scene_pointer=int(scene_pointer),
                target_name=str(target_name),
                target_pointer=int(target_pointer),
            )
            self._token = token
            self._state = OperationState.RUNNING
            self._phase = ""
            self._status = str(status)
            self._progress = 0.0
            self._frame_current = 0
            self._frame_total = 0
            self._step_current = 0
            self._step_total = 0
            self._step_progress = 0.0
            self._step_started_monotonic = 0.0
            return token

    def require(
        self,
        kind: OperationKind,
        owner: str,
        **context: Any,
    ) -> OperationToken:
        token = self.acquire(kind, owner, **context)
        if token is None:
            snapshot = self.snapshot()
            label = snapshot.status or snapshot.kind or "another operation"
            raise OperationBusy(f"Media to Pose is busy: {label}")
        return token

    def owns(self, token: OperationToken | None) -> bool:
        if token is None:
            return False
        with self._lock:
            return self._owns_locked(token)

    def active_token(self) -> OperationToken | None:
        with self._lock:
            self._reap_locked()
            return self._token

    def attach_worker(
        self,
        token: OperationToken,
        *,
        thread: threading.Thread | None = None,
        process=None,
    ) -> bool:
        process_to_cancel = None
        with self._lock:
            if not self._owns_locked(token):
                return False
            if thread is not None:
                self._worker = thread
            if process is not None:
                self._process = process
                if token.cancel_event.is_set():
                    process_to_cancel = process
        if process_to_cancel is not None:
            self._cancel_process(process_to_cancel)
        return True

    def _cancel_process(self, process) -> None:
        if getattr(process, "poll", lambda: 0)() is not None:
            return
        try:
            process.terminate()
        except OSError:
            return
        with self._lock:
            if self._cancel_escalation_started:
                return
            self._cancel_escalation_started = True

        def escalate() -> None:
            deadline = time.monotonic() + PROCESS_TERMINATE_GRACE_SECONDS
            while (
                getattr(process, "poll", lambda: 0)() is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.05)
            if getattr(process, "poll", lambda: 0)() is None:
                try:
                    getattr(process, "kill", lambda: None)()
                except OSError as exc:
                    LOGGER.warning("Could not kill cancelled subprocess: %s", exc)

        threading.Thread(
            target=escalate,
            daemon=True,
            name="MediaToPoseProcessCancellation",
        ).start()

    def update(
        self,
        token: OperationToken,
        *,
        status: str | None = None,
        phase: str | None = None,
        progress: float | None = None,
        frame_current: int | None = None,
        frame_total: int | None = None,
        step_current: int | None = None,
        step_total: int | None = None,
        step_progress: float | None = None,
    ) -> bool:
        with self._lock:
            if not self._owns_locked(token):
                return False
            if self._state not in {
                OperationState.RUNNING,
                OperationState.CANCELLING,
            }:
                return False
            if status is not None:
                self._status = str(status)
            if phase is not None:
                self._phase = str(phase)
            if progress is not None:
                self._progress = min(max(float(progress), 0.0), 1.0)
            if frame_total is not None:
                self._frame_total = max(int(frame_total), 0)
                self._frame_current = min(self._frame_current, self._frame_total)
            if frame_current is not None:
                self._frame_current = max(int(frame_current), 0)
                if self._frame_total:
                    self._frame_current = min(self._frame_current, self._frame_total)
            if step_total is not None:
                self._step_total = max(int(step_total), 0)
                self._step_current = min(self._step_current, self._step_total)
            if step_current is not None:
                next_step = max(int(step_current), 0)
                if self._step_total:
                    next_step = min(next_step, self._step_total)
                if next_step != self._step_current:
                    self._step_progress = 0.0
                    self._step_started_monotonic = time.monotonic()
                self._step_current = next_step
            if step_progress is not None:
                value = min(max(float(step_progress), 0.0), 1.0)
                if self._step_current and not self._step_started_monotonic:
                    self._step_started_monotonic = time.monotonic()
                self._step_progress = max(self._step_progress, value)
            return True

    def request_cancel(
        self,
        token: OperationToken | None = None,
        *,
        status: str = "Cancelling",
    ) -> bool:
        with self._lock:
            current = self._token
            if current is None or (token is not None and not self._owns_locked(token)):
                return False
            current.cancel_event.set()
            self._state = OperationState.CANCELLING
            self._status = str(status)
            process = self._process
        if process is not None and getattr(process, "poll", lambda: 0)() is None:
            self._cancel_process(process)
        return True

    def finish(
        self,
        token: OperationToken,
        state: OperationState,
        status: str,
    ) -> bool:
        if state not in {
            OperationState.SUCCEEDED,
            OperationState.CANCELLED,
            OperationState.FAILED,
        }:
            raise ValueError(f"Invalid terminal operation state: {state}")
        with self._lock:
            if not self._owns_locked(token):
                return False
            self._state = state
            self._status = str(status)
            if state == OperationState.SUCCEEDED:
                self._progress = 1.0
                self._frame_current = self._frame_total
                self._step_current = self._step_total
                self._step_progress = 1.0
            return True

    def release(self, token: OperationToken) -> bool:
        with self._lock:
            if not self._owns_locked(token):
                return False
            if not self._worker_stopped_locked():
                return False
            self._clear_locked()
            return True

    def snapshot(self) -> OperationSnapshot:
        with self._lock:
            self._reap_locked()
            token = self._token
            if token is None:
                return OperationSnapshot()
            step_eta_seconds = -1.0
            if self._step_progress >= 1.0:
                step_eta_seconds = 0.0
            elif self._step_progress > 0.0 and self._step_started_monotonic:
                elapsed = max(time.monotonic() - self._step_started_monotonic, 0.0)
                if elapsed >= 0.5:
                    step_eta_seconds = (
                        elapsed * (1.0 - self._step_progress) / self._step_progress
                    )
            return OperationSnapshot(
                active=True,
                generation=token.generation,
                kind=token.kind.value,
                owner=token.owner,
                state=self._state.value,
                phase=self._phase,
                status=self._status,
                progress=self._progress,
                frame_current=self._frame_current,
                frame_total=self._frame_total,
                step_current=self._step_current,
                step_total=self._step_total,
                step_progress=self._step_progress,
                step_eta_seconds=step_eta_seconds,
                scene_name=token.scene_name,
                target_name=token.target_name,
                cancel_requested=token.cancel_event.is_set(),
            )

    def is_busy(self) -> bool:
        return self.snapshot().active

    def shutdown(self, *, join_timeout_seconds: float = 2.0) -> bool:
        token = self.active_token()
        if token is None:
            return True
        self.request_cancel(token, status="Cancelling during add-on shutdown")
        with self._lock:
            worker = self._worker
            process = self._process
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=max(0.0, float(join_timeout_seconds)))
        if process is not None and getattr(process, "poll", lambda: 0)() is None:
            try:
                process.wait(timeout=max(0.0, float(join_timeout_seconds)))
            except (OSError, subprocess.TimeoutExpired, TimeoutError):
                try:
                    process.kill()
                except OSError as exc:
                    LOGGER.warning("Could not kill subprocess during shutdown: %s", exc)
        self.finish(token, OperationState.CANCELLED, "Cancelled during shutdown")
        return self.release(token)


COORDINATOR = OperationCoordinator()


__all__ = [
    "COORDINATOR",
    "OperationCoordinator",
    "OperationKind",
    "OperationSnapshot",
    "OperationState",
    "OperationToken",
]
