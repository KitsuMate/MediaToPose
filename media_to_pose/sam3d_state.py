"""Generation-safe background-job references for the SAM 3D subsystem."""

from __future__ import annotations

import threading
from enum import Enum

from .operation import COORDINATOR


class SamJobSlot(str, Enum):
    INSTALL = "installation"
    MODEL = "model"


class SamJobRegistry:
    """Retain status objects without granting them operation ownership."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[SamJobSlot, object] = {}

    def assign(self, slot: SamJobSlot, job: object) -> bool:
        """Record a job only while its generation owns the coordinator."""
        token = getattr(job, "token", None)
        if token is None or not COORDINATOR.owns(token):
            return False
        with self._lock:
            current = self._jobs.get(slot)
            if current is not None and current is not job:
                thread = getattr(current, "thread", None)
                if thread is not None and thread.is_alive():
                    return False
            self._jobs[slot] = job
            return True

    def current(self, slot: SamJobSlot) -> object | None:
        with self._lock:
            return self._jobs.get(slot)

    def owns(self, slot: SamJobSlot, job: object) -> bool:
        token = getattr(job, "token", None)
        with self._lock:
            return bool(
                self._jobs.get(slot) is job
                and token is not None
                and COORDINATOR.owns(token)
            )

    def clear(self, slot: SamJobSlot, expected: object | None = None) -> bool:
        with self._lock:
            current = self._jobs.get(slot)
            if current is None or (expected is not None and current is not expected):
                return False
            del self._jobs[slot]
            return True

    def jobs(self) -> tuple[object, ...]:
        with self._lock:
            return tuple(self._jobs.values())


SAM_JOBS = SamJobRegistry()


__all__ = ["SAM_JOBS", "SamJobRegistry", "SamJobSlot"]
