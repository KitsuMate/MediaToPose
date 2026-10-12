"""Guard that keeps programmatic mapping edits from becoming user locks."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

_MANUAL_LOCK_SUPPRESSED = ContextVar(
    "media_to_pose_manual_lock_suppressed",
    default=False,
)


@contextmanager
def suppress_manual_locks():
    """Prevent programmatic edits from becoming user-locked mappings."""
    token = _MANUAL_LOCK_SUPPRESSED.set(True)
    try:
        yield
    finally:
        _MANUAL_LOCK_SUPPRESSED.reset(token)


def manual_locks_suppressed() -> bool:
    """Whether mapping edits are currently programmatic."""
    return _MANUAL_LOCK_SUPPRESSED.get()
