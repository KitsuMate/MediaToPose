"""Typed errors raised across module boundaries.

Every error carries a stable ``code`` string (for JSON results and logs) and
an ``exit_code`` used by the command line. Each class also derives from the
built-in exception it replaced, so existing ``except ValueError`` or
``except RuntimeError`` handlers keep working while callers migrate.
"""

from __future__ import annotations

from typing import Any


class MtpError(Exception):
    """Base class for all Media to Pose errors."""

    code = "MTP_ERROR"
    exit_code = 1

    def __init__(self, message: str = "", **details: Any) -> None:
        super().__init__(message)
        self.message = str(message)
        self.details = dict(details)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = {
                key: value
                if isinstance(value, (str, int, float, bool, type(None)))
                else str(value)
                for key, value in self.details.items()
            }
        return payload


class InvalidRequest(MtpError, ValueError):
    """Arguments, configuration or presets that cannot be used."""

    code = "INVALID_REQUEST"
    exit_code = 2


class MediaError(MtpError, ValueError):
    """Source media or detections that cannot be read or used."""

    code = "MEDIA_ERROR"
    exit_code = 3


class ModelMissing(MtpError, RuntimeError):
    """An optional model or bundled dependency is not installed or usable."""

    code = "MODEL_MISSING"
    exit_code = 4


class DependencyMissing(ModelMissing, ImportError):
    """A bundled Python dependency (wheel) could not be imported."""

    code = "DEPENDENCY_MISSING"


class MappingIncomplete(MtpError, ValueError):
    """The rig mapping does not cover the requested capture scope."""

    code = "MAPPING_INCOMPLETE"
    exit_code = 5


class OperationBusy(MtpError, RuntimeError):
    """Another operation owns the process-wide lease or install lock."""

    code = "OPERATION_BUSY"
    exit_code = 6


class Cancelled(MtpError, InterruptedError):
    """The operation was cancelled before it finished."""

    code = "CANCELLED"
    exit_code = 7


class BackendError(MtpError, RuntimeError):
    """A model backend, runtime or correction step failed."""

    code = "BACKEND_ERROR"
    exit_code = 8


class CaptureFailed(MtpError, RuntimeError):
    """A capture started but did not finish; Blender data was rolled back."""

    code = "CAPTURE_FAILED"
    exit_code = 9


ERROR_CLASSES: tuple[type[MtpError], ...] = (
    MtpError,
    InvalidRequest,
    MediaError,
    ModelMissing,
    DependencyMissing,
    MappingIncomplete,
    OperationBusy,
    Cancelled,
    BackendError,
    CaptureFailed,
)


__all__ = [
    "ERROR_CLASSES",
    "BackendError",
    "Cancelled",
    "CaptureFailed",
    "DependencyMissing",
    "InvalidRequest",
    "MappingIncomplete",
    "MediaError",
    "ModelMissing",
    "MtpError",
    "OperationBusy",
]
