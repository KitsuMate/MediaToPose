"""Shared cross-process safety for downloadable model bundles."""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import suppress
from pathlib import Path

from .core.errors import BackendError, OperationBusy


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
    """Exclusive, stale-safe installation lease shared by Blender processes."""

    def __init__(self, path: Path, model_label: str):
        self.path = Path(path)
        self.model_label = str(model_label)
        self.identifier = uuid.uuid4().hex
        self.acquired = False

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
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                try:
                    existing = json.loads(self.path.read_text(encoding="utf8"))
                except (OSError, ValueError, TypeError):
                    existing = {}
                pid = int(existing.get("pid") or 0)
                stale = not _pid_running(pid)
                if not stale:
                    raise OperationBusy(
                        f"Another Blender process is managing {self.model_label}"
                    ) from None
                try:
                    self.path.unlink()
                except OSError as exc:
                    raise BackendError(
                        f"A stale {self.model_label} installation lock could not be removed"
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
        try:
            current = json.loads(self.path.read_text(encoding="utf8"))
        except (OSError, ValueError, TypeError):
            current = {}
        if current.get("identifier") == self.identifier:
            with suppress(FileNotFoundError):
                self.path.unlink()
        self.acquired = False


def model_installation_lock(root: Path, label: str) -> InstallationLock:
    root = Path(root)
    return InstallationLock(root.parent / f".{root.name}.install.lock", label)
