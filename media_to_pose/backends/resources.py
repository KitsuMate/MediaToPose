"""Free disk and memory checks before a model is downloaded or loaded.

Shared by every model backend. Shortfalls raise ``InsufficientResources``
with the numbers, so the panel and agents can say what is missing.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..core.errors import BackendError, InsufficientResources

GIB = 1024**3


def closest_existing_directory(path: Path) -> Path:
    """``path`` or its nearest existing parent, whose volume will hold it."""
    candidate = Path(path).resolve(strict=False)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise BackendError(f"No accessible parent directory exists for {path}")
    return candidate


def free_disk_bytes(path: Path) -> int:
    """Free bytes on the volume that holds (or will hold) ``path``."""
    return int(shutil.disk_usage(closest_existing_directory(path)).free)


def same_volume(first: Path, second: Path) -> bool:
    return (
        closest_existing_directory(first).stat().st_dev
        == closest_existing_directory(second).stat().st_dev
    )


def _gib(value: int) -> str:
    return f"{value / GIB:.1f} GB"


def require_disk(
    path: Path,
    needed_bytes: int,
    label: str,
    *,
    fitting_variant: str = "",
) -> None:
    """Raise ``InsufficientResources`` when ``path``'s volume lacks the space."""
    available = free_disk_bytes(path)
    if available >= needed_bytes:
        return
    raise InsufficientResources(
        f"{label} needs {_gib(needed_bytes)} of free disk space; "
        f"{_gib(available)} is free",
        resource="disk",
        needed_bytes=int(needed_bytes),
        available_bytes=available,
        path=str(closest_existing_directory(path)),
        fitting_variant=fitting_variant,
    )


__all__ = [
    "GIB",
    "closest_existing_directory",
    "free_disk_bytes",
    "require_disk",
    "same_volume",
]
