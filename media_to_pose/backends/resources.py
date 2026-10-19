"""Free disk and available memory checks before a model is downloaded or loaded.

Shared by every model backend. Shortfalls raise ``InsufficientResources``
with the numbers, so the panel and agents can say what is missing.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from ..core.errors import BackendError, InsufficientResources
from ..core.log import get_logger

LOGGER = get_logger(__name__)
GIB = 1024**3
MEMINFO = Path("/proc/meminfo")


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


def _windows_available_ram() -> int:
    assert sys.platform == "win32"
    import ctypes
    from ctypes import wintypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [  # noqa: RUF012 - ctypes layout
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError(ctypes.get_last_error(), "GlobalMemoryStatusEx failed")
    return int(status.ullAvailPhys)


def _linux_available_ram(meminfo: str) -> int:
    for line in meminfo.splitlines():
        name, _, value = line.partition(":")
        if name == "MemAvailable":
            return int(value.split()[0]) * 1024
    raise OSError("/proc/meminfo has no MemAvailable")


def _sysctl_int(name: str) -> int:
    import ctypes
    import ctypes.util

    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    value = ctypes.c_uint64(0)
    size = ctypes.c_size_t(ctypes.sizeof(value))
    if libc.sysctlbyname(
        name.encode("ascii"), ctypes.byref(value), ctypes.byref(size), None, 0
    ):
        raise OSError(ctypes.get_errno(), f"sysctl {name} failed")
    return int(value.value) if size.value == 8 else int(value.value & 0xFFFFFFFF)


def _macos_available_ram() -> int:
    # The kernel's own memory-pressure gauge: the share of memory that can
    # be handed out without paging, which includes reclaimable file cache.
    return _sysctl_int("hw.memsize") * _sysctl_int("kern.memorystatus_level") // 100


def available_ram_bytes(platform: str = "") -> int | None:
    """Physical memory available to a new allocation; ``None`` when unknown."""
    platform = platform or sys.platform
    try:
        if platform == "win32":
            return _windows_available_ram()
        if platform.startswith("linux"):
            return _linux_available_ram(MEMINFO.read_text(encoding="ascii"))
        if platform == "darwin":
            return _macos_available_ram()
    except (OSError, ValueError, AttributeError) as exc:
        LOGGER.warning("Available memory could not be read: %s", exc)
    return None


def require_ram(
    needed_bytes: int,
    label: str,
    *,
    fitting_variant: str = "",
) -> None:
    """Raise ``InsufficientResources`` when less memory is available.

    An unknown amount (an unsupported platform or a failed query) passes:
    loading then fails, or succeeds, on its own.
    """
    available = available_ram_bytes()
    if available is None or available >= needed_bytes:
        return
    raise InsufficientResources(
        f"{label} needs {_gib(needed_bytes)} of available memory; "
        f"{_gib(available)} is available. Close other applications and retry",
        resource="ram",
        needed_bytes=int(needed_bytes),
        available_bytes=available,
        fitting_variant=fitting_variant,
    )


__all__ = [
    "GIB",
    "available_ram_bytes",
    "closest_existing_directory",
    "free_disk_bytes",
    "require_disk",
    "require_ram",
    "same_volume",
]
