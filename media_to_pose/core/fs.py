"""Atomic file and directory replacement.

Every write goes to a temporary sibling first and is moved over the target
with ``Path.replace``, so readers see the old or the new file, never a
partial one. These are the only atomic writers in the package.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np


def _temporary_sibling(path: Path) -> tuple[int, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    return descriptor, Path(temporary)


def _write_atomically(
    path: Path, write: Callable[[Any], None], *, binary: bool, private: bool
) -> None:
    path = Path(path)
    descriptor, temporary = _temporary_sibling(path)
    try:
        mode = "wb" if binary else "w"
        encoding = None if binary else "utf8"
        with os.fdopen(descriptor, mode, encoding=encoding) as handle:
            write(handle)
        if private:
            with suppress(OSError):
                temporary.chmod(stat.S_IRUSR | stat.S_IWUSR)
        temporary.replace(path)
    except BaseException:
        with suppress(OSError):
            temporary.unlink()
        raise


def atomic_write_bytes(path: Path, data: bytes, *, private: bool = False) -> None:
    _write_atomically(
        path, lambda handle: handle.write(data), binary=True, private=private
    )


def atomic_write_text(path: Path, text: str, *, private: bool = False) -> None:
    _write_atomically(
        path, lambda handle: handle.write(text), binary=False, private=private
    )


def atomic_json(
    path: Path,
    value: Any,
    *,
    private: bool = False,
    indent: int | None = 2,
    sort_keys: bool = True,
    ensure_ascii: bool = True,
) -> None:
    """Write indented JSON; ``private`` limits the file to its owner."""
    _write_atomically(
        path,
        lambda handle: json.dump(
            value,
            handle,
            indent=indent,
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
        ),
        binary=False,
        private=private,
    )


def atomic_npz(
    path: Path, arrays: Mapping[str, np.ndarray], *, compressed: bool = True
) -> None:
    """Write an npz that loads with ``allow_pickle=False``."""
    for name, array in arrays.items():
        if np.asarray(array).dtype == object:
            raise TypeError(f"npz array {name!r} would need pickle")
    save: Any = np.savez_compressed if compressed else np.savez
    _write_atomically(
        path,
        lambda handle: save(handle, **dict(arrays)),
        binary=True,
        private=False,
    )


def read_json(path: Path) -> Any:
    """Read UTF-8 JSON; ``None`` when the file is missing or malformed."""
    try:
        return json.loads(Path(path).read_text(encoding="utf8"))
    except (OSError, ValueError, TypeError):
        return None


def remove_tree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def replace_directory(
    staging: Path,
    target: Path,
    previous: Path,
    *,
    remove: Callable[[Path], None] = remove_tree,
) -> None:
    """Swap a fully written ``staging`` directory into ``target``.

    The old target is parked at ``previous`` and restored if the swap fails,
    then removed. ``staging`` is consumed on success.
    """
    staging, target, previous = Path(staging), Path(target), Path(previous)
    if previous.exists():
        remove(previous)
    if target.exists():
        target.replace(previous)
    try:
        staging.replace(target)
    except OSError:
        if previous.exists() and not target.exists():
            previous.replace(target)
        raise
    if previous.exists():
        remove(previous)


__all__ = [
    "atomic_json",
    "atomic_npz",
    "atomic_write_bytes",
    "atomic_write_text",
    "read_json",
    "remove_tree",
    "replace_directory",
]
