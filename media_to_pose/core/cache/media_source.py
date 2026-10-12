"""The selected source media: identity, existence and the retained still copy."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from .. import paths
from ..fs import atomic_json, replace_directory
from ..log import get_logger
from .store import (
    _LOCK,
    CACHE_SCHEMA_VERSION,
    MANIFEST_NAME,
    _safe_remove,
    cache_root,
    invalidate_ui_state,
    managed_source_directory,
)

LOGGER = get_logger(__name__)


IMAGE_EXTENSIONS = {
    ".bmp",
    ".exr",
    ".gif",
    ".hdr",
    ".jpeg",
    ".jpg",
    ".png",
    ".tga",
    ".tif",
    ".tiff",
    ".webp",
}


def _canonical_media_path(value: str) -> str:
    raw = str(value or "").strip().strip('"')
    if not raw:
        return ""
    try:
        return str(paths.expand_user_path(raw).resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return ""


def existing_media_path(value: str) -> str:
    """Return a canonical existing file path without changing Blender data."""
    canonical = _canonical_media_path(value)
    if not canonical:
        return ""
    resolved = Path(canonical)
    return str(resolved) if resolved.is_file() else ""


def _source_media(path: str) -> dict:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _stored_source_media(settings) -> dict | None:
    try:
        value = json.loads(str(getattr(settings, "media_identity_json", "") or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    path = _canonical_media_path(str(value.get("path", "")))
    if not path:
        return None
    display = _canonical_media_path(str(getattr(settings, "media_path", "") or ""))
    if display and os.path.normcase(path) != os.path.normcase(display):
        return None
    try:
        return {
            "path": path,
            "size": int(value["size"]),
            "mtime_ns": int(value["mtime_ns"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _managed_source() -> tuple[dict | None, str]:
    directory = managed_source_directory(create_root=False)
    try:
        manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf8"))
        source = manifest["source"]
        filename = str(manifest["file"])
        candidate = (directory / filename).resolve()
        if directory.resolve() not in candidate.parents or not candidate.is_file():
            return None, ""
        return {
            "path": str(source["path"]),
            "size": int(source["size"]),
            "mtime_ns": int(source["mtime_ns"]),
        }, str(candidate)
    except (OSError, ValueError, TypeError, KeyError):
        return None, ""


def _same_source(left: dict | None, right: dict | None) -> bool:
    if not left or not right:
        return False
    try:
        return (
            os.path.normcase(str(left["path"])) == os.path.normcase(str(right["path"]))
            and int(left["size"]) == int(right["size"])
            and int(left["mtime_ns"]) == int(right["mtime_ns"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def retain_still_source(settings, path: str) -> bool:
    """Atomically retain the latest compressed still-image source."""
    if not bool(getattr(settings, "keep_detection_cache", False)):
        return False
    selected = existing_media_path(path)
    if not selected or Path(selected).suffix.casefold() not in IMAGE_EXTENSIONS:
        return False
    source = _source_media(selected)
    limit_bytes = max(
        0,
        int(getattr(settings, "cache_limit_mb", 512)) * 1024 * 1024,
    )
    if limit_bytes and int(source["size"]) > limit_bytes:
        return False
    root = cache_root(create=True)
    staging = Path(tempfile.mkdtemp(prefix=".source-", dir=str(root)))
    filename = f"media{Path(selected).suffix.casefold()}"
    try:
        shutil.copy2(selected, staging / filename)
        atomic_json(
            staging / MANIFEST_NAME,
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "source": source,
                "file": filename,
            },
        )
        active = managed_source_directory(create_root=False)
        with _LOCK:
            replace_directory(
                staging, active, root / ".source-previous", remove=_safe_remove
            )
        settings.media_identity_json = json.dumps(source, sort_keys=True)
        invalidate_ui_state()
        return True
    except OSError:
        if staging.exists():
            _safe_remove(staging)
        raise


def record_media_selection(
    settings,
    path: str,
    *,
    assign_path: bool = True,
) -> str:
    """Record one canonical media selection and its exact source identity.

    This is the sole writer for media identity.  ``assign_path=False`` is used
    by the RNA update callback when a user edits the path field directly; the
    explicit file-selection operator uses the default and canonicalizes the
    displayed path as well.
    """
    selected = existing_media_path(path)
    if not selected:
        return ""
    if assign_path and str(settings.media_path) != selected:
        # Assignment invokes the RNA callback, which records the identity with
        # ``assign_path=False``. Avoid doing the same file work twice.
        settings.media_path = selected
        return selected
    source = _source_media(selected)
    settings.media_identity_json = json.dumps(source, sort_keys=True)
    retain_still_source(settings, selected)
    return selected
