"""Versioned, single-input provider-neutral detection cache.

The cache is intentionally external to the .blend. Only the scene preference
that enables it is persistent Blender data. Frames are stored in independently
compressed NumPy chunks so a preview never needs to load a complete clip.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import bpy

from .constants import CAPTURE_CONFIDENCE
from .core.cache.frames import (
    compact_frame,
    read_chunk,
    write_chunk,
)
from .core.errors import InvalidRequest, MediaError
from .core.fs import atomic_json, replace_directory
from .core.log import get_logger
from .models import (
    CanonicalFrame,
    MediaInfo,
)

LOGGER = get_logger(__name__)

CACHE_SCHEMA_VERSION = 7
CHUNK_SIZE = 120
CURRENT_DIRECTORY = "current"
SOURCE_DIRECTORY = "source"
CAMERA_MOTION_DIRECTORY = "camera_motion"
MANIFEST_NAME = "manifest.json"
_LOCK = threading.RLock()
_UI_STATE_TTL_SECONDS = 0.5
_UI_STATE_CACHE: dict[tuple, tuple[float, MediaSourceState, CacheStatus]] = {}
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


@dataclass(frozen=True)
class CacheStatus:
    key: str = ""
    matches: bool = False
    complete: bool = False
    frame_count: int = 0
    cached_frames: int = 0
    size_bytes: int = 0
    media_path: str = ""
    media_kind: str = ""
    fps: float = 0.0
    duration_seconds: float = 0.0
    width: int = 0
    height: int = 0
    configuration: str = ""
    message: str = "Not analyzed"


@dataclass(frozen=True)
class MediaSourceState:
    state: str = "EMPTY"
    display_path: str = ""
    analysis_path: str = ""
    source_media: dict | None = None
    cache_key: str = ""
    cache_matches: bool = False
    message: str = "Choose an image or video"

    @property
    def usable(self) -> bool:
        return self.state in {
            "AVAILABLE",
            "MANAGED_COPY",
            "CACHED_DETECTION",
        }


def cache_root(*, create: bool = True) -> Path:
    value = bpy.utils.user_resource(
        "SCRIPTS",
        path="media_to_pose/detection_cache",
        create=create,
    )
    return Path(value)


def current_directory(*, create_root: bool = True) -> Path:
    return cache_root(create=create_root) / CURRENT_DIRECTORY


def managed_source_directory(*, create_root: bool = True) -> Path:
    return cache_root(create=create_root) / SOURCE_DIRECTORY


def camera_motion_directory(*, create_root: bool = True) -> Path:
    """Return the active derived camera-track sidecar directory."""
    return cache_root(create=create_root) / CAMERA_MOTION_DIRECTORY / CURRENT_DIRECTORY


def _canonical_media_path(value: str) -> str:
    raw = str(value or "").strip().strip('"')
    if not raw:
        return ""
    try:
        return str(Path(bpy.path.abspath(raw)).expanduser().resolve(strict=False))
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


def resolve_media_source(settings) -> MediaSourceState:
    raw = str(getattr(settings, "media_path", "") or "")
    display = _canonical_media_path(raw)
    if not raw.strip():
        return MediaSourceState()
    selected = existing_media_path(raw)
    if selected:
        source = _source_media(selected)
        return MediaSourceState(
            state="AVAILABLE",
            display_path=selected,
            analysis_path=selected,
            source_media=source,
            message="Source available",
        )

    stored = _stored_source_media(settings)
    if bool(getattr(settings, "keep_detection_cache", False)) and stored:
        managed_media, managed_path = _managed_source()
        if managed_path and _same_source(stored, managed_media):
            return MediaSourceState(
                state="MANAGED_COPY",
                display_path=display,
                analysis_path=managed_path,
                source_media=stored,
                message="Using retained source image",
            )
        try:
            key, _identity = cache_identity_from_media(
                stored,
                settings.provider,
                options_from_settings(settings),
            )
        except (OSError, ValueError, TypeError):
            key = ""
        manifest = _read_manifest()
        if key and manifest and manifest.get("key") == key:
            return MediaSourceState(
                state="CACHED_DETECTION",
                display_path=display,
                source_media=stored,
                cache_key=key,
                cache_matches=True,
                message="Using retained detection results",
            )
    return MediaSourceState(
        state="MISSING",
        display_path=display or raw,
        source_media=stored,
        message=f"Selected file no longer exists: {display or raw}",
    )


def _ui_state_key(settings) -> tuple:
    return (
        str(getattr(settings, "media_path", "") or ""),
        str(getattr(settings, "media_identity_json", "") or ""),
        bool(getattr(settings, "keep_detection_cache", False)),
        str(getattr(settings, "provider", "MEDIAPIPE")),
        bool(getattr(settings, "use_body", False)),
        bool(getattr(settings, "use_hands", False)),
        bool(getattr(settings, "use_face", False)),
        bool(getattr(settings, "mirror_input", False)),
        str(getattr(settings, "sam3d_model_variant", "VITH")),
        str(getattr(settings, "sam3d_device", "AUTO")),
    )


def invalidate_ui_state() -> None:
    """Discard media/cache UI data after settings or cached files change."""
    with _LOCK:
        _UI_STATE_CACHE.clear()


def ui_state_for_settings(settings) -> tuple[MediaSourceState, CacheStatus]:
    """Share bounded filesystem work between panels during redraw bursts."""
    key = _ui_state_key(settings)
    now = time.monotonic()
    with _LOCK:
        cached = _UI_STATE_CACHE.get(key)
        if cached and now - cached[0] < _UI_STATE_TTL_SECONDS:
            return cached[1], cached[2]
    source = resolve_media_source(settings)
    status = status_for_settings(settings, media_source=source)
    with _LOCK:
        if len(_UI_STATE_CACHE) >= 16:
            _UI_STATE_CACHE.clear()
        _UI_STATE_CACHE[key] = (now, source, status)
    return source, status


def clear_unrecoverable_media_selection(settings) -> bool:
    """Clear a stale path unless an exact retained source or cache can recover it."""
    if resolve_media_source(settings).state != "MISSING":
        return False
    settings.media_identity_json = ""
    settings.media_path = ""
    settings.status = "Ready"
    settings.status_kind = "READY"
    return True


def _safe_remove(path: Path) -> None:
    root = cache_root(create=True).resolve()
    resolved = path.resolve()
    if resolved == root or root not in resolved.parents:
        raise InvalidRequest(f"Refusing to remove path outside cache root: {path}")
    if resolved.is_dir():
        shutil.rmtree(resolved, ignore_errors=True)
    elif resolved.exists():
        resolved.unlink()


def clear_cache() -> None:
    with _LOCK:
        for candidate in (
            current_directory(create_root=True),
            managed_source_directory(create_root=True),
            cache_root(create=True) / CAMERA_MOTION_DIRECTORY,
        ):
            if candidate.exists():
                _safe_remove(candidate)
        invalidate_ui_state()


def clear_camera_motion_cache() -> None:
    """Delete only the derived camera-motion track."""
    with _LOCK:
        candidate = cache_root(create=True) / CAMERA_MOTION_DIRECTORY
        if candidate.exists():
            _safe_remove(candidate)
        invalidate_ui_state()


def cache_usage_bytes() -> int:
    """Return bytes owned by retained detections, sources, and camera tracks."""
    return sum(
        _path_size(candidate)
        for candidate in (
            current_directory(create_root=False),
            managed_source_directory(create_root=False),
            cache_root(create=False) / CAMERA_MOTION_DIRECTORY,
        )
    )


def discard_temporary(path: str) -> None:
    candidate = Path(path)
    if candidate.exists():
        _safe_remove(candidate)


def _read_manifest(directory: Path | None = None) -> dict | None:
    path = (directory or current_directory(create_root=False)) / MANIFEST_NAME
    try:
        value = json.loads(path.read_text(encoding="utf8"))
    except (OSError, ValueError, TypeError):
        return None
    if value.get("schema_version") != CACHE_SCHEMA_VERSION:
        return None
    return value


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return sum(
            candidate.stat().st_size
            for candidate in path.rglob("*")
            if candidate.is_file()
        )
    except OSError:
        return 0


@lru_cache(maxsize=4)
def _model_identity(model_directory: str) -> tuple[tuple[str, int, int], ...]:
    directory = Path(model_directory)
    result = []
    try:
        for path in sorted(directory.glob("*")):
            if path.is_file():
                stat = path.stat()
                result.append((path.name, stat.st_size, stat.st_mtime_ns))
    except OSError as exc:
        LOGGER.warning("Could not fingerprint bundled model files: %s", exc)
    return tuple(result)


def cache_identity(
    path: str,
    provider_id: str,
    options: dict,
) -> tuple[str, dict]:
    return cache_identity_from_media(
        _source_media(path),
        provider_id,
        options,
    )


def cache_identity_from_media(
    media: dict,
    provider_id: str,
    options: dict,
) -> tuple[str, dict]:
    from .providers import descriptor

    provider = descriptor(provider_id)
    fingerprint = provider.cache_fingerprint(options)
    source = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "media": {
            "path": str(media["path"]),
            "size": int(media["size"]),
            "mtime_ns": int(media["mtime_ns"]),
        },
        "provider": provider_id,
        "provider_identity": fingerprint,
        "models": (
            _model_identity(options.get("model_directory", ""))
            if provider_id == "MEDIAPIPE"
            else []
        ),
        "capture": {
            "body": bool(options.get("use_body")),
            "hands": bool(options.get("use_hands")),
            "face": bool(options.get("use_face")),
            "mirror": bool(options.get("mirror")),
            "confidence": round(float(options.get("confidence", 0.5)), 6),
            "perspective_correction": str(
                options.get("perspective_correction", "AUTO")
            ),
            "manual_fov_override": bool(options.get("manual_fov_override", False)),
            "manual_vertical_fov": round(
                float(options.get("manual_vertical_fov", 45.0)), 6
            ),
            "camera_bundle_revision": str(options.get("camera_bundle_revision", "")),
            "camera_precision": str(options.get("camera_precision", "")),
        },
    }
    encoded = json.dumps(
        source,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf8")
    return hashlib.sha256(encoded).hexdigest(), source


def options_from_settings(settings) -> dict:
    options = {
        "model_directory": str(Path(__file__).parent / "assets" / "models"),
        "use_body": bool(settings.use_body),
        "use_hands": bool(settings.use_hands),
        "use_face": bool(settings.use_face and settings.provider == "MEDIAPIPE"),
        "mirror": bool(settings.mirror_input),
        "confidence": CAPTURE_CONFIDENCE,
        "smoothing": float(settings.smoothing),
        "perspective_correction": str(
            getattr(settings, "perspective_correction", "AUTO")
        ),
        "manual_fov_override": bool(getattr(settings, "manual_fov_override", False)),
        "manual_vertical_fov": float(getattr(settings, "manual_vertical_fov", 45.0)),
        "anycalib_device": str(getattr(settings, "anycalib_device", "AUTO")),
    }
    if getattr(settings, "perspective_correction", "AUTO") == "AUTO":
        from . import anycalib_backend

        snapshot = anycalib_backend.backend_snapshot(
            getattr(settings, "anycalib_device", "AUTO")
        )
        options.update(
            {
                "camera_bundle_revision": (
                    str(anycalib_backend.MODEL_SPEC.get("revision") or "")
                    if snapshot.installed
                    else ""
                ),
                "camera_precision": snapshot.precision if snapshot.installed else "",
                "anycalib_installed": bool(snapshot.installed),
                "anycalib_bundle_directory": (
                    str(anycalib_backend.active_directory())
                    if snapshot.installed
                    else ""
                ),
            }
        )
    if settings.provider == "SAM3D_BODY":
        from .sam3d_backend import provider_options_from_settings

        options.update(provider_options_from_settings(settings))
    return options


def _identity_options_from_settings(settings) -> dict:
    """Return only values that contribute to a detection-cache identity."""
    options = {
        "model_directory": str(Path(__file__).parent / "assets" / "models"),
        "use_body": bool(settings.use_body),
        "use_hands": bool(settings.use_hands),
        "use_face": bool(settings.use_face and settings.provider == "MEDIAPIPE"),
        "mirror": bool(settings.mirror_input),
        "confidence": CAPTURE_CONFIDENCE,
        "perspective_correction": str(
            getattr(settings, "perspective_correction", "AUTO")
        ),
        "manual_fov_override": bool(getattr(settings, "manual_fov_override", False)),
        "manual_vertical_fov": float(getattr(settings, "manual_vertical_fov", 45.0)),
    }
    if getattr(settings, "perspective_correction", "AUTO") == "AUTO":
        from . import anycalib_backend

        snapshot = anycalib_backend.backend_snapshot(
            getattr(settings, "anycalib_device", "AUTO")
        )
        options.update(
            {
                "camera_bundle_revision": (
                    str(anycalib_backend.MODEL_SPEC.get("revision") or "")
                    if snapshot.installed
                    else ""
                ),
                "camera_precision": snapshot.precision if snapshot.installed else "",
            }
        )
    if settings.provider == "SAM3D_BODY":
        options.update(
            {
                "variant": str(settings.sam3d_model_variant),
                "device": str(settings.sam3d_device),
            }
        )
    return options


def identity_from_settings(
    settings,
    *,
    media_source: MediaSourceState | None = None,
) -> tuple[str, dict] | None:
    source = media_source or resolve_media_source(settings)
    if not source.source_media:
        return None
    try:
        return cache_identity_from_media(
            source.source_media,
            settings.provider,
            _identity_options_from_settings(settings),
        )
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _configuration_label(identity: dict) -> str:
    capture = identity.get("capture", {})
    enabled = [
        label
        for key, label in (
            ("body", "Body"),
            ("hands", "Hands"),
            ("face", "Face"),
        )
        if capture.get(key)
    ]
    return ", ".join(enabled) or "None"


def status_for_settings(
    settings,
    *,
    media_source: MediaSourceState | None = None,
) -> CacheStatus:
    identity = identity_from_settings(settings, media_source=media_source)
    if identity is None:
        return CacheStatus(message="Choose existing media")
    key, _source = identity
    manifest = _read_manifest()
    if not manifest or manifest.get("key") != key:
        return CacheStatus(key=key, message="Not analyzed")
    return _status_from_manifest(manifest, matches=True)


def media_info_from_status(status: CacheStatus) -> MediaInfo:
    return MediaInfo(
        path=status.media_path,
        kind=status.media_kind,
        fps=status.fps,
        frame_count=status.frame_count,
        duration_seconds=status.duration_seconds,
        width=status.width,
        height=status.height,
    )


def _status_from_manifest(manifest: dict, *, matches: bool) -> CacheStatus:
    media = manifest.get("media_info", {})
    mode = manifest.get("mode", "POINTS")
    cached = (
        int(manifest.get("frame_count", 0))
        if mode == "SEQUENCE"
        else len(manifest.get("points", {}))
    )
    return CacheStatus(
        key=str(manifest.get("key", "")),
        matches=matches,
        complete=mode == "SEQUENCE" and bool(manifest.get("complete")),
        frame_count=int(media.get("frame_count", 1)),
        cached_frames=cached,
        size_bytes=int(manifest.get("size_bytes", 0)),
        media_path=str(media.get("path", "")),
        media_kind=str(media.get("kind", "")),
        fps=float(media.get("fps", 0.0)),
        duration_seconds=float(media.get("duration_seconds", 0.0)),
        width=int(media.get("width", 0)),
        height=int(media.get("height", 0)),
        configuration=_configuration_label(manifest.get("identity", {})),
        message=(
            "Complete" if mode == "SEQUENCE" and manifest.get("complete") else "Sparse"
        ),
    )


def _media_dict(media_info: MediaInfo) -> dict:
    return {
        "path": media_info.path,
        "kind": media_info.kind,
        "fps": float(media_info.fps),
        "frame_count": int(media_info.frame_count),
        "duration_seconds": float(media_info.duration_seconds),
        "width": int(media_info.width),
        "height": int(media_info.height),
    }


def _promote(staging: Path) -> None:
    root = cache_root(create=True)
    with _LOCK:
        replace_directory(
            staging,
            root / CURRENT_DIRECTORY,
            root / ".previous",
            remove=_safe_remove,
        )
    invalidate_ui_state()


class SequenceWriter:
    def __init__(
        self,
        key: str,
        identity: dict,
        media_info: MediaInfo,
        limit_bytes: int,
    ):
        root = cache_root(create=True)
        self.directory = Path(tempfile.mkdtemp(prefix=".sequence-", dir=str(root)))
        self.key = key
        self.identity = identity
        self.media_info = media_info
        self.limit_bytes = max(0, int(limit_bytes))
        self.retained_source_bytes = _path_size(
            managed_source_directory(create_root=False)
        )
        self.buffer: list[CanonicalFrame] = []
        self.chunks: list[dict] = []
        self.count = 0
        self.written_bytes = 0
        self.overflow = bool(
            self.limit_bytes and self.retained_source_bytes > self.limit_bytes
        )

    def add(self, frame: CanonicalFrame) -> None:
        self.buffer.append(compact_frame(frame))
        if len(self.buffer) >= CHUNK_SIZE:
            self._flush()

    def _flush(self) -> None:
        if not self.buffer:
            return
        name = f"chunk_{len(self.chunks):06d}.npz"
        start = self.count
        frames = self.buffer
        self.buffer = []
        write_chunk(self.directory / name, frames)
        try:
            self.written_bytes += (self.directory / name).stat().st_size
        except OSError:
            self.overflow = True
        self.count += len(frames)
        self.chunks.append({"file": name, "start": start, "count": len(frames)})
        if (
            self.limit_bytes
            and self.retained_source_bytes + self.written_bytes > self.limit_bytes
        ):
            self.overflow = True

    def finish(self) -> tuple[Path, bool]:
        self._flush()
        if self.count == 0:
            self.discard()
            raise MediaError("No detectable frames were produced")
        manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "key": self.key,
            "identity": self.identity,
            "media_info": _media_dict(self.media_info),
            "mode": "SEQUENCE",
            "complete": True,
            "frame_count": self.count,
            "chunks": self.chunks,
            "size_bytes": self.written_bytes + self.retained_source_bytes,
        }
        atomic_json(self.directory / MANIFEST_NAME, manifest)
        if not self.overflow:
            _promote(self.directory)
            invalidate_ui_state()
            return current_directory(create_root=False), True
        return self.directory, False

    def discard(self) -> None:
        if self.directory.exists():
            _safe_remove(self.directory)


def store_point(
    key: str,
    identity: dict,
    media_info: MediaInfo,
    source_index: int,
    frame: CanonicalFrame,
) -> None:
    with _LOCK:
        current = current_directory(create_root=True)
        manifest = _read_manifest(current)
        if manifest and manifest.get("key") == key:
            if manifest.get("mode") == "SEQUENCE" and manifest.get("complete"):
                return
            directory = current
        else:
            root = cache_root(create=True)
            directory = Path(tempfile.mkdtemp(prefix=".points-", dir=str(root)))
            manifest = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "key": key,
                "identity": identity,
                "media_info": _media_dict(media_info),
                "mode": "POINTS",
                "complete": False,
                "points": {},
                "size_bytes": _path_size(managed_source_directory(create_root=False)),
            }
        points = directory / "points"
        filename = f"point_{max(0, int(source_index)):09d}.npz"
        existing = manifest.setdefault("points", {}).get(str(int(source_index)), {})
        existing_path = directory / str(existing.get("file", ""))
        try:
            existing_bytes = (
                existing_path.stat().st_size if existing_path.is_file() else 0
            )
        except OSError:
            existing_bytes = 0
        write_chunk(points / filename, [frame])
        try:
            written_bytes = (points / filename).stat().st_size
        except OSError:
            written_bytes = 0
        manifest["points"][str(int(source_index))] = {
            "file": f"points/{filename}",
            "timestamp_ms": int(frame.timestamp_ms),
        }
        manifest["size_bytes"] = max(
            0,
            int(manifest.get("size_bytes", 0)) - existing_bytes + written_bytes,
        )
        atomic_json(directory / MANIFEST_NAME, manifest)
        if directory != current:
            _promote(directory)
        invalidate_ui_state()


def load_frame(key: str, source_index: int) -> CanonicalFrame | None:
    directory = current_directory(create_root=False)
    manifest = _read_manifest(directory)
    if not manifest or manifest.get("key") != key:
        return None
    try:
        if manifest.get("mode") == "SEQUENCE":
            index = max(
                0,
                min(
                    int(source_index),
                    max(0, int(manifest.get("frame_count", 1)) - 1),
                ),
            )
            for chunk in manifest.get("chunks", []):
                start = int(chunk["start"])
                count = int(chunk["count"])
                if start <= index < start + count:
                    return read_chunk(directory / chunk["file"])[index - start]
            return None
        point = manifest.get("points", {}).get(str(int(source_index)))
        if not point:
            return None
        frames = read_chunk(directory / point["file"])
        return frames[0] if frames else None
    except (OSError, ValueError, KeyError, TypeError, UnicodeError):
        return None


def iter_frames(key: str) -> Iterator[CanonicalFrame] | None:
    directory = current_directory(create_root=False)
    manifest = _read_manifest(directory)
    if (
        not manifest
        or manifest.get("key") != key
        or manifest.get("mode") != "SEQUENCE"
        or not manifest.get("complete")
    ):
        return None

    def generate() -> Iterator[CanonicalFrame]:
        for chunk in manifest.get("chunks", []):
            yield from read_chunk(directory / chunk["file"])

    return generate()


def iter_directory(directory: str) -> Iterator[CanonicalFrame]:
    path = Path(directory)
    manifest = _read_manifest(path)
    if not manifest:
        return iter(())

    def generate() -> Iterator[CanonicalFrame]:
        for chunk in manifest.get("chunks", []):
            yield from read_chunk(path / chunk["file"])

    return generate()
