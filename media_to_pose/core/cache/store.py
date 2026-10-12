"""Retained detections on disk: manifests, chunked frames and size limits."""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .. import paths
from ..errors import InvalidRequest, MediaError
from ..fs import atomic_json, replace_directory
from ..log import get_logger
from ..types import CanonicalFrame, MediaInfo
from .frames import compact_frame, read_chunk, write_chunk

LOGGER = get_logger(__name__)
CACHE_SCHEMA_VERSION = 7
CHUNK_SIZE = 120
CURRENT_DIRECTORY = "current"
SOURCE_DIRECTORY = "source"
CAMERA_MOTION_DIRECTORY = "camera_motion"
MANIFEST_NAME = "manifest.json"
_LOCK = threading.RLock()


_UI_STATE_CACHE: dict[tuple, tuple[float, MediaSourceState, CacheStatus]] = {}


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
    return paths.user_directory("detection_cache", create=create)


def current_directory(*, create_root: bool = True) -> Path:
    return cache_root(create=create_root) / CURRENT_DIRECTORY


def managed_source_directory(*, create_root: bool = True) -> Path:
    return cache_root(create=create_root) / SOURCE_DIRECTORY


def camera_motion_directory(*, create_root: bool = True) -> Path:
    """Return the active derived camera-track sidecar directory."""
    return cache_root(create=create_root) / CAMERA_MOTION_DIRECTORY / CURRENT_DIRECTORY


def invalidate_ui_state() -> None:
    """Discard media/cache UI data after settings or cached files change."""
    with _LOCK:
        _UI_STATE_CACHE.clear()


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


def current_manifest() -> dict | None:
    """The manifest of the retained detections, or ``None`` when empty."""
    return _read_manifest()


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
