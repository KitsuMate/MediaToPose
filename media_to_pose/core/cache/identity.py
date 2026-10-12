"""The cache key: media identity, provider fingerprint and capture options."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from ..log import get_logger
from .media_source import _source_media
from .store import CACHE_SCHEMA_VERSION

LOGGER = get_logger(__name__)


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
    *,
    provider_identity,
) -> tuple[str, dict]:
    return cache_identity_from_media(
        _source_media(path),
        provider_id,
        options,
        provider_identity=provider_identity,
    )


def cache_identity_from_media(
    media: dict,
    provider_id: str,
    options: dict,
    *,
    provider_identity,
) -> tuple[str, dict]:
    """Return the cache key and identity for ``media`` captured with ``options``.

    ``provider_identity`` is the provider's own fingerprint of the options
    that change its output; the session layer looks it up.
    """
    source = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "media": {
            "path": str(media["path"]),
            "size": int(media["size"]),
            "mtime_ns": int(media["mtime_ns"]),
        },
        "provider": provider_id,
        "provider_identity": provider_identity,
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
