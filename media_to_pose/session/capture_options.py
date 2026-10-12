"""Capture settings to provider options, cache identity and cache status."""

from __future__ import annotations

import time

from ..constants import CAPTURE_CONFIDENCE
from ..core import paths
from ..core.cache import identity as cache_identity
from ..core.cache.media_source import (
    _canonical_media_path,
    _managed_source,
    _same_source,
    _source_media,
    _stored_source_media,
    existing_media_path,
)
from ..core.cache.store import (
    _LOCK,
    _UI_STATE_CACHE,
    CacheStatus,
    MediaSourceState,
    _read_manifest,
    _status_from_manifest,
)
from ..providers import descriptor

_UI_STATE_TTL_SECONDS = 0.5


def cache_identity_from_media(
    media: dict,
    provider_id: str,
    options: dict,
) -> tuple[str, dict]:
    """Return the cache key and identity, fingerprinted by the provider."""
    return cache_identity.cache_identity_from_media(
        media,
        provider_id,
        options,
        provider_identity=descriptor(provider_id).cache_fingerprint(options),
    )


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


def options_from_settings(settings) -> dict:
    options = {
        "model_directory": str(paths.BUNDLED_MODELS),
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
        from ..backends.anycalib import backend as anycalib_backend

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
        from ..backends.sam3d.backend import provider_options_from_settings

        options.update(provider_options_from_settings(settings))
    return options


def _identity_options_from_settings(settings) -> dict:
    """Return only values that contribute to a detection-cache identity."""
    options = {
        "model_directory": str(paths.BUNDLED_MODELS),
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
        from ..backends.anycalib import backend as anycalib_backend

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
