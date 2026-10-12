"""Which capture providers and optional models are available."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
from typing import Any

from ..backends.anycalib import backend as anycalib_backend
from ..backends.rohm import backend as rohm_backend
from ..backends.sam3d import backend as sam3d_backend
from ..core.requests import CaptureSettings
from ..providers import DESCRIPTORS
from ..session import capture_options
from .common import resolve_scene


def _plain(snapshot) -> dict[str, Any]:
    data = (
        asdict(snapshot)
        if is_dataclass(snapshot) and not isinstance(snapshot, type)
        else dict(snapshot)
    )
    return {
        key: value
        if isinstance(value, (str, int, float, bool, type(None)))
        else str(value)
        for key, value in data.items()
    }


def list_providers(*, scene=None) -> list[dict[str, Any]]:
    """Capture providers, the parts they capture and whether they can run."""
    settings = CaptureSettings.from_rna(resolve_scene(scene).mtp_settings)
    providers = []
    for identifier, descriptor in DESCRIPTORS.items():
        options = capture_options.options_from_settings(
            replace(settings, provider=identifier)
        )
        try:
            available, message = descriptor.dependency_status(options)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            available, message = False, str(exc)
        providers.append(
            {
                "id": identifier,
                "label": descriptor.label,
                "description": descriptor.description,
                "parts": sorted(part.lower() for part in descriptor.capabilities),
                "available": bool(available),
                "message": str(message),
            }
        )
    return providers


def model_status(*, scene=None) -> dict[str, dict[str, Any]]:
    """Install and load state of the optional models, with download sizes."""
    settings = resolve_scene(scene).mtp_settings
    sam = _plain(
        sam3d_backend.backend_snapshot(
            settings.sam3d_model_variant, settings.sam3d_device
        )
    )
    sam["download_bytes"] = int(
        sam3d_backend.model_spec(settings.sam3d_model_variant)["download_bytes"]
    )
    sam["loaded"] = sam3d_backend.is_model_loaded()
    anycalib = _plain(anycalib_backend.backend_snapshot(settings.anycalib_device))
    anycalib["loaded"] = anycalib_backend.is_model_loaded()
    rohm = _plain(rohm_backend.backend_snapshot(settings.rohm_device))
    rohm["loaded"] = rohm_backend.is_model_loaded()
    return {
        "SAM3D_BODY": sam,
        "ANYCALIB": anycalib,
        "ROHM": rohm,
    }


__all__ = ["list_providers", "model_status"]
