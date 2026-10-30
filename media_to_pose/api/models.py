"""Which capture providers and optional models are available."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
from typing import Any

from ..backends.anycalib import backend as anycalib_backend
from ..backends.kimodo import backend as kimodo_backend
from ..backends.llm2vec import backend as llm2vec_backend
from ..backends.rohm import backend as rohm_backend
from ..backends.sam3d import backend as sam3d_backend
from ..core.dependencies import refresh_dependency_snapshot
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
    # The panel fills the snapshot lazily; a script may be the first caller.
    refresh_dependency_snapshot()
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
    """Install and load state of the optional models, with download sizes.

    ``KIMODO`` and ``LLM2VEC`` also report the variant, device, memory need
    (``min_ram_bytes``) and licence state (``license_state`` is ``ACCEPTED``
    or ``REQUIRED`` for the pinned revision).
    """
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
    # Generation settings arrive with the Generate panel; until then the
    # denoiser reports the AUTO variant and device.
    kimodo = _plain(kimodo_backend.backend_snapshot("AUTO", "AUTO"))
    kimodo["loaded"] = kimodo_backend.is_model_loaded()
    llm2vec = _plain(llm2vec_backend.backend_snapshot())
    llm2vec["loaded"] = llm2vec_backend.is_model_loaded()
    return {
        "SAM3D_BODY": sam,
        "ANYCALIB": anycalib,
        "ROHM": rohm,
        "KIMODO": kimodo,
        "LLM2VEC": llm2vec,
    }


__all__ = ["list_providers", "model_status"]
