"""Which capture providers and optional models are available, and installs."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, is_dataclass, replace
from typing import Any

import bpy

from ..backends import licenses
from ..backends.anycalib import backend as anycalib_backend
from ..backends.kimodo import backend as kimodo_backend
from ..backends.llm2vec import backend as llm2vec_backend
from ..backends.rohm import backend as rohm_backend
from ..backends.sam3d import backend as sam3d_backend
from ..core.coordinator import COORDINATOR
from ..core.dependencies import refresh_dependency_snapshot
from ..core.errors import BackendError, Cancelled, InvalidRequest
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


MODELS = ("SAM3D_BODY", "ANYCALIB", "ROHM", "KIMODO", "LLM2VEC")


def _start_install(
    model: str,
    variant: str,
    device: str,
    storage_parent: str,
    accept_license: str,
    source: licenses.AcceptanceSource,
):
    online = bool(bpy.app.online_access)
    if model == "KIMODO":
        return kimodo_backend.start_install(
            variant or "AUTO",
            device or "AUTO",
            storage_parent,
            online_access=online,
            accept_license=accept_license,
            source=source,
        )
    if model == "LLM2VEC":
        return llm2vec_backend.start_install(
            storage_parent,
            online_access=online,
            accept_license=accept_license,
            source=source,
        )
    if model == "SAM3D_BODY":
        return sam3d_backend.start_install(
            variant or "VITH", device or "AUTO", storage_parent, online_access=online
        )
    if model == "ANYCALIB":
        return anycalib_backend.start_install(
            storage_parent, device or "AUTO", online_access=online
        )
    return rohm_backend.start_install(storage_parent, online_access=online)


def install_model(
    model: str,
    *,
    variant: str = "",
    device: str = "",
    storage_parent: str = "",
    accept_license: str = "",
    timeout: float | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    source: licenses.AcceptanceSource = licenses.AcceptanceSource.API,
) -> dict[str, Any]:
    """Download and verify one optional model, blocking until it finishes.

    ``model`` is one of ``MODELS``. Kimodo and LLM2Vec raise
    ``LicenseRequired`` until their licence is accepted: show the user the
    licence from the error's details and pass ``accept_license`` (the exact
    value in those details) only after they agree. Files already downloaded
    are reused, so an interrupted install resumes. ``progress`` receives
    ``{"progress", "phase", "status"}`` while the download runs. Returns the
    model's ``model_status`` entry.
    """
    name = str(model or "").upper().replace("-", "_")
    name = {"SAM3D": "SAM3D_BODY", "SAM": "SAM3D_BODY"}.get(name, name)
    if name not in MODELS:
        raise InvalidRequest(
            f"Unknown model {model!r}; use one of {', '.join(MODELS)}",
            parameter="model",
        )
    job = _start_install(
        name,
        str(variant or "").upper(),
        str(device or "").upper(),
        str(storage_parent or ""),
        accept_license,
        source,
    )
    deadline = None if timeout is None else time.monotonic() + float(timeout)
    last = None
    try:
        while job.alive():
            if deadline is not None and time.monotonic() > deadline:
                job.cancel.set()
                deadline = None
            snapshot = job.snapshot()
            report = (round(snapshot.progress, 3), snapshot.phase, snapshot.status)
            if progress is not None and report != last:
                progress(
                    {"progress": report[0], "phase": report[1], "status": report[2]}
                )
            last = report
            time.sleep(0.25)
        job.thread.join()
    finally:
        if job.token is not None:
            COORDINATOR.release(job.token)
    snapshot = job.snapshot()
    if snapshot.outcome == "CANCELLED":
        raise Cancelled(snapshot.status or f"{name} installation cancelled")
    if snapshot.outcome != "DONE":
        raise BackendError(snapshot.error or snapshot.status or f"{name} failed")
    return model_status()[name]


__all__ = ["MODELS", "install_model", "list_providers", "model_status"]
