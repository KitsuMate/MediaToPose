"""Licence acceptance for downloadable models, recorded per model revision.

A model whose licence must be accepted (Kimodo under the NVIDIA Open Model
License, LLM2Vec under the Meta Llama 3 Community License) is installed only
after the user accepts it. The acceptance is stored in the model storage
folder's marker JSON, keyed by licence and revision, with the time and where
it was given (panel, command line or Python API). A new pinned revision asks
again.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ..core.errors import InvalidRequest, LicenseRequired
from .lifecycle import ManagedStorage

MARKER_FIELD = "licenses"


class AcceptanceSource(str, Enum):
    PANEL = "PANEL"
    CLI = "CLI"
    API = "API"


@dataclass(frozen=True)
class ModelLicense:
    """A licence a model download is subject to.

    ``files`` are the licence and notice files installed next to the model.
    """

    license_id: str
    name: str
    summary: str
    url: str
    files: tuple[str, ...] = ("LICENSE", "NOTICE")

    def accept_value(self, revision: str) -> str:
        """The exact value ``--accept-license`` takes for ``revision``."""
        return f"{self.license_id}@{revision}"


def acceptance(
    storage: ManagedStorage, root: Path, license: ModelLicense, revision: str
) -> dict[str, str] | None:
    """The recorded acceptance of ``license`` for ``revision``, if any."""
    accepted = storage.read_marker(root).get(MARKER_FIELD)
    if not isinstance(accepted, dict):
        return None
    entry = accepted.get(license.accept_value(revision))
    return dict(entry) if isinstance(entry, dict) else None


def record_acceptance(
    storage: ManagedStorage,
    root: Path,
    license: ModelLicense,
    revision: str,
    source: AcceptanceSource,
) -> dict[str, str]:
    """Store that the user accepted ``license`` for ``revision``."""
    entry = {
        "license_id": license.license_id,
        "revision": revision,
        "accepted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": AcceptanceSource(source).value,
    }
    accepted = storage.read_marker(root).get(MARKER_FIELD)
    accepted = dict(accepted) if isinstance(accepted, dict) else {}
    accepted[license.accept_value(revision)] = entry
    storage.update_marker(root, **{MARKER_FIELD: accepted})
    return entry


def ensure_accepted(
    storage: ManagedStorage,
    root: Path,
    license: ModelLicense,
    revision: str,
    *,
    accept: str = "",
    source: AcceptanceSource = AcceptanceSource.API,
) -> dict[str, str]:
    """Return the acceptance, recording ``accept`` when it names this revision.

    Raises ``LicenseRequired`` when the licence was never accepted and
    ``accept`` is empty, and ``InvalidRequest`` when ``accept`` names another
    licence or revision.
    """
    expected = license.accept_value(revision)
    if accept and accept != expected:
        raise InvalidRequest(
            f"--accept-license {accept!r} does not match {expected!r}",
            accept_license=expected,
        )
    if accept:
        return record_acceptance(storage, root, license, revision, source)
    existing = acceptance(storage, root, license, revision)
    if existing is not None:
        return existing
    raise LicenseRequired(
        f"{license.name} must be accepted before this model is downloaded",
        license_id=license.license_id,
        name=license.name,
        summary=license.summary,
        url=license.url,
        revision=revision,
        accept_license=expected,
    )


__all__ = [
    "MARKER_FIELD",
    "AcceptanceSource",
    "ModelLicense",
    "acceptance",
    "ensure_accepted",
    "record_acceptance",
]
