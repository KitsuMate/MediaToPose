"""What a generation reports: the plan of a dry run and the result."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

GENERATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class GeneratePlan:
    """What ``generate`` would do; ``dry_run`` returns it without loading
    any model."""

    armature: str
    frame_range: tuple[float, float]
    model_frames: int
    windows: list[dict[str, Any]]
    segments: list[dict[str, Any]]
    guides: list[dict[str, Any]]
    needs_text_encoder: bool
    warnings: list[dict[str, str]]
    recipe: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dry_run"] = True
        payload["schema_version"] = GENERATE_SCHEMA_VERSION
        return payload


@dataclass(frozen=True)
class GenerateResult:
    """A finished generation.

    ``guides`` lists each guide with its ``guide_error_m`` (how far the
    model's output was from it, in rig metres, before the warp) and
    ``guide_warp_deg`` / ``guide_warp_m`` (the correction that made keyed
    frames exact). ``warnings`` are ``{code, message}``.
    """

    status: str
    message: str
    armature: str
    action: str
    slot: str
    output_mode: str
    nla_track: str
    timeline_frame_range: tuple[float, float]
    model_frames: int
    windows: list[dict[str, Any]]
    segments: list[dict[str, Any]]
    seed: int
    steps: int
    variant: str
    device: str
    guides: list[dict[str, Any]]
    keys_written: int
    seconds: dict[str, float]
    warnings: list[dict[str, str]] = field(default_factory=list)
    recipe: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == "SUCCEEDED"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = GENERATE_SCHEMA_VERSION
        return payload


__all__ = ["GENERATE_SCHEMA_VERSION", "GeneratePlan", "GenerateResult"]
