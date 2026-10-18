"""The structured result of one capture, as returned by the API and CLI.

``CaptureResult.to_dict`` is the JSON document agents read; its layout is
versioned by ``RESULT_SCHEMA_VERSION``. Frame numbers named ``source_*``
are 0-based frames of the media; ``timeline`` frames are scene frames.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.fs import atomic_json
from ..core.report import CaptureReport, frame_ranges
from ..core.requests import FaceOutput
from ..core.types import CanonicalFrame

RESULT_SCHEMA_VERSION = 1

# Stable codes for the free-text warnings the pipeline produces, matched by
# message prefix. Agents should branch on the code, not on the message.
WARNING_CODES: tuple[tuple[str, str], ...] = (
    ("Detection Filter rejected", "DETECTIONS_REJECTED"),
    ("Detection Filter locally repaired", "GAPS_FILLED"),
    ("Manual cleanup mask", "MANUAL_MASK"),
    ("Mostly reconstructed", "LOW_EVIDENCE"),
    ("Ground Contacts", "GROUND_CONTACTS"),
    ("Experimental RoHM Correction", "ROHM_APPLIED"),
    ("RoHM", "ROHM"),
    ("Camera", "CAMERA_MOTION"),
    ("Root motion", "ROOT_MOTION"),
    ("Face", "FACE_OUTPUT"),
)


def warning_code(message: str) -> str:
    text = str(message).strip()
    for prefix, code in WARNING_CODES:
        if text.startswith(prefix):
            return code
    return "CAPTURE_WARNING"


def _mean_confidence(landmarks) -> float:
    values = [
        float(landmark.confidence)
        for landmark in landmarks
        if landmark is not None and landmark.valid
    ]
    return sum(values) / len(values) if values else 0.0


def part_confidence(frame: CanonicalFrame) -> dict[str, float]:
    """Mean landmark confidence per captured part, 0.0 when undetected."""
    hands = [_mean_confidence(side.values()) for side in frame.hands.values() if side]
    face = _mean_confidence(frame.face_landmarks)
    if not face and frame.blendshapes:
        face = 1.0
    return {
        "body": round(_mean_confidence(frame.body.values()), 4),
        "hands": round(sum(hands) / len(hands), 4) if hands else 0.0,
        "face": round(face, 4),
    }


class FaceSummary:
    """Per capture: the strength each mapped expression was sent, and what
    had no target.

    Strengths are the detections through the mapping's input range, as a
    share of its output range (0 to 1, -1 to 1 for signed controls); for
    video, Face Expression Cleanup changes them further before they are
    keyed, and ``evaluate`` measures what the rig shows.
    """

    def __init__(self, entries, face_output: FaceOutput):
        self.entries = entries
        self.face_output = FaceOutput(face_output)
        self.frames = 0
        self.peak: dict[str, float] = {}
        self.total: dict[str, float] = {}
        self.unmapped: set[str] = set()

    @classmethod
    def for_request(cls, request) -> FaceSummary | None:
        if not request.use_face or request.provider_id != "MEDIAPIPE":
            return None
        return cls(request.mapping_entries, request.face_output)

    def observe(self, frame: CanonicalFrame) -> None:
        if not frame.blendshapes:
            return
        # Imported here: the solver needs Blender, this module does not.
        from ..retarget.solve.face_values import mapped_expressions

        self.frames += 1
        applied, unmapped = mapped_expressions(
            self.entries, dict(frame.blendshapes), self.face_output
        )
        for name, value in applied.items():
            # Signed controls peak on whichever side they reach furthest.
            if abs(value) > abs(self.peak.get(name, 0.0)) or name not in self.peak:
                self.peak[name] = value
            self.total[name] = self.total.get(name, 0.0) + value
        self.unmapped.update(unmapped)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": str(self.face_output),
            "frames_with_face": self.frames,
            "expressions": {
                name: {
                    "peak": round(self.peak[name], 4),
                    "mean": round(self.total[name] / self.frames, 4),
                }
                for name in sorted(self.peak)
            },
            "unmapped_expressions": sorted(self.unmapped),
        }


@dataclass
class CaptureResult:
    """What a capture produced; see the module docstring for frame numbers."""

    status: str
    message: str
    mode: str
    provider: str
    media: dict[str, Any]
    armature: str
    action: str = ""
    slot: str = ""
    output_mode: str = "NEW_ACTION"
    nla_track: str = ""
    source_frame_range: list[int] | None = None
    timeline_frame_range: list[float] | None = None
    frames_processed: int = 0
    cached: bool = False
    cache_key: str = ""
    confidence: dict[str, Any] = field(default_factory=dict)
    rejected_frames: list[list[int]] = field(default_factory=list)
    filled_frames: list[list[int]] = field(default_factory=list)
    contact_frames: list[list[int]] = field(default_factory=list)
    cleanup_steps: list[dict[str, Any]] = field(default_factory=list)
    mapping_issues: list[dict[str, Any]] = field(default_factory=list)
    face_output_warning: str = ""
    face: dict[str, Any] = field(default_factory=dict)
    warnings: list[dict[str, str]] = field(default_factory=list)
    schema_version: int = RESULT_SCHEMA_VERSION

    @property
    def succeeded(self) -> bool:
        return self.status == "SUCCEEDED"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        atomic_json(destination, self.to_dict(), indent=2, sort_keys=False)
        return destination


def _ranges(frames) -> list[list[int]]:
    return [[first, last] for first, last in frame_ranges(frames)]


def cleanup_steps(request, report: CaptureReport) -> list[dict[str, Any]]:
    """Every enabled clean-up module with its settings and statistics."""
    config = request.postprocess_config
    steps = []
    for module in config.modules if config is not None else ():
        if not module.enabled:
            continue
        recorded = report.steps.get(module.module_id)
        step = {
            "module_id": module.module_id,
            "display_name": module.display_name,
            "parameters": dict(module.values),
            "statistics": {},
        }
        if recorded is not None:
            data = recorded.to_dict()
            step["parameters"].update(data["parameters"])
            step["statistics"] = data["statistics"]
        steps.append(step)
    return steps


def mapping_issues(armature) -> list[dict[str, Any]]:
    if armature is None:
        return []
    return [
        {
            "semantic_id": entry.semantic_id,
            "section": entry.section,
            "status": entry.status,
            "required": bool(entry.required),
            "message": entry.message,
        }
        for entry in armature.mtp_rig_config.mappings
        if entry.enabled and entry.status in {"WARNING", "ERROR", "MISSING"}
    ]


def _action_and_slot(armature, output_mode: str, action_name: str):
    animation_data = getattr(armature, "animation_data", None)
    if animation_data is None:
        return action_name, ""
    if output_mode == "NLA_STRIP":
        for track in reversed(tuple(animation_data.nla_tracks)):
            for strip in track.strips:
                if strip.action is not None and strip.action.name == action_name:
                    slot = getattr(strip, "action_slot", None)
                    return action_name, getattr(slot, "identifier", "") or ""
        return action_name, ""
    action = animation_data.action
    slot = getattr(animation_data, "action_slot", None)
    return (
        action.name if action is not None else action_name,
        getattr(slot, "identifier", "") or "",
    )


def build_capture_result(session, *, armature=None) -> CaptureResult:
    """Collect the result of a finished ``CaptureSession``."""
    request = session.request
    outcome = session.outcome
    report = session.report
    media = request.media_info
    action, slot = ("", "")
    if armature is not None and session.state == "SUCCEEDED":
        action, slot = _action_and_slot(
            armature, str(request.output_mode), outcome.action_name
        )
    frames = session.frame_confidence
    per_part = {}
    for part in ("body", "hands", "face"):
        values = [item[part] for item in frames]
        per_part[part] = round(sum(values) / len(values), 4) if values else 0.0
    timeline = session.timeline_frames
    transaction = session.transaction
    return CaptureResult(
        status=str(session.state),
        message=session.status,
        mode=str(request.mode),
        provider=request.provider_id,
        media={
            "path": request.media_path,
            "kind": str(request.media_kind),
            "fps": float(getattr(media, "fps", 0.0) or 0.0),
            "frame_count": int(getattr(media, "frame_count", 0) or 0),
            "width": int(getattr(media, "width", 0) or 0),
            "height": int(getattr(media, "height", 0) or 0),
            "mirror": bool(request.options.get("mirror", False)),
        },
        armature=request.target_name,
        action=action,
        slot=slot,
        output_mode=str(request.output_mode),
        nla_track=transaction.nla_track_name if transaction is not None else "",
        source_frame_range=(
            [frames[0]["source_frame"], frames[-1]["source_frame"] + 1]
            if frames
            else None
        ),
        timeline_frame_range=list(timeline) if timeline else None,
        frames_processed=int(outcome.frames),
        cached=bool(outcome.cached),
        cache_key=request.cache_key if request.keep_detection_cache else "",
        confidence={"per_part": per_part, "per_frame": frames},
        rejected_frames=_ranges(report.rejected_frames),
        filled_frames=_ranges(report.filled_frames),
        contact_frames=_ranges(report.contact_frames),
        cleanup_steps=cleanup_steps(request, report),
        mapping_issues=mapping_issues(armature),
        face_output_warning=next(
            (message for kind, message in session.notices if kind == "WARNING"), ""
        ),
        face=session.face_summary.to_dict() if session.face_summary else {},
        warnings=[
            {"code": warning_code(message), "message": message}
            for message in outcome.warnings
        ],
    )


__all__ = [
    "RESULT_SCHEMA_VERSION",
    "CaptureResult",
    "FaceSummary",
    "build_capture_result",
    "cleanup_steps",
    "mapping_issues",
    "part_confidence",
    "warning_code",
]
