"""Export the retained detections for inspection or editing, and import them.

``export_detections(path)`` writes two files next to each other:

- ``<path>.json``: every frame with its body, hand and auxiliary landmarks,
  blendshapes, camera and root observations, in the detection cache's
  JSON-safe format. A landmark is ``[[x, y, z], [u, v], confidence, valid]``
  with ``x, y, z`` in the provider's metric space and ``u, v`` in 0..1 image
  coordinates.
- ``<path>.npz``: the same body and hand landmarks as dense arrays
  (``body_position`` ``[T, J, 3]``, ``body_image``, ``body_confidence``,
  ``body_valid`` and ``hand_<SIDE>_*`` alike, plus ``*_names``,
  ``timestamps_ms`` and ``source_frames``) for numeric analysis.

``import_detections(path)`` reads the JSON back, applies the npz arrays on
top when the npz exists, and stores the frames as the retained detections
under the same cache key, so ``retarget_cached`` with the same media and
settings re-applies the edited detections.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..core.cache import store as cache_store
from ..core.cache.frames import frame_from_plain, frame_to_plain
from ..core.errors import InvalidRequest, MediaError
from ..core.fs import atomic_json, atomic_npz
from ..core.types import CanonicalFrame, Landmark, MediaInfo
from .common import output_path

DETECTIONS_SCHEMA_VERSION = 1


def _paths(path) -> tuple[Path, Path]:
    base = output_path(path)
    if base is None:
        raise InvalidRequest("Give a file path", parameter="path")
    if base.suffix.lower() in {".json", ".npz"}:
        base = base.with_suffix("")
    return base.with_suffix(".json"), base.with_suffix(".npz")


def _retained_frames(manifest: dict) -> list[tuple[int, CanonicalFrame]]:
    key = str(manifest.get("key", ""))
    if manifest.get("mode") == "SEQUENCE":
        frames = cache_store.iter_frames(key) or ()
        return list(enumerate(frames))
    result = []
    for index in sorted(int(value) for value in manifest.get("points", {})):
        frame = cache_store.load_frame(key, index)
        if frame is not None:
            result.append((index, frame))
    return result


def _dense(frames: list[tuple[int, CanonicalFrame]], groups) -> dict[str, Any]:
    """Arrays for each landmark group: ``{prefix: lambda frame: landmarks}``."""
    arrays: dict[str, Any] = {
        "timestamps_ms": np.asarray([f.timestamp_ms for _i, f in frames], np.int64),
        "source_frames": np.asarray([index for index, _f in frames], np.int64),
    }
    for prefix, select in groups.items():
        names = sorted({name for _index, frame in frames for name in select(frame)})
        if not names:
            continue
        count = len(frames)
        position = np.zeros((count, len(names), 3), np.float32)
        image = np.zeros((count, len(names), 2), np.float32)
        confidence = np.zeros((count, len(names)), np.float32)
        valid = np.zeros((count, len(names)), bool)
        for row, (_index, frame) in enumerate(frames):
            points = select(frame)
            for column, name in enumerate(names):
                landmark = points.get(name)
                if landmark is None:
                    continue
                position[row, column] = landmark.position
                image[row, column] = landmark.image
                confidence[row, column] = landmark.confidence
                valid[row, column] = landmark.valid
        arrays[f"{prefix}_names"] = np.asarray(names)
        arrays[f"{prefix}_position"] = position
        arrays[f"{prefix}_image"] = image
        arrays[f"{prefix}_confidence"] = confidence
        arrays[f"{prefix}_valid"] = valid
    return arrays


def _groups(frames) -> dict:
    sides = sorted({side for _index, frame in frames for side in frame.hands})
    groups = {"body": lambda frame: frame.body}
    for side in sides:
        groups[f"hand_{side}"] = lambda frame, side=side: frame.hands.get(side, {})
    return groups


def export_detections(path) -> dict[str, Any]:
    """Write the retained detections; returns the paths and a summary."""
    manifest = cache_store.current_manifest()
    if not manifest:
        raise MediaError("There are no retained detections to export")
    frames = _retained_frames(manifest)
    if not frames:
        raise MediaError("The retained detections contain no frames")
    json_path, npz_path = _paths(path)
    document = {
        "schema_version": DETECTIONS_SCHEMA_VERSION,
        "cache_key": manifest["key"],
        "identity": manifest.get("identity", {}),
        "media": manifest.get("media_info", {}),
        "mode": manifest.get("mode", "SEQUENCE"),
        "frames": [
            {"source_frame": index, **frame_to_plain(frame)} for index, frame in frames
        ],
    }
    atomic_json(json_path, document, indent=1, sort_keys=False)
    atomic_npz(npz_path, _dense(frames, _groups(frames)))
    return {
        "json": str(json_path),
        "npz": str(npz_path),
        "frames": len(frames),
        "cache_key": manifest["key"],
    }


def _apply_dense(frames: list[CanonicalFrame], npz_path: Path) -> None:
    with np.load(npz_path, allow_pickle=False) as data:
        if len(data["timestamps_ms"]) != len(frames):
            raise InvalidRequest(
                f"{npz_path.name} has {len(data['timestamps_ms'])} frames; "
                f"the JSON has {len(frames)}",
                parameter="path",
            )
        for key in data.files:
            if not key.endswith("_names"):
                continue
            prefix = key[: -len("_names")]
            names = [str(name) for name in data[key]]
            position = data[f"{prefix}_position"]
            image = data[f"{prefix}_image"]
            confidence = data[f"{prefix}_confidence"]
            valid = data[f"{prefix}_valid"]
            for row, frame in enumerate(frames):
                if prefix == "body":
                    points = frame.body
                else:
                    points = frame.hands.setdefault(prefix[len("hand_") :], {})
                for column, name in enumerate(names):
                    x, y, z = (float(v) for v in position[row, column])
                    u, v = (float(item) for item in image[row, column])
                    points[name] = Landmark(
                        position=(x, y, z),
                        image=(u, v),
                        confidence=float(confidence[row, column]),
                        valid=bool(valid[row, column]),
                    )


def import_detections(path) -> dict[str, Any]:
    """Store edited detections as the retained detections.

    ``path`` names the ``.json`` export (or the shared base name); an
    ``.npz`` beside it overrides the body and hand landmarks.
    """
    json_path, npz_path = _paths(path)
    try:
        document = json.loads(json_path.read_text(encoding="utf8"))
    except (OSError, ValueError) as exc:
        raise InvalidRequest(f"Could not read {json_path}: {exc}") from exc
    if document.get("schema_version") != DETECTIONS_SCHEMA_VERSION:
        raise InvalidRequest(
            f"Unsupported detections schema {document.get('schema_version')!r}",
            parameter="path",
        )
    try:
        entries = sorted(document["frames"], key=lambda item: item["source_frame"])
        frames = [frame_from_plain(item) for item in entries]
        key = str(document["cache_key"])
        identity = dict(document["identity"])
        media = MediaInfo(**document["media"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidRequest(f"Malformed detections file: {exc}") from exc
    if npz_path.is_file():
        _apply_dense(frames, npz_path)
    if document.get("mode") == "POINTS":
        for item, frame in zip(entries, frames, strict=True):
            cache_store.store_point(key, identity, media, item["source_frame"], frame)
        persisted = True
    else:
        writer = cache_store.SequenceWriter(key, identity, media, 0)
        try:
            for frame in frames:
                writer.add(frame)
            _directory, persisted = writer.finish()
        except BaseException:
            writer.discard()
            raise
    return {"frames": len(frames), "cache_key": key, "retained": bool(persisted)}


__all__ = ["DETECTIONS_SCHEMA_VERSION", "export_detections", "import_detections"]
