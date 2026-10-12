"""JSON-safe frame codec and chunked npz files (no pickle).

A chunk stores the UTF-8 JSON of each frame back to back in one ``uint8``
array plus an ``offsets`` array, inside a compressed npz that is always read
with ``allow_pickle=False``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np

from ...models import (
    CameraIntrinsicsObservation,
    CanonicalFrame,
    JointFrameObservation,
    Landmark,
    Mat3,
    RootMotionObservation,
)
from ..fs import atomic_npz


def compact_frame(frame: CanonicalFrame) -> CanonicalFrame:
    """Copy a frame without the dense 3D face mesh (kept as image overlay)."""
    return CanonicalFrame(
        timestamp_ms=int(frame.timestamp_ms),
        body=dict(frame.body),
        hands={side: dict(points) for side, points in frame.hands.items()},
        auxiliary=dict(frame.auxiliary),
        joint_frames=dict(frame.joint_frames),
        face_landmarks=[],
        face_overlay=list(
            frame.face_overlay or [point.image for point in frame.face_landmarks]
        ),
        blendshapes=dict(frame.blendshapes),
        camera=frame.camera,
        root_motion=frame.root_motion,
        warnings=list(frame.warnings),
    )


def landmark_to_plain(value: Landmark) -> tuple:
    return (
        tuple(float(item) for item in value.position),
        tuple(float(item) for item in value.image),
        float(value.confidence),
        bool(value.valid),
    )


def landmark_from_plain(value) -> Landmark:
    return Landmark(
        position=tuple(value[0]),
        image=tuple(value[1]),
        confidence=float(value[2]),
        valid=bool(value[3]),
    )


def _camera_to_plain(camera: CameraIntrinsicsObservation | None) -> dict | None:
    if camera is None:
        return None
    return {
        "width": camera.width,
        "height": camera.height,
        "fx": camera.fx,
        "fy": camera.fy,
        "cx": camera.cx,
        "cy": camera.cy,
        "horizontal_fov": camera.horizontal_fov,
        "vertical_fov": camera.vertical_fov,
        "confidence": camera.confidence,
        "source": camera.source,
        "bundle_revision": camera.bundle_revision,
        "precision": camera.precision,
        "valid": camera.valid,
    }


def _root_to_plain(root: RootMotionObservation | None) -> dict | None:
    if root is None:
        return None
    return {
        "translation": list(root.translation),
        "rotation_matrix": (
            list(root.rotation_matrix) if root.rotation_matrix is not None else None
        ),
        "focal_length": root.focal_length,
        "scale": root.scale,
        "confidence": root.confidence,
        "valid": root.valid,
    }


def frame_to_plain(frame: CanonicalFrame, *, compact: bool = True) -> dict:
    """Return JSON-safe data; ``compact`` drops the 3D face mesh."""
    source = compact_frame(frame) if compact else frame
    plain = {
        "timestamp_ms": int(source.timestamp_ms),
        "body": {name: landmark_to_plain(value) for name, value in source.body.items()},
        "hands": {
            side: {name: landmark_to_plain(value) for name, value in points.items()}
            for side, points in source.hands.items()
        },
        "auxiliary": {
            name: landmark_to_plain(value) for name, value in source.auxiliary.items()
        },
        "joint_frames": {
            name: (
                tuple(float(item) for item in value.rotation_matrix),
                float(value.confidence),
                bool(value.valid),
            )
            for name, value in source.joint_frames.items()
        },
        "face_overlay": [tuple(point) for point in source.face_overlay],
        "blendshapes": dict(source.blendshapes),
        "camera": _camera_to_plain(source.camera),
        "root_motion": _root_to_plain(source.root_motion),
        "warnings": list(source.warnings),
    }
    if not compact and source.face_landmarks:
        plain["face_landmarks"] = [
            landmark_to_plain(value) for value in source.face_landmarks
        ]
    return plain


def _camera_from_plain(value: dict | None) -> CameraIntrinsicsObservation | None:
    if value is None:
        return None
    return CameraIntrinsicsObservation(
        width=int(value["width"]),
        height=int(value["height"]),
        fx=float(value["fx"]),
        fy=float(value["fy"]),
        cx=float(value["cx"]),
        cy=float(value["cy"]),
        horizontal_fov=float(value["horizontal_fov"]),
        vertical_fov=float(value["vertical_fov"]),
        confidence=float(value.get("confidence", 1.0)),
        source=str(value.get("source", "FALLBACK")),
        bundle_revision=str(value.get("bundle_revision", "")),
        precision=str(value.get("precision", "")),
        valid=bool(value.get("valid", True)),
    )


def _root_from_plain(value: dict | None) -> RootMotionObservation | None:
    if value is None:
        return None
    x, y, z = (float(item) for item in value["translation"])
    rotation = value.get("rotation_matrix")
    return RootMotionObservation(
        translation=(x, y, z),
        rotation_matrix=(
            cast(Mat3, tuple(float(item) for item in rotation))
            if rotation is not None
            else None
        ),
        focal_length=float(value.get("focal_length", 0.0)),
        scale=float(value.get("scale", 1.0)),
        confidence=float(value.get("confidence", 1.0)),
        valid=bool(value.get("valid", True)),
    )


def frame_from_plain(value: dict) -> CanonicalFrame:
    return CanonicalFrame(
        timestamp_ms=int(value["timestamp_ms"]),
        body={
            name: landmark_from_plain(landmark)
            for name, landmark in value.get("body", {}).items()
        },
        hands={
            side: {
                name: landmark_from_plain(landmark) for name, landmark in points.items()
            }
            for side, points in value.get("hands", {}).items()
        },
        auxiliary={
            name: landmark_from_plain(landmark)
            for name, landmark in value.get("auxiliary", {}).items()
        },
        joint_frames={
            name: JointFrameObservation(
                rotation_matrix=tuple(frame_value[0]),
                confidence=float(frame_value[1]),
                valid=bool(frame_value[2]),
            )
            for name, frame_value in value.get("joint_frames", {}).items()
        },
        face_landmarks=[
            landmark_from_plain(landmark)
            for landmark in value.get("face_landmarks", [])
        ],
        face_overlay=[tuple(point) for point in value.get("face_overlay", [])],
        blendshapes={
            name: float(score) for name, score in value.get("blendshapes", {}).items()
        },
        camera=_camera_from_plain(value.get("camera")),
        root_motion=_root_from_plain(value.get("root_motion")),
        warnings=list(value.get("warnings", [])),
    )


def encode_frames(
    frames: list[CanonicalFrame], *, compact: bool = True
) -> dict[str, np.ndarray]:
    """Return the ``payload`` and ``offsets`` arrays for one chunk."""
    blobs = [
        json.dumps(
            frame_to_plain(frame, compact=compact), separators=(",", ":")
        ).encode("utf8")
        for frame in frames
    ]
    offsets = [0]
    for blob in blobs:
        offsets.append(offsets[-1] + len(blob))
    return {
        "payload": np.frombuffer(b"".join(blobs), dtype=np.uint8),
        "offsets": np.asarray(offsets, dtype=np.int64),
    }


def decode_frames(payload: np.ndarray, offsets: np.ndarray) -> list[CanonicalFrame]:
    raw = memoryview(np.ascontiguousarray(payload))
    return [
        frame_from_plain(
            json.loads(
                bytes(raw[int(offsets[index]) : int(offsets[index + 1])]).decode("utf8")
            )
        )
        for index in range(len(offsets) - 1)
    ]


def write_chunk(
    path: Path, frames: list[CanonicalFrame], *, compact: bool = True
) -> None:
    atomic_npz(path, encode_frames(frames, compact=compact))


def read_chunk(path: Path) -> list[CanonicalFrame]:
    with np.load(str(path), allow_pickle=False) as archive:
        return decode_frames(archive["payload"], archive["offsets"])


__all__ = [
    "compact_frame",
    "decode_frames",
    "encode_frames",
    "frame_from_plain",
    "frame_to_plain",
    "landmark_from_plain",
    "landmark_to_plain",
    "read_chunk",
    "write_chunk",
]
