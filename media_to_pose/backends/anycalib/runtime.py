"""Small ONNX Runtime adapter for the published AnyCalib neural field."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ...core.camera import calibration as camera_calibration
from ..sam3d import ort_runtime as sam3d_ort_runtime

BUNDLE_CONTRACT = "media_to_pose.anycalib_onnx.v1"
SUPPORTED_BUNDLE_CONTRACTS = frozenset(
    {
        BUNDLE_CONTRACT,
        "vision_to_pose.anycalib_onnx.v1",
    }
)
RUNNER = "onnxruntime.anycalib_tangent.v1"
UPSTREAM_REVISION = "cc0399c4f563ceffb3edcb02342febf150c53893"
CHECKPOINT_REVISION = "660ab9b7924ccca12d1e353585594e34f1251341"


@dataclass(frozen=True)
class Asset:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class Bundle:
    root: Path
    precision: str
    graph: Asset
    download_bytes: int
    upstream_revision: str
    checkpoint_revision: str


@dataclass(frozen=True)
class Diagnostics:
    requested_mode: str
    actual_ep: str
    actual_device: str
    fallback: bool
    precision: str
    download_bytes: int
    installed_bytes: int
    upstream_revision: str


def _safe_path(root: Path, relative: object) -> Path:
    value = PurePosixPath(str(relative))
    if value.is_absolute() or ".." in value.parts or "\\" in str(relative):
        raise RuntimeError(f"Unsafe AnyCalib asset path: {relative!r}")
    resolved = root.joinpath(*value.parts).resolve(False)
    if root.resolve(False) not in resolved.parents:
        raise RuntimeError(f"AnyCalib asset escapes its bundle: {relative!r}")
    return resolved


def _asset(value: Mapping[str, object]) -> Asset:
    result = Asset(
        path=str(value.get("path") or ""),
        bytes=int(value.get("bytes") or 0),
        sha256=str(value.get("sha256") or "").casefold(),
    )
    if not result.path or result.bytes <= 0 or len(result.sha256) != 64:
        raise RuntimeError("AnyCalib manifest contains an invalid graph declaration")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bundle_manifest(
    root: str | Path,
    *,
    expected_upstream_revision: str = "",
) -> Bundle:
    """Parse and validate bundle metadata without requiring downloaded assets."""
    directory = Path(root)
    manifest = directory / "manifest.json"
    try:
        value = json.loads(manifest.read_text(encoding="utf8"))
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"Invalid AnyCalib manifest: {exc}") from exc
    if (
        value.get("contract") not in SUPPORTED_BUNDLE_CONTRACTS
        or value.get("runner") != RUNNER
    ):
        raise RuntimeError("Unsupported AnyCalib bundle contract")
    upstream = str(value.get("upstream_revision") or "")
    checkpoint = str(value.get("checkpoint_revision") or "")
    if not sam3d_ort_runtime.is_immutable_revision(upstream):
        raise RuntimeError("AnyCalib upstream revision is missing or invalid")
    if expected_upstream_revision and upstream != str(expected_upstream_revision):
        raise RuntimeError("AnyCalib upstream revision does not match the plugin pin")
    if not sam3d_ort_runtime.is_immutable_revision(checkpoint):
        raise RuntimeError("AnyCalib checkpoint revision is missing or invalid")
    precision = str(value.get("precision") or "").casefold()
    if precision not in {"fp16", "fp32"}:
        raise RuntimeError("Unsupported AnyCalib precision")
    graph = _asset(value.get("graph") or {})
    _safe_path(directory, graph.path)
    return Bundle(
        root=directory,
        precision=precision,
        graph=graph,
        download_bytes=graph.bytes,
        upstream_revision=upstream,
        checkpoint_revision=checkpoint,
    )


def load_bundle(
    root: str | Path,
    *,
    verify_files: bool = True,
    expected_upstream_revision: str = "",
) -> Bundle:
    """Load an installed bundle, requiring its graph even for a fast check."""
    bundle = load_bundle_manifest(
        root,
        expected_upstream_revision=expected_upstream_revision,
    )
    graph_path = _safe_path(bundle.root, bundle.graph.path)
    if not graph_path.is_file():
        raise RuntimeError(f"AnyCalib graph is missing: {bundle.graph.path}")
    if verify_files:
        if graph_path.stat().st_size != bundle.graph.bytes:
            raise RuntimeError("AnyCalib graph byte size does not match its manifest")
        if _sha256(graph_path) != bundle.graph.sha256:
            raise RuntimeError("AnyCalib graph failed SHA-256 verification")
    return bundle


def target_size(height: int, width: int) -> tuple[int, int]:
    aspect = max(0.5, min(float(height) / max(float(width), 1.0), 2.0))
    target_width = math.sqrt(102_400.0 / aspect)
    target_height = aspect * target_width
    return round(target_height / 14.0) * 14, round(target_width / 14.0) * 14


def _cubic_kernel(distance, np):
    absolute = np.abs(distance)
    first = 1.25 * absolute**3 - 2.25 * absolute**2 + 1.0
    second = -0.75 * absolute**3 + 3.75 * absolute**2 - 6.0 * absolute + 3.0
    return np.where(absolute <= 1.0, first, np.where(absolute < 2.0, second, 0.0))


def _resize_axis(image, output_size: int, axis: int, np, *, antialias: bool):
    input_size = int(image.shape[axis])
    output_size = int(output_size)
    if input_size == output_size:
        return image
    scale = float(output_size) / float(input_size)
    kernel_scale = scale if antialias and scale < 1.0 else 1.0
    support = 2.0 / kernel_scale
    taps = int(math.ceil(2.0 * support)) + 2
    centers = (np.arange(output_size, dtype=np.float64) + 0.5) / scale - 0.5
    starts = np.floor(centers - support).astype(np.int64)
    indexes = starts[:, None] + np.arange(taps, dtype=np.int64)[None]
    weights = kernel_scale * _cubic_kernel(
        (indexes.astype(np.float64) - centers[:, None]) * kernel_scale, np
    )
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0e-15)
    clipped = np.clip(indexes, 0, input_size - 1)
    if axis == 1:
        sampled = image[:, clipped, :]
        return np.sum(
            sampled * weights[None, :, :, None], axis=2, dtype=np.float64
        ).astype(np.float32)
    sampled = image[clipped, :, :]
    return np.sum(sampled * weights[:, :, None, None], axis=1, dtype=np.float64).astype(
        np.float32
    )


def _resize_bicubic(
    image, output_height: int, output_width: int, np, *, antialias: bool
):
    # PyTorch applies separable horizontal and vertical cubic filters.
    resized = _resize_axis(image, output_width, 1, np, antialias=antialias)
    return _resize_axis(resized, output_height, 0, np, antialias=antialias)


def prepare_image(image_rgb, cv2, np):
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"AnyCalib input must be HxWx3, got {image.shape}")
    original_height, original_width = image.shape[:2]
    target_height, target_width = target_size(original_height, original_width)
    working = image.astype(np.float32) / np.float32(255.0)
    height, width = original_height, original_width
    scale_1_x = scale_1_y = 1.0
    if height < target_height or width < target_width:
        scale = max(target_height / height, target_width / width)
        resized_width = max(target_width, int(math.floor(width * scale)))
        resized_height = max(target_height, int(math.floor(height * scale)))
        working = _resize_bicubic(
            working,
            resized_height,
            resized_width,
            np,
            antialias=False,
        )
        working = np.clip(working, 0.0, 1.0)
        scale_1_x = resized_width / width
        scale_1_y = resized_height / height
        height, width = resized_height, resized_width
    shift_x = shift_y = 0
    target_aspect = target_width / target_height
    if width / height > target_aspect:
        crop = round(width - height * target_aspect)
        left = crop // 2
        working = working[:, left : width - crop + left]
        shift_x = -left
    else:
        crop = round(height - width / target_aspect)
        top = crop // 2
        working = working[top : height - crop + top, :]
        shift_y = -top
    cropped_height, cropped_width = working.shape[:2]
    working = _resize_bicubic(
        working,
        target_height,
        target_width,
        np,
        antialias=True,
    )
    working = np.clip(working, 0.0, 1.0)
    scale_2_x = target_width / cropped_width
    scale_2_y = target_height / cropped_height
    scale_xy = (scale_1_x * scale_2_x, scale_1_y * scale_2_y)
    shift_xy = (shift_x * scale_2_x, shift_y * scale_2_y)
    tensor = np.ascontiguousarray(working.transpose(2, 0, 1)[None], dtype=np.float32)
    return tensor, scale_xy, shift_xy


class AnyCalibSession:
    def __init__(self, bundle: Bundle, requested_mode: object = "AUTO"):
        self.bundle = bundle
        self.requested_mode = sam3d_ort_runtime.normalize_mode(requested_mode)
        ort = sam3d_ort_runtime.import_onnxruntime()
        available = sam3d_ort_runtime.available_execution_providers(ort)
        provider, device, fallback = sam3d_ort_runtime.select_execution_provider(
            self.requested_mode, available
        )
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        graph_path = _safe_path(bundle.root, bundle.graph.path)
        self.session = ort.InferenceSession(
            str(graph_path), sess_options=options, providers=[provider]
        )
        self.input_name = self.session.get_inputs()[0].name
        outputs = self.session.get_outputs()
        if len(outputs) != 1:
            raise RuntimeError("AnyCalib graph must expose one tangent-field output")
        self.output_name = outputs[0].name
        self.diagnostics = Diagnostics(
            requested_mode=self.requested_mode,
            actual_ep=provider,
            actual_device=device,
            fallback=fallback,
            precision=bundle.precision,
            download_bytes=bundle.download_bytes,
            installed_bytes=bundle.graph.bytes,
            upstream_revision=bundle.upstream_revision,
        )

    def calibrate(self, image_rgb, cv2, np, *, bundle_revision: str = ""):
        height, width = image_rgb.shape[:2]
        tensor, scale_xy, shift_xy = prepare_image(image_rgb, cv2, np)
        tangent = self.session.run([self.output_name], {self.input_name: tensor})[0]
        return camera_calibration.fit_tangent_field(
            tangent,
            original_width=width,
            original_height=height,
            scale_xy=scale_xy,
            shift_xy=shift_xy,
            np=np,
            bundle_revision=bundle_revision,
            precision=self.bundle.precision,
        )

    def close(self) -> None:
        self.session = None


__all__ = [
    "AnyCalibSession",
    "BUNDLE_CONTRACT",
    "CHECKPOINT_REVISION",
    "RUNNER",
    "UPSTREAM_REVISION",
    "load_bundle",
    "prepare_image",
    "target_size",
]
