"""Mesh-free ONNX Runtime execution for an optional immutable RoHM bundle."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from ...core.errors import BackendError

MODEL_FPS = 30.0
SOURCE_WINDOW_FRAMES = 145
MODEL_WINDOW_FRAMES = 144
FEATURE_COUNT = 294
TRAJECTORY_COUNT = 22
POSE_COUNT = 272
TRAJECTORY_ABSOLUTE_INDICES = np.asarray(
    (0, 2, 3, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18), dtype=np.int64
)
POSE_START = 22
BETAS_SLICE = slice(280, 290)
FOOT_CONTACT_SLICE = slice(290, 294)
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".pkl", ".pickle"}


class RohmRuntimeError(BackendError):
    """A RoHM bundle contract or inference failure."""


@dataclass(frozen=True)
class RuntimeDiagnostics:
    requested_mode: str
    actual_ep: str
    actual_device: str
    fallback: bool
    precision: str
    download_bytes: int
    installed_bytes: int
    upstream_revision: str


@dataclass(frozen=True)
class Asset:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class Bundle:
    root: Path
    assets: dict[str, Asset]
    precision: str
    download_bytes: int
    upstream_revision: str

    def resolve(self, relative: str) -> Path:
        try:
            asset = self.assets[relative]
        except KeyError as exc:
            raise RohmRuntimeError(f"RoHM manifest has no asset {relative}") from exc
        return _resolved_member(self.root, asset.path)


def _resolved_member(root: Path, relative: str) -> Path:
    raw = str(relative)
    path = PurePosixPath(raw)
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in raw
        or "\x00" in raw
        or any(":" in part for part in path.parts)
    ):
        raise RohmRuntimeError(f"unsafe RoHM bundle path: {relative!r}")
    resolved_root = root.resolve(strict=False)
    resolved = resolved_root.joinpath(*path.parts).resolve(strict=False)
    if resolved_root not in resolved.parents:
        raise RohmRuntimeError(f"RoHM bundle path escapes its root: {relative!r}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bundle(
    root: Path,
    *,
    verify_files: bool = True,
    expected_upstream_revision: str = "",
) -> Bundle:
    root = root.resolve(strict=True)
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise RohmRuntimeError(f"could not read RoHM manifest: {exc}") from exc
    if manifest.get("schema") != 1:
        raise RohmRuntimeError("unsupported RoHM manifest schema")
    upstream_revision = str(manifest.get("upstream_revision", ""))
    if expected_upstream_revision and upstream_revision != expected_upstream_revision:
        raise RohmRuntimeError("RoHM upstream revision does not match this release")
    raw_assets = manifest.get("assets")
    if not isinstance(raw_assets, dict):
        raise RohmRuntimeError("RoHM manifest assets must be an object")
    assets = {}
    installed_bytes = 0
    for relative, value in raw_assets.items():
        if not isinstance(value, dict):
            raise RohmRuntimeError(f"invalid RoHM asset entry {relative}")
        suffix = PurePosixPath(relative).suffix.casefold()
        if suffix in FORBIDDEN_SUFFIXES:
            raise RohmRuntimeError(f"unsafe RoHM asset type: {relative}")
        asset = Asset(
            path=str(relative),
            bytes=int(value.get("bytes", -1)),
            sha256=str(value.get("sha256", "")).casefold(),
        )
        if asset.bytes < 0 or len(asset.sha256) != 64:
            raise RohmRuntimeError(f"invalid RoHM asset metadata: {relative}")
        if verify_files:
            path = _resolved_member(root, asset.path)
            if not path.is_file() or path.stat().st_size != asset.bytes:
                raise RohmRuntimeError(f"RoHM asset size mismatch: {relative}")
            if _sha256(path) != asset.sha256:
                raise RohmRuntimeError(f"RoHM asset checksum mismatch: {relative}")
        installed_bytes += asset.bytes
        assets[str(relative)] = asset
    if verify_files:
        expected_files = {"manifest.json", *assets}
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if path.suffix.casefold() in FORBIDDEN_SUFFIXES:
                raise RohmRuntimeError(f"unsafe file in RoHM bundle: {relative}")
            if relative not in expected_files:
                raise RohmRuntimeError(f"unmanifested file in RoHM bundle: {relative}")
    required = {
        "graphs/rohm_posenet_fp32.onnx",
        "graphs/rohm_trajnet_control_fp32.onnx",
        "metadata/normalizers.npz",
        "metadata/scheduler_posenet.npz",
        "metadata/scheduler_trajnet.npz",
        "metadata/representation.json",
        "reports/denoiser_parity.json",
        "licenses/RoHM-LICENSE.md",
    }
    if not required.issubset(assets):
        raise RohmRuntimeError(
            f"RoHM bundle is missing assets: {sorted(required - set(assets))}"
        )
    declared_download = int(manifest.get("download_bytes", -1))
    if declared_download != installed_bytes:
        raise RohmRuntimeError("RoHM manifest download size does not match its assets")
    if verify_files:
        parity = json.loads(
            _resolved_member(root, "reports/denoiser_parity.json").read_text(
                encoding="utf-8"
            )
        )
        for name in ("pose_fp32", "traj_gate_zero_fp32", "traj_gate_one_fp32"):
            result = parity.get("parity", {}).get(name, {})
            if float(result.get("atol", 1.0)) > 1.0e-4:
                raise RohmRuntimeError(f"RoHM parity tolerance is too loose: {name}")
            if float(result.get("rtol", 1.0)) > 1.0e-4:
                raise RohmRuntimeError(f"RoHM parity tolerance is too loose: {name}")
    return Bundle(
        root=root,
        assets=assets,
        precision=str(manifest.get("precision", "")),
        download_bytes=declared_download,
        upstream_revision=upstream_revision,
    )


def _safe_npz(
    path: Path, expected: dict[str, tuple[int, ...]]
) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected):
                raise RohmRuntimeError(f"unexpected arrays in {path.name}")
            result = {
                name: np.asarray(archive[name], dtype=np.float64) for name in expected
            }
    except (OSError, ValueError, TypeError) as exc:
        raise RohmRuntimeError(f"could not load {path.name}: {exc}") from exc
    for name, shape in expected.items():
        if result[name].shape != shape or not np.isfinite(result[name]).all():
            raise RohmRuntimeError(f"invalid {path.name} array {name}")
    return result


def visibility_vector(joint_visibility: np.ndarray) -> np.ndarray:
    visible = np.asarray(joint_visibility, dtype=bool)
    if visible.ndim != 2 or visible.shape[1] != 22:
        raise RohmRuntimeError("RoHM visibility must have shape [T, 22]")
    result = np.ones((len(visible), FEATURE_COUNT), dtype=np.float32)
    result[:, 22:88] = np.repeat(visible, 3, axis=1)
    result[:, 88:154] = np.repeat(visible, 3, axis=1)
    result[:, 154:280] = np.repeat(visible[:, 1:], 6, axis=1)
    result[:, FOOT_CONTACT_SLICE] = 0.0
    left = visible[:, 7] & visible[:, 10]
    right = visible[:, 8] & visible[:, 11]
    result[left, 290:292] = 1.0
    result[right, 292:294] = 1.0
    return result


def _provider(requested: object, available: Sequence[str]) -> tuple[str, str, bool]:
    from ..sam3d import ort_runtime as sam3d_ort_runtime

    return sam3d_ort_runtime.select_execution_provider(requested, available)


class RohmSession:
    """Own the two static RoHM graphs and mesh-free diffusion scheduler."""

    def __init__(
        self,
        bundle: Bundle,
        requested_mode: object,
        *,
        ort_module=None,
        session_factory=None,
    ):
        if ort_module is None:
            from ... import dependency_bootstrap

            ort_module = dependency_bootstrap.import_onnxruntime()
        available = tuple(ort_module.get_available_providers())
        provider, device, fallback = _provider(requested_mode, available)
        providers = [provider]
        if (
            provider == "WebGpuExecutionProvider"
            and "CPUExecutionProvider" in available
        ):
            providers.append("CPUExecutionProvider")
        options = ort_module.SessionOptions()
        options.log_severity_level = 3
        factory = session_factory or ort_module.InferenceSession
        self.pose_session = factory(
            str(bundle.resolve("graphs/rohm_posenet_fp32.onnx")),
            sess_options=options,
            providers=providers,
        )
        self.traj_session = factory(
            str(bundle.resolve("graphs/rohm_trajnet_control_fp32.onnx")),
            sess_options=options,
            providers=providers,
        )
        normalizers = _safe_npz(
            bundle.resolve("metadata/normalizers.npz"),
            {"mean": (FEATURE_COUNT,), "std": (FEATURE_COUNT,)},
        )
        self.mean = normalizers["mean"].astype(np.float32)
        self.std = normalizers["std"].astype(np.float32)
        if np.any(self.std <= 0.0):
            raise RohmRuntimeError("RoHM normalizer scales must be positive")
        self.pose_scheduler = _safe_npz(
            bundle.resolve("metadata/scheduler_posenet.npz"),
            {
                "posterior_mean_coef1": (1000,),
                "posterior_mean_coef2": (1000,),
                "posterior_variance": (1000,),
            },
        )
        self.traj_scheduler = _safe_npz(
            bundle.resolve("metadata/scheduler_trajnet.npz"),
            {
                "posterior_mean_coef1": (100,),
                "posterior_mean_coef2": (100,),
                "posterior_variance": (100,),
            },
        )
        installed = sum(asset.bytes for asset in bundle.assets.values())
        self.diagnostics = RuntimeDiagnostics(
            requested_mode=str(requested_mode).upper(),
            actual_ep=provider,
            actual_device=device,
            fallback=fallback,
            precision=bundle.precision,
            download_bytes=bundle.download_bytes,
            installed_bytes=installed,
            upstream_revision=bundle.upstream_revision,
        )

    @staticmethod
    def _iter_sample(
        session,
        scheduler: dict[str, np.ndarray],
        shape: tuple[int, ...],
        condition: np.ndarray,
        rng: np.random.RandomState,
        *,
        control_condition: np.ndarray | None = None,
        control_gate: float = 0.0,
    ) -> Iterator[tuple[int, int]]:
        """Run one diffusion sample and release Blender after every ONNX call."""
        sample = rng.standard_normal(shape).astype(np.float32)
        step_count = len(scheduler["posterior_variance"])
        for timestep in range(step_count - 1, -1, -1):
            feeds = {
                "x_t": sample,
                "cond": condition,
            }
            if control_condition is None:
                feeds["timesteps"] = np.asarray([timestep], dtype=np.int64)
            else:
                feeds.update(
                    {
                        "control_cond": control_condition,
                        "time": np.asarray([timestep], dtype=np.int64),
                        "control_gate": np.asarray([control_gate], dtype=np.float32),
                    }
                )
            try:
                predicted = np.asarray(session.run(None, feeds)[0], dtype=np.float32)
            except Exception as exc:
                raise RohmRuntimeError(f"RoHM ONNX inference failed: {exc}") from exc
            if predicted.shape != shape or not np.isfinite(predicted).all():
                raise RohmRuntimeError("RoHM graph returned an invalid tensor")
            mean = (
                scheduler["posterior_mean_coef1"][timestep] * predicted
                + scheduler["posterior_mean_coef2"][timestep] * sample
            )
            noise = rng.standard_normal(shape).astype(np.float32)
            if timestep:
                variance = max(float(scheduler["posterior_variance"][timestep]), 0.0)
                sample = (mean + math.sqrt(variance) * noise).astype(np.float32)
            else:
                sample = mean.astype(np.float32)
            yield step_count - timestep, step_count
        return sample

    @staticmethod
    def _sample(
        session,
        scheduler: dict[str, np.ndarray],
        shape: tuple[int, ...],
        condition: np.ndarray,
        rng: np.random.RandomState,
        *,
        control_condition: np.ndarray | None = None,
        control_gate: float = 0.0,
    ) -> np.ndarray:
        iterator = RohmSession._iter_sample(
            session,
            scheduler,
            shape,
            condition,
            rng,
            control_condition=control_condition,
            control_gate=control_gate,
        )
        while True:
            try:
                next(iterator)
            except StopIteration as stopped:
                return stopped.value

    def iter_run_trajnet(
        self,
        condition: np.ndarray,
        control_condition: np.ndarray,
        rng: np.random.RandomState,
        *,
        control_gate: float,
    ) -> Iterator[tuple[int, int]]:
        condition = np.asarray(condition, dtype=np.float32)
        control = np.asarray(control_condition, dtype=np.float32)
        if condition.shape != (1, MODEL_WINDOW_FRAMES, 13):
            raise RohmRuntimeError("TrajNet condition must have shape [1, 144, 13]")
        if control.shape != (1, MODEL_WINDOW_FRAMES, POSE_COUNT):
            raise RohmRuntimeError("TrajNet control must have shape [1, 144, 272]")
        return (
            yield from self._iter_sample(
                self.traj_session,
                self.traj_scheduler,
                condition.shape,
                condition,
                rng,
                control_condition=control,
                control_gate=control_gate,
            )
        )

    def run_trajnet(
        self,
        condition: np.ndarray,
        control_condition: np.ndarray,
        rng: np.random.RandomState,
        *,
        control_gate: float,
    ) -> np.ndarray:
        iterator = self.iter_run_trajnet(
            condition,
            control_condition,
            rng,
            control_gate=control_gate,
        )
        while True:
            try:
                next(iterator)
            except StopIteration as stopped:
                return stopped.value

    def iter_run_posenet(
        self, condition: np.ndarray, rng: np.random.RandomState
    ) -> Iterator[tuple[int, int]]:
        condition = np.asarray(condition, dtype=np.float32)
        shape = (1, FEATURE_COUNT, 1, MODEL_WINDOW_FRAMES)
        if condition.shape != shape:
            raise RohmRuntimeError("PoseNet condition must have shape [1, 294, 1, 144]")
        return (
            yield from self._iter_sample(
                self.pose_session,
                self.pose_scheduler,
                shape,
                condition,
                rng,
            )
        )

    def run_posenet(
        self, condition: np.ndarray, rng: np.random.RandomState
    ) -> np.ndarray:
        iterator = self.iter_run_posenet(condition, rng)
        while True:
            try:
                next(iterator)
            except StopIteration as stopped:
                return stopped.value

    def normalize(self, representation: np.ndarray) -> np.ndarray:
        value = np.asarray(representation, dtype=np.float32)
        if value.shape[-1] != FEATURE_COUNT or not np.isfinite(value).all():
            raise RohmRuntimeError("invalid RoHM representation")
        return ((value - self.mean) / self.std).astype(np.float32)

    def denormalize(self, representation: np.ndarray) -> np.ndarray:
        value = np.asarray(representation, dtype=np.float32)
        if value.shape[-1] != FEATURE_COUNT or not np.isfinite(value).all():
            raise RohmRuntimeError("invalid normalized RoHM representation")
        return (value * self.std + self.mean).astype(np.float32)
