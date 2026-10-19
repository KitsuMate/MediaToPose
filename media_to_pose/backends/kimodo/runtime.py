"""ONNX Runtime execution of the Kimodo denoiser.

One ``KimodoSession`` holds one ONNX Runtime session for the installed
variant. Input dtypes are read from the session, so the fp16, fp32 and int8
graphs share this code. In AUTO mode an accelerator failure at run time
recreates the session on CPU once and retries, as SAM 3D Body does.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ...core.errors import BackendError
from ...core.log import get_logger
from ...core.motion.kimodo import rep
from ...core.motion.kimodo.sampler import BRANCHES, DenoiseInputs
from .. import lifecycle
from ..sam3d import ort_runtime

LOGGER = get_logger(__name__)

CPU_PROVIDER = "CPUExecutionProvider"
INPUT_NAMES = (
    "motion",
    "observed_motion",
    "motion_valid",
    "constraint_mask",
    "text_embedding",
    "timestep",
    "first_heading_angle",
)
STATS_FILES = {
    "global_root_mean": "stats/motion/global_root/mean.npy",
    "global_root_std": "stats/motion/global_root/std.npy",
    "body_mean": "stats/motion/body/mean.npy",
    "body_std": "stats/motion/body/std.npy",
}
_ORT_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(bool)": np.bool_,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}


class KimodoRuntimeError(BackendError):
    """A Kimodo bundle contract or inference failure."""


@dataclass(frozen=True)
class KimodoBundle:
    """An installed bundle: one denoiser graph plus statistics and licences."""

    root: Path
    variant: str
    graph: str
    revision: str
    files: tuple[lifecycle.BundleFile, ...]

    @property
    def installed_bytes(self) -> int:
        return sum(item.bytes for item in self.files)

    def resolve(self, relative: str) -> Path:
        return lifecycle.contained_path(self.root, relative, label="Kimodo bundle")


def graph_variant(path: str) -> str:
    """``onnx/model_fp16.onnx`` → ``FP16``."""
    stem = Path(path).stem
    return stem.removeprefix("model_").upper() if stem.startswith("model_") else ""


def load_bundle(root: Path, *, verify_files: bool = True) -> KimodoBundle:
    """Read ``root/manifest.json`` and check that every listed file is intact."""
    try:
        revision, files = lifecycle.read_code_manifest(
            root, verify_files=verify_files, label="Kimodo"
        )
    except BackendError as exc:
        raise KimodoRuntimeError(str(exc)) from exc
    graphs = [item.path for item in files if item.path.endswith(".onnx")]
    if len(graphs) != 1 or not graph_variant(graphs[0]):
        raise KimodoRuntimeError("A Kimodo bundle must hold exactly one denoiser")
    missing = sorted(set(STATS_FILES.values()) - {item.path for item in files})
    if missing:
        raise KimodoRuntimeError(f"Kimodo bundle lacks {', '.join(missing)}")
    return KimodoBundle(
        Path(root), graph_variant(graphs[0]), graphs[0], revision, files
    )


def load_stats(bundle: KimodoBundle) -> rep.FeatureStats:
    """The upstream normalisation statistics, read without pickle."""
    arrays = {}
    for name, relative in STATS_FILES.items():
        try:
            arrays[name] = np.load(bundle.resolve(relative), allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise KimodoRuntimeError(f"Kimodo statistics {relative}: {exc}") from exc
    return rep.FeatureStats.from_parts(**arrays)


@dataclass(frozen=True)
class RuntimeDiagnostics:
    requested_mode: str
    actual_ep: str
    actual_device: str
    fallback: bool
    variant: str
    revision: str
    installed_bytes: int


class KimodoSession:
    """The denoiser of one installed bundle on one execution provider.

    ``ort_module`` and ``session_factory`` are injectable for tests.
    """

    def __init__(
        self,
        bundle: KimodoBundle,
        requested_mode: object,
        *,
        ort_module: Any = None,
        session_factory: Any = None,
        sys_platform: str | None = None,
    ) -> None:
        self.bundle = bundle
        self.requested_mode = ort_runtime.normalize_mode(requested_mode)
        self._ort = ort_module or ort_runtime.import_onnxruntime()
        self._factory = session_factory or self._ort.InferenceSession
        self._available = ort_runtime.available_execution_providers(self._ort)
        provider, device, fallback = ort_runtime.select_execution_provider(
            self.requested_mode, self._available, sys_platform=sys_platform
        )
        self.stats = load_stats(bundle)
        self._session = self._create(provider)
        self.diagnostics = RuntimeDiagnostics(
            requested_mode=self.requested_mode,
            actual_ep=provider,
            actual_device=device,
            fallback=fallback,
            variant=bundle.variant,
            revision=bundle.revision,
            installed_bytes=bundle.installed_bytes,
        )

    def _create(self, provider: str):
        options = self._ort.SessionOptions()
        options.log_severity_level = 3
        options.graph_optimization_level = (
            self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        providers = [provider]
        if provider != CPU_PROVIDER and CPU_PROVIDER in self._available:
            providers.append(CPU_PROVIDER)
        try:
            session = self._factory(
                str(self.bundle.resolve(self.bundle.graph)),
                sess_options=options,
                providers=providers,
            )
        except Exception as exc:  # ONNX Runtime raises its own exception types.
            LOGGER.exception("Kimodo session creation failed")
            raise KimodoRuntimeError(
                f"Kimodo session could not be created on {provider}: {exc}"
            ) from exc
        inputs = {item.name: item.type for item in session.get_inputs()}
        missing = sorted(set(INPUT_NAMES) - set(inputs))
        if missing:
            raise KimodoRuntimeError(f"Kimodo graph lacks inputs {missing}")
        unknown = {
            name: kind for name, kind in inputs.items() if kind not in _ORT_DTYPES
        }
        if unknown:
            raise KimodoRuntimeError(f"Kimodo graph has unsupported inputs {unknown}")
        self._dtypes = {name: _ORT_DTYPES[inputs[name]] for name in INPUT_NAMES}
        self._output = session.get_outputs()[0].name
        return session

    def _feeds(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            name: np.ascontiguousarray(values[name], dtype=self._dtypes[name])
            for name in INPUT_NAMES
        }

    def _can_fall_back(self) -> bool:
        return (
            self.requested_mode == "AUTO"
            and self.diagnostics.actual_ep != CPU_PROVIDER
            and CPU_PROVIDER in self._available
        )

    def _fall_back_to_cpu(self, error: Exception) -> None:
        LOGGER.warning(
            "Kimodo accelerator inference failed; retrying on CPU: %s", error
        )
        self._session = None
        self._session = self._create(CPU_PROVIDER)
        self.diagnostics = replace(
            self.diagnostics,
            actual_ep=CPU_PROVIDER,
            actual_device="cpu",
            fallback=True,
        )

    def denoise(
        self,
        motion: np.ndarray,
        observed: np.ndarray,
        mask: np.ndarray,
        valid: np.ndarray,
        text: np.ndarray,
        timestep: np.ndarray,
        heading: np.ndarray,
    ) -> np.ndarray:
        """One three-branch call; returns the predicted clean motion [3, T, 369]."""
        motion = np.asarray(motion)
        frames = motion.shape[1] if motion.ndim == 3 else 0
        expected = (BRANCHES, frames, rep.FEATURE_COUNT)
        if motion.shape != expected or not 2 <= frames <= rep.MAX_WINDOW_FRAMES:
            raise KimodoRuntimeError(
                f"Kimodo motion must be [3, T, 369] with T in 2..300, got {motion.shape}"
            )
        feeds = self._feeds(
            {
                "motion": motion,
                "observed_motion": observed,
                "motion_valid": valid,
                "constraint_mask": mask,
                "text_embedding": text,
                "timestep": timestep,
                "first_heading_angle": heading,
            }
        )
        try:
            output = self._session.run([self._output], feeds)[0]
        except Exception as exc:  # ONNX Runtime raises its own exception types.
            if not self._can_fall_back():
                LOGGER.exception("Kimodo inference failed")
                raise KimodoRuntimeError(f"Kimodo inference failed: {exc}") from exc
            self._fall_back_to_cpu(exc)
            try:
                output = self._session.run([self._output], self._feeds(feeds))[0]
            except Exception as cpu_exc:  # As above.
                LOGGER.exception("Kimodo CPU retry failed")
                raise KimodoRuntimeError(
                    f"Kimodo inference failed on the accelerator and on CPU: {cpu_exc}"
                ) from cpu_exc
        result = np.asarray(output, dtype=np.float32)
        if result.shape != expected or not np.all(np.isfinite(result)):
            raise KimodoRuntimeError("Kimodo returned an invalid tensor")
        return result

    def __call__(self, inputs: DenoiseInputs) -> np.ndarray:
        """The sampler's ``DenoiseFn``."""
        return self.denoise(
            inputs.motion,
            inputs.observed_motion,
            inputs.constraint_mask,
            inputs.motion_valid,
            inputs.text_embedding,
            inputs.timestep,
            inputs.first_heading_angle,
        )

    def close(self) -> None:
        self._session = None


__all__ = [
    "INPUT_NAMES",
    "STATS_FILES",
    "KimodoBundle",
    "KimodoRuntimeError",
    "KimodoSession",
    "RuntimeDiagnostics",
    "graph_variant",
    "load_bundle",
    "load_stats",
]
