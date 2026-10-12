"""Manifest-driven ONNX Runtime adapter for SAM 3D Body.

The extension treats converted model bundles as an external, versioned
contract. Every component graph must be declared for the split NumPy runner,
and an invalid bundle fails before a session is created.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from .core.errors import BackendError

BUNDLE_SCHEMA_VERSION = 1
BUNDLE_CONTRACT = "media-to-pose.sam3d.raw.v1"
SUPPORTED_BUNDLE_CONTRACTS = frozenset(
    {
        BUNDLE_CONTRACT,
        "vision-to-pose.sam3d.raw.v1",
    }
)
SPLIT_NUMPY_RUNNER = "onnxruntime.split_numpy.v1"
REQUIRED_OUTPUT_KEYS = (
    "pred_keypoints_3d",
    "pred_keypoints_2d",
    "pred_cam_t",
    "focal_length",
    "global_rot",
    "scale_params",
    "pred_joint_coords",
    "pred_global_rots",
)

# Exact public tensor boundary captured from the pinned ViT-H reference
# implementation. Every suffix follows the leading person axis.
OUTPUT_TENSOR_CONTRACTS = {
    "pred_keypoints_3d": ("float32", (70, 3)),
    "pred_keypoints_2d": ("float32", (70, 2)),
    "pred_cam_t": ("float32", (3,)),
    "focal_length": ("float32", ()),
    "global_rot": ("float32", (3,)),
    "scale_params": ("float32", (28,)),
    "pred_joint_coords": ("float32", (127, 3)),
    "pred_global_rots": ("float32", (127, 3, 3)),
}

# These are the MHR output joints whose orientation frames are consumed by
# ``SAM3DBodyProvider``. Keeping the mapping beside the bundle contract makes
# it impossible for runtime validation and provider conversion to drift.
MHR_OUTPUT_JOINTS = {
    "MHR_SPINE_0": 34,
    "MHR_SPINE_1": 35,
    "MHR_SPINE_2": 36,
    "MHR_SPINE_3": 37,
    "MHR_CLAVICLE_R": 38,
    "MHR_UPPER_ARM_R": 39,
    "MHR_LOWER_ARM_R": 40,
    "MHR_HAND_R": 41,
    "MHR_CLAVICLE_L": 74,
    "MHR_UPPER_ARM_L": 75,
    "MHR_LOWER_ARM_L": 76,
    "MHR_HAND_L": 77,
    "MHR_NECK": 112,
    "MHR_HEAD": 115,
    "MHR_EYE_R": 122,
    "MHR_EYE_L": 124,
    "MHR_HEAD_TIP": 126,
}
REQUIRED_NEUTRAL_ROTATION_INDICES = frozenset(MHR_OUTPUT_JOINTS.values())
_MHR_JOINT_COUNT = 127
_ROTATION_MATRIX_TOLERANCE = 1.0e-4

_WEBGPU_PROVIDERS = (
    "WebGpuExecutionProvider",
    "WebGPUExecutionProvider",
)
_COREML_PROVIDERS = ("CoreMLExecutionProvider",)
_CPU_PROVIDER = "CPUExecutionProvider"
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-fA-F]{40,64}$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_GRAPH_ROLE = re.compile(r"^[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*$")
_LOGICAL_TENSOR_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_EP_REGISTRATION_LOCK = threading.RLock()
_REGISTERED_ORT_MODULES: set[int] = set()


class OrtRuntimeError(BackendError):
    """Base error for an unavailable runtime or incompatible model bundle."""


class BundleContractError(OrtRuntimeError):
    """Raised when a downloaded graph bundle violates the public contract."""


@dataclass(frozen=True)
class RuntimeDiagnostics:
    requested_mode: str
    actual_ep: str
    actual_device: str
    fallback: bool
    model_revision: str
    precision: str
    available_providers: tuple[str, ...] = ()
    hardware_vendor_id: int = 0
    hardware_device_id: int = 0
    accelerator_nodes_verified: bool = False
    accelerator_node_count: int = 0
    profiled_node_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BundleFile:
    path: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class GraphSpec(BundleFile):
    role: str
    inputs: Mapping[str, str]
    outputs: Mapping[str, str]


@dataclass(frozen=True)
class BundleManifest:
    root: Path
    variant: str
    model_revision: str
    precision: str
    runner: str
    graphs: tuple[GraphSpec, ...]
    artifacts: tuple[BundleFile, ...]
    neutral_rotations: Mapping[int, tuple[float, ...]]

    def graph_for_role(self, role: str) -> GraphSpec:
        """Return one graph by its exact, manifest-declared role."""
        if not isinstance(role, str) or not role:
            raise BundleContractError("Graph role lookup requires a non-empty string")
        for graph in self.graphs:
            if graph.role == role:
                return graph
        raise BundleContractError(f"Bundle has no graph with role {role!r}")

    def graph_path_for_role(self, role: str) -> Path:
        """Resolve a role to a verified member path inside the bundle root."""
        return _resolved_member(self.root, self.graph_for_role(role).path)

    @property
    def files(self) -> tuple[BundleFile, ...]:
        """Every downloadable graph and ordinary data artifact in the bundle."""
        return (*self.graphs, *self.artifacts)


def normalize_mode(value: object) -> str:
    """Validate and return one of the public AUTO/GPU/CPU modes."""
    selected = str(value or "AUTO").upper()
    if selected not in {"AUTO", "GPU", "CPU"}:
        raise OrtRuntimeError(f"Unsupported SAM device mode {selected!r}")
    return selected


def is_immutable_revision(value: object) -> bool:
    return bool(_IMMUTABLE_REVISION.fullmatch(str(value or "")))


def _gpu_candidates(sys_platform: str | None = None) -> tuple[str, ...]:
    platform_name = sys_platform or sys.platform
    if platform_name == "darwin":
        return (*_COREML_PROVIDERS, *_WEBGPU_PROVIDERS)
    if platform_name == "win32" or platform_name.startswith("linux"):
        return _WEBGPU_PROVIDERS
    return ()


def select_execution_provider(
    requested_mode: object,
    available_providers: Sequence[str],
    *,
    sys_platform: str | None = None,
) -> tuple[str, str, bool]:
    """Choose one accelerator deterministically, with an explicit CPU fallback."""
    mode = normalize_mode(requested_mode)
    available = tuple(str(value) for value in available_providers)
    by_folded = {value.casefold(): value for value in available}
    if mode != "CPU":
        for candidate in _gpu_candidates(sys_platform):
            selected = by_folded.get(candidate.casefold())
            if selected:
                device = "coreml" if "coreml" in selected.casefold() else "webgpu"
                return selected, device, False
        if mode == "GPU":
            expectation = (
                "CoreMLExecutionProvider"
                if (sys_platform or sys.platform) == "darwin"
                else "WebGpuExecutionProvider"
            )
            raise OrtRuntimeError(
                f"GPU mode requires {expectation}, but ONNX Runtime does not "
                "expose it on this machine"
            )
    cpu = by_folded.get(_CPU_PROVIDER.casefold())
    if cpu:
        return cpu, "cpu", mode != "CPU"
    expectation = (
        "CoreMLExecutionProvider"
        if (sys_platform or sys.platform) == "darwin"
        else "WebGpuExecutionProvider"
    )
    raise OrtRuntimeError(
        "ONNX Runtime exposes neither the requested accelerator "
        f"({expectation}) nor CPUExecutionProvider"
    )


def _resolved_member(root: Path, relative: str) -> Path:
    raw = str(relative)
    posix = PurePosixPath(raw)
    if (
        posix.is_absolute()
        or ".." in posix.parts
        or not posix.parts
        or "\\" in raw
        or "\x00" in raw
        or any(":" in part for part in posix.parts)
    ):
        raise BundleContractError(f"Unsafe graph bundle path: {relative!r}")
    resolved_root = root.resolve(strict=False)
    resolved = resolved_root.joinpath(*posix.parts).resolve(strict=False)
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise BundleContractError(f"Graph bundle path escapes its root: {relative!r}")
    return resolved


def _string_mapping(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise BundleContractError(f"Bundle {label} must be an object")
    if not value:
        raise BundleContractError(f"Bundle {label} must not be empty")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not _LOGICAL_TENSOR_NAME.fullmatch(key):
            raise BundleContractError(
                f"Bundle {label} contains an invalid logical tensor name"
            )
        if (
            not isinstance(item, str)
            or not item
            or item != item.strip()
            or len(item) > 512
            or any(ord(character) < 32 for character in item)
        ):
            raise BundleContractError(
                f"Bundle {label} contains an invalid ONNX tensor name"
            )
        result[key] = item
    if len(set(result.values())) != len(result):
        raise BundleContractError(
            f"Bundle {label} maps multiple logical names to one ONNX tensor"
        )
    return result


def _bundle_file(value: object, label: str) -> BundleFile:
    if not isinstance(value, dict):
        raise BundleContractError(f"Bundle {label} must be an object")
    path_value = value.get("path")
    if not isinstance(path_value, str):
        raise BundleContractError(f"Bundle {label} path must be a string")
    path = path_value
    _resolved_member(Path(), path)
    digest_value = value.get("sha256")
    digest = digest_value if isinstance(digest_value, str) else ""
    byte_value = value.get("bytes")
    if not isinstance(byte_value, int) or isinstance(byte_value, bool):
        raise BundleContractError(f"Bundle {label} byte count is invalid")
    byte_count = byte_value
    if not _SHA256.fullmatch(digest):
        raise BundleContractError(f"Bundle {label} SHA-256 is missing or invalid")
    if byte_count <= 0:
        raise BundleContractError(f"Bundle {label} byte count must be positive")
    return BundleFile(path=path, sha256=digest.casefold(), bytes=byte_count)


def _graph_role(value: object, label: str) -> str:
    if not isinstance(value, str) or not _GRAPH_ROLE.fullmatch(value):
        raise BundleContractError(
            f"Bundle {label} role must be a non-empty dotted identifier"
        )
    return value


def _json_object_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise BundleContractError(f"Bundle JSON repeats object key {key!r}")
        result[key] = value
    return result


def _neutral_rotations(value: object) -> dict[int, tuple[float, ...]]:
    if not isinstance(value, dict):
        raise BundleContractError("neutral_rotations must be an object")
    result: dict[int, tuple[float, ...]] = {}
    for raw_index, raw_matrix in value.items():
        try:
            index = int(raw_index)
            matrix = tuple(float(item) for item in raw_matrix)
        except (TypeError, ValueError) as exc:
            raise BundleContractError(
                f"Invalid neutral rotation for joint {raw_index!r}"
            ) from exc
        if index < 0 or index >= _MHR_JOINT_COUNT or len(matrix) != 9:
            raise BundleContractError(
                f"Neutral rotation for joint {raw_index!r} must be a 3x3 matrix"
            )
        if index in result:
            raise BundleContractError(
                f"neutral_rotations repeats normalized joint index {index}"
            )
        if not all(math.isfinite(item) for item in matrix):
            raise BundleContractError(
                f"Neutral rotation for joint {index} contains a non-finite value"
            )
        rows = (matrix[0:3], matrix[3:6], matrix[6:9])
        for row_index, row in enumerate(rows):
            for column_index, column in enumerate(rows):
                expected = 1.0 if row_index == column_index else 0.0
                dot = sum(left * right for left, right in zip(row, column, strict=True))
                if abs(dot - expected) > _ROTATION_MATRIX_TOLERANCE:
                    raise BundleContractError(
                        f"Neutral rotation for joint {index} is not orthonormal"
                    )
        determinant = (
            matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
            - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
            + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6])
        )
        if abs(determinant - 1.0) > _ROTATION_MATRIX_TOLERANCE:
            raise BundleContractError(
                f"Neutral rotation for joint {index} is not a proper rotation"
            )
        result[index] = matrix
    missing = sorted(REQUIRED_NEUTRAL_ROTATION_INDICES - set(result))
    if missing:
        raise BundleContractError(
            "neutral_rotations is missing provider-consumed MHR joints: "
            + ", ".join(str(index) for index in missing)
        )
    return result


def _validate_output_tensors(np, arrays: Mapping[str, Any], person_count: int) -> None:
    """Reject any inference result outside the exact provider tensor contract."""
    missing = sorted(set(OUTPUT_TENSOR_CONTRACTS) - set(arrays))
    extra = sorted(set(arrays) - set(OUTPUT_TENSOR_CONTRACTS))
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unsupported " + ", ".join(extra))
        raise OrtRuntimeError(
            "ONNX outputs violate the graph contract: " + "; ".join(details)
        )
    for key, value in arrays.items():
        expected_dtype, suffix = OUTPUT_TENSOR_CONTRACTS[key]
        expected_shape = (person_count, *suffix)
        if value.dtype != np.dtype(expected_dtype):
            raise OrtRuntimeError(
                f"ONNX output {key!r} must have dtype {expected_dtype}; "
                f"received {value.dtype}"
            )
        if tuple(value.shape) != expected_shape:
            raise OrtRuntimeError(
                f"ONNX output {key!r} must have shape {expected_shape}, "
                "including the leading person axis; received "
                f"{tuple(value.shape)}"
            )
        if not bool(np.isfinite(value).all()):
            raise OrtRuntimeError(f"ONNX output {key!r} contains a non-finite value")


def load_bundle_manifest(
    path: os.PathLike[str] | str,
    *,
    expected_variant: str = "",
    expected_model_revision: str = "",
    verify_files: bool = True,
) -> BundleManifest:
    manifest_path = Path(path)
    try:
        payload = json.loads(
            manifest_path.read_text(encoding="utf8"),
            object_pairs_hook=_json_object_no_duplicates,
        )
    except BundleContractError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise BundleContractError(
            f"Could not read ONNX bundle manifest: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise BundleContractError("ONNX bundle manifest must contain an object")
    if payload.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise BundleContractError(
            "Unsupported ONNX bundle schema; publish a compatible immutable bundle"
        )
    if payload.get("contract") not in SUPPORTED_BUNDLE_CONTRACTS:
        raise BundleContractError("Unsupported SAM 3D Body bundle contract")
    runner_value = payload.get("runner")
    runner = runner_value if isinstance(runner_value, str) else ""
    if runner != SPLIT_NUMPY_RUNNER:
        raise BundleContractError(f"Unsupported SAM 3D Body bundle runner {runner!r}")
    variant_value = payload.get("variant")
    revision_value = payload.get("model_revision")
    precision_value = payload.get("precision")
    variant = variant_value if isinstance(variant_value, str) else ""
    revision = revision_value if isinstance(revision_value, str) else ""
    precision = precision_value if isinstance(precision_value, str) else ""
    if expected_variant and variant != expected_variant:
        raise BundleContractError(
            f"Bundle variant is {variant!r}, expected {expected_variant!r}"
        )
    if not is_immutable_revision(revision):
        raise BundleContractError(
            "Bundle model_revision must be an immutable commit SHA"
        )
    if (
        expected_model_revision
        and revision.casefold() != expected_model_revision.casefold()
    ):
        raise BundleContractError(
            "Bundle model revision does not match the pinned source checkpoint"
        )
    if not precision:
        raise BundleContractError("Bundle precision is missing")
    raw_graphs = payload.get("graphs")
    if not isinstance(raw_graphs, list):
        raise BundleContractError("Bundle graphs must be an array")
    if len(raw_graphs) < 2:
        raise BundleContractError("Split NumPy bundle must declare multiple graphs")
    parsed_graphs: list[GraphSpec] = []
    for index, raw_graph in enumerate(raw_graphs):
        label = f"graph {index}"
        if not isinstance(raw_graph, dict):
            raise BundleContractError(f"Bundle {label} must be an object")
        role = _graph_role(raw_graph.get("role"), label)
        graph_file = _bundle_file(raw_graph, label)
        if PurePosixPath(graph_file.path).suffix.casefold() != ".onnx":
            raise BundleContractError(f"Bundle {label} must reference an ONNX file")
        inputs = _string_mapping(raw_graph.get("inputs"), f"{label} inputs")
        outputs = _string_mapping(raw_graph.get("outputs"), f"{label} outputs")
        parsed_graphs.append(
            GraphSpec(
                path=graph_file.path,
                sha256=graph_file.sha256,
                bytes=graph_file.bytes,
                role=role,
                inputs=MappingProxyType(inputs),
                outputs=MappingProxyType(outputs),
            )
        )
    graphs = tuple(parsed_graphs)
    folded_roles = [graph.role.casefold() for graph in graphs]
    if len(set(folded_roles)) != len(folded_roles):
        raise BundleContractError("Bundle contains duplicate graph roles")
    raw_artifacts = payload.get("artifacts", [])
    if not isinstance(raw_artifacts, list):
        raise BundleContractError("Bundle artifacts must be an array")
    artifacts = tuple(
        _bundle_file(value, f"artifact {index}")
        for index, value in enumerate(raw_artifacts)
    )
    folded_paths = [item.path.casefold() for item in (*graphs, *artifacts)]
    if len(set(folded_paths)) != len(folded_paths):
        raise BundleContractError("Bundle contains duplicate file paths")
    bundle = BundleManifest(
        root=manifest_path.parent,
        variant=variant,
        model_revision=revision.casefold(),
        precision=precision,
        runner=runner,
        graphs=graphs,
        artifacts=artifacts,
        neutral_rotations=MappingProxyType(
            _neutral_rotations(payload.get("neutral_rotations"))
        ),
    )
    if verify_files:
        for bundle_file in bundle.files:
            local_file = _resolved_member(bundle.root, bundle_file.path)
            if not local_file.is_file():
                raise BundleContractError(f"Bundle file is missing: {bundle_file.path}")
            if local_file.stat().st_size != bundle_file.bytes:
                raise BundleContractError(
                    f"Bundle file size differs: {bundle_file.path}"
                )
            digest_value = hashlib.sha256()
            with local_file.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest_value.update(chunk)
            if digest_value.hexdigest() != bundle_file.sha256:
                raise BundleContractError(
                    f"Bundle file hash differs: {bundle_file.path}"
                )
    return bundle


def import_onnxruntime():
    try:
        ort = importlib.import_module("onnxruntime")
    except Exception as exc:
        raise OrtRuntimeError(
            "The platform ONNX Runtime wheel is unavailable. Reinstall the complete "
            "platform-specific Media to Pose extension."
        ) from exc
    _register_webgpu_plugin(ort)
    return ort


def _register_webgpu_plugin(ort_module) -> None:
    if sys.platform == "darwin":
        return
    identity = id(ort_module)
    with _EP_REGISTRATION_LOCK:
        if identity in _REGISTERED_ORT_MODULES:
            return
        try:
            existing = tuple(
                str(value) for value in ort_module.get_available_providers()
            )
        except Exception:
            existing = ()
        if any("webgpu" in value.casefold() for value in existing):
            _REGISTERED_ORT_MODULES.add(identity)
            return
        register_library = getattr(
            ort_module,
            "register_execution_provider_library",
            None,
        )
        if not callable(register_library):
            _REGISTERED_ORT_MODULES.add(identity)
            return
        try:
            plugin = importlib.import_module("onnxruntime_ep_webgpu")
            register_library(
                "media_to_pose_webgpu",
                plugin.get_library_path(),
            )
        except ModuleNotFoundError:
            # Some ORT distributions compile WebGPU directly into the main
            # wheel, while macOS intentionally uses CoreML.  The provider
            # selection error below names a genuinely missing accelerator.
            _REGISTERED_ORT_MODULES.add(identity)
            return
        except Exception as exc:
            raise OrtRuntimeError(
                f"Could not register the bundled WebGPU execution provider: {exc}"
            ) from exc
        try:
            registered = tuple(
                str(value) for value in ort_module.get_available_providers()
            )
        except Exception as exc:
            raise OrtRuntimeError(
                f"Could not verify the registered WebGPU provider: {exc}"
            ) from exc
        if not any("webgpu" in value.casefold() for value in registered):
            raise OrtRuntimeError(
                "The bundled WebGPU provider library loaded but did not register "
                "an execution provider"
            )
        _REGISTERED_ORT_MODULES.add(identity)


def available_execution_providers(ort_module=None) -> tuple[str, ...]:
    ort = ort_module or import_onnxruntime()
    _register_webgpu_plugin(ort)
    try:
        return tuple(str(value) for value in ort.get_available_providers())
    except Exception as exc:
        raise OrtRuntimeError(f"Could not query ONNX Runtime providers: {exc}") from exc


def _hardware_device(ort_module, execution_provider: str):
    if "webgpu" not in execution_provider.casefold():
        return None
    getter = getattr(ort_module, "get_ep_devices", None)
    if not callable(getter):
        return None
    try:
        devices = [
            item
            for item in getter()
            if str(getattr(item, "ep_name", "")).casefold()
            == execution_provider.casefold()
        ]
    except Exception as exc:
        raise OrtRuntimeError(f"Could not enumerate WebGPU adapters: {exc}") from exc
    if not devices:
        return None

    def rank(item) -> tuple[int, int, int]:
        hardware = getattr(item, "device", None)
        vendor = int(getattr(hardware, "vendor_id", 0) or 0)
        device_id = int(getattr(hardware, "device_id", 0) or 0)
        kind = str(getattr(hardware, "type", "")).casefold()
        # Prefer a discrete NVIDIA adapter over an integrated AMD/Intel one.
        # Vendor ranking remains deterministic when ORT exposes no integrated
        # versus discrete flag (the current native WebGPU device API does not).
        vendor_rank = {0x10DE: 300, 0x1002: 200, 0x8086: 100}.get(vendor, 0)
        discrete_hint = 1 if "discrete" in kind else 0
        return discrete_hint, vendor_rank, device_id

    return max(devices, key=rank)


def _hardware_ids(device) -> tuple[int, int]:
    hardware = getattr(device, "device", None)
    return (
        int(getattr(hardware, "vendor_id", 0) or 0),
        int(getattr(hardware, "device_id", 0) or 0),
    )


def probe_runtime(
    requested_mode: object,
    *,
    ort_module=None,
    sys_platform: str | None = None,
) -> RuntimeDiagnostics:
    available = available_execution_providers(ort_module)
    provider, device, fallback = select_execution_provider(
        requested_mode,
        available,
        sys_platform=sys_platform,
    )
    hardware = _hardware_device(ort_module or import_onnxruntime(), provider)
    vendor_id, device_id = _hardware_ids(hardware)
    return RuntimeDiagnostics(
        requested_mode=normalize_mode(requested_mode),
        actual_ep=provider,
        actual_device=device,
        fallback=fallback,
        model_revision="",
        precision="",
        available_providers=available,
        hardware_vendor_id=vendor_id,
        hardware_device_id=device_id,
    )


def create_estimator(
    bundle: BundleManifest,
    requested_mode: object,
    *,
    ort_module=None,
    session_factory=None,
    sys_platform: str | None = None,
    verify_accelerator_execution: bool = False,
):
    """Create the model-specific estimator declared by the bundle runner."""

    common = {
        "ort_module": ort_module,
        "session_factory": session_factory,
        "sys_platform": sys_platform,
        "verify_accelerator_execution": verify_accelerator_execution,
    }
    if bundle.runner != SPLIT_NUMPY_RUNNER:
        raise OrtRuntimeError(f"Unsupported ONNX bundle runner {bundle.runner!r}")
    module_name = (
        f"{__package__}.sam3d_vith_runtime" if __package__ else "sam3d_vith_runtime"
    )
    try:
        module = importlib.import_module(module_name)
        estimator_type = module.VithEstimator
    except Exception as exc:
        raise OrtRuntimeError(f"Could not load the ViT-H orchestrator: {exc}") from exc
    return estimator_type(bundle, requested_mode, **common)


def load_estimator(
    paths: Mapping[str, str],
    variant: str,
    requested_mode: object,
    *,
    ort_module=None,
    session_factory=None,
    verify_accelerator_execution: bool = False,
):
    manifest_path = str(paths.get("bundle_manifest_path") or "")
    if not manifest_path:
        raise OrtRuntimeError(
            "Installed ONNX bundle manifest is unavailable; repair the provider"
        )
    bundle = load_bundle_manifest(
        manifest_path,
        expected_variant=str(variant),
        expected_model_revision=str(paths.get("model_revision") or ""),
        verify_files=True,
    )
    return create_estimator(
        bundle,
        requested_mode,
        ort_module=ort_module,
        session_factory=session_factory,
        verify_accelerator_execution=bool(verify_accelerator_execution),
    )


__all__ = [
    "BUNDLE_CONTRACT",
    "BUNDLE_SCHEMA_VERSION",
    "BundleFile",
    "BundleContractError",
    "BundleManifest",
    "GraphSpec",
    "MHR_OUTPUT_JOINTS",
    "OUTPUT_TENSOR_CONTRACTS",
    "OrtRuntimeError",
    "REQUIRED_NEUTRAL_ROTATION_INDICES",
    "REQUIRED_OUTPUT_KEYS",
    "RuntimeDiagnostics",
    "SPLIT_NUMPY_RUNNER",
    "available_execution_providers",
    "create_estimator",
    "is_immutable_revision",
    "load_bundle_manifest",
    "load_estimator",
    "normalize_mode",
    "probe_runtime",
    "select_execution_provider",
]
