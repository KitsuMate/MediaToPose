"""The estimator the SAM 3D Body provider calls, and a bundle smoke runner."""

from __future__ import annotations

import gc
import json
import os
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ....core.log import get_logger
from .. import ort_runtime as _ort_runtime
from .common import OUTPUT_KEYS, GraphExecutionError, VithRuntimeError
from .graphs import required_graph_roles
from .runner import VithSplitRunner

LOGGER = get_logger(__name__)


class VithEstimator:
    """Plugin-facing persistent split estimator with EP/device verification."""

    def __init__(
        self,
        bundle,
        requested_mode: object,
        *,
        ort_module=None,
        session_factory=None,
        sys_platform: str | None = None,
        verify_accelerator_execution: bool = False,
    ) -> None:
        runtime = _ort_runtime
        if bundle.runner != runtime.SPLIT_NUMPY_RUNNER:
            raise VithRuntimeError("VithEstimator requires onnxruntime.split_numpy.v1")
        self.bundle = bundle
        self.requested_mode = runtime.normalize_mode(requested_mode)
        self._ort = ort_module or runtime.import_onnxruntime()
        runtime._register_webgpu_plugin(self._ort)
        available = runtime.available_execution_providers(self._ort)
        self._available_providers = tuple(available)
        selected, device, fallback = runtime.select_execution_provider(
            self.requested_mode, available, sys_platform=sys_platform
        )
        cpu_available = any(
            name.casefold() == "cpuexecutionprovider" for name in available
        )
        factory = session_factory or self._ort.InferenceSession
        self._session_factory = factory
        hardware = runtime._hardware_device(self._ort, selected)
        self._profile_requested = bool(
            verify_accelerator_execution
            and selected.casefold() != "cpuexecutionprovider"
        )
        self._profile_failure: VithRuntimeError | None = None
        self.accelerator_profile_evidence: dict[str, object] = {}
        self._profile_prefixes: dict[str, Path] = {}
        self._runner = None
        self._sessions: dict[str, object] = {}
        try:
            self._sessions = self._create_sessions(
                bundle,
                factory,
                selected,
                available,
                hardware,
                profile=self._profile_requested,
            )
        except Exception as exc:
            self._sessions.clear()
            if (
                self.requested_mode == "AUTO"
                and selected.casefold() != "cpuexecutionprovider"
                and cpu_available
            ):
                try:
                    self._sessions = self._create_sessions(
                        bundle,
                        factory,
                        "CPUExecutionProvider",
                        available,
                        None,
                        profile=False,
                    )
                except Exception as cpu_exc:
                    raise VithRuntimeError(
                        "Could not create all split SAM sessions with "
                        f"{selected} or CPU fallback: {cpu_exc}"
                    ) from cpu_exc
                selected, device, fallback = "CPUExecutionProvider", "cpu", True
                hardware = None
                self._profile_requested = False
            else:
                raise VithRuntimeError(
                    f"Could not create all split SAM sessions with {selected}: {exc}"
                ) from exc
        actual_providers = self._actual_session_providers()
        unexpected = {
            provider
            for provider in actual_providers
            if provider.casefold() != selected.casefold()
        }
        if unexpected:
            cpu_only = all(
                provider.casefold() == "cpuexecutionprovider" for provider in unexpected
            )
            if self.requested_mode == "GPU" or not cpu_only:
                self.close()
                raise VithRuntimeError(
                    "Split sessions selected unexpected providers: "
                    + ", ".join(sorted(unexpected))
                )
            self._dispose_sessions(self._sessions, profiled=self._profile_requested)
            try:
                self._sessions = self._create_sessions(
                    bundle,
                    factory,
                    "CPUExecutionProvider",
                    available,
                    None,
                    profile=False,
                )
            except Exception as cpu_exc:
                raise VithRuntimeError(
                    "Could not recreate every split SAM session on the CPU "
                    f"after accelerator fallback: {cpu_exc}"
                ) from cpu_exc
            selected, device, fallback = "CPUExecutionProvider", "cpu", True
            hardware = None
            self._profile_requested = False
            if not self._all_sessions_use("CPUExecutionProvider"):
                self._dispose_sessions(self._sessions, profiled=False)
                raise VithRuntimeError(
                    "CPU fallback did not recreate every split session exclusively "
                    "on CPUExecutionProvider"
                )
        vendor_id, device_id = runtime._hardware_ids(hardware)
        self.diagnostics = runtime.RuntimeDiagnostics(
            requested_mode=self.requested_mode,
            actual_ep=selected,
            actual_device=device,
            fallback=fallback,
            model_revision=bundle.model_revision,
            precision=bundle.precision,
            available_providers=available,
            hardware_vendor_id=vendor_id,
            hardware_device_id=device_id,
        )
        self.neutral_rotations = dict(bundle.neutral_rotations)
        try:
            self._runner = VithSplitRunner(bundle, self._sessions)
        except Exception as exc:
            self._dispose_sessions(
                self._sessions, profiled=bool(self._profile_requested)
            )
            self._profile_requested = False
            self._ort = None
            if isinstance(exc, VithRuntimeError):
                raise
            raise VithRuntimeError(
                f"Could not initialize the split NumPy orchestrator: {exc}"
            ) from exc

    def _actual_session_providers(self) -> list[str]:
        actual: list[str] = []
        for session in self._sessions.values():
            try:
                providers = tuple(session.get_providers())
            except (AttributeError, RuntimeError, TypeError):
                providers = ()
            if providers:
                actual.append(str(providers[0]))
        return actual

    def _all_sessions_use(self, provider: str) -> bool:
        actual = self._actual_session_providers()
        return len(actual) == len(self._sessions) and all(
            value.casefold() == provider.casefold() for value in actual
        )

    def _delete_profile_files(self) -> None:
        for prefix in tuple(self._profile_prefixes.values()):
            for candidate in prefix.parent.glob(prefix.name + "*"):
                if candidate.is_file():
                    candidate.unlink(missing_ok=True)
        self._profile_prefixes.clear()

    def _dispose_sessions(self, sessions: dict[str, object], *, profiled: bool) -> None:
        for session in tuple(sessions.values()):
            profile_path = None
            if profiled:
                try:
                    profile_path = Path(session.end_profiling())
                except Exception:  # ONNX Runtime raises its own exception types.
                    LOGGER.exception("Could not end ONNX Runtime profiling")
                finally:
                    if profile_path is not None and profile_path.is_file():
                        profile_path.unlink(missing_ok=True)
        self._delete_profile_files()
        sessions.clear()
        gc.collect()

    def _create_sessions(
        self,
        bundle,
        factory,
        selected: str,
        available: Sequence[str],
        hardware,
        *,
        profile: bool,
    ) -> dict[str, object]:
        providers = [selected]
        if selected.casefold() != "cpuexecutionprovider" and any(
            name.casefold() == "cpuexecutionprovider" for name in available
        ):
            providers.append("CPUExecutionProvider")
        sessions: dict[str, object] = {}
        options_type = getattr(self._ort, "SessionOptions", None)
        try:
            for ordinal, graph in enumerate(bundle.graphs):
                options = options_type() if options_type is not None else None
                if profile:
                    if options is None:
                        raise VithRuntimeError(
                            "This ORT build cannot profile split accelerator sessions"
                        )
                    options.enable_profiling = True
                    prefix = Path(tempfile.gettempdir()) / (
                        f"mtp-sam3d-vith-{os.getpid()}-"
                        f"{threading.get_ident()}-{ordinal}"
                    )
                    options.profile_file_prefix = str(prefix)
                    self._profile_prefixes[graph.role] = prefix
                path = str(bundle.graph_path_for_role(graph.role))
                add_devices = (
                    getattr(options, "add_provider_for_devices", None)
                    if options is not None
                    else None
                )
                if hardware is not None and callable(add_devices):
                    add_devices(
                        [hardware],
                        {"preferredLayout": "NHWC", "enableGraphCapture": "0"},
                    )
                    sessions[graph.role] = factory(path, sess_options=options)
                elif options is not None:
                    sessions[graph.role] = factory(
                        path, sess_options=options, providers=providers
                    )
                else:
                    sessions[graph.role] = factory(path, providers=providers)
        except Exception:
            self._dispose_sessions(sessions, profiled=profile)
            raise
        return sessions

    def _verify_accelerator_profiles(self, *, full_inference: bool) -> None:
        if self._profile_failure is not None:
            raise self._profile_failure
        if not self._profile_requested or not full_inference:
            # A body-only request may run while profiling is active, but it can
            # never certify the release's full-hand 22-graph execution path.
            return

        expected_roles = set(required_graph_roles())
        executed_roles = set(self._runner.graphs.executed_roles)
        role_counts: dict[str, dict[str, object]] = {}
        errors: list[str] = []
        actual = self.diagnostics.actual_ep.casefold()
        if not any(name in actual for name in ("webgpu", "coreml")):
            errors.append(
                "selected provider is not a supported WebGPU/CoreML accelerator"
            )
        if executed_roles != expected_roles:
            missing = sorted(expected_roles - executed_roles)
            extra = sorted(executed_roles - expected_roles)
            errors.append(
                "full inference graph execution mismatch "
                f"(missing={missing}, extra={extra})"
            )

        for role, session in self._sessions.items():
            profile_path = None
            providers: list[str] = []
            try:
                profile_path = Path(session.end_profiling())
                payload = json.loads(profile_path.read_text(encoding="utf8"))
                if not isinstance(payload, list):
                    raise ValueError("profile root is not an event array")
                providers = [
                    str(item.get("args", {}).get("provider", ""))
                    for item in payload
                    if isinstance(item, dict)
                    and isinstance(item.get("args"), dict)
                    and item.get("args", {}).get("provider")
                ]
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                errors.append(f"{role}: could not read profile: {exc}")
            finally:
                if profile_path is not None and profile_path.is_file():
                    profile_path.unlink(missing_ok=True)
            provider_counts: dict[str, int] = {}
            for provider in providers:
                provider_counts[provider] = provider_counts.get(provider, 0) + 1
            accelerated = sum(provider.casefold() == actual for provider in providers)
            role_counts[role] = {
                "accelerator_nodes": accelerated,
                "profiled_nodes": len(providers),
                "providers": dict(sorted(provider_counts.items())),
            }
            if role in expected_roles and accelerated < 1:
                errors.append(f"{role}: no compute executed on the accelerator")
        self._profile_requested = False
        self._delete_profile_files()

        accelerated = sum(
            int(value["accelerator_nodes"]) for value in role_counts.values()
        )
        total = sum(int(value["profiled_nodes"]) for value in role_counts.values())
        fraction = accelerated / max(total, 1)
        minimum = 32 if "webgpu" in actual else 1
        minimum_fraction = 0.25 if "webgpu" in actual else 0.0
        if accelerated < minimum or fraction < minimum_fraction:
            errors.append(
                "aggregate accelerator execution was insufficient: "
                f"{accelerated}/{total} nodes"
            )

        self.accelerator_profile_evidence = {
            "passed": not errors,
            "provider": self.diagnostics.actual_ep,
            "full_inference": True,
            "executed_roles": sorted(executed_roles),
            "accelerator_nodes": accelerated,
            "profiled_nodes": total,
            "accelerator_fraction": fraction,
            "roles": role_counts,
        }
        if errors:
            failure = VithRuntimeError(
                "Could not verify full split accelerator execution: "
                + "; ".join(errors)
            )
            self._profile_failure = failure
            self._dispose_sessions(self._sessions, profiled=False)
            poisoned_runner = self._runner
            self._runner = None
            if poisoned_runner is not None:
                poisoned_runner.close()
            raise failure

        from dataclasses import replace

        self.diagnostics = replace(
            self.diagnostics,
            accelerator_nodes_verified=True,
            accelerator_node_count=accelerated,
            profiled_node_count=total,
        )

    def _can_fallback_after_inference_error(self) -> bool:
        return (
            self.requested_mode == "AUTO"
            and self.diagnostics.actual_ep.casefold() != "cpuexecutionprovider"
            and any(
                provider.casefold() == "cpuexecutionprovider"
                for provider in self._available_providers
            )
        )

    def _fallback_to_cpu(self, accelerator_error: Exception) -> None:
        old_runner = self._runner
        self._runner = None
        if old_runner is not None:
            old_runner.close()
        self._dispose_sessions(self._sessions, profiled=bool(self._profile_requested))
        self._profile_requested = False
        self._profile_failure = None
        try:
            self._sessions = self._create_sessions(
                self.bundle,
                self._session_factory,
                "CPUExecutionProvider",
                self._available_providers,
                None,
                profile=False,
            )
            if not self._all_sessions_use("CPUExecutionProvider"):
                raise VithRuntimeError(
                    "not every recreated session selected CPUExecutionProvider"
                )
            self._runner = VithSplitRunner(self.bundle, self._sessions)
        except Exception as cpu_exc:
            self._dispose_sessions(self._sessions, profiled=False)
            raise VithRuntimeError(
                "Accelerator inference failed and the complete CPU retry could "
                f"not be initialized: {cpu_exc}"
            ) from cpu_exc

        from dataclasses import replace

        self.diagnostics = replace(
            self.diagnostics,
            actual_ep="CPUExecutionProvider",
            actual_device="cpu",
            fallback=True,
            hardware_vendor_id=0,
            hardware_device_id=0,
            accelerator_nodes_verified=False,
            accelerator_node_count=0,
            profiled_node_count=0,
        )
        self.accelerator_profile_evidence = {
            "passed": False,
            "fallback": "CPUExecutionProvider",
            "accelerator_error": str(accelerator_error),
        }

    def _run_output(
        self,
        image_rgb: object,
        boxes: np.ndarray,
        *,
        include_hands: bool,
        camera_intrinsics: object | None = None,
    ) -> dict[str, np.ndarray]:
        if self._profile_failure is not None:
            raise self._profile_failure
        try:
            output = self._runner.run(
                image_rgb,
                boxes,
                include_hands=include_hands,
                camera_intrinsics=camera_intrinsics,
            )
        except GraphExecutionError as exc:
            if self._can_fallback_after_inference_error():
                self._fallback_to_cpu(exc)
                output = self._runner.run(
                    image_rgb,
                    boxes,
                    include_hands=include_hands,
                    camera_intrinsics=camera_intrinsics,
                )
            else:
                if self._profile_requested:
                    self._dispose_sessions(self._sessions, profiled=True)
                    self._profile_requested = False
                    poisoned_runner = self._runner
                    self._runner = None
                    if poisoned_runner is not None:
                        poisoned_runner.close()
                    failure = VithRuntimeError(
                        "Accelerator inference failed before execution could be "
                        f"verified: {exc}"
                    )
                    self._profile_failure = failure
                    raise failure from exc
                raise
        _ort_runtime._validate_output_tensors(np, output, len(boxes))
        self._verify_accelerator_profiles(full_inference=include_hands)
        return output

    def process_one_image(
        self,
        image_rgb,
        *,
        bboxes,
        inference_type: str = "full",
        camera_intrinsics=None,
    ) -> list[dict[str, np.ndarray]]:
        boxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
        if not len(boxes):
            return []
        if inference_type not in {"full", "body"}:
            raise VithRuntimeError(
                "Split estimator inference_type must be 'full' or 'body'"
            )
        output = self._run_output(
            image_rgb,
            boxes,
            include_hands=inference_type == "full",
            camera_intrinsics=camera_intrinsics,
        )
        return [
            {key: output[key][index] for key in OUTPUT_KEYS}
            for index in range(len(boxes))
        ]

    def close(self) -> None:
        # End active profiles solely to obtain and delete their files.  Close
        # must never leave potentially identifying temp paths behind.
        sessions = getattr(self, "_sessions", {})
        self._dispose_sessions(
            sessions,
            profiled=bool(getattr(self, "_profile_requested", False)),
        )
        self._profile_requested = False
        runner = getattr(self, "_runner", None)
        if runner is not None:
            runner.close()
        self._runner = None
        self._sessions = {}
        self._ort = None
        gc.collect()


def run_bundle(
    *,
    manifest_path: str | Path,
    image_rgb: object,
    bboxes_xyxy: object,
    include_hands: bool,
    providers: Sequence[str] | None,
    camera_intrinsics: object | None = None,
) -> dict[str, np.ndarray]:
    """One-shot publication/parity entry point with the stable 8-output API."""

    runtime = _ort_runtime
    bundle = runtime.load_bundle_manifest(manifest_path, verify_files=True)
    requested = "CPU"
    if providers and any(
        str(provider).casefold() != "cpuexecutionprovider" for provider in providers
    ):
        requested = "GPU"
    estimator = VithEstimator(
        bundle,
        requested,
        verify_accelerator_execution=requested == "GPU",
    )
    try:
        boxes = np.asarray(bboxes_xyxy, dtype=np.float32).reshape(-1, 4)
        return estimator._run_output(
            image_rgb,
            boxes,
            include_hands=bool(include_hands),
            camera_intrinsics=camera_intrinsics,
        )
    finally:
        estimator.close()
