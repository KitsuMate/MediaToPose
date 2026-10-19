"""Download, verify, license and load the optional Kimodo denoiser.

The denoiser comes from ``KitsuMate/Kimodo-SOMA-RP-v1.1-ONNX`` and the
normalisation statistics from ``nvidia/Kimodo-SOMA-RP-v1.1``, both at pinned
commits. One variant is installed at a time. The NVIDIA Open Model License
must be accepted for the pinned revision before anything is downloaded, and
its LICENSE and NOTICE are installed next to the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ...core.coordinator import OperationKind
from ...core.errors import (
    Cancelled,
    InsufficientResources,
    InvalidRequest,
    ModelMissing,
    MtpError,
)
from .. import licenses, lifecycle, resources
from ..lifecycle import (
    BackgroundJob,
    BundleFile,
    JobKind,
    JobSnapshot,
    ManagedStorage,
    ModelBackend,
    PinnedBundle,
)
from ..sam3d import backend as sam3d_backend
from ..sam3d import ort_runtime
from . import runtime

LABEL = "Kimodo"
REPOSITORY = "KitsuMate/Kimodo-SOMA-RP-v1.1-ONNX"
REVISION = "7520fac9267951878686b416c52c4eda55832e1e"
STATS_REPOSITORY = "nvidia/Kimodo-SOMA-RP-v1.1"
STATS_REVISION = "6c9233af1180b8151e3c4703477104af5dce9dd5"
# Decision 1 is pending: until a manifest with the statistics is published in
# REPOSITORY, the file list lives in ``bundle_files`` below.
MANIFEST_PATH = ""
MODEL_STORAGE_FOLDER = "media_to_pose_kimodo"
STORAGE = ManagedStorage(MODEL_STORAGE_FOLDER, "kimodo_models", LABEL)
LICENSE = licenses.ModelLicense(
    license_id="nvidia-open-model-license",
    name="NVIDIA Open Model License",
    summary=(
        "Kimodo-SOMA-RP v1.1 by NVIDIA, converted to ONNX by KitsuMate, is "
        "distributed under the NVIDIA Open Model License. Read the licence "
        "before accepting it."
    ),
    url=(
        "https://www.nvidia.com/en-us/agreements/enterprise-software/"
        "nvidia-open-model-license/"
    ),
)


class KimodoVariant(str, Enum):
    AUTO = "AUTO"
    FP16 = "FP16"
    FP32 = "FP32"
    INT8 = "INT8"


@dataclass(frozen=True)
class VariantSpec:
    label: str
    description: str
    graph: BundleFile
    min_ram_bytes: int


VARIANTS: dict[KimodoVariant, VariantSpec] = {
    KimodoVariant.FP16: VariantSpec(
        "FP16",
        "Half precision; the fastest on WebGPU and CoreML",
        BundleFile(
            "onnx/model_fp16.onnx",
            587_590_790,
            "c4a4aee64a6e7a2d1551d6382d7cd89e35f3d5e0e50b6c62728bc5563b839497",
        ),
        2 * resources.GIB,
    ),
    KimodoVariant.FP32: VariantSpec(
        "FP32",
        "Full precision; the fastest and most exact on CPU",
        BundleFile(
            "onnx/model_fp32.onnx",
            1_174_634_698,
            "07341f1d89bea7a2a392a63040f39c048f8b0638ee1f50a5a0988bb8a9d7109b",
        ),
        2 * resources.GIB,
    ),
    KimodoVariant.INT8: VariantSpec(
        "INT8",
        "8-bit weights; half the memory, slower on CPU",
        BundleFile(
            "onnx/model_int8.onnx",
            305_933_441,
            "4c59427c7d9a33feddd50a184b806ff135b53408c1772ccd0f2a8fd1f8a44930",
        ),
        1 * resources.GIB,
    ),
}


def _stats_file(relative: str, size: int, sha256: str) -> BundleFile:
    return BundleFile(
        relative,
        size,
        sha256,
        repository=STATS_REPOSITORY,
        revision=STATS_REVISION,
    )


SHARED_FILES: tuple[BundleFile, ...] = (
    BundleFile(
        "LICENSE",
        10_229,
        "11ccdcb47db87ac30b63765b5e5154403026c4fbb52c19919d99d912ec454e19",
    ),
    BundleFile(
        "NOTICE",
        215,
        "31a9247658055dfbd9259fbab083ee154ab9c86c3719a6a142bebf644de89f6a",
    ),
    _stats_file(
        runtime.STATS_FILES["global_root_mean"],
        168,
        "6c110980f9a0386e57a8c186f8eac7f55dc05bade92f44c4b391450ada598da1",
    ),
    _stats_file(
        runtime.STATS_FILES["global_root_std"],
        168,
        "fb15aeed7143f5f0754611f0acf5fa5f55d1f6492a49c23e490fc62f7bf11fb0",
    ),
    _stats_file(
        runtime.STATS_FILES["body_mean"],
        3_040,
        "d42f35f8bb7ef52fdec66ae29c5cf590bd038f4f1aabd2d560d1d0f78663ed6f",
    ),
    _stats_file(
        runtime.STATS_FILES["body_std"],
        3_040,
        "965bbad497726ef4e6c46b519827e2cbcb29f974214b25718d20a725b82279ed",
    ),
)


def bundle_files(variant: KimodoVariant) -> list[BundleFile]:
    """Every file of one variant's bundle: the graph, statistics and licences."""
    return [VARIANTS[variant].graph, *SHARED_FILES]


def download_bytes(variant: KimodoVariant) -> int:
    return sum(item.bytes for item in bundle_files(variant))


@dataclass(frozen=True)
class BackendSnapshot:
    installed: bool = False
    state: str = "NOT_INSTALLED"
    message: str = "Kimodo is not installed"
    variant: str = ""
    requested_variant: str = "AUTO"
    requested_mode: str = "AUTO"
    actual_ep: str = ""
    actual_device: str = ""
    fallback: bool = False
    download_bytes: int = 0
    installed_bytes: int = 0
    min_ram_bytes: int = 0
    revision: str = REVISION
    license_id: str = LICENSE.license_id
    license_state: str = "REQUIRED"
    license_accepted_at: str = ""
    license_url: str = LICENSE.url
    model_directory: str = ""


class KimodoBackend(ModelBackend):
    label = LABEL
    install_operation = OperationKind.KIMODO_INSTALL
    load_operation = OperationKind.KIMODO_MODEL_LOAD
    unload_operation = OperationKind.KIMODO_MODEL_UNLOAD
    uninstall_operation = OperationKind.KIMODO_UNINSTALL


BACKEND = KimodoBackend()


def kimodo_variant_enum_items():
    return (
        ("AUTO", "Auto", "FP16 with a GPU, FP32 on CPU"),
        *(
            (variant.value, spec.label, spec.description)
            for variant, spec in VARIANTS.items()
        ),
    )


def normalize_variant(value: object) -> KimodoVariant:
    try:
        return KimodoVariant(str(value or "AUTO").upper())
    except ValueError:
        raise InvalidRequest(f"Unknown Kimodo variant {value!r}") from None


def resolve_variant(variant: object, device: object = "AUTO") -> KimodoVariant:
    """A concrete variant: AUTO is FP16 with an accelerator and FP32 on CPU."""
    selected = normalize_variant(variant)
    if selected is not KimodoVariant.AUTO:
        return selected
    try:
        available = ort_runtime.available_execution_providers()
        _provider, actual, _fallback = ort_runtime.select_execution_provider(
            device, available
        )
    except MtpError:
        actual = "cpu"
    return KimodoVariant.FP32 if actual == "cpu" else KimodoVariant.FP16


# Storage ---------------------------------------------------------------------


def backend_root(*, create: bool = True) -> Path:
    return lifecycle.backend_root("kimodo", create=create)


def _config_path(*, create: bool = True) -> Path:
    return backend_root(create=create) / "storage.json"


def configured_storage_parent() -> Path:
    configured = lifecycle.read_storage_parent(_config_path(create=False))
    if configured:
        return lifecycle.resolve_user_path(configured)
    return sam3d_backend.preferred_model_storage_parent().resolve(False)


def model_storage_root(parent: str = "") -> Path:
    selected = (
        lifecycle.resolve_user_path(parent) if parent else configured_storage_parent()
    )
    return STORAGE.root(selected)


def active_directory() -> Path:
    return model_storage_root() / "active"


def backend_present() -> bool:
    return (active_directory() / "manifest.json").is_file()


def pinned_bundle(variant: KimodoVariant) -> PinnedBundle:
    return PinnedBundle(
        label=LABEL,
        repository=REPOSITORY,
        revision=REVISION,
        manifest_path=MANIFEST_PATH,
        list_files=lambda _staging: bundle_files(variant),
        verify=runtime.load_bundle,
    )


# Licence ---------------------------------------------------------------------


def license_acceptance(parent: str = "") -> dict[str, str] | None:
    return licenses.acceptance(STORAGE, model_storage_root(parent), LICENSE, REVISION)


def accept_license(
    accept: str, source: licenses.AcceptanceSource, parent: str = ""
) -> dict[str, str]:
    """Record the user's acceptance; ``accept`` is ``LICENSE.accept_value``."""
    entry = licenses.ensure_accepted(
        STORAGE,
        model_storage_root(parent),
        LICENSE,
        REVISION,
        accept=accept,
        source=source,
    )
    invalidate_backend_snapshot()
    return entry


# Install -----------------------------------------------------------------------


def install_status_snapshot() -> JobSnapshot:
    return BACKEND.install_status_snapshot()


def model_load_status_snapshot() -> JobSnapshot:
    return BACKEND.model_load_status_snapshot()


def invalidate_backend_snapshot() -> None:
    BACKEND.invalidate()


def start_install(
    variant: object = "AUTO",
    device: object = "AUTO",
    storage_parent: str = "",
    *,
    online_access: bool,
    accept_license: str = "",
    source: licenses.AcceptanceSource = licenses.AcceptanceSource.API,
) -> BackgroundJob:
    """Start downloading one Kimodo variant on a worker thread.

    ``online_access`` is ``bpy.app.online_access``. Raises
    ``LicenseRequired`` unless the licence was accepted for this revision
    before, or ``accept_license`` accepts it now.
    """
    lifecycle.require_online_access(online_access, LABEL)
    root = model_storage_root(storage_parent)
    licenses.ensure_accepted(
        STORAGE, root, LICENSE, REVISION, accept=accept_license, source=source
    )
    selected = resolve_variant(variant, device)
    config = _config_path()

    def work(job: BackgroundJob) -> str:
        unload_model()
        try:
            return lifecycle.install_pinned_bundle(
                job,
                pinned_bundle(selected),
                STORAGE,
                root,
                config,
                online_access=online_access,
            )
        finally:
            invalidate_backend_snapshot()

    return BACKEND.start_job(
        JobKind.INSTALL,
        OperationKind.KIMODO_INSTALL,
        "kimodo-install",
        lifecycle.run_job,
        (work,),
        label=f"Starting {LABEL} {selected.value} download",
        kwargs={
            "describe_error": lambda exc, _phase: f"Kimodo installation failed: {exc}"
        },
    )


def cancel_install() -> bool:
    return BACKEND.cancel_job(JobKind.INSTALL, "Cancelling Kimodo download")


# Status -------------------------------------------------------------------------


def _license_fields() -> dict[str, str]:
    entry = license_acceptance()
    if entry is None:
        return {"license_state": "REQUIRED"}
    return {
        "license_state": "ACCEPTED",
        "license_accepted_at": str(entry.get("accepted_at", "")),
    }


def _build_snapshot(requested_variant: str, requested_mode: str) -> BackendSnapshot:
    common = {
        "requested_variant": requested_variant,
        "requested_mode": requested_mode,
        "model_directory": str(active_directory()),
        **_license_fields(),
    }
    session = BACKEND.session
    if session is not None:
        diagnostic = session.diagnostics
        spec = VARIANTS[KimodoVariant(diagnostic.variant)]
        return BackendSnapshot(
            installed=True,
            state="LOADED",
            message=f"Kimodo {diagnostic.variant} · Loaded",
            variant=diagnostic.variant,
            actual_ep=diagnostic.actual_ep,
            actual_device=diagnostic.actual_device,
            fallback=diagnostic.fallback,
            installed_bytes=diagnostic.installed_bytes,
            min_ram_bytes=spec.min_ram_bytes,
            **common,
        )
    if not backend_present():
        wanted = resolve_variant(requested_variant, requested_mode)
        return BackendSnapshot(
            variant=wanted.value,
            download_bytes=download_bytes(wanted),
            min_ram_bytes=VARIANTS[wanted].min_ram_bytes,
            **common,
        )
    try:
        bundle = runtime.load_bundle(active_directory(), verify_files=False)
        available = ort_runtime.available_execution_providers()
        provider, device, fallback = ort_runtime.select_execution_provider(
            requested_mode, available
        )
    except (MtpError, OSError, ValueError) as exc:
        return BackendSnapshot(
            state="ERROR", message=f"Kimodo is invalid; use Repair ({exc})", **common
        )
    return BackendSnapshot(
        installed=True,
        state="READY",
        message=(
            f"Kimodo {bundle.variant} · {device.upper()}"
            + (" · CPU fallback" if fallback else "")
            + " · Ready"
        ),
        variant=bundle.variant,
        actual_ep=provider,
        actual_device=device,
        fallback=fallback,
        installed_bytes=bundle.installed_bytes,
        min_ram_bytes=VARIANTS[KimodoVariant(bundle.variant)].min_ram_bytes,
        **common,
    )


def backend_snapshot(
    requested_variant: object = "AUTO", requested_mode: object = "AUTO"
) -> BackendSnapshot:
    variant = normalize_variant(requested_variant).value
    mode = ort_runtime.normalize_mode(requested_mode)
    return BACKEND.cached_snapshot(
        (variant, mode), lambda: _build_snapshot(variant, mode)
    )


# Load ----------------------------------------------------------------------------


def _require_ram(variant: KimodoVariant) -> None:
    spec = VARIANTS[variant]
    try:
        resources.require_ram(spec.min_ram_bytes, f"Kimodo {variant.value}")
    except InsufficientResources as exc:
        small = VARIANTS[KimodoVariant.INT8].min_ram_bytes
        if (
            variant is not KimodoVariant.INT8
            and exc.details["available_bytes"] >= small
        ):
            exc.details["fitting_variant"] = KimodoVariant.INT8.value
        raise


def verify_bundle(bundle_directory: str = "") -> runtime.KimodoBundle:
    invalidate_backend_snapshot()
    directory = Path(bundle_directory) if bundle_directory else active_directory()
    if not (directory / "manifest.json").is_file():
        raise ModelMissing("Kimodo is not installed")
    return runtime.load_bundle(directory)


def loaded_session() -> runtime.KimodoSession:
    session = BACKEND.session
    if session is None:
        raise ModelMissing("Kimodo is not loaded")
    return session


def is_model_loaded() -> bool:
    return BACKEND.is_model_loaded()


def ensure_model_loaded(
    device: object = "AUTO", bundle_directory: str = ""
) -> runtime.KimodoSession:
    """The loaded denoiser on ``device``, loading the installed bundle if needed."""
    requested = ort_runtime.normalize_mode(device)
    current = BACKEND.session
    if current is not None and current.requested_mode == requested:
        return current
    unload_model()
    bundle = verify_bundle(bundle_directory)
    _require_ram(KimodoVariant(bundle.variant))
    session = BACKEND.set_session(runtime.KimodoSession(bundle, requested))
    BACKEND.last_diagnostic = "Bundle verification and runtime session creation passed"
    return session


def start_load(device: object = "AUTO") -> BackgroundJob:
    bundle_directory = str(active_directory())

    def work(job: BackgroundJob) -> str:
        job.update(0.15, "Verifying model files", "Verification")
        job.check_cancelled("Kimodo load cancelled")
        job.update(0.45, "Creating the ONNX Runtime session", "Loading")
        ensure_model_loaded(device, bundle_directory)
        if job.cancel.is_set():
            unload_model()
            raise Cancelled("Kimodo load cancelled")
        job.update(1.0, "Kimodo loaded", "Complete")
        return "Kimodo loaded"

    def describe(exc: BaseException, _phase: str) -> str:
        BACKEND.last_diagnostic = str(exc)
        return f"Kimodo loading failed: {exc}"

    return BACKEND.start_job(
        JobKind.MODEL,
        OperationKind.KIMODO_MODEL_LOAD,
        "kimodo-load",
        lifecycle.run_job,
        (work,),
        label="Loading Kimodo",
        kwargs={"describe_error": describe},
    )


def unload_model() -> None:
    session = BACKEND.take_session()
    if session is not None:
        session.close()


def last_diagnostic() -> str:
    return BACKEND.last_diagnostic


def verify(device: object = "AUTO") -> str:
    """Re-hash every file and create a session, then restore the load state."""

    def action() -> str:
        was_loaded = is_model_loaded()
        try:
            verify_bundle()
            ensure_model_loaded(device)
            BACKEND.last_diagnostic = (
                "All files passed size/SHA-256 verification and runtime session "
                "creation passed"
            )
        except (MtpError, OSError, ValueError) as exc:
            BACKEND.last_diagnostic = str(exc)
            raise
        finally:
            if not was_loaded:
                unload_model()
        return backend_snapshot("AUTO", device).message

    return BACKEND.run_exclusive(
        OperationKind.KIMODO_MODEL_LOAD,
        "kimodo-verify",
        action,
        status="Verifying Kimodo",
    )


def unload() -> str:
    def action() -> str:
        unload_model()
        return "Kimodo unloaded"

    return BACKEND.run_exclusive(
        OperationKind.KIMODO_MODEL_UNLOAD,
        "kimodo-unload",
        action,
        status="Unloading Kimodo",
    )


def uninstall() -> str:
    def action() -> str:
        root = model_storage_root().resolve(False)
        with lifecycle.model_installation_lock(root, LABEL):
            unload_model()
            try:
                STORAGE.remove(root)
            finally:
                invalidate_backend_snapshot()
        return "Kimodo uninstalled"

    return BACKEND.run_exclusive(
        OperationKind.KIMODO_UNINSTALL,
        "kimodo-uninstall",
        action,
        status="Uninstalling Kimodo",
    )


def shutdown() -> None:
    BACKEND.shutdown()
    unload_model()


__all__ = [
    "BACKEND",
    "LICENSE",
    "REPOSITORY",
    "REVISION",
    "STATS_REPOSITORY",
    "STATS_REVISION",
    "VARIANTS",
    "BackendSnapshot",
    "KimodoVariant",
    "accept_license",
    "backend_present",
    "backend_snapshot",
    "bundle_files",
    "cancel_install",
    "download_bytes",
    "ensure_model_loaded",
    "install_status_snapshot",
    "is_model_loaded",
    "kimodo_variant_enum_items",
    "last_diagnostic",
    "license_acceptance",
    "loaded_session",
    "model_load_status_snapshot",
    "pinned_bundle",
    "resolve_variant",
    "shutdown",
    "start_install",
    "start_load",
    "uninstall",
    "unload",
    "verify",
]
