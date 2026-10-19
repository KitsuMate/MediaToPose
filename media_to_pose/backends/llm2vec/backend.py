"""Download, verify, license and run the optional LLM2Vec text encoder.

The int4 ONNX encoder (5.8 GB) comes from
``KitsuMate/Llama-3-LLM2Vec-MNTP-Supervised-ONNX`` at a pinned commit. The
Meta Llama 3 Community License must be accepted for that revision before
anything is downloaded; LICENSE, LICENSE-LLM2VEC and NOTICE are installed
next to the model. Prompts go through the embedding cache first, so a cached
prompt never loads the encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...core.cache.embeddings import EmbeddingCache, EmbeddingKey
from ...core.coordinator import OperationKind
from ...core.errors import ModelMissing, MtpError
from ...core.motion.kimodo import prompt as prompt_layout
from .. import licenses, lifecycle
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
from . import runtime

LABEL = "LLM2Vec text encoder"
REPOSITORY = "KitsuMate/Llama-3-LLM2Vec-MNTP-Supervised-ONNX"
REVISION = "883615966398adc12e787d9b3a2dc4da26d4b44c"
VARIANT = "INT4"
# Decision 2 is pending: until a manifest is published in REPOSITORY, the
# file list lives in ``bundle_files`` below.
MANIFEST_PATH = ""
MODEL_STORAGE_FOLDER = "media_to_pose_llm2vec"
STORAGE = ManagedStorage(MODEL_STORAGE_FOLDER, "llm2vec_models", "LLM2Vec")
BUILT_WITH = "Built with Meta Llama 3"
LICENSE = licenses.ModelLicense(
    license_id="meta-llama-3-community-license",
    name="Meta Llama 3 Community License",
    summary=(
        "The LLM2Vec text encoder is Meta Llama 3 8B Instruct with McGill "
        "NLP's LLM2Vec adapter (MIT), converted to ONNX by KitsuMate. It is "
        "distributed under the Meta Llama 3 Community License and Acceptable "
        "Use Policy. Read the licence before accepting it. Built with Meta "
        "Llama 3."
    ),
    url="https://www.llama.com/llama3/license/",
    files=("LICENSE", "LICENSE-LLM2VEC", "NOTICE"),
)
FILES: tuple[BundleFile, ...] = (
    BundleFile(
        runtime.GRAPH,
        28_734_230,
        "d3ce3a3a23f927c2f18bcea950a1cef6557c73c4dc6b9efa5ffeba8d2ba5bd25",
    ),
    BundleFile(
        runtime.GRAPH_DATA,
        5_810_176_000,
        "97beb9d2a6b48076512c83cc8b7fac84418e5b4cef3b3ab7ac45fa6c5bce9c8b",
    ),
    BundleFile(
        runtime.TOKENIZER,
        9_085_671,
        "52b583ee1699c6ca4468f70885bd5f96dd1f7517292362a67581038afc04f4d8",
    ),
    BundleFile(
        "tokenizer_config.json",
        51_042,
        "0655c52e874137e4a455228155d982c4ea18419f461a10ceb57054995fec4223",
    ),
    BundleFile(
        "LICENSE",
        7_801,
        "475211637354ce4c14b9c3dacccbefbaba735fe1b0db97d6c4fd11cd1356819f",
    ),
    BundleFile(
        "LICENSE-LLM2VEC",
        1_067,
        "e69268828e67ffd1d3393a2d3530f97df6deaa0b961039fc971abea28cec8715",
    ),
    BundleFile(
        "NOTICE",
        401,
        "72cc8ff3394f16204a008e313ea6520ce3b3f1a9f6457b439944c70946b79e5c",
    ),
)


def bundle_files() -> list[BundleFile]:
    """Every file of the encoder bundle."""
    return list(FILES)


DOWNLOAD_BYTES = sum(item.bytes for item in FILES)


@dataclass(frozen=True)
class BackendSnapshot:
    installed: bool = False
    state: str = "NOT_INSTALLED"
    message: str = "The LLM2Vec text encoder is not installed"
    variant: str = VARIANT
    actual_device: str = "cpu"
    download_bytes: int = DOWNLOAD_BYTES
    installed_bytes: int = 0
    min_ram_bytes: int = runtime.MIN_RAM_BYTES
    revision: str = REVISION
    license_id: str = LICENSE.license_id
    license_state: str = "REQUIRED"
    license_accepted_at: str = ""
    license_url: str = LICENSE.url
    attribution: str = BUILT_WITH
    model_directory: str = ""


class LLM2VecBackend(ModelBackend):
    label = LABEL
    install_operation = OperationKind.LLM2VEC_INSTALL
    load_operation = OperationKind.LLM2VEC_ENCODE
    unload_operation = OperationKind.LLM2VEC_ENCODE
    uninstall_operation = OperationKind.LLM2VEC_UNINSTALL


BACKEND = LLM2VecBackend()


# Storage ---------------------------------------------------------------------


def backend_root(*, create: bool = True) -> Path:
    return lifecycle.backend_root("llm2vec", create=create)


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


def pinned_bundle() -> PinnedBundle:
    return PinnedBundle(
        label="LLM2Vec",
        repository=REPOSITORY,
        revision=REVISION,
        manifest_path=MANIFEST_PATH,
        list_files=lambda _staging: bundle_files(),
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


def invalidate_backend_snapshot() -> None:
    BACKEND.invalidate()


def start_install(
    storage_parent: str = "",
    *,
    online_access: bool,
    accept_license: str = "",
    source: licenses.AcceptanceSource = licenses.AcceptanceSource.API,
) -> BackgroundJob:
    """Start downloading the encoder on a worker thread.

    ``online_access`` is ``bpy.app.online_access``. Raises
    ``LicenseRequired`` unless the licence was accepted for this revision
    before, or ``accept_license`` accepts it now. An interrupted download
    resumes on the next call.
    """
    lifecycle.require_online_access(online_access, LABEL)
    root = model_storage_root(storage_parent)
    licenses.ensure_accepted(
        STORAGE, root, LICENSE, REVISION, accept=accept_license, source=source
    )
    config = _config_path()

    def work(job: BackgroundJob) -> str:
        unload_model()
        try:
            return lifecycle.install_pinned_bundle(
                job,
                pinned_bundle(),
                STORAGE,
                root,
                config,
                online_access=online_access,
            )
        finally:
            invalidate_backend_snapshot()

    return BACKEND.start_job(
        JobKind.INSTALL,
        OperationKind.LLM2VEC_INSTALL,
        "llm2vec-install",
        lifecycle.run_job,
        (work,),
        label="Starting the LLM2Vec download",
        kwargs={
            "describe_error": lambda exc, _phase: f"LLM2Vec installation failed: {exc}"
        },
    )


def cancel_install() -> bool:
    return BACKEND.cancel_job(JobKind.INSTALL, "Cancelling the LLM2Vec download")


# Status -------------------------------------------------------------------------


def _build_snapshot() -> BackendSnapshot:
    entry = license_acceptance()
    common = {
        "model_directory": str(active_directory()),
        "license_state": "ACCEPTED" if entry else "REQUIRED",
        "license_accepted_at": str((entry or {}).get("accepted_at", "")),
    }
    if not backend_present():
        return BackendSnapshot(**common)
    try:
        bundle = runtime.load_bundle(active_directory(), verify_files=False)
    except (MtpError, OSError, ValueError) as exc:
        return BackendSnapshot(
            state="ERROR",
            message=f"The LLM2Vec text encoder is invalid; use Repair ({exc})",
            **common,
        )
    encoder = BACKEND.session
    loaded = encoder is not None and encoder.loaded
    return BackendSnapshot(
        installed=True,
        state="LOADED" if loaded else "READY",
        message="LLM2Vec · CPU · " + ("Loaded" if loaded else "Ready"),
        installed_bytes=bundle.installed_bytes,
        **common,
    )


def backend_snapshot() -> BackendSnapshot:
    return BACKEND.cached_snapshot((), _build_snapshot)


# Encoding --------------------------------------------------------------------------


def embedding_key(text: str) -> EmbeddingKey:
    """The cache key of an already sanitised prompt."""
    return EmbeddingKey(REVISION, VARIANT, prompt_layout.TEMPLATE_VERSION, text)


def ensure_encoder(*, keep_loaded: bool = False) -> runtime.LLM2VecEncoder:
    """The encoder of the installed bundle; its session opens on demand."""
    encoder = BACKEND.session
    if encoder is None:
        if not backend_present():
            raise ModelMissing("The LLM2Vec text encoder is not installed")
        bundle = runtime.load_bundle(active_directory(), verify_files=False)
        encoder = BACKEND.set_session(runtime.LLM2VecEncoder(bundle))
    encoder.keep_loaded = keep_loaded
    return encoder


def encode_prompt(
    prompt: str,
    *,
    cache: EmbeddingCache | None = None,
    keep_loaded: bool = False,
) -> runtime.Encoding | None:
    """The embedding of ``prompt``: from the cache, else from the encoder.

    ``None`` for an empty prompt, which needs no encoder. A new embedding is
    stored in the cache. The encoder session is released afterwards unless
    ``keep_loaded`` is set.
    """
    text = prompt_layout.sanitize(prompt)
    if not text:
        return None
    cache = cache or EmbeddingCache()
    key = embedding_key(text)
    cached = cache.load(key)
    if cached is not None:
        return runtime.Encoding(cached.vector, cached.text, cached.truncated)
    try:
        encoding = ensure_encoder(keep_loaded=keep_loaded).encode_prompt(text)
    finally:
        invalidate_backend_snapshot()
    cache.store(key, encoding.vector, text=encoding.text, truncated=encoding.truncated)
    return encoding


def is_model_loaded() -> bool:
    encoder = BACKEND.session
    return bool(encoder is not None and encoder.loaded)


def unload_model() -> None:
    encoder = BACKEND.take_session()
    if encoder is not None:
        encoder.close()


def verify() -> str:
    """Re-hash every installed file."""

    def action() -> str:
        if not backend_present():
            raise ModelMissing("The LLM2Vec text encoder is not installed")
        runtime.load_bundle(active_directory())
        invalidate_backend_snapshot()
        return "LLM2Vec files verified"

    return BACKEND.run_exclusive(
        OperationKind.LLM2VEC_ENCODE, "llm2vec-verify", action, status="Verifying"
    )


def unload() -> str:
    def action() -> str:
        unload_model()
        return "LLM2Vec unloaded"

    return BACKEND.run_exclusive(
        OperationKind.LLM2VEC_ENCODE, "llm2vec-unload", action, status="Unloading"
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
        return "LLM2Vec uninstalled"

    return BACKEND.run_exclusive(
        OperationKind.LLM2VEC_UNINSTALL,
        "llm2vec-uninstall",
        action,
        status="Uninstalling the LLM2Vec text encoder",
    )


def shutdown() -> None:
    BACKEND.shutdown()
    unload_model()


__all__ = [
    "BACKEND",
    "BUILT_WITH",
    "DOWNLOAD_BYTES",
    "LICENSE",
    "REPOSITORY",
    "REVISION",
    "VARIANT",
    "BackendSnapshot",
    "accept_license",
    "backend_present",
    "backend_snapshot",
    "bundle_files",
    "cancel_install",
    "embedding_key",
    "encode_prompt",
    "ensure_encoder",
    "install_status_snapshot",
    "is_model_loaded",
    "license_acceptance",
    "pinned_bundle",
    "shutdown",
    "start_install",
    "uninstall",
    "unload",
    "verify",
]
