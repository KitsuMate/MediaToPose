"""The LLM2Vec text encoder: tokenizer plus an int4 ONNX graph on CPU.

The graph takes ``input_ids``, ``attention_mask`` and ``pooling_mask``
(int64, [1, 64]) and returns one pooled [1, 4096] vector. The prompt layout
(sanitising, the Llama 3 template, left padding and the pooling mask) comes
from ``core.motion.kimodo.prompt``; this module only tokenizes and runs.

The ONNX session takes about 4.5 GB of memory, so it is created on demand
and released straight after encoding unless the encoder is kept loaded.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...core import dependencies
from ...core.errors import BackendError
from ...core.log import get_logger
from ...core.motion.kimodo import prompt as prompt_layout
from .. import lifecycle, resources

LOGGER = get_logger(__name__)

GRAPH = "onnx/model_int4_b128_fp32act_s64.onnx"
GRAPH_DATA = f"{GRAPH}_data"
TOKENIZER = "tokenizer.json"
EMBEDDING_WIDTH = 4096
# K0 measured a 4.7 GB peak working set while loading and encoding; this
# leaves headroom for Blender itself.
MIN_RAM_BYTES = 6 * resources.GIB


class LLM2VecRuntimeError(BackendError):
    """An LLM2Vec bundle contract or inference failure."""


@dataclass(frozen=True)
class EncoderBundle:
    root: Path
    revision: str
    files: tuple[lifecycle.BundleFile, ...]

    @property
    def installed_bytes(self) -> int:
        return sum(item.bytes for item in self.files)

    def resolve(self, relative: str) -> Path:
        return lifecycle.contained_path(self.root, relative, label="LLM2Vec bundle")


def load_bundle(root: Path, *, verify_files: bool = True) -> EncoderBundle:
    """Read ``root/manifest.json`` and check that every listed file is intact."""
    try:
        revision, files = lifecycle.read_code_manifest(
            root, verify_files=verify_files, label="LLM2Vec"
        )
    except BackendError as exc:
        raise LLM2VecRuntimeError(str(exc)) from exc
    missing = sorted({GRAPH, GRAPH_DATA, TOKENIZER} - {item.path for item in files})
    if missing:
        raise LLM2VecRuntimeError(f"LLM2Vec bundle lacks {', '.join(missing)}")
    return EncoderBundle(Path(root), revision, files)


def load_tokenizer(
    bundle: EncoderBundle, tokenizers_module: Any = None
) -> prompt_layout.Tokenize:
    """``tokenize(text, add_special_tokens) -> ids`` from ``tokenizer.json``."""
    module = tokenizers_module or dependencies.import_tokenizers()
    try:
        tokenizer = module.Tokenizer.from_file(str(bundle.resolve(TOKENIZER)))
    except Exception as exc:  # The Rust tokenizer raises its own exception types.
        LOGGER.exception("LLM2Vec tokenizer could not be loaded")
        raise LLM2VecRuntimeError(f"LLM2Vec tokenizer: {exc}") from exc

    def tokenize(text: str, add_special_tokens: bool) -> list[int]:
        return list(tokenizer.encode(text, add_special_tokens=add_special_tokens).ids)

    return tokenize


@dataclass(frozen=True)
class Encoding:
    """A pooled [4096] float32 vector and the sanitised text it encodes."""

    vector: np.ndarray
    text: str
    truncated: bool


class LLM2VecEncoder:
    """Tokenizer and on-demand CPU session for one installed bundle.

    ``keep_loaded`` holds the session between prompts (batch work); by
    default it is released after every ``encode``. ``ort_module``,
    ``session_factory`` and ``tokenize`` are injectable for tests.
    """

    def __init__(
        self,
        bundle: EncoderBundle,
        *,
        keep_loaded: bool = False,
        ort_module: Any = None,
        session_factory: Any = None,
        tokenize: prompt_layout.Tokenize | None = None,
    ) -> None:
        self.bundle = bundle
        self.keep_loaded = keep_loaded
        self._ort = ort_module
        self._factory = session_factory
        self._tokenize = tokenize or load_tokenizer(bundle)
        self._session: Any = None
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def open(self) -> None:
        """Create the ONNX session after checking available memory."""
        with self._lock:
            if self._session is not None:
                return
            resources.require_ram(MIN_RAM_BYTES, "The LLM2Vec text encoder")
            ort = self._ort or dependencies.import_onnxruntime()
            factory = self._factory or ort.InferenceSession
            options = ort.SessionOptions()
            options.log_severity_level = 3
            try:
                self._session = factory(
                    str(self.bundle.resolve(GRAPH)),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
            except Exception as exc:  # ONNX Runtime raises its own exception types.
                LOGGER.exception("LLM2Vec session creation failed")
                raise LLM2VecRuntimeError(
                    f"The LLM2Vec session could not be created: {exc}"
                ) from exc

    def close(self) -> None:
        with self._lock:
            self._session = None

    def encode_prompt(self, prompt: str) -> Encoding:
        """Encode one prompt; raises ``InvalidRequest`` for an empty prompt."""
        inputs = prompt_layout.encoder_inputs(prompt, self._tokenize)
        with self._lock:
            self.open()
            try:
                output = self._session.run(
                    None,
                    {
                        "input_ids": inputs.input_ids,
                        "attention_mask": inputs.attention_mask,
                        "pooling_mask": inputs.pooling_mask,
                    },
                )[0]
            except Exception as exc:  # ONNX Runtime raises its own exception types.
                LOGGER.exception("LLM2Vec inference failed")
                raise LLM2VecRuntimeError(f"LLM2Vec inference failed: {exc}") from exc
            finally:
                if not self.keep_loaded:
                    self.close()
        vector = np.asarray(output, dtype=np.float32).reshape(-1)
        if vector.shape != (EMBEDDING_WIDTH,) or not np.all(np.isfinite(vector)):
            raise LLM2VecRuntimeError("LLM2Vec returned an invalid embedding")
        return Encoding(vector, inputs.text, inputs.truncated)

    def encode(self, prompt: str) -> np.ndarray:
        """The pooled [4096] float32 embedding of ``prompt``."""
        return self.encode_prompt(prompt).vector


__all__ = [
    "EMBEDDING_WIDTH",
    "GRAPH",
    "MIN_RAM_BYTES",
    "TOKENIZER",
    "EncoderBundle",
    "Encoding",
    "LLM2VecEncoder",
    "LLM2VecRuntimeError",
    "load_bundle",
    "load_tokenizer",
]
