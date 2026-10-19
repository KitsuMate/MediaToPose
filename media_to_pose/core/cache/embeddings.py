"""Prompt embedding cache: one npz plus a JSON manifest per prompt.

A text embedding depends on the encoder revision and variant, the prompt
template version and the sanitised prompt; the cache key is the SHA-256 of
those four joined by ``|``. A cached prompt never loads the 5.8 GB encoder.
Entries are about 16 KB and live in ``embeddings/`` under the detection
cache root. The JSON is written after the npz and repeats the key fields, so
an entry counts only when both files exist and agree. When a size limit is
given, the least recently used entries are removed beyond it.
"""

from __future__ import annotations

import hashlib
import os
import threading
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..fs import atomic_json, atomic_npz, read_json
from ..log import get_logger
from . import store

LOGGER = get_logger(__name__)

SCHEMA_VERSION = 1
EMBEDDING_WIDTH = 4096
DIRECTORY = "embeddings"
_LOCK = threading.RLock()


@dataclass(frozen=True)
class EmbeddingKey:
    """What an embedding depends on; ``prompt`` is already sanitised."""

    encoder_revision: str
    variant: str
    template_version: int
    prompt: str

    @property
    def digest(self) -> str:
        text = "|".join(
            (
                self.encoder_revision,
                self.variant,
                str(int(self.template_version)),
                self.prompt,
            )
        )
        return hashlib.sha256(text.encode("utf8")).hexdigest()

    def fields(self) -> dict[str, object]:
        return {
            "encoder_revision": self.encoder_revision,
            "variant": self.variant,
            "template_version": int(self.template_version),
            "prompt": self.prompt,
        }


@dataclass(frozen=True)
class CachedEmbedding:
    """A pooled [4096] float32 vector, with the text that was encoded.

    ``text`` differs from the key's prompt when the prompt was truncated to
    fit the encoder.
    """

    vector: np.ndarray
    text: str
    truncated: bool


def default_directory(*, create: bool = True) -> Path:
    return store.cache_root(create=create) / DIRECTORY


def _vector(value: object) -> np.ndarray | None:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape != (EMBEDDING_WIDTH,) or not np.all(np.isfinite(vector)):
        return None
    return vector


class EmbeddingCache:
    """Embeddings keyed by ``EmbeddingKey`` in ``directory``.

    ``limit_bytes`` (0 for none) bounds the folder; the least recently used
    entries are removed first.
    """

    def __init__(self, directory: Path | None = None, *, limit_bytes: int = 0):
        self.directory = Path(directory) if directory else default_directory()
        self.limit_bytes = max(0, int(limit_bytes))

    def _paths(self, key: EmbeddingKey) -> tuple[Path, Path]:
        stem = self.directory / key.digest
        return stem.with_suffix(".npz"), stem.with_suffix(".json")

    def load(self, key: EmbeddingKey) -> CachedEmbedding | None:
        arrays_path, manifest_path = self._paths(key)
        manifest = read_json(manifest_path)
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("key") != key.fields()
        ):
            return None
        try:
            with np.load(arrays_path, allow_pickle=False) as arrays:
                vector = _vector(arrays["embedding"])
        except (OSError, ValueError, KeyError) as exc:
            LOGGER.warning("Ignoring unreadable cached embedding: %s", exc)
            return None
        if vector is None:
            return None
        with suppress(OSError):
            os.utime(manifest_path)  # Recently used: pruned last.
        return CachedEmbedding(
            vector,
            str(manifest.get("text") or key.prompt),
            bool(manifest.get("truncated")),
        )

    def store(
        self,
        key: EmbeddingKey,
        vector: np.ndarray,
        *,
        text: str = "",
        truncated: bool = False,
    ) -> CachedEmbedding:
        checked = _vector(vector)
        if checked is None:
            raise ValueError(f"Embeddings must be {EMBEDDING_WIDTH} finite values")
        arrays_path, manifest_path = self._paths(key)
        with _LOCK:
            self.directory.mkdir(parents=True, exist_ok=True)
            atomic_npz(arrays_path, {"embedding": checked})
            atomic_json(
                manifest_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "key": key.fields(),
                    "text": text or key.prompt,
                    "truncated": bool(truncated),
                },
            )
            self.prune()
        return CachedEmbedding(checked, text or key.prompt, bool(truncated))

    def _entries(self) -> list[tuple[float, int, Path]]:
        entries = []
        for manifest_path in self.directory.glob("*.json"):
            arrays_path = manifest_path.with_suffix(".npz")
            try:
                used = manifest_path.stat().st_mtime
                size = manifest_path.stat().st_size + (
                    arrays_path.stat().st_size if arrays_path.exists() else 0
                )
            except OSError:
                continue
            entries.append((used, size, manifest_path))
        return sorted(entries)

    def usage_bytes(self) -> int:
        return sum(size for _used, size, _path in self._entries())

    def prune(self) -> None:
        """Remove the least recently used entries beyond ``limit_bytes``."""
        if not self.limit_bytes:
            return
        with _LOCK:
            entries = self._entries()
            total = sum(size for _used, size, _path in entries)
            for _used, size, manifest_path in entries[:-1]:
                if total <= self.limit_bytes:
                    break
                for path in (manifest_path, manifest_path.with_suffix(".npz")):
                    with suppress(OSError):
                        path.unlink()
                total -= size

    def clear(self) -> None:
        with _LOCK:
            for pattern in ("*.json", "*.npz"):
                for path in self.directory.glob(pattern):
                    with suppress(OSError):
                        path.unlink()


__all__ = [
    "EMBEDDING_WIDTH",
    "SCHEMA_VERSION",
    "CachedEmbedding",
    "EmbeddingCache",
    "EmbeddingKey",
    "default_directory",
]
