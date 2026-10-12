"""Temporary, append-only frame stream stored as chunked npz files.

Used where a capture must hold every detected frame before retargeting (the
worker hand-off without a retained cache, and automatic root motion, which
needs the whole clip before the first frame is solved). Frames are encoded
losslessly with the cache codec; nothing is pickled.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

from ...models import CanonicalFrame
from ..fs import remove_tree
from .frames import read_chunk, write_chunk

DEFAULT_CHUNK_FRAMES = 64


class FrameSpool:
    """Write frames once, then read them back in order any number of times."""

    def __init__(
        self,
        *,
        prefix: str = "media_to_pose_frames_",
        directory: Path | None = None,
        chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    ) -> None:
        self.directory = Path(
            tempfile.mkdtemp(prefix=prefix, dir=str(directory) if directory else None)
        )
        self.chunk_frames = max(1, int(chunk_frames))
        self._buffer: list[CanonicalFrame] = []
        self._chunks: list[Path] = []
        self._count = 0
        self._closed = False

    def __len__(self) -> int:
        return self._count

    def __enter__(self) -> FrameSpool:
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

    def append(self, frame: CanonicalFrame) -> None:
        if self._closed:
            raise ValueError("Frame spool is closed")
        self._buffer.append(frame)
        self._count += 1
        if len(self._buffer) >= self.chunk_frames:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        path = self.directory / f"chunk_{len(self._chunks):06d}.npz"
        write_chunk(path, self._buffer, compact=False)
        self._chunks.append(path)
        self._buffer = []

    def __iter__(self) -> Iterator[CanonicalFrame]:
        if self._closed:
            raise ValueError("Frame spool is closed")
        self.flush()
        for path in tuple(self._chunks):
            yield from read_chunk(path)

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._buffer = []
        remove_tree(self.directory)


__all__ = ["DEFAULT_CHUNK_FRAMES", "FrameSpool"]
