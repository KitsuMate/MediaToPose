"""Where Media to Pose keeps per-user data, and how user paths are expanded.

Inside Blender the extension stores caches and model settings under the
user scripts directory and accepts ``//`` paths relative to the open
.blend file. The Blender layer installs those two behaviours with
``configure`` when the package is imported; plain Python (unit tests,
tools) falls back to ``~/.media_to_pose`` and ordinary paths.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

# The installed extension folder and the files shipped inside it.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_MODELS = PACKAGE_ROOT / "assets" / "models"
MANIFEST_PATH = PACKAGE_ROOT / "blender_manifest.toml"

UserDirectory = Callable[[str, bool], "str | None"]
PathExpander = Callable[[str], str]

_user_directory: UserDirectory | None = None
_expand_blend_path: PathExpander | None = None


def configure(
    user_directory: UserDirectory | None,
    expand_blend_path: PathExpander | None,
) -> None:
    """Install the host's user-data directory and ``//`` path expansion."""
    global _user_directory, _expand_blend_path
    _user_directory = user_directory
    _expand_blend_path = expand_blend_path


def user_directory(relative: str, *, create: bool = True) -> Path:
    """Return ``<user data>/media_to_pose/<relative>``."""
    relative = relative.strip("/")
    if _user_directory is not None:
        value = _user_directory(f"media_to_pose/{relative}", create)
        if value:
            return Path(value)
    root = (
        Path(os.environ.get("MTP_USER_DATA") or Path.home() / ".media_to_pose")
        / relative
    )
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def expand_user_path(value: str) -> Path:
    """Expand ``~``, environment variables and ``//`` .blend-relative paths."""
    expanded = os.path.expandvars(str(Path(str(value).strip()).expanduser()))
    if _expand_blend_path is not None:
        expanded = _expand_blend_path(expanded)
    return Path(expanded)


__all__ = [
    "BUNDLED_MODELS",
    "MANIFEST_PATH",
    "PACKAGE_ROOT",
    "configure",
    "expand_user_path",
    "user_directory",
]
