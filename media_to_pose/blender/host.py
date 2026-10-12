"""Connect the Blender-free layers to Blender's user directories and paths."""

from __future__ import annotations

import bpy

from ..core import paths


def _user_resource(relative: str, create: bool) -> str | None:
    return bpy.utils.user_resource("SCRIPTS", path=relative, create=create) or None


def install() -> None:
    """Store user data under Blender's scripts folder and expand ``//`` paths."""
    paths.configure(_user_resource, bpy.path.abspath)


__all__ = ["install"]
