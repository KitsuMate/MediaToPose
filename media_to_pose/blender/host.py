"""Connect the Blender-free layers to Blender's user directories and paths."""

from __future__ import annotations

import bpy

from ..core import paths

# The extension's own package name (bl_ext.<repository>.media_to_pose inside
# Blender), which add-on preferences use as their bl_idname.
ADDON_PACKAGE = __package__.rpartition(".")[0]


def _user_resource(relative: str, create: bool) -> str | None:
    return bpy.utils.user_resource("SCRIPTS", path=relative, create=create) or None


def install() -> None:
    """Store user data under Blender's scripts folder and expand ``//`` paths."""
    paths.configure(_user_resource, bpy.path.abspath)


__all__ = ["ADDON_PACKAGE", "install"]
