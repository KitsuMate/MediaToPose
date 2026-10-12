"""Media to Pose Blender extension."""

from __future__ import annotations

import importlib.util

# The package is importable without Blender so the plain unit tier can load
# its pure modules. Inside Blender, registration lives in addon_lifecycle.
if importlib.util.find_spec("bpy") is not None:
    from .addon_lifecycle import register, unregister

    __all__ = ["register", "unregister"]
