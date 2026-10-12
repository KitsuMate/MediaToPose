"""Context access for callbacks Blender runs without a context argument.

Operators, panels and property callbacks receive ``context`` and use it.
Timers and app handlers do not, so they read the global context through
these helpers instead of touching ``bpy.context`` directly.
"""

from __future__ import annotations

import bpy


def active_scene():
    """The scene of the active window, or ``None`` while Blender has none."""
    return getattr(bpy.context, "scene", None)


def active_settings():
    """The active scene's Media to Pose settings, or ``None``."""
    scene = active_scene()
    return getattr(scene, "mtp_settings", None) if scene else None


def redraw_view3d() -> None:
    """Tag every 3D Viewport in every window for redraw."""
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
