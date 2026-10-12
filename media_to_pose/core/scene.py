"""The Blender scene an operation works in, passed explicitly to lower layers.

Only operators and the API read ``bpy.context``. They build one
``SceneContext`` at their boundary and hand it down, so retargeting and
cleanup never depend on whatever happens to be active when they run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SceneContext:
    scene: Any
    view_layer: Any
    preferences: Any = None

    @classmethod
    def from_context(cls, context: Any) -> SceneContext:
        """Capture the scene, view layer and preferences of a Blender context."""
        return cls(
            scene=context.scene,
            view_layer=context.view_layer,
            preferences=getattr(context, "preferences", None),
        )

    @property
    def fps(self) -> float:
        render = self.scene.render
        return float(render.fps) / float(render.fps_base)

    def update(self) -> None:
        """Re-evaluate the view layer so pose matrices reflect recent edits."""
        self.view_layer.update()

    def set_frame(self, value: float) -> None:
        """Move the scene to a possibly fractional frame."""
        integer = math.floor(float(value))
        self.scene.frame_set(integer, subframe=float(value) - integer)


__all__ = ["SceneContext"]
