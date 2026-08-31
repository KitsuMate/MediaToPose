"""Media to Pose Blender extension."""

from __future__ import annotations

import bpy
from bpy.app.handlers import persistent

from . import (
    anycalib_backend,
    anycalib_ort_runtime,
    cache_preview,
    camera_calibration,
    camera_motion_correction,
    operators_capture,
    operators_mapping,
    properties,
    rohm_backend,
    rohm_correction,
    rohm_ort_runtime,
    sam3d_backend,
    ui,
)
from .operation import COORDINATOR
from .rig_planning import clear_rig_solve_plans

__all__ = [
    "anycalib_ort_runtime",
    "camera_calibration",
    "camera_motion_correction",
    "rohm_correction",
    "rohm_ort_runtime",
]

MODULES = (
    properties,
    operators_mapping,
    sam3d_backend,
    rohm_backend,
    anycalib_backend,
    operators_capture,
    ui,
)

_scene_state_pending = True
_STATE_TIMER_KEY = "_media_to_pose_state_watch_timer"


def _reset_scene_state(data=None) -> bool:
    """Reset transient scene state when Blender data access is available.

    Extension registration can run under Blender's ``_RestrictData`` guard,
    which intentionally has no ``scenes`` collection. Returning ``False``
    lets the regular timer retry after registration without treating this
    expected install context as an error.
    """
    global _scene_state_pending
    from . import mapping
    from .detection_cache import clear_unrecoverable_media_selection

    data = bpy.data if data is None else data
    scenes = getattr(data, "scenes", None)
    if scenes is None:
        _scene_state_pending = True
        return False
    try:
        for scene in scenes:
            settings = getattr(scene, "mtp_settings", None)
            if settings is not None:
                settings.status = "Ready"
                settings.status_kind = "READY"
                settings.progress = 0.0
                clear_unrecoverable_media_selection(settings)
                mapping.refresh_auto_mapping(settings.target_armature)
    except (AttributeError, ReferenceError, RuntimeError):
        _scene_state_pending = True
        return False
    _scene_state_pending = False
    return True


@persistent
def _initialize_scene_state(_unused) -> None:
    _reset_scene_state()
    cache_preview.schedule_refresh()


def _state_watch_timer():
    """Refresh non-mutating background state used by the Capture panel."""
    try:
        if _scene_state_pending:
            _reset_scene_state()
    except (AttributeError, ReferenceError, RuntimeError) as exc:
        print(f"[Media to Pose] State refresh deferred: {exc}")
    return 0.75


def _remove_runtime_hooks() -> None:
    previous_timer = bpy.app.driver_namespace.pop(
        _STATE_TIMER_KEY,
        None,
    )
    for callback in (previous_timer, _state_watch_timer):
        if callback is not None and bpy.app.timers.is_registered(callback):
            bpy.app.timers.unregister(callback)
    for callback in list(bpy.app.handlers.load_post):
        if callback is _initialize_scene_state or (
            getattr(callback, "__module__", "") == __name__
            and getattr(callback, "__name__", "") == "_initialize_scene_state"
        ):
            bpy.app.handlers.load_post.remove(callback)


def _cleanup_previous_registration() -> None:
    """Remove a complete or partially registered earlier module instance."""
    _remove_runtime_hooks()
    cache_preview.unregister()
    for module in reversed(MODULES):
        module.unregister()


def register() -> None:
    global _scene_state_pending
    _scene_state_pending = True
    _cleanup_previous_registration()
    registered_modules = []
    try:
        for module in MODULES:
            module.register()
            registered_modules.append(module)
        bpy.app.driver_namespace[_STATE_TIMER_KEY] = _state_watch_timer
        bpy.app.timers.register(
            _state_watch_timer,
            first_interval=0.2,
            persistent=True,
        )
        bpy.app.handlers.load_post.append(_initialize_scene_state)
        # This is a no-op during Blender's restricted extension-install
        # context; the timer retries after normal access is restored.
        _reset_scene_state()
        cache_preview.schedule_refresh()
    except Exception:
        _remove_runtime_hooks()
        cache_preview.unregister()
        for module in reversed(registered_modules):
            module.unregister()
        raise


def unregister() -> None:
    global _scene_state_pending
    _scene_state_pending = False
    COORDINATOR.shutdown(join_timeout_seconds=2.0)
    clear_rig_solve_plans()
    _cleanup_previous_registration()


if __name__ == "__main__":
    register()
