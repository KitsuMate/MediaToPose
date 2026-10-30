"""Add-on registration, scene state reset and runtime hooks."""

from __future__ import annotations

from contextlib import suppress

import bpy
from bpy.app.handlers import persistent

from ..api import cli
from ..backends.anycalib import runtime as anycalib_ort_runtime
from ..backends.rohm import runtime as rohm_ort_runtime
from ..core.camera import calibration as camera_calibration
from ..core.camera import motion_correction as camera_motion_correction
from ..core.coordinator import COORDINATOR
from ..core.log import configure_logging, get_logger
from ..core.motion import rohm_correction
from ..rig.planning import clear_rig_solve_plans
from . import host, properties, ui
from . import preview as cache_preview
from .operators import capture as operators_capture
from .operators import generate as operators_generate
from .operators import mapping as operators_mapping
from .operators import models as model_operators

LOGGER = get_logger(__name__)

# Point the Blender-free layers at Blender's user directories before any
# cache or model path is resolved.
host.install()

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
    model_operators,
    operators_capture,
    operators_generate,
    ui,
)

_scene_state_pending = True
_STATE_TIMER_KEY = "_media_to_pose_state_watch_timer"
# The handle of the registered command line, kept in the driver namespace so
# a reloaded module can still remove the previous registration.
_CLI_KEY = "_media_to_pose_cli_command"


def _reset_scene_state(data=None) -> bool:
    """Reset transient scene state when Blender data access is available.

    Extension registration can run under Blender's ``_RestrictData`` guard,
    which intentionally has no ``scenes`` collection. Returning ``False``
    lets the regular timer retry after registration without treating this
    expected install context as an error.
    """
    global _scene_state_pending
    from ..rig import mapping
    from ..session.capture_options import clear_unrecoverable_media_selection

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
                mapping.refresh_auto_mapping(
                    settings.target_armature, settings=settings
                )
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
        LOGGER.debug("State refresh deferred: %s", exc)
    return 0.75


def _register_cli() -> None:
    _unregister_cli()
    register_command = getattr(bpy.utils, "register_cli_command", None)
    if register_command is not None:
        bpy.app.driver_namespace[_CLI_KEY] = register_command(cli.COMMAND, cli.execute)


def _unregister_cli() -> None:
    handle = bpy.app.driver_namespace.pop(_CLI_KEY, None)
    if handle is not None:
        with suppress(RuntimeError, ValueError):
            bpy.utils.unregister_cli_command(handle)


def _remove_runtime_hooks() -> None:
    _unregister_cli()
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
    configure_logging()
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
        _register_cli()
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
