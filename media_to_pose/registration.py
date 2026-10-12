"""Idempotent Blender RNA class registration helpers."""

from __future__ import annotations

from collections.abc import Iterable

import bpy


def _registered_type(cls):
    # Most dynamically registered add-on types are intentionally not exposed
    # as attributes on ``bpy.types``. RNA's subclass registry is authoritative
    # for PropertyGroup, Operator, Panel, Menu, and UIList alike.
    return bpy.types.bpy_struct.bl_rna_get_subclass_py(
        cls.__name__,
        None,
    )


def unregister_classes(classes: Iterable[type]) -> None:
    """Unregister matching RNA classes in reverse dependency order."""
    for cls in reversed(tuple(classes)):
        existing = _registered_type(cls)
        if existing is None:
            continue
        try:
            bpy.utils.unregister_class(existing)
        except RuntimeError as exc:
            message = str(exc).lower()
            if "missing bl_rna" not in message and "not registered" not in message:
                raise


def register_classes(classes: Iterable[type]) -> tuple[type, ...]:
    """Replace stale matching types and register all classes atomically."""
    classes = tuple(classes)
    unregister_classes(classes)
    registered = []
    try:
        for cls in classes:
            bpy.utils.register_class(cls)
            registered.append(cls)
    except Exception:
        unregister_classes(registered)
        raise
    return tuple(registered)
