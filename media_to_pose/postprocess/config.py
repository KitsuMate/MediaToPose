"""Module settings stored on the armature: ensure, compile, save and load."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from ..core.errors import InvalidRequest
from ..core.requests import PostProcessConfig, PostProcessModuleConfig
from .settings_schema import (
    MODULE_SETTING_FIELDS,
    MODULES_BY_ID,
    POST_PROCESSING_DEFAULT_ENABLED,
    POST_PROCESSING_MODULES,
    validate_settings,
)


def ensure_module_configs(config) -> bool:
    """Ensure the config contains exactly the registered correction modules."""
    registered_ids = {item[0] for item in POST_PROCESSING_MODULES}
    seen_ids = set()
    try:
        for index in reversed(range(len(config.post_processing_modules))):
            module_id = config.post_processing_modules[index].module_id
            if module_id not in registered_ids or module_id in seen_ids:
                config.post_processing_modules.remove(index)
            else:
                seen_ids.add(module_id)
    except (AttributeError, RuntimeError):
        return False
    existing = {item.module_id: item for item in config.post_processing_modules}
    for module_id, name, description in POST_PROCESSING_MODULES:
        item = existing.get(module_id)
        try:
            if item is None:
                item = config.post_processing_modules.add()
                item.module_id = module_id
                item.enabled = POST_PROCESSING_DEFAULT_ENABLED[module_id]
            item.display_name = name
            item.description = description
        except (AttributeError, RuntimeError):
            # Linked/read-only armatures and Blender's UI draw context do not
            # permit ID writes. Operators may retry from a writable context;
            # panel drawing must remain read-only.
            return False
    return True


def module_config(config, module_id: str):
    ensure_module_configs(config)
    return next(
        (
            item
            for item in config.post_processing_modules
            if item.module_id == module_id
        ),
        None,
    )


def ordered_module_configs(config) -> list[object]:
    """Return existing settings in fixed order without mutating Blender IDs."""
    configured = {item.module_id: item for item in config.post_processing_modules}
    return [
        configured[module_id]
        for module_id, _name, _description in POST_PROCESSING_MODULES
        if module_id in configured
    ]


def compile_postprocess_config(
    config,
    *,
    diagnostic_mode: bool = False,
) -> PostProcessConfig:
    """Snapshot Blender RNA settings once for one capture operation."""
    if isinstance(config, PostProcessConfig):
        return config
    ensure_module_configs(config)
    modules = []
    for item in ordered_module_configs(config):
        fields = MODULE_SETTING_FIELDS.get(item.module_id, ())
        modules.append(
            PostProcessModuleConfig(
                module_id=str(item.module_id),
                display_name=str(item.display_name or item.module_id),
                enabled=bool(item.enabled),
                values={name: getattr(item, name) for name in fields},
            )
        )
    return PostProcessConfig(
        detected_profile=str(config.detected_profile),
        modules=tuple(modules),
        diagnostic_mode=bool(diagnostic_mode),
    )


def serialize_module_configs(config) -> list[dict]:
    ensure_module_configs(config)
    return [
        {
            "module_id": item.module_id,
            "display_name": item.display_name,
            "description": item.description,
            "enabled": item.enabled,
            "settings": {
                field_name: getattr(item, field_name)
                for field_name in MODULE_SETTING_FIELDS[item.module_id]
            },
        }
        for item in config.post_processing_modules
    ]


def load_module_configs(config, payload: Iterable[dict]) -> None:
    ensure_module_configs(config)
    existing = {item.module_id: item for item in config.post_processing_modules}
    for data in payload:
        module_id = str(data.get("module_id", ""))
        item = existing.get(module_id)
        if item is None:
            raise ValueError(
                f"Unsupported post-processing module {module_id or '<missing>'}"
            )
        item.display_name = str(
            data.get("display_name", item.display_name or module_id)
        )
        item.description = str(data.get("description", item.description))
        item.enabled = bool(data.get("enabled", True))
        settings = dict(data.get("settings", {}))
        for key, value in settings.items():
            if key in MODULE_SETTING_FIELDS[module_id] and hasattr(item, key):
                setattr(item, key, value)
    ensure_module_configs(config)


def reset_module_defaults(config, module_id: str) -> bool:
    item = module_config(config, module_id)
    if item is None:
        return False
    # RNA's unset restores the PropertyGroup declaration defaults.
    for field_name in MODULE_SETTING_FIELDS[module_id]:
        item.property_unset(field_name)
    item.enabled = POST_PROCESSING_DEFAULT_ENABLED[module_id]
    item.last_warning = ""
    item.status = "VALID"
    return True


def normalize_module_payload(payload: Any) -> list[dict[str, Any]]:
    """Validate module settings in any accepted shape and list them.

    ``payload`` is the ``post_processing`` list of a saved config, or a
    mapping ``{module_id: {"enabled": bool, "settings": {...}}}``, or a
    mapping ``{module_id: {setting: value}}``. Every module id and setting
    is checked against the settings schema; values are coerced to their
    declared types. Returns ``[{"module_id", "enabled"?, "settings"}]``.
    """
    if isinstance(payload, Mapping):
        items = []
        for module_id, value in payload.items():
            if not isinstance(value, Mapping):
                raise InvalidRequest(
                    f"Settings for {module_id} must be an object",
                    parameter=str(module_id),
                )
            if "settings" in value or "enabled" in value:
                items.append({"module_id": module_id, **value})
            else:
                items.append({"module_id": module_id, "settings": dict(value)})
    elif isinstance(payload, Iterable) and not isinstance(payload, (str, bytes)):
        items = [dict(item) for item in payload]
    else:
        raise InvalidRequest("Post-processing settings must be a list or an object")
    result = []
    for item in items:
        module_id = str(item.get("module_id", ""))
        if module_id not in MODULES_BY_ID:
            raise InvalidRequest(
                f"Unknown post-processing module {module_id or '<missing>'}",
                parameter="module_id",
            )
        normalized: dict[str, Any] = {
            "module_id": module_id,
            "settings": validate_settings(module_id, dict(item.get("settings", {}))),
        }
        if "enabled" in item:
            if not isinstance(item["enabled"], bool):
                raise InvalidRequest(
                    f"{module_id}.enabled must be true or false",
                    parameter="enabled",
                )
            normalized["enabled"] = item["enabled"]
        result.append(normalized)
    return result


def apply_module_overrides(
    config: PostProcessConfig, payload: Any
) -> PostProcessConfig:
    """Return ``config`` with validated module settings from ``payload``.

    Nothing is written to Blender data; see ``normalize_module_payload`` for
    the accepted shapes.
    """
    overrides = {item["module_id"]: item for item in normalize_module_payload(payload)}
    modules = []
    for module in config.modules:
        override = overrides.get(module.module_id)
        if override is None:
            modules.append(module)
            continue
        modules.append(
            replace(
                module,
                enabled=override.get("enabled", module.enabled),
                values={**module.values, **override["settings"]},
            )
        )
    return replace(config, modules=tuple(modules))
