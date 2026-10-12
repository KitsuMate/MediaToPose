"""Read and change a rig's capture configuration as JSON-ready data."""

from __future__ import annotations

import json
from typing import Any

from ..core.errors import InvalidRequest
from ..postprocess import (
    load_module_configs,
    module_config,
    normalize_module_payload,
    serialize_module_configs,
    settings_schema,
)
from ..rig import mapping, presets
from .common import resolve_armature, resolve_scene


def list_postprocessors(armature=None) -> list[dict[str, Any]]:
    """Every clean-up module with its parameters, types, ranges and defaults.

    With ``armature``, each module also carries ``enabled`` and ``values``,
    the rig's current settings.
    """
    modules = settings_schema.describe()
    if armature is None:
        return modules
    target = resolve_armature(armature)
    current = {
        item["module_id"]: item
        for item in serialize_module_configs(target.mtp_rig_config)
    }
    for module in modules:
        stored = current.get(module["module_id"])
        if stored is not None:
            module["enabled"] = bool(stored["enabled"])
            module["values"] = stored["settings"]
    return modules


def get_config(armature) -> dict[str, Any]:
    """The rig's complete configuration: profile, mapping, FK/IK switches and
    clean-up modules, in the same format as the panel's preset export."""
    target = resolve_armature(armature)
    # Round-trip through JSON so the result holds plain Python values only.
    return json.loads(json.dumps(presets.serialize_config(target)))


def _with_enabled_state(config, modules: list[dict[str, Any]]) -> list[dict]:
    """Keep each module's enabled state unless the payload sets it."""
    result = []
    for item in modules:
        stored = module_config(config, item["module_id"])
        result.append(
            {
                "module_id": item["module_id"],
                "enabled": item.get(
                    "enabled", bool(stored.enabled) if stored else True
                ),
                "settings": item["settings"],
            }
        )
    return result


def set_config(armature, config: dict[str, Any], *, scene=None) -> dict[str, Any]:
    """Apply a configuration and return the rig's resulting configuration.

    ``config`` is either a complete configuration from ``get_config`` (it
    carries ``schema_version``), or a partial one with only
    ``post_processing``, in any shape ``capture(config=...)`` accepts.
    Every clean-up setting is validated against the settings schema before
    anything changes.
    """
    target = resolve_armature(armature)
    if not isinstance(config, dict):
        raise InvalidRequest("config must be an object", parameter="config")
    payload = dict(config)
    rig_config = target.mtp_rig_config
    modules = (
        _with_enabled_state(
            rig_config, normalize_module_payload(payload["post_processing"])
        )
        if "post_processing" in payload
        else None
    )
    settings = getattr(resolve_scene(scene), "mtp_settings", None)
    if "schema_version" in payload:
        if modules is not None:
            payload["post_processing"] = modules
        presets.load_config(target, payload, settings=settings)
    else:
        unknown = sorted(set(payload) - {"post_processing"})
        if unknown:
            raise InvalidRequest(
                "A partial config may only contain post_processing; got "
                + ", ".join(unknown),
                parameter="config",
            )
        if modules:
            load_module_configs(rig_config, modules)
        mapping.validate_mapping(target, settings=settings)
    return get_config(target)


__all__ = ["get_config", "list_postprocessors", "set_config"]
