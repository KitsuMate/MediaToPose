"""Enforce the package layering rules described in AGENTS.md.

Every module of the extension is parsed (never imported). Each import is
resolved to a package-relative module name and checked against these rules:

- Imports point down the layer order
  ``blender → api → session → (retarget, postprocess, rig, providers,
  backends, evaluate) → core``.
- Nothing imports ``blender/`` except the package ``__init__`` and
  ``blender/`` itself.
- ``core/`` imports only the standard library, numpy and ``core/``. The one
  exception is an optional import inside ``try``/``except ImportError`` that
  has a pure-Python fallback (``core.geometry`` uses ``mathutils`` when
  Blender provides it).
- ``bpy`` and ``mathutils`` are used only where the layer allows them.
- Only ``blender/operators/``, ``api/`` and ``blender/context.py`` (for
  timers and handlers, which get no context argument) read ``bpy.context``.

``ALLOWLIST`` records known violations. Entries must be removed as the code
moves into the target layout; stale entries fail the test so the list only
ever shrinks.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "media_to_pose"
NOT_PACKAGE = {"wheels", "assets"}

LEVELS = {
    "core": 0,
    "retarget": 4,
    "postprocess": 4,
    "rig": 4,
    "providers": 4,
    "backends": 4,
    "evaluate": 4,
    "session": 5,
    "api": 6,
    "blender": 7,
    "root": 8,
}
BPY_LAYERS = {
    "blender",
    "api",
    "session",
    "retarget",
    "rig",
    "evaluate",
    "root",
}
MATHUTILS_LAYERS = BPY_LAYERS | {"postprocess"}
CORE_THIRD_PARTY = {"numpy"}
BLENDER_MODULES = {"bpy", "mathutils", "bmesh", "gpu", "gpu_extras", "blf", "bl_ui"}

# Modules that still live outside the target layout, mapped to the layer
# they will move into. Empty once Phase 7 has finished.
LEGACY_LAYERS: dict[str, str] = {
    "addon_lifecycle": "blender",
    "animation_cleanup": "core",
    "cache_preview": "blender",
    "camera_calibration": "core",
    "camera_calibration_pipeline": "core",
    "camera_motion_correction": "core",
    "capture_state": "session",
    "constants": "core",
    "dependency_bootstrap": "core",
    "geometry": "core",
    "grounded_cleanup": "core",
    "media_io": "core",
    "models": "core",
    "motion_filter": "core",
    "operation": "core",
    "operators_capture": "blender",
    "operators_mapping": "blender",
    "properties": "blender",
    "registration": "blender",
    "rig_planning": "rig",
    "rohm_correction": "core",
    "ui": "blender",
}

# (module, rule, detail) tuples for known violations.
ALLOWLIST: set[tuple[str, str, str]] = set()


def _package_files() -> dict[str, Path]:
    files = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(PACKAGE_ROOT)
        if relative.parts[0] in NOT_PACKAGE:
            continue
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        files[".".join(parts)] = path
    return files


def layer_of(module: str) -> str:
    if not module:
        return "root"
    head = module.split(".", 1)[0]
    if head in LEVELS:
        return head
    if module in LEGACY_LAYERS:
        return LEGACY_LAYERS[module]
    if head in LEGACY_LAYERS:
        return LEGACY_LAYERS[head]
    raise AssertionError(f"{module}: no layer; add it to the target layout")


def _resolve_relative(
    module: str, is_package: bool, level: int, name: str | None
) -> str:
    parts = module.split(".") if module else []
    if not is_package:
        parts = parts[:-1]
    if level > 1:
        parts = parts[: len(parts) - (level - 1)]
    if name:
        parts.extend(name.split("."))
    return ".".join(parts)


def _closest_module(name: str, modules: set[str]) -> str:
    while name and name not in modules:
        name = name.rpartition(".")[0]
    return name


class _ImportVisitor(ast.NodeVisitor):
    def __init__(self, module: str, is_package: bool, modules: set[str]):
        self.module = module
        self.is_package = is_package
        self.modules = modules
        self.internal: set[str] = set()
        self.external: set[str] = set()
        self.optional: set[str] = set()
        self.reads_context = False
        self.bpy_attributes: set[str] = set()
        self._optional_depth = 0

    def visit_Try(self, node: ast.Try) -> None:
        caught = set()
        for handler in node.handlers:
            names = (
                handler.type.elts
                if isinstance(handler.type, ast.Tuple)
                else [handler.type]
            )
            caught.update(name.id for name in names if isinstance(name, ast.Name))
        optional = bool(caught & {"ImportError", "ModuleNotFoundError"})
        self._optional_depth += optional
        for statement in node.body:
            self.visit(statement)
        self._optional_depth -= optional
        for statement in [*node.handlers, *node.orelse, *node.finalbody]:
            self.visit(statement)

    def _external(self, name: str) -> None:
        name = name.split(".", 1)[0]
        (self.optional if self._optional_depth else self.external).add(name)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._external(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level == 0:
            self._external(str(node.module))
            return
        base = _resolve_relative(self.module, self.is_package, node.level, node.module)
        for alias in node.names:
            candidate = f"{base}.{alias.name}" if base else alias.name
            if candidate in self.modules:
                self.internal.add(candidate)
            else:
                self.internal.add(_closest_module(base, self.modules))

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == "bpy":
            self.bpy_attributes.add(node.attr)
            if node.attr == "context":
                self.reads_context = True
        self.generic_visit(node)


def collect_violations() -> set[tuple[str, str, str]]:
    files = _package_files()
    modules = set(files)
    violations: set[tuple[str, str, str]] = set()
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    for module, path in files.items():
        tree = ast.parse(path.read_text(encoding="utf8"), filename=str(path))
        visitor = _ImportVisitor(module, path.name == "__init__.py", modules)
        visitor.visit(tree)
        layer = layer_of(module)
        for target in sorted(visitor.internal):
            if target == module or not target:
                continue
            target_layer = layer_of(target)
            if target_layer == "blender" and layer not in {"blender", "root"}:
                violations.add((module, "imports-blender", target))
            elif LEVELS[target_layer] > LEVELS[layer]:
                violations.add((module, "upward-import", target))
            elif layer == "core" and target_layer != "core":
                violations.add((module, "core-imports-layer", target))
        if layer != "core":
            visitor.external |= visitor.optional
        for name in sorted(visitor.external):
            if name in BLENDER_MODULES:
                allowed = MATHUTILS_LAYERS if name == "mathutils" else BPY_LAYERS
                if layer not in allowed:
                    violations.add((module, f"uses-{name}", name))
            elif (
                layer == "core" and name not in stdlib and name not in CORE_THIRD_PARTY
            ):
                violations.add((module, "core-third-party", name))
        if visitor.reads_context and not (
            module.startswith(("blender.operators", "api"))
            or module == "blender.context"
            or layer == "root"
        ):
            violations.add((module, "reads-bpy-context", "bpy.context"))
    return violations


class LayeringTests(unittest.TestCase):
    def test_layering_rules(self) -> None:
        violations = collect_violations()
        new = sorted(violations - ALLOWLIST)
        stale = sorted(ALLOWLIST - violations)
        message = []
        if new:
            message.append("New layering violations:")
            message.extend(f"  {entry!r}," for entry in new)
        if stale:
            message.append("Fixed violations; remove them from ALLOWLIST:")
            message.extend(f"  {entry!r}," for entry in stale)
        self.assertFalse(message, "\n".join(message))

    def test_every_module_has_a_layer(self) -> None:
        for module in _package_files():
            self.assertIn(layer_of(module), LEVELS)


if __name__ == "__main__":
    unittest.main()
