"""Enforce the package layering rules described in AGENTS.md.

Every module of the extension is parsed (never imported). Each import is
resolved to a package-relative module name and checked against these rules:

- Imports point down the layer order
  ``blender → api → session → (retarget, postprocess, rig, providers,
  backends, evaluate) → core``.
- Nothing imports ``blender/`` except the package ``__init__`` and
  ``blender/`` itself.
- ``core/`` imports only the standard library, numpy and ``core/``.
- ``bpy`` and ``mathutils`` are used only where the layer allows them.
- Only ``blender/operators/`` and ``api/`` read ``bpy.context``.

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
PACKAGE_ROOT = (
    ROOT / "media_to_pose"
    if (ROOT / "media_to_pose" / "__init__.py").exists()
    else ROOT
)
NOT_PACKAGE = {"wheels", "assets", "tests", "ci", "docs", "dist", ".workspace"}

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
BPY_LAYERS = {"blender", "api", "session", "retarget", "rig", "evaluate", "root"}
MATHUTILS_LAYERS = BPY_LAYERS | {"postprocess"}
CORE_THIRD_PARTY = {"numpy"}
BLENDER_MODULES = {"bpy", "mathutils", "bmesh", "gpu", "gpu_extras", "blf", "bl_ui"}

# Modules that still live outside the target layout, mapped to the layer
# they will move into. Empty once Phase 7 has finished.
LEGACY_LAYERS: dict[str, str] = {
    "addon_lifecycle": "blender",
    "animation_cleanup": "core",
    "anycalib_backend": "backends",
    "anycalib_ort_runtime": "backends",
    "cache_preview": "blender",
    "camera_calibration": "core",
    "camera_calibration_pipeline": "core",
    "camera_motion_correction": "core",
    "capture_state": "session",
    "constants": "core",
    "dependency_bootstrap": "core",
    "detection_cache": "core",
    "grounded_cleanup": "core",
    "mapping": "rig",
    "media_io": "core",
    "model_lifecycle": "backends",
    "models": "core",
    "motion_filter": "core",
    "operation": "core",
    "operators_capture": "blender",
    "operators_mapping": "blender",
    "pose_translation_solver": "retarget",
    "post_processing": "postprocess",
    "properties": "blender",
    "registration": "blender",
    "retarget": "retarget",
    "rig_planning": "rig",
    "rohm_backend": "backends",
    "rohm_correction": "core",
    "rohm_ort_runtime": "backends",
    "sam3d_backend": "backends",
    "sam3d_mhr_runtime": "backends",
    "sam3d_ort_runtime": "backends",
    "sam3d_state": "providers",
    "sam3d_vith_runtime": "backends",
    "target_pose": "retarget",
    "ui": "blender",
}

# (module, rule, detail) tuples for known violations.
ALLOWLIST: set[tuple[str, str, str]] = {
    ("animation_cleanup", "uses-mathutils", "mathutils"),
    ("anycalib_backend", "imports-blender", "registration"),
    ("anycalib_backend", "uses-bpy", "bpy"),
    ("cache_preview", "reads-bpy-context", "bpy.context"),
    ("camera_calibration_pipeline", "upward-import", "anycalib_backend"),
    ("detection_cache", "upward-import", "anycalib_backend"),
    ("detection_cache", "upward-import", "providers"),
    ("detection_cache", "upward-import", "sam3d_backend"),
    ("detection_cache", "uses-bpy", "bpy"),
    ("grounded_cleanup", "uses-mathutils", "mathutils"),
    ("mapping", "imports-blender", "properties"),
    ("mapping", "reads-bpy-context", "bpy.context"),
    ("operators_capture", "reads-bpy-context", "bpy.context"),
    ("pose_translation_solver", "reads-bpy-context", "bpy.context"),
    ("retarget", "reads-bpy-context", "bpy.context"),
    ("rohm_backend", "imports-blender", "registration"),
    ("rohm_backend", "uses-bpy", "bpy"),
    ("rohm_correction", "upward-import", "rohm_ort_runtime"),
    ("rohm_correction", "uses-mathutils", "mathutils"),
    ("sam3d_backend", "imports-blender", "registration"),
    ("sam3d_backend", "uses-bpy", "bpy"),
    ("target_pose", "reads-bpy-context", "bpy.context"),
    ("ui", "reads-bpy-context", "bpy.context"),
}


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
        self.reads_context = False

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.external.add(alias.name.split(".", 1)[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level == 0:
            self.external.add(str(node.module).split(".", 1)[0])
            return
        base = _resolve_relative(self.module, self.is_package, node.level, node.module)
        for alias in node.names:
            candidate = f"{base}.{alias.name}" if base else alias.name
            if candidate in self.modules:
                self.internal.add(candidate)
            else:
                self.internal.add(_closest_module(base, self.modules))

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            node.attr == "context"
            and isinstance(node.value, ast.Name)
            and node.value.id == "bpy"
        ):
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
            module.startswith(("blender.operators", "api")) or layer == "root"
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
