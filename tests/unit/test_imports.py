"""Every relative import in the package points at an existing module or name.

Imports inside functions only fail when that code path runs, so moving a
module can silently break rarely used paths. This test resolves all of
them statically.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "media_to_pose"
SKIP = {"wheels", "assets"}


def _modules() -> dict[str, Path]:
    modules = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        relative = path.relative_to(PACKAGE)
        if relative.parts[0] in SKIP:
            continue
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules[".".join(parts)] = path
    return modules


def _top_level_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf8"))
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
            names.update(targets)
            # Lazy packages re-export names through ``__getattr__``.
            if "_EXPORTS" in targets and isinstance(node.value, ast.Dict):
                names.update(
                    key.value
                    for key in node.value.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(
                (alias.asname or alias.name).split(".")[0] for alias in node.names
            )
        elif isinstance(node, (ast.Try, ast.If)):
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    names.update(
                        (alias.asname or alias.name).split(".")[0]
                        for alias in child.names
                    )
                elif isinstance(child, ast.Assign):
                    names.update(t.id for t in child.targets if isinstance(t, ast.Name))
    return names


class RelativeImportTests(unittest.TestCase):
    def test_every_relative_import_resolves(self) -> None:
        modules = _modules()
        exported = {name: _top_level_names(path) for name, path in modules.items()}
        problems = []
        for module, path in modules.items():
            is_package = path.name == "__init__.py"
            tree = ast.parse(path.read_text(encoding="utf8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.level == 0:
                    continue
                parts = module.split(".") if module else []
                if not is_package:
                    parts = parts[:-1]
                if node.level > 1:
                    parts = parts[: len(parts) - (node.level - 1)]
                base = ".".join(
                    [*parts, *(node.module.split(".") if node.module else [])]
                )
                if base not in modules:
                    problems.append(f"{module}:{node.lineno} imports missing {base!r}")
                    continue
                for alias in node.names:
                    candidate = f"{base}.{alias.name}" if base else alias.name
                    if candidate in modules or alias.name in exported[base]:
                        continue
                    problems.append(
                        f"{module}:{node.lineno} imports {alias.name!r} "
                        f"missing from {base or 'the package'!r}"
                    )
        self.assertFalse(problems, "\n".join(problems))


class SourceLocationTests(unittest.TestCase):
    def test_only_core_paths_locates_files_from_its_own_path(self) -> None:
        # Moving a module changes what ``__file__`` points at, so package
        # resources are located through core.paths only.
        offenders = [
            name
            for name, path in _modules().items()
            if name != "core.paths" and "__file__" in path.read_text(encoding="utf8")
        ]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
