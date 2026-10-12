"""docs/agents.md covers the whole scripting surface it describes."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from media_to_pose.core import errors
from media_to_pose.session import results

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "media_to_pose"
DOC = (ROOT / "docs" / "agents.md").read_text(encoding="utf8")


def _module(path: str) -> ast.Module:
    return ast.parse((PACKAGE / path).read_text(encoding="utf8"))


def _api_functions() -> list[str]:
    for node in _module("api/__init__.py").body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            names = ast.literal_eval(node.value)
            return [name for name in names if name[0].islower()]
    raise AssertionError("api.__all__ not found")


def _is_add_parser(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_parser"
    )


def _cli_commands() -> set[str]:
    """Names passed to ``add_parser``, directly or through a ``for`` tuple."""
    commands = set()
    for node in ast.walk(_module("api/cli.py")):
        if _is_add_parser(node) and isinstance(node.args[0], ast.Constant):
            commands.add(node.args[0].value)
        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Tuple)
            and any(_is_add_parser(item) for item in ast.walk(node))
        ):
            for row in node.iter.elts:
                if isinstance(row, ast.Tuple) and isinstance(row.elts[0], ast.Constant):
                    commands.add(row.elts[0].value)
    return commands


def _metric_fields() -> list[str]:
    for node in _module("evaluate/metrics.py").body:
        if isinstance(node, ast.ClassDef) and node.name == "Metrics":
            return [
                item.target.id
                for item in node.body
                if isinstance(item, ast.AnnAssign)
                and isinstance(item.target, ast.Name)
                and item.target.id
                not in {"armature", "action", "frame_range", "frames", "fps"}
                and item.target.id not in {"schema_version", "body_height_m"}
            ]
    raise AssertionError("Metrics not found")


class AgentDocsTests(unittest.TestCase):
    def test_every_api_function_is_documented(self) -> None:
        missing = [name for name in _api_functions() if f"api.{name}" not in DOC]
        self.assertEqual(missing, [])

    def test_every_command_is_documented(self) -> None:
        commands = _cli_commands()
        self.assertGreater(len(commands), 10)
        missing = sorted(name for name in commands if f"`{name}" not in DOC)
        self.assertEqual(missing, [])

    def test_every_error_code_and_exit_code_is_listed(self) -> None:
        for cls in errors.ERROR_CLASSES:
            with self.subTest(code=cls.code):
                self.assertIn(f"`{cls.code}`", DOC)
                self.assertIn(f"| {cls.exit_code} |", DOC)

    def test_every_metric_and_warning_code_is_explained(self) -> None:
        for name in _metric_fields():
            with self.subTest(metric=name):
                self.assertIn(f"`{name}`", DOC)
        for _prefix, code in results.WARNING_CODES:
            with self.subTest(code=code):
                self.assertIn(f"`{code}`", DOC)

    def test_readme_points_to_the_agent_guide(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf8")
        self.assertIn("docs/agents.md", readme)
        self.assertIn("-c media_to_pose", readme)


if __name__ == "__main__":
    unittest.main()
