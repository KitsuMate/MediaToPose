"""Shared bootstrap and fixture helpers for Blender-hosted tests."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import sys
import tomllib
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
SOURCE_PACKAGE_NAME = "media_to_pose_test"
INSTALLED_PACKAGE_NAME = "bl_ext.user_default.media_to_pose"
PACKAGE_NAME = (
    INSTALLED_PACKAGE_NAME
    if os.environ.get("MTP_TEST_INSTALLED_EXTENSION") == "1"
    else SOURCE_PACKAGE_NAME
)


def _package_name() -> str:
    return PACKAGE_NAME


def load_addon() -> ModuleType:
    """Load the installed CI package or the local source checkout."""
    package_name = _package_name()
    existing = sys.modules.get(package_name)
    if existing is not None:
        return existing
    if package_name == INSTALLED_PACKAGE_NAME:
        return importlib.import_module(package_name)
    spec = importlib.util.spec_from_file_location(
        package_name,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to construct the Media to Pose test package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


def register_addon() -> ModuleType:
    module = load_addon()
    module.register()
    return module


def unregister_addon() -> None:
    module = sys.modules.get(_package_name())
    if module is not None:
        module.unregister()


def fixture_manifest() -> dict:
    with (FIXTURES / "fixtures.toml").open("rb") as stream:
        return tomllib.load(stream)


def fixture_path(fixture_id: str) -> Path:
    for entry in fixture_manifest()["fixture"]:
        if entry["id"] == fixture_id:
            return FIXTURES / entry["path"]
    raise KeyError(f"Unknown fixture: {fixture_id}")


def verify_fixture(fixture_id: str) -> Path:
    manifest = fixture_manifest()
    entry = next(item for item in manifest["fixture"] if item["id"] == fixture_id)
    path = FIXTURES / entry["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise AssertionError(f"Fixture {fixture_id} hash mismatch: {digest}")
    return path
