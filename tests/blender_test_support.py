"""Shared bootstrap and fixture helpers for Blender-hosted tests."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
import sys
import tomllib
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "media_to_pose"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
GOLDEN = FIXTURES / "golden"
GOLDEN_SCHEMA_VERSION = 1
SOURCE_PACKAGE_NAME = "media_to_pose"
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
        SOURCE / "__init__.py",
        submodule_search_locations=[str(SOURCE)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to construct the Media to Pose test package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # Never leave a half-initialised package behind: later tests would
        # see a module without register() instead of the real import error.
        sys.modules.pop(package_name, None)
        raise
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
    payload = path.read_bytes()
    if entry.get("media_type") in {"application/json", "text/plain"}:
        payload = payload.replace(b"\r\n", b"\n")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != entry["sha256"]:
        raise AssertionError(f"Fixture {fixture_id} hash mismatch: {digest}")
    return path


def scene_context():
    """The explicit scene context retargeting needs, from the active context."""
    import bpy

    return load_addon().core.scene.SceneContext.from_context(bpy.context)


def golden_snapshot(test: unittest.TestCase, name: str, actual: dict) -> dict | None:
    """The stored golden snapshot ``name`` to compare ``actual`` with.

    With ``MTP_UPDATE_GOLDEN=1`` (``--update-golden``) ``actual`` is written
    instead and None is returned.
    """
    path = GOLDEN / f"{name}.json"
    if os.environ.get("MTP_UPDATE_GOLDEN") == "1":
        payload = {"schema_version": GOLDEN_SCHEMA_VERSION, "case": name, **actual}
        GOLDEN.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        return None
    if not path.is_file():
        test.fail(f"Missing golden snapshot {path}; run with --update-golden")
    expected = json.loads(path.read_text(encoding="utf-8"))
    test.assertEqual(expected["schema_version"], GOLDEN_SCHEMA_VERSION)
    return expected


def assert_pose_close(
    test: unittest.TestCase,
    name: str,
    expected: dict,
    actual: dict,
    *,
    location_tolerance: float,
    degrees: float = 0.1,
) -> None:
    """Bone ``rotation`` (quaternions) and ``location`` snapshots agree."""
    from mathutils import Quaternion

    test.assertEqual(set(expected), set(actual), name)
    for bone, values in expected.items():
        want = Quaternion(values["rotation"])
        have = Quaternion(actual[bone]["rotation"])
        angle = math.degrees(want.rotation_difference(have).angle)
        test.assertLess(min(angle, 360.0 - angle), degrees, f"{name}: {bone}")
        for a, b in zip(values["location"], actual[bone]["location"], strict=True):
            test.assertAlmostEqual(
                a, b, delta=location_tolerance, msg=f"{name}: {bone}"
            )


def open_rig(addon: ModuleType, fixture_id: str, rig_name: str, **settings):
    """Open a rig fixture, register, auto-map the rig and return it.

    ``settings`` are set on the scene's capture settings first.
    """
    import bpy

    bpy.ops.wm.open_mainfile(filepath=str(fixture_path(fixture_id)))
    addon.register()
    rig = bpy.data.objects[rig_name]
    scene_settings = bpy.context.scene.mtp_settings
    for name, value in settings.items():
        setattr(scene_settings, name, value)
    scene_settings.target_armature = rig
    addon.api.auto_map(rig.name)
    return rig


def face_settings(face_output: str, **extra) -> SimpleNamespace:
    """Retarget settings for MediaPipe face-only capture."""
    return SimpleNamespace(
        provider="MEDIAPIPE",
        use_body=False,
        use_hands=False,
        use_face=True,
        face_output=face_output,
        **extra,
    )


def apply_face(addon: ModuleType, rig, blendshapes: dict, face_output: str) -> None:
    """Pose ``rig`` from one frame of face coefficients."""
    import bpy

    frame = addon.core.types.CanonicalFrame(timestamp_ms=0, blendshapes=blendshapes)
    addon.retarget.apply_frame(
        rig,
        frame,
        face_settings(face_output),
        post_process=False,
        scene_context=scene_context(),
    )
    bpy.context.view_layer.update()


def add_facs_targets(addon: ModuleType, rig) -> None:
    """Give the FaceFX mesh real FACS AU keys and map them.

    FaceFX's ``facs_bs_*`` keys are ARKit coefficients, not FACS AUs; each
    generated AU key moves a distinct vertex.
    """
    import bpy

    mesh = bpy.data.objects["FaceFX"]
    for index, (name, _label) in enumerate(addon.core.constants.FACS_ACTION_UNITS):
        block = mesh.shape_key_add(name=name, from_mix=False)
        block.data[index].co.z += 0.01
    addon.api.auto_map(rig.name)


def run_cli(addon: ModuleType, *argv: str) -> tuple[int, str]:
    """Run a ``media_to_pose`` command; its exit code and printed output."""
    output = io.StringIO()
    with redirect_stdout(output):
        code = addon.api.cli.execute(list(argv))
    return code, output.getvalue()
