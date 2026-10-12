"""Install, exercise, remove, and reinstall one platform release archive."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = Path(__file__).with_name("release_smoke_blender.py")
RIG_FIXTURE = ROOT / "tests" / "fixtures" / "rigs" / "vroid_test.blend"


def _run(command: list[str], environment: dict[str, str]) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    output = completed.stdout + completed.stderr
    print(output, end="")
    if completed.returncode:
        raise RuntimeError(
            f"Command exited {completed.returncode}: {' '.join(command)}"
        )
    return output


def _blender_command(blender: Path, *arguments: str) -> list[str]:
    return [str(blender), "--background", *arguments]


def _assert_smoke_markers(output: str) -> None:
    if "MTP_ORT_DIAGNOSTICS=" not in output:
        raise AssertionError("Installed extension ONNX Runtime diagnostics are missing")
    if "MTP_RELEASE_SMOKE_OK" not in output:
        raise AssertionError("Installed extension smoke marker is missing")
    if os.environ.get("MTP_ROHM_BUNDLE") and "MTP_ROHM_RUNTIME_OK=" not in output:
        raise AssertionError("Installed extension RoHM runtime marker is missing")


def _assert_command_output(output: str) -> None:
    """The installed package registered ``-c media_to_pose`` and it answered."""
    lines = [line for line in output.splitlines() if line.startswith('{"armature"')]
    if not lines:
        raise AssertionError("media_to_pose command printed no rig description")
    described = json.loads(lines[-1])
    if not described["sections"]["BODY"]["mapped"]:
        raise AssertionError("media_to_pose command mapped no body bones")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--require-accelerator",
        action="store_true",
        help="Require the platform GPU provider to execute the tiny ONNX graph",
    )
    arguments = parser.parse_args()
    blender = arguments.blender.resolve()
    archive = arguments.archive.resolve()
    if not blender.is_file():
        raise FileNotFoundError(blender)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    with zipfile.ZipFile(archive) as package:
        expected_version = str(
            tomllib.loads(package.read("blender_manifest.toml").decode("utf8"))[
                "version"
            ]
        )

    with tempfile.TemporaryDirectory(prefix="mtp-release-smoke-") as temporary:
        user_root = Path(temporary)
        environment = os.environ.copy()
        # The smoke must resolve native modules from the installed extension,
        # never from a developer shell or another Python installation.
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        for name in ("config", "scripts", "datafiles", "extensions", "cache"):
            path = user_root / name
            path.mkdir()
            environment[f"BLENDER_USER_{name.upper()}"] = str(path)
        environment["MTP_EXPECTED_VERSION"] = expected_version
        if arguments.require_accelerator:
            environment["MTP_REQUIRE_ACCELERATOR"] = "1"
        (user_root / "extensions" / "blender_org").mkdir()

        _run(
            _blender_command(
                blender,
                "--factory-startup",
                "--command",
                "extension",
                "validate",
                str(archive),
            ),
            environment,
        )
        install = _run(
            _blender_command(
                blender,
                "--command",
                "extension",
                "install-file",
                "-r",
                "user_default",
                "-e",
                str(archive),
            ),
            environment,
        )
        if 'STATUS Installed "media_to_pose"' not in install:
            raise AssertionError("Blender did not report a successful install")
        if not any((user_root / "config").rglob("userpref.blend")):
            raise AssertionError("Extension enable was not persisted")

        smoke = _run(
            _blender_command(
                blender,
                "--python-exit-code",
                "1",
                "--python",
                str(SMOKE_SCRIPT),
            ),
            environment,
        )
        _assert_smoke_markers(smoke)
        command = _run(
            _blender_command(
                blender,
                str(RIG_FIXTURE),
                "--command",
                "media_to_pose",
                "describe-rig",
                "--armature",
                "Armature",
                "--json",
            ),
            environment,
        )
        _assert_command_output(command)

        removed = _run(
            _blender_command(
                blender,
                "--command",
                "extension",
                "remove",
                "media_to_pose",
            ),
            environment,
        )
        if 'STATUS Removed "media_to_pose"' not in removed:
            raise AssertionError("Blender did not report a successful uninstall")
        installed_directory = (
            user_root / "extensions" / "user_default" / "media_to_pose"
        )
        if installed_directory.exists():
            raise AssertionError("Uninstall left the extension directory behind")

        reinstall = _run(
            _blender_command(
                blender,
                "--command",
                "extension",
                "install-file",
                "-r",
                "user_default",
                "-e",
                str(archive),
            ),
            environment,
        )
        if 'STATUS Installed "media_to_pose"' not in reinstall:
            raise AssertionError("Blender did not report a successful reinstall")
        verification = _run(
            _blender_command(
                blender,
                "--python-exit-code",
                "1",
                "--python",
                str(SMOKE_SCRIPT),
                "--",
                "--verify-only",
            ),
            environment,
        )
        _assert_smoke_markers(verification)

    print("MTP_RELEASE_LIFECYCLE_OK")


if __name__ == "__main__":
    main()
