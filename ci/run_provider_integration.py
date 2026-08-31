"""Install one archive and run real provider inference against its fixture."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLENDER_SCRIPT = Path(__file__).with_name("provider_integration_blender.py")
RIG_FIXTURE = ROOT / "tests" / "fixtures" / "rigs" / "vroid_test.blend"
IMAGES = {
    "mediapipe": ROOT
    / "tests"
    / "fixtures"
    / "media"
    / "mediapipe"
    / "ref_pose_realistic.png",
    "sam3d": ROOT / "tests" / "fixtures" / "media" / "sam" / "ref_pose_stylized.jpg",
}


def _run(
    command: list[str],
    environment: dict[str, str],
    timeout_seconds: float,
) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    output = completed.stdout + completed.stderr
    print(output, end="")
    if completed.returncode:
        raise RuntimeError(
            f"Command exited {completed.returncode}: {' '.join(command)}"
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--provider",
        choices=tuple(IMAGES),
        required=True,
    )
    parser.add_argument("--sam-user-scripts", type=Path)
    parser.add_argument("--sam-device", default="AUTO")
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    arguments = parser.parse_args()
    blender = arguments.blender.resolve()
    archive = arguments.archive.resolve()
    if not blender.is_file():
        raise FileNotFoundError(blender)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if arguments.provider == "sam3d" and arguments.sam_user_scripts is None:
        raise ValueError("SAM integration requires --sam-user-scripts")

    with tempfile.TemporaryDirectory(prefix="mtp-provider-integration-") as temporary:
        user_root = Path(temporary)
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        for name in ("config", "scripts", "datafiles", "extensions", "cache"):
            path = user_root / name
            path.mkdir()
            environment[f"BLENDER_USER_{name.upper()}"] = str(path)
        if arguments.provider == "sam3d":
            scripts = arguments.sam_user_scripts.resolve()
            if not scripts.is_dir():
                raise FileNotFoundError(scripts)
            environment["BLENDER_USER_SCRIPTS"] = str(scripts)
        (user_root / "extensions" / "blender_org").mkdir()

        install = _run(
            [
                str(blender),
                "--background",
                "--command",
                "extension",
                "install-file",
                "-r",
                "user_default",
                "-e",
                str(archive),
            ],
            environment,
            arguments.timeout_seconds,
        )
        if 'STATUS Installed "media_to_pose"' not in install:
            raise AssertionError("Blender did not report a successful install")
        output = _run(
            [
                str(blender),
                "--background",
                "--factory-startup",
                str(RIG_FIXTURE),
                "--python-exit-code",
                "1",
                "--python",
                str(BLENDER_SCRIPT),
                "--",
                "--provider",
                arguments.provider,
                "--image",
                str(IMAGES[arguments.provider]),
                "--sam-device",
                arguments.sam_device,
            ],
            environment,
            arguments.timeout_seconds,
        )
        if "MTP_PROVIDER_INTEGRATION_OK" not in output:
            raise AssertionError("Provider integration marker is missing")


if __name__ == "__main__":
    main()
