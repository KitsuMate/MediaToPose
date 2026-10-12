"""Validate all platform-split Media to Pose release archives."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import stat
import sys
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "tests" / "fixtures" / "benchmarks" / "distribution_0_1_0.json"
ARCHIVE_TARGET_BYTES = 300 * 1024 * 1024
ARCHIVE_HARD_MAX_BYTES = int(1.9 * 1024**3)
ROHM_COMPRESSED_BUDGET_BYTES = 512 * 1024
UNEXPECTED_MEMBER_LIMIT_BYTES = 1024 * 1024
ARCHIVE_BASELINE_BYTES = {
    "windows_x64": 137_535_113,
    "linux_x64": 170_460_483,
    "macos_arm64": 142_029_193,
}
PLATFORMS = {
    "windows_x64": {
        "manifest": "windows-x64",
        "sys_platform": "win32",
        "platform_system": "Windows",
        "platform_machine": "AMD64",
        "os_name": "nt",
    },
    "linux_x64": {
        "manifest": "linux-x64",
        "sys_platform": "linux",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "os_name": "posix",
    },
    "macos_arm64": {
        "manifest": "macos-arm64",
        "sys_platform": "darwin",
        "platform_system": "Darwin",
        "platform_machine": "arm64",
        "os_name": "posix",
    },
}
ROOT_FILES = {
    "__init__.py",
    "animation_cleanup.py",
    "anycalib_backend.py",
    "anycalib_ort_runtime.py",
    "blender_manifest.toml",
    "cache_preview.py",
    "camera_calibration.py",
    "camera_calibration_pipeline.py",
    "camera_motion_correction.py",
    "capture_state.py",
    "constants.py",
    "dependency_bootstrap.py",
    "detection_cache.py",
    "grounded_cleanup.py",
    "LICENSE",
    "mapping.py",
    "media_io.py",
    "model_lifecycle.py",
    "models.py",
    "motion_filter.py",
    "operation.py",
    "operators_capture.py",
    "operators_mapping.py",
    "pose_translation_solver.py",
    "post_processing.py",
    "properties.py",
    "README.md",
    "registration.py",
    "retarget.py",
    "rig_planning.py",
    "rohm_backend.py",
    "rohm_correction.py",
    "rohm_ort_runtime.py",
    "sam3d_backend.py",
    "sam3d_mhr_runtime.py",
    "sam3d_ort_runtime.py",
    "sam3d_state.py",
    "sam3d_vith_runtime.py",
    "target_pose.py",
    "THIRD_PARTY_NOTICES.md",
    "ui.py",
}
NESTED_FILES = {
    "providers/__init__.py",
    "providers/base.py",
    "providers/mediapipe_provider.py",
    "providers/sam3d_body_provider.py",
    "assets/models/README.md",
    "assets/models/face_landmarker.task",
    "assets/models/hand_landmarker.task",
    "assets/models/pose_landmarker_full.task",
}
BLENDER_PROVIDED_REQUIREMENTS = {"numpy"}
INTENTIONALLY_PRUNED_REQUIREMENTS = {
    # MediaPipe declares its audio capture stack unconditionally. Vision To
    # Pose uses only Tasks image/video inference; real Blender import and
    # end-to-end release smoke tests guard this explicit pruning exception.
    "cffi",
    "pycparser",
    "sounddevice",
}
EXPECTED_ONNX_RUNTIME = {
    "windows_x64": ("onnxruntime-webgpu", Version("1.27.0")),
    "linux_x64": ("onnxruntime-webgpu", Version("1.27.0")),
    "macos_arm64": ("onnxruntime", Version("1.23.2")),
}
FORBIDDEN_DISTRIBUTIONS = {
    "chumpy",
    "human-body-prior",
    "onnxruntime-gpu",
    "open3d",
    "pyrender",
    "pytorch-triton",
    "scipy",
    "smplx",
    "torch",
    "torchaudio",
    "torchvision",
    "trimesh",
    "triton",
}
FORBIDDEN_DISTRIBUTION_PREFIXES = (
    "cupy",
    "cuda-",
    "nvidia-",
    "torch-",
)
FORBIDDEN_PARTS = {
    ".git",
    ".github",
    ".artifacts",
    ".workspace",
    "AGENTS.md",
    "tests",
    "tools",
    "dist",
    "__pycache__",
}
FORBIDDEN_RELEASE_SUFFIXES = {
    ".ckpt",
    ".onnx",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
}
FORBIDDEN_RELEASE_TOKENS = {
    "benchmark",
    "checkpoints",
    "conversion",
    "prox",
    "smpl",
    "smpl-x",
    "smplx",
}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _wheel_metadata(archive: zipfile.ZipFile, member: str):
    with zipfile.ZipFile(io.BytesIO(archive.read(member))) as wheel:
        metadata_name = next(
            (name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")),
            None,
        )
        if metadata_name is None:
            raise AssertionError(f"{member}: wheel has no METADATA")
        metadata = Parser().parsestr(
            wheel.read(metadata_name).decode("utf-8", "replace")
        )
        if wheel.testzip() is not None:
            raise AssertionError(f"{member}: wheel CRC failure")
        return metadata


def _compatible_wheel(filename: str, platform: str) -> bool:
    value = filename.casefold()
    if value.endswith("-none-any.whl"):
        return True
    if platform == "windows_x64":
        return value.endswith("-win_amd64.whl")
    if platform == "linux_x64":
        return "manylinux" in value and value.endswith("_x86_64.whl")
    return "macosx" in value and (value.endswith(("_arm64.whl", "_universal2.whl")))


def _validate_member(name: str) -> str:
    if "\\" in name or "\x00" in name:
        raise AssertionError(f"unsafe archive path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise AssertionError(f"unsafe archive path: {name!r}")
    if FORBIDDEN_PARTS.intersection(part.casefold() for part in path.parts):
        raise AssertionError(f"forbidden release content: {name}")
    if name.endswith((".pyc", ".pyo")):
        raise AssertionError(f"bytecode leaked into release: {name}")
    suffix = path.suffix.casefold()
    if suffix in FORBIDDEN_RELEASE_SUFFIXES:
        raise AssertionError(f"forbidden model or pickle payload: {name}")
    tokens = {
        token
        for part in path.parts
        for token in part.casefold().replace("_", "-").split("-")
    }
    if FORBIDDEN_RELEASE_TOKENS.intersection(tokens):
        raise AssertionError(f"forbidden bundled model content: {name}")
    return str(path)


def _validate_member_sizes(infos: list[zipfile.ZipInfo]) -> tuple[list[dict], int]:
    members = []
    rohm_compressed_bytes = 0
    for info in infos:
        if info.is_dir():
            continue
        name = str(PurePosixPath(info.filename))
        allowed_large = (name.startswith("wheels/") and name.endswith(".whl")) or (
            name.startswith("assets/models/") and name.endswith(".task")
        )
        if info.file_size > UNEXPECTED_MEMBER_LIMIT_BYTES and not allowed_large:
            raise AssertionError(
                f"unexpected member over 1 MiB: {name} ({info.file_size} bytes)"
            )
        if PurePosixPath(name).name.casefold().startswith("rohm") or name.startswith(
            "assets/rohm/"
        ):
            rohm_compressed_bytes += info.compress_size
        members.append(
            {
                "name": name,
                "bytes": info.file_size,
                "compressed_bytes": info.compress_size,
            }
        )
    if rohm_compressed_bytes > ROHM_COMPRESSED_BUDGET_BYTES:
        raise AssertionError(
            "RoHM code and metadata exceed the 512 KiB compressed budget: "
            f"{rohm_compressed_bytes} bytes"
        )
    members.sort(key=lambda value: (-value["compressed_bytes"], value["name"]))
    return members, rohm_compressed_bytes


def _validate_allowlist(names: set[str]) -> None:
    for name in names:
        if name in ROOT_FILES or name in NESTED_FILES:
            continue
        if name.startswith("wheels/") and name.endswith(".whl"):
            continue
        raise AssertionError(f"release member is not allowlisted: {name}")
    missing = (ROOT_FILES | NESTED_FILES) - names
    if missing:
        raise AssertionError(f"release is missing required members: {sorted(missing)}")


def _validate_dependency_closure(
    archive: zipfile.ZipFile,
    wheel_members: list[str],
    platform: str,
) -> dict[str, str]:
    environment = default_environment()
    environment.update(PLATFORMS[platform])
    environment.update(
        {
            "python_version": "3.13",
            "python_full_version": "3.13.0",
            "implementation_name": "cpython",
            "platform_python_implementation": "CPython",
            "extra": "",
        }
    )
    metadata = {member: _wheel_metadata(archive, member) for member in wheel_members}
    available = {
        canonicalize_name(value["Name"]): Version(value["Version"])
        for value in metadata.values()
    }
    for member, value in metadata.items():
        for raw_requirement in value.get_all("Requires-Dist", []):
            requirement = Requirement(raw_requirement)
            if requirement.marker is not None and not requirement.marker.evaluate(
                environment
            ):
                continue
            name = canonicalize_name(requirement.name)
            if name in (
                BLENDER_PROVIDED_REQUIREMENTS | INTENTIONALLY_PRUNED_REQUIREMENTS
            ):
                continue
            installed = available.get(name)
            if installed is None:
                raise AssertionError(
                    f"{member}: missing transitive wheel {requirement.name}"
                )
            if requirement.specifier and installed not in requirement.specifier:
                raise AssertionError(
                    f"{member}: {requirement} is not satisfied by {installed}"
                )
    return {name: str(version) for name, version in sorted(available.items())}


def _validate_archive_size(path: Path, platform: str) -> int:
    size = path.stat().st_size
    if size >= ARCHIVE_HARD_MAX_BYTES:
        raise AssertionError(
            f"{path.name}: {size} bytes reaches the hard 1.9 GiB release limit"
        )
    if size > ARCHIVE_TARGET_BYTES:
        raise AssertionError(
            f"{path.name}: {size} bytes exceeds the 300 MiB release budget"
        )
    baseline = ARCHIVE_BASELINE_BYTES[platform]
    if size > baseline:
        raise AssertionError(
            f"{path.name}: {size} bytes exceeds its release baseline "
            f"of {baseline} bytes"
        )
    return size


def _is_forbidden_distribution(name: str) -> bool:
    if name in FORBIDDEN_DISTRIBUTIONS:
        return True
    if name.startswith(FORBIDDEN_DISTRIBUTION_PREFIXES):
        return True
    return any(token in name for token in ("cuda", "cudnn", "nccl"))


def _validate_onnx_runtime(
    dependencies: dict[str, str], platform: str, archive_name: str
) -> dict[str, object]:
    forbidden = sorted(
        name for name in dependencies if _is_forbidden_distribution(name)
    )
    if forbidden:
        raise AssertionError(
            f"{archive_name}: forbidden Torch/CUDA distributions: {forbidden}"
        )
    runtime_packages = {
        name: Version(version)
        for name, version in dependencies.items()
        if name == "onnxruntime" or name.startswith("onnxruntime-")
    }
    if len(runtime_packages) != 1:
        raise AssertionError(
            f"{archive_name}: expected exactly one ONNX Runtime distribution, "
            f"found {sorted(runtime_packages)}"
        )
    actual_name, actual_version = next(iter(runtime_packages.items()))
    expected_name, expected_version = EXPECTED_ONNX_RUNTIME[platform]
    if (actual_name, actual_version) != (expected_name, expected_version):
        raise AssertionError(
            f"{archive_name}: expected {expected_name}=={expected_version}, "
            f"found {actual_name}=={actual_version}"
        )
    return {"distribution": actual_name, "version": str(actual_version)}


def validate_archive(path: Path, platform: str, source_manifest: dict) -> dict:
    archive_bytes = _validate_archive_size(path, platform)
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        member_sizes, rohm_compressed_bytes = _validate_member_sizes(infos)
        raw_names = [info.filename for info in infos if not info.is_dir()]
        normalized = [_validate_member(name) for name in raw_names]
        folded = [name.casefold() for name in normalized]
        if len(folded) != len(set(folded)):
            raise AssertionError(f"{path.name}: duplicate archive destinations")
        for info in infos:
            mode = int(info.external_attr) >> 16
            if stat.S_ISLNK(mode):
                raise AssertionError(f"{path.name}: symlink member {info.filename}")
        corrupt = archive.testzip()
        if corrupt is not None:
            raise AssertionError(f"{path.name}: CRC failure in {corrupt}")
        names = set(normalized)
        _validate_allowlist(names)
        embedded_manifest = tomllib.loads(
            archive.read("blender_manifest.toml").decode("utf-8")
        )
        comparable_manifest = copy.deepcopy(embedded_manifest)
        generated = comparable_manifest.get("build", {}).pop("generated", None)
        if comparable_manifest != source_manifest:
            raise AssertionError(f"{path.name}: embedded manifest differs from source")
        if not isinstance(generated, dict):
            raise AssertionError(f"{path.name}: split-build metadata is missing")
        if generated.get("platforms") != [PLATFORMS[platform]["manifest"]]:
            raise AssertionError(f"{path.name}: generated platform is inconsistent")
        wheel_members = sorted(name for name in names if name.startswith("wheels/"))
        declared = {PurePosixPath(value).name for value in generated.get("wheels", [])}
        packaged = {PurePosixPath(member).name for member in wheel_members}
        if packaged != declared:
            raise AssertionError(
                f"{path.name}: packaged wheels differ from generated manifest"
            )
        for member in wheel_members:
            filename = PurePosixPath(member).name
            if filename not in declared:
                raise AssertionError(f"{path.name}: undeclared wheel {filename}")
            if not _compatible_wheel(filename, platform):
                raise AssertionError(
                    f"{path.name}: incompatible wheel for {platform}: {filename}"
                )
            source_wheel = ROOT / "wheels" / filename
            if _sha256(archive.read(member)) != _sha256(source_wheel.read_bytes()):
                raise AssertionError(f"{path.name}: wheel changed during packaging")
        dependencies = _validate_dependency_closure(
            archive,
            wheel_members,
            platform,
        )
        runtime = _validate_onnx_runtime(dependencies, platform, path.name)
        source_hashes = {
            name: _sha256(archive.read(name))
            for name in sorted(names)
            if name != "blender_manifest.toml" and not name.startswith("wheels/")
        }
        for name, archived_hash in source_hashes.items():
            source = ROOT / PurePosixPath(name)
            if not source.is_file():
                raise AssertionError(
                    f"{path.name}: release source no longer exists: {name}"
                )
            if _sha256(source.read_bytes()) != archived_hash:
                raise AssertionError(f"{path.name}: release source is stale: {name}")
        return {
            "archive": path.name,
            "bytes": archive_bytes,
            "size_budget": {
                "target_bytes": ARCHIVE_TARGET_BYTES,
                "hard_max_bytes": ARCHIVE_HARD_MAX_BYTES,
                "target_headroom_bytes": ARCHIVE_TARGET_BYTES - archive_bytes,
            },
            "members": len(names),
            "member_sizes": member_sizes,
            "rohm_compressed_bytes": rohm_compressed_bytes,
            "wheels": len(wheel_members),
            "dependencies": dependencies,
            "intentionally_pruned_requirements": sorted(
                INTENTIONALLY_PRUNED_REQUIREMENTS
            ),
            "onnx_runtime": runtime,
            "source_hashes": source_hashes,
        }


def _growth(
    current: Path,
    platform: str,
    dist_directory: Path,
    maximum_growth: float,
) -> float | None:
    candidates = sorted(dist_directory.glob(f"media_to_pose-*-{platform}.zip"))
    previous = [path for path in candidates if path != current]
    if previous:
        baseline = max(previous, key=lambda path: path.stat().st_mtime)
        baseline_bytes = baseline.stat().st_size
        baseline_label = baseline.name
    else:
        baseline_data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        baseline_bytes = int(baseline_data["archives"][platform]["bytes"])
        baseline_label = str(baseline_data["version"])
    ratio = current.stat().st_size / baseline_bytes - 1.0
    if ratio > maximum_growth:
        raise AssertionError(
            f"{current.name}: unexplained growth {ratio:.1%} exceeds "
            f"{maximum_growth:.1%} relative to {baseline_label}"
        )
    return ratio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path, default=ROOT / "dist")
    parser.add_argument(
        "--report",
        type=Path,
        help="Optionally write the full machine-readable validation report",
    )
    parser.add_argument("--max-growth", type=float, default=0.05)
    parser.add_argument(
        "--platform",
        action="append",
        choices=tuple(PLATFORMS),
        help="Validate only this platform (repeatable)",
    )
    arguments = parser.parse_args()
    with (ROOT / "blender_manifest.toml").open("rb") as stream:
        manifest = tomllib.load(stream)
    version = str(manifest["version"])
    reports = []
    selected_platforms = arguments.platform or list(PLATFORMS)
    for platform in selected_platforms:
        path = arguments.dist / f"media_to_pose-{version}-{platform}.zip"
        if not path.is_file():
            raise FileNotFoundError(f"Missing release archive: {path}")
        report = validate_archive(path, platform, manifest)
        report["growth"] = _growth(
            path,
            platform,
            arguments.dist,
            arguments.max_growth,
        )
        reports.append(report)
    common = reports[0]["source_hashes"]
    for report in reports[1:]:
        if report["source_hashes"] != common:
            raise AssertionError(
                f"{report['archive']}: common source hashes differ by platform"
            )
    payload = {"version": version, "archives": reports}
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf8",
        )
    print(f"Validated {len(reports)} release archive(s) for Media to Pose {version}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"DISTRIBUTION_VALIDATION_FAILED: {exc}", file=sys.stderr)
        raise
