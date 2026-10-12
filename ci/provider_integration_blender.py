"""Run real provider inference and bounded VRoid smoke gates inside Blender."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import bpy
from mathutils import Matrix

MODULE_NAME = "bl_ext.user_default.media_to_pose"
MEDIAPIPE_SPINE_TANGENT_LIMIT_DEGREES = 10.0
MEDIAPIPE_SPINE_DRIFT_LIMIT = 0.075
SAM_SPINE_TANGENT_LIMIT_DEGREES = 5.0
SAM_SPINE_DRIFT_LIMIT = 0.05
FINGER_ENDPOINT_LIMIT = 0.12


def _arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("mediapipe", "sam3d"), required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--sam-device", default="AUTO")
    return parser.parse_args(values)


def _installed_addon():
    if not hasattr(bpy.types.Scene, "mtp_settings"):
        enabled = bpy.ops.preferences.addon_enable(module=MODULE_NAME)
        if enabled != {"FINISHED"}:
            raise RuntimeError(f"Could not enable installed extension: {enabled}")
    addon = importlib.import_module(MODULE_NAME)
    extension_root = Path(os.environ["BLENDER_USER_EXTENSIONS"]).resolve()
    module_path = Path(addon.__file__).resolve()
    if not module_path.is_relative_to(extension_root):
        raise AssertionError(
            f"Loaded source checkout instead of archive: {module_path}"
        )
    return addon


def _detect(addon, provider_name: str, image: Path, sam_device: str):
    common = {
        "use_body": True,
        "use_hands": True,
        "use_face": False,
        "mirror": False,
        "confidence": 0.35,
        "smoothing": 0.0,
        "video_mode": False,
    }
    if provider_name == "mediapipe":
        provider = addon.providers.MediaPipeProvider(
            model_directory=str(Path(addon.__file__).parent / "assets" / "models"),
            **common,
        )
    else:
        options = addon.backends.sam3d.backend.runtime_paths()
        provider = addon.providers.SAM3DBodyProvider(
            variant="VITH",
            device=sam_device,
            verify_accelerator_execution=sam_device.upper() == "GPU",
            **common,
            **options,
        )
    try:
        return provider.detect_image(str(image))
    finally:
        provider.close()


def _spine_quality(addon, armature, frame):
    entries = addon.mapping.mapping_by_id(armature.mtp_rig_config)
    samples, inferred = addon.postprocess.CORRECTION_REGISTRY[
        "SPINE_CORRECTION"
    ]._source_curve(SimpleNamespace(frame=frame))
    channels = []
    for semantic_id in ("SPINE_01", "SPINE_02", "CHEST"):
        bone = addon.retarget.evaluated_semantic_bone(
            armature,
            entries,
            semantic_id,
        )
        if bone is not None and all(
            existing[1].name != bone.name for existing in channels
        ):
            channels.append((semantic_id, bone))
    if not samples or len(channels) != 3:
        raise AssertionError("Could not resolve the three VRoid spine controls")
    lengths = [max(float(bone.length), 1.0e-8) for _name, bone in channels]
    total = sum(lengths)
    cursor = 0.0
    expected_cursor = channels[0][1].head.copy()
    source_basis = addon.retarget.source_root_basis(frame)
    target_basis = addon.retarget.target_basis(armature, entries)
    neck = addon.retarget.evaluated_semantic_bone(armature, entries, "NECK")
    anatomical = [bone for _name, bone in channels] + [neck]
    tangents = []
    drift = []
    for index, ((_semantic_id, bone), length) in enumerate(
        zip(channels, lengths, strict=False)
    ):
        start_fraction = cursor / total
        cursor += length
        end_fraction = cursor / total
        source_segment = addon.postprocess.base._arc_point(
            samples, end_fraction
        ) - addon.postprocess.base._arc_point(samples, start_fraction)
        intended = addon.retarget.solve.frames._to_target_direction(
            source_segment,
            source_basis,
            target_basis,
        ).normalized()
        endpoint = anatomical[index + 1].head
        actual = endpoint - bone.head
        tangents.append(math.degrees(actual.angle(intended)))
        expected_cursor += intended * actual.length
        drift.append((endpoint - expected_cursor).length / total)
    return bool(inferred), tangents, drift


def _retarget_and_assert(addon, provider_name: str, frame) -> dict:
    armature = next(
        (obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"),
        None,
    )
    if armature is None:
        raise AssertionError("VRoid fixture has no armature")
    armature.mtp_rig_config.mappings.clear()
    addon.mapping.auto_map(armature, force_profile=True)
    addon.retarget.prepare_limb_ownership(armature)
    for bone in armature.pose.bones:
        bone.matrix_basis = Matrix.Identity(4)
    settings = SimpleNamespace(
        provider="MEDIAPIPE" if provider_name == "mediapipe" else "SAM3D_BODY",
        use_body=True,
        use_hands=True,
        use_face=False,
        face_output="ARKIT",
    )
    changed_bones, changed_shapes = addon.retarget.apply_frame(
        armature,
        frame,
        settings,
        scene_context=addon.core.scene.SceneContext.from_context(bpy.context),
    )
    if changed_bones != 51 or changed_shapes != 0:
        raise AssertionError(
            f"Unexpected retarget output: {changed_bones} bones, "
            f"{changed_shapes} shapes"
        )
    inferred, tangents, drift = _spine_quality(addon, armature, frame)
    if provider_name == "mediapipe":
        if (
            not inferred
            or max(tangents) >= MEDIAPIPE_SPINE_TANGENT_LIMIT_DEGREES
            or max(drift) >= MEDIAPIPE_SPINE_DRIFT_LIMIT
        ):
            raise AssertionError(
                f"MediaPipe spine quality failed: tangent={tangents}, drift={drift}"
            )
    elif (
        inferred
        or max(tangents) >= SAM_SPINE_TANGENT_LIMIT_DEGREES
        or max(drift) >= SAM_SPINE_DRIFT_LIMIT
    ):
        raise AssertionError(
            f"SAM spine quality failed: tangent={tangents}, drift={drift}"
        )
    head_errors = addon.retarget.diagnostics.normalized_head_position_errors(
        armature, frame
    )
    if not head_errors or max(head_errors.values()) >= 0.01:
        raise AssertionError(f"Head endpoint quality failed: {head_errors}")
    entries = addon.mapping.mapping_by_id(armature.mtp_rig_config)
    finger_errors = addon.retarget.diagnostics.finger_world_position_errors(
        armature,
        entries,
        frame,
    )
    if len(finger_errors) != 30 or max(finger_errors.values()) >= FINGER_ENDPOINT_LIMIT:
        raise AssertionError(
            f"Finger endpoint quality failed: count={len(finger_errors)}, "
            f"maximum={max(finger_errors.values(), default=math.inf)}"
        )
    return {
        "changed_bones": changed_bones,
        "valid_body": sum(point.valid for point in frame.body.values()),
        "hands": sorted(frame.hands),
        "spine_tangent_max_degrees": max(tangents),
        "spine_drift_max_body_scale": max(drift),
        "head_error_max_body_scale": max(head_errors.values()),
        "finger_error_max_body_scale": max(finger_errors.values()),
    }


def main() -> None:
    arguments = _arguments()
    image = arguments.image.resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    addon = _installed_addon()
    available, message = addon.dependency_bootstrap.ensure_bundled_dependencies()
    if not available:
        raise RuntimeError(message)
    frame = _detect(addon, arguments.provider, image, arguments.sam_device)
    valid_body = sum(point.valid for point in frame.body.values())
    minimum_body = 33 if arguments.provider == "mediapipe" else 27
    if valid_body < minimum_body:
        raise AssertionError(
            f"{arguments.provider} produced {valid_body} valid body landmarks; "
            f"expected at least {minimum_body}"
        )
    if set(frame.hands) != {"LEFT", "RIGHT"}:
        raise AssertionError(f"Both hands were not detected: {sorted(frame.hands)}")
    report = _retarget_and_assert(addon, arguments.provider, frame)
    if arguments.provider == "sam3d":
        diagnostics = addon.backends.sam3d.backend.runtime_diagnostics(
            "VITH",
            arguments.sam_device,
        )
        if arguments.sam_device.upper() == "GPU":
            if diagnostics.get("actual_device") == "cpu" or diagnostics.get("fallback"):
                raise AssertionError(
                    f"GPU integration silently fell back: {diagnostics}"
                )
            if not diagnostics.get("accelerator_nodes_verified"):
                raise AssertionError(
                    "The real SAM graph did not pass accelerator node profiling"
                )
        report["runtime"] = diagnostics
    print("MTP_PROVIDER_INTEGRATION_OK", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
