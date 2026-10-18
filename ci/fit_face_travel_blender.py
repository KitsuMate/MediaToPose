"""Fit face bone travel to a rig's own ARKit shape keys.

The developer tool behind ``rig/face_defaults.FACE_CONTROL_TRAVEL``: for
each face bone control with a matching ARKit key, find the output that moves
the meshes closest to the key at full strength (least squares over every
vertex), and how much of the key's motion that explains. Controls that
explain less than a quarter (an Auto-Rig Pro lip centre against ARKit's
lip-side keys) keep a hand-set travel.

    blender --background --factory-startup --python ci/fit_face_travel_blender.py \
        -- [--fixture facefx_face_rig] [--armature FaceFX.Rig]

Prints JSON: per control and side, the fitted travel as a share of the eye
distance (the unit of ``FACE_CONTROL_TRAVEL``) and the share explained.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion

ROOT = Path(__file__).resolve().parents[1]
# Control -> (ARKit keys of the positive side, keys of the negative side).
FIT_SHAPES = {
    "FACE_JAW_OPEN": (("jawOpen",), ()),
    "FACE_JAW_SIDE": (("jawRight",), ("jawLeft",)),
    "FACE_BROW_INNER": (("browInnerUp",), ()),
    "FACE_BROW_OUTER_L": (("browOuterUpLeft",), ("browDownLeft",)),
    "FACE_BROW_OUTER_R": (("browOuterUpRight",), ("browDownRight",)),
    "FACE_LID_L": (("eyeBlinkLeft",), ("eyeWideLeft",)),
    "FACE_LID_R": (("eyeBlinkRight",), ("eyeWideRight",)),
    "FACE_CHEEK_L": (("cheekSquintLeft",), ()),
    "FACE_CHEEK_R": (("cheekSquintRight",), ()),
    "FACE_MOUTH_CORNER_L": (("mouthSmileLeft",), ("mouthFrownLeft",)),
    "FACE_MOUTH_CORNER_R": (("mouthSmileRight",), ("mouthFrownRight",)),
    "FACE_LIP_TOP": (("mouthUpperUpLeft", "mouthUpperUpRight"), ()),
    "FACE_LIP_BOTTOM": (("mouthLowerDownLeft", "mouthLowerDownRight"), ()),
}
# Samples as multiples of the current default; the best is refined.
SAMPLES = (-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
REFINE_STEPS = 16
MIN_EXPLAINED = 0.25


def _arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="facefx_face_rig")
    parser.add_argument("--armature", default="FaceFX.Rig")
    return parser.parse_args(values)


class _Rig:
    """The armature, its mapping and its meshes' evaluated vertices."""

    def __init__(self, addon, armature):
        self.armature = armature
        package = addon.__name__
        self.space_module = importlib.import_module(f"{package}.rig.face_space")
        self.schemes = importlib.import_module(f"{package}.rig.face_schemes")
        mapping = importlib.import_module(f"{package}.rig.mapping")
        self.entries = mapping.mapping_by_id(armature.mtp_rig_config)
        self.space = self.space_module.face_frame(armature, self.entries)
        self.meshes = [
            obj
            for obj in self.schemes.associated_meshes(armature)
            if obj.data.shape_keys is not None or obj.find_armature() == armature
        ]

    def positions(self) -> np.ndarray:
        layer = bpy.context.view_layer
        layer.update()
        chunks = []
        for obj in self.meshes:
            data = obj.evaluated_get(layer.depsgraph).data
            coordinates = np.empty(len(data.vertices) * 3)
            data.vertices.foreach_get("co", coordinates)
            chunks.append(coordinates.reshape(-1, 3))
        return np.concatenate(chunks)

    def reset(self) -> None:
        for bone in self.armature.pose.bones:
            bone.matrix_basis = Matrix.Identity(4)
        for obj in self.meshes:
            for block in obj.data.shape_keys.key_blocks if obj.data.shape_keys else ():
                block.value = 0.0

    def pose(self, bone, entry, value: float) -> None:
        if entry.channel_mode == "LOCATION":
            offset = self.space_module.face_axis(self.space, entry.axis) * value
            bone.matrix_basis = Matrix.Translation(
                self.space_module.to_bone(bone, offset)
            )
        else:
            axis = self.space_module.bone_axis(entry.axis)
            bone.matrix_basis = (
                Quaternion(axis, math.radians(value)).to_matrix().to_4x4()
            )

    def key_target(self, names) -> np.ndarray | None:
        blocks = []
        for name in names:
            entry = self.entries.get(f"ARKIT_{name}")
            if entry is None or not entry.enabled or not entry.target_name:
                return None
            blocks += [
                b for _o, b in self.schemes.matching_shape_blocks(self.armature, entry)
            ]
        if not blocks:
            return None
        for block in blocks:
            block.value = 1.0
        target = self.positions()
        for block in blocks:
            block.value = 0.0
        return target


def _fit(rig: _Rig, bone, entry, target, rest, default: float) -> dict:
    def error(value: float) -> float:
        rig.pose(bone, entry, value)
        difference = rig.positions() - target
        return float(np.einsum("ij,ij->", difference, difference))

    unit = default if abs(default) > 1.0e-9 else 1.0
    samples = [unit * factor for factor in sorted(SAMPLES)]
    errors = [error(value) for value in samples]
    best = min(range(len(samples)), key=errors.__getitem__)
    low = samples[max(best - 1, 0)]
    high = samples[min(best + 1, len(samples) - 1)]
    # Golden-section search between the best sample's neighbours.
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    for _step in range(REFINE_STEPS):
        a, b = high - ratio * (high - low), low + ratio * (high - low)
        if error(a) < error(b):
            high = b
        else:
            low = a
    output = (low + high) / 2.0
    fit_error = min(error(output), errors[best])
    rest_error = float(np.einsum("ij,ij->", rest - target, rest - target))
    rig.pose(bone, entry, 0.0)
    explained = 1.0 - fit_error / rest_error if rest_error > 0.0 else 0.0
    return {
        "travel": round(output / rig.space.scale, 4)
        if entry.channel_mode == "LOCATION"
        else None,
        "output": round(output, 6),
        "default_output": round(default, 6),
        "explained": round(explained, 4),
        "accepted": explained >= MIN_EXPLAINED,
    }


def main() -> None:
    arguments = _arguments()
    sys.path.insert(0, str(ROOT))
    from tests.blender_test_support import fixture_path, register_addon

    addon = register_addon()
    bpy.ops.wm.open_mainfile(filepath=str(fixture_path(arguments.fixture)))
    addon.register()
    armature = bpy.data.objects[arguments.armature]
    addon.api.auto_map(armature.name)
    defaults = importlib.import_module(f"{addon.__name__}.rig.face_defaults")
    rig = _Rig(addon, armature)
    armature.data.pose_position = "POSE"
    rig.reset()
    rest = rig.positions()
    report = {"face_scale": round(rig.space.scale, 6), "controls": {}}
    for semantic_id, (positive, negative) in FIT_SHAPES.items():
        entry = rig.entries.get(semantic_id)
        bone = armature.pose.bones.get(entry.target_name) if entry else None
        if bone is None or entry.channel_mode not in {"LOCATION", "ROTATION"}:
            continue
        ranges = defaults.default_ranges(armature, rig.entries, entry)
        sides = {"full": (positive, ranges["output_full"])}
        if negative and defaults.is_signed(semantic_id):
            sides["negative"] = (negative, ranges["output_negative"])
        fits = {}
        for side, (names, default) in sides.items():
            target = rig.key_target(names)
            if target is not None:
                fits[side] = _fit(rig, bone, entry, target, rest, default)
        if fits:
            report["controls"][semantic_id] = fits
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
