"""Generation guides from the rig: keyed poses, root waypoints and curves.

Runs on the main thread before anything is staged. Keyed frames of the
armature's own action slot are classified by which mapped channels carry
keys there:

- most mapped body bones rotated → a full-body guide (the evaluated pose,
  converted to SOMA on SOMA's proportions);
- only the pelvis or root location → a root waypoint (with heading when the
  pelvis rotation is keyed too);
- anything else (fingers, face, custom properties) is not a guide and is
  left as it is.

Explicit waypoints and a curve path add root guides on top of the keys.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from mathutils import Vector

from ..core.errors import InvalidRequest
from ..core.motion import retime, soma
from ..core.motion.kimodo import guides as model_guides
from ..core.scene import SceneContext
from .keyframes import _action_fcurves
from .samples import _restore_sequence_state, _sequence_state
from .soma_pose import SomaRigBinding, current_soma_pose

FULL_BODY_SHARE = 0.5
_BONE_PATH = re.compile(r'^pose\.bones\["((?:[^"\\]|\\.)*)"\]\.(\w+)$')
_ROTATIONS = {"rotation_quaternion", "rotation_euler", "rotation_axis_angle"}
_LIMBS = frozenset(("LEFT_HAND", "RIGHT_HAND", "LEFT_FOOT", "RIGHT_FOOT"))
_FEET = tuple(
    index
    for chain in ("LEFT_FOOT", "RIGHT_FOOT")
    for index in soma.EFFECTOR_CHAINS[chain]
)


class GuideKind(str, Enum):
    FULL_BODY = "FULL_BODY"
    ROOT = "ROOT"


class GuideSource(str, Enum):
    KEYS = "KEYS"
    WAYPOINT = "WAYPOINT"
    PATH = "PATH"


@dataclass(frozen=True)
class SceneGuide:
    """One guide at a scene frame, in Kimodo space.

    ``basis`` holds the keyed bones' local transforms at that frame
    (location, quaternion), which the apply step restores exactly.
    """

    frame: float
    kind: GuideKind
    source: GuideSource
    positions: np.ndarray | None = None
    root_xz: np.ndarray | None = None
    heading: float | None = None
    rotations: np.ndarray | None = None
    basis: Mapping[str, tuple[tuple[float, ...], tuple[float, ...]]] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict:
        payload = {
            "frame": self.frame,
            "kind": self.kind.value,
            "source": self.source.value,
            "keyed_bones": sorted(self.basis),
        }
        if self.root_xz is not None:
            payload["root_xz"] = [round(float(v), 4) for v in self.root_xz]
        if self.heading is not None:
            payload["heading_deg"] = round(math.degrees(self.heading), 2)
        return payload


def _slot_fcurves(armature) -> Iterable[object]:
    """F-Curves of the armature's action, in its own slot only."""
    animation = armature.animation_data
    action = animation.action if animation else None
    if action is None:
        return ()
    slot = getattr(animation, "action_slot", None)
    curves = []
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                if slot is None or bag.slot == slot:
                    curves.extend(bag.fcurves)
    return curves or list(_action_fcurves(action))


def keyed_channels(
    armature, first: float, last: float
) -> dict[float, tuple[set[str], set[str]]]:
    """Per keyed frame in range: bones with rotation keys, bones with
    location keys."""
    frames: dict[float, tuple[set[str], set[str]]] = {}
    for curve in _slot_fcurves(armature):
        match = _BONE_PATH.match(curve.data_path)
        if not match:
            continue
        bone, prop = match.group(1).replace('\\"', '"'), match.group(2)
        if prop not in _ROTATIONS and prop != "location":
            continue
        for point in curve.keyframe_points:
            frame = float(point.co.x)
            if first - 1.0e-4 <= frame <= last + 1.0e-4:
                rotated, located = frames.setdefault(round(frame, 4), (set(), set()))
                (rotated if prop in _ROTATIONS else located).add(bone)
    return frames


def _basis(armature, names: Iterable[str]) -> dict:
    result = {}
    for name in names:
        bone = armature.pose.bones.get(name)
        if bone is None:
            continue
        location, rotation, _scale = bone.matrix_basis.decompose()
        result[name] = (tuple(location), tuple(rotation.normalized()))
    return result


def read_key_guides(
    binding: SomaRigBinding,
    first: float,
    last: float,
    *,
    scene_context: SceneContext,
    ignored: Iterable[float] = (),
) -> list[SceneGuide]:
    """Guides from the keys of the armature's action between ``first`` and
    ``last`` (scene frames, inclusive)."""
    armature = binding.armature
    skipped = {round(float(frame), 4) for frame in ignored}
    mapped = {bone.name for bone in binding.bones.values()}
    pelvis = binding.pelvis.name
    placement = {pelvis} | ({binding.root.name} if binding.root is not None else set())
    state = _sequence_state(armature, scene_context=scene_context)
    guides: list[SceneGuide] = []
    try:
        for frame, (rotated, located) in sorted(
            keyed_channels(armature, first, last).items()
        ):
            if frame in skipped:
                continue
            body = rotated & mapped
            if len(body) >= FULL_BODY_SHARE * len(mapped):
                kind = GuideKind.FULL_BODY
            elif placement & located:
                kind = GuideKind.ROOT
            else:
                continue
            scene_context.set_frame(frame)
            rotations, positions = current_soma_pose(binding)
            if not placement & located:
                # No keyed height: stand the pose on the ground, as the
                # model expects, instead of leaving it at the rest height.
                positions = positions.copy()
                positions[:, 1] -= positions[list(_FEET), 1].min()
            keyed = (rotated | located) & (mapped | placement)
            if kind is GuideKind.FULL_BODY:
                guides.append(
                    SceneGuide(
                        frame,
                        kind,
                        GuideSource.KEYS,
                        positions=positions,
                        root_xz=positions[soma.ROOT, [0, 2]].copy(),
                        rotations=rotations,
                        basis=_basis(armature, keyed),
                    )
                )
            else:
                heading = (
                    float(soma.heading_angle(positions))
                    if placement & rotated
                    else None
                )
                guides.append(
                    SceneGuide(
                        frame,
                        kind,
                        GuideSource.KEYS,
                        root_xz=positions[soma.ROOT, [0, 2]].copy(),
                        heading=heading,
                        basis=_basis(armature, keyed),
                    )
                )
    finally:
        _restore_sequence_state(armature, state, scene_context=scene_context)
    return guides


def _heading_between(start: np.ndarray, end: np.ndarray) -> float | None:
    delta = np.asarray(end) - np.asarray(start)
    if float(np.hypot(delta[0], delta[1])) < 1.0e-4:
        return None
    return float(math.atan2(delta[0], delta[1]))


def _world_to_kimodo_xz(binding: SomaRigBinding, point) -> np.ndarray:
    local = binding.armature.matrix_world.inverted() @ Vector(point)
    return binding.to_kimodo(np.array(local[:]))[[0, 2]]


def waypoint_guides(
    binding: SomaRigBinding,
    waypoints: Sequence[Sequence[float]],
    *,
    face_travel: bool = True,
) -> list[SceneGuide]:
    """Root guides from ``(frame, x, y)`` world positions on the ground.

    With ``face_travel`` each waypoint faces the next one (the last faces
    away from the previous one).
    """
    points = []
    for waypoint in waypoints:
        values = [float(value) for value in waypoint]
        if len(values) not in {3, 4}:
            raise InvalidRequest(
                "A waypoint is FRAME, X, Y (world units)", parameter="waypoints"
            )
        frame, x, y = values[:3]
        points.append((frame, _world_to_kimodo_xz(binding, (x, y, 0.0))))
    points.sort(key=lambda item: item[0])
    guides = []
    for index, (frame, xz) in enumerate(points):
        heading = None
        if face_travel and len(points) > 1:
            if index + 1 < len(points):
                heading = _heading_between(xz, points[index + 1][1])
            else:
                heading = _heading_between(points[index - 1][1], xz)
        guides.append(
            SceneGuide(frame, GuideKind.ROOT, GuideSource.WAYPOINT, None, xz, heading)
        )
    return guides


def path_guides(
    binding: SomaRigBinding,
    curve,
    first: float,
    last: float,
    *,
    scene_context: SceneContext,
    face_travel: bool = True,
) -> list[SceneGuide]:
    """A dense root track along a curve object, at constant speed."""
    if getattr(curve, "type", "") != "CURVE":
        raise InvalidRequest(f"{getattr(curve, 'name', curve)!r} is not a curve")
    depsgraph = scene_context.view_layer.depsgraph
    evaluated = curve.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        world = [evaluated.matrix_world @ vertex.co for vertex in mesh.vertices]
    finally:
        evaluated.to_mesh_clear()
    if len(world) < 2:
        raise InvalidRequest(f"Curve {curve.name!r} has no length")
    track = np.array([_world_to_kimodo_xz(binding, point) for point in world])
    lengths = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(track, axis=0), axis=1)))
    )
    if lengths[-1] <= 1.0e-6:
        raise InvalidRequest(f"Curve {curve.name!r} has no length")
    frames = np.arange(math.ceil(first), math.floor(last) + 1, dtype=np.float64)
    if len(frames) < 2:
        raise InvalidRequest("The frame range is too short for a path")
    distance = (frames - frames[0]) / (frames[-1] - frames[0]) * lengths[-1]
    xs = np.interp(distance, lengths, track[:, 0])
    zs = np.interp(distance, lengths, track[:, 1])
    guides = []
    for index, frame in enumerate(frames):
        heading = None
        if face_travel:
            ahead = min(index + 1, len(frames) - 1)
            behind = ahead - 1
            heading = _heading_between((xs[behind], zs[behind]), (xs[ahead], zs[ahead]))
        guides.append(
            SceneGuide(
                float(frame),
                GuideKind.ROOT,
                GuideSource.PATH,
                None,
                np.array((xs[index], zs[index])),
                heading,
            )
        )
    return guides


def model_guide_list(
    guides: Sequence[SceneGuide], first: float, fps: float, frame_count: int
) -> tuple[list[model_guides.Guide], list[str]]:
    """Kimodo guides at 30 fps frame indices, and collision warnings.

    Two scene guides on one model frame keep the first; full-body guides
    come before root guides, so a keyed pose wins over a waypoint.
    """
    ordered = sorted(
        guides, key=lambda guide: (guide.kind is GuideKind.ROOT, guide.frame)
    )
    used: dict[int, float] = {}
    result: list[model_guides.Guide] = []
    warnings: list[str] = []
    for guide in ordered:
        index = min(
            max(retime.model_index(guide.frame, first, fps), 0), frame_count - 1
        )
        if guide.kind is GuideKind.FULL_BODY:
            if index in used:
                warnings.append(
                    f"Guides at frames {used[index]:g} and {guide.frame:g} share one "
                    "model frame; the first is used"
                )
                continue
            used[index] = guide.frame
            result.append(
                model_guides.FullBodyGuide(index, guide.positions, guide.root_xz)
            )
            if guide.rotations is not None:
                # Hand and foot rotations pin the limb twist, which joint
                # positions alone leave free (as upstream's transitions do).
                result.append(
                    model_guides.EffectorGuide(
                        index,
                        _LIMBS,
                        guide.positions,
                        guide.rotations,
                        guide.root_xz,
                    )
                )
        else:
            result.append(model_guides.RootGuide(index, guide.root_xz, guide.heading))
    return result, warnings


__all__ = [
    "GuideKind",
    "GuideSource",
    "SceneGuide",
    "keyed_channels",
    "model_guide_list",
    "path_guides",
    "read_key_guides",
    "waypoint_guides",
]
