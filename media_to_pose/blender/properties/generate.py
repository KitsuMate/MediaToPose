"""Scene settings of the Generate panel (Kimodo motion generation)."""

from __future__ import annotations

from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Object, PropertyGroup


def _is_curve(_settings, obj: Object) -> bool:
    return obj.type == "CURVE"


class MTPGenerateSettings(PropertyGroup):
    prompt: StringProperty(
        name="Prompt",
        description=(
            'What the person does, for example "A person walks forward and '
            'waves." Leave empty to generate from the keyed poses alone'
        ),
    )
    use_scene_range: BoolProperty(
        name="Scene Range",
        description="Generate over the scene's frame range",
        default=True,
    )
    frame_first: IntProperty(name="First", default=1)
    frame_last: IntProperty(name="Last", default=120)
    use_keys: BoolProperty(
        name="Keys as Guides",
        description=(
            "Keyed full poses in the range are reached exactly; keys of only "
            "the pelvis or root location are waypoints"
        ),
        default=True,
    )
    path: PointerProperty(
        name="Path",
        description="A curve the character follows at constant speed",
        type=Object,
        poll=_is_curve,
    )
    face_travel: BoolProperty(
        name="Face Along Path",
        description="Turn towards the direction of travel",
        default=True,
    )
    seed: IntProperty(name="Seed", default=0, min=0)
    steps: IntProperty(
        name="Steps",
        description="Denoising steps: 25 is fast, 50 the default, 100 the most detailed",
        default=50,
        min=1,
        max=1000,
    )
    text_weight: FloatProperty(
        name="Prompt Strength", default=2.0, min=0.0, soft_max=5.0
    )
    guide_weight: FloatProperty(
        name="Guide Strength", default=2.0, min=0.0, soft_max=5.0
    )
    blend_frames: FloatProperty(
        name="Blend Frames",
        description="Frames over which keyed poses blend into the generated motion",
        default=8.0,
        min=0.0,
        soft_max=30.0,
    )
    output_mode: EnumProperty(
        name="Output",
        items=(
            ("NEW_ACTION", "New Action", "Key into a copy of the current action"),
            ("REPLACE", "Replace", "Key into the current action"),
            ("NLA_STRIP", "NLA Strip", "Put the motion on a new NLA track"),
        ),
        default="NEW_ACTION",
    )
    device: EnumProperty(
        name="Device",
        items=(
            ("AUTO", "Automatic", "GPU when available"),
            ("GPU", "GPU", "WebGPU or CoreML"),
            ("CPU", "CPU", "Processor only"),
        ),
        default="AUTO",
    )
    show_advanced: BoolProperty(name="Advanced", default=False)
    status: StringProperty(default="")
    status_kind: StringProperty(default="INFO")
    progress: FloatProperty(default=0.0, min=0.0, max=1.0, subtype="FACTOR")


__all__ = ["MTPGenerateSettings"]
