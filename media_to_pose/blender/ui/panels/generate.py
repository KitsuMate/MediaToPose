"""The Generate panel: Kimodo motion from a prompt, keyed poses and paths."""

from __future__ import annotations

from bpy.types import Panel

from ....backends.kimodo import backend as kimodo_backend
from ....backends.llm2vec import backend as llm2vec_backend
from ....core.coordinator import COORDINATOR
from ..widgets import _draw_wrapped_labels

_STATUS_ICONS = {"ERROR": "ERROR", "WARNING": "ERROR", "READY": "CHECKMARK"}


def _draw_models(layout) -> bool:
    box = layout.box()
    box.label(text="Models", icon="PACKAGE")
    kimodo = kimodo_backend.backend_present()
    encoder = llm2vec_backend.backend_present()
    row = box.row()
    if kimodo:
        row.label(text="Kimodo motion model ready", icon="CHECKMARK")
    else:
        row.operator("mtp.kimodo_install", icon="IMPORT")
    row = box.row()
    if encoder:
        row.label(text="Text encoder ready (Built with Meta Llama 3)", icon="CHECKMARK")
    else:
        row.operator("mtp.llm2vec_install", icon="IMPORT")
        box.label(text="Without it, generate from keyed poses only", icon="INFO")
    for module in (kimodo_backend, llm2vec_backend):
        job = module.install_status_snapshot()
        if job.running:
            box.progress(factor=job.progress, text=job.status or "Downloading")
    return kimodo


class MTP_PT_generate(Panel):
    bl_idname = "MTP_PT_generate"
    bl_label = "Generate Motion"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Media to Pose"
    bl_order = 3

    def draw(self, context):
        layout = self.layout
        settings = context.scene.mtp_generate
        busy = COORDINATOR.is_busy()
        ready = _draw_models(layout)

        column = layout.column()
        column.enabled = ready and not busy
        column.prop(settings, "prompt", text="", icon="TEXT")
        if not settings.prompt.strip():
            column.label(text="No prompt: motion follows the guides only", icon="INFO")

        box = column.box()
        box.label(text="Range", icon="TIME")
        box.prop(settings, "use_scene_range")
        if not settings.use_scene_range:
            row = box.row(align=True)
            row.prop(settings, "frame_first")
            row.prop(settings, "frame_last")

        box = column.box()
        box.label(text="Guides", icon="KEYFRAME")
        box.prop(settings, "use_keys")
        box.prop(settings, "path")
        if settings.path is not None:
            box.prop(settings, "face_travel")

        row = column.row(align=True)
        row.prop(settings, "seed")
        row.prop(settings, "output_mode", text="")
        column.prop(
            settings,
            "show_advanced",
            icon="DOWNARROW_HLT" if settings.show_advanced else "RIGHTARROW",
        )
        if settings.show_advanced:
            advanced = column.box()
            for name in (
                "steps",
                "text_weight",
                "guide_weight",
                "blend_frames",
                "device",
            ):
                advanced.prop(settings, name)

        layout.operator("mtp.generate", icon="PLAY")
        if settings.status_kind == "PROGRESS" and busy:
            layout.progress(factor=settings.progress, text=settings.status)
        elif settings.status:
            _draw_wrapped_labels(
                layout,
                settings.status,
                icon=_STATUS_ICONS.get(settings.status_kind, "INFO"),
            )


__all__ = ["MTP_PT_generate"]
