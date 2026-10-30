"""Generate motion with Kimodo, and install its two models with their licences.

``MTP_OT_generate`` converts the panel settings into a ``GenerateRequest``
once, then drives a ``GenerateSession`` from a modal timer; Esc cancels and
rolls the animation back.
"""

from __future__ import annotations

import bpy
from bpy.props import BoolProperty
from bpy.types import Operator

from ...api.rig import prepare_rig
from ...backends import licenses
from ...backends.kimodo import backend as kimodo_backend
from ...backends.llm2vec import backend as llm2vec_backend
from ...core.coordinator import COORDINATOR
from ...core.errors import MtpError, OperationBusy
from ...core.requests import OutputMode
from ...core.scene import SceneContext
from ...session.generate_session import GenerateRequest, GenerateSession
from ..classes import register_classes, unregister_classes
from .models import BUSY, ModelJobModal, _tag_redraw


def _status(settings, message: str, kind: str = "INFO", progress: float | None = None):
    settings.status = message
    settings.status_kind = kind
    if progress is not None:
        settings.progress = progress


def _target(context):
    """The panels' Target Character, else the active armature."""
    target = context.scene.mtp_settings.target_armature
    if target is not None and target.name in context.view_layer.objects:
        return target
    active = context.active_object
    return active if active is not None and active.type == "ARMATURE" else None


def _request(context, armature) -> GenerateRequest:
    scene, settings = context.scene, context.scene.mtp_generate
    first, last = (
        (scene.frame_start, scene.frame_end)
        if settings.use_scene_range
        else (settings.frame_first, settings.frame_last)
    )
    return GenerateRequest(
        armature=armature.name,
        first=float(first),
        last=float(last),
        prompt=str(settings.prompt),
        use_keys=bool(settings.use_keys),
        path=settings.path.name if settings.path is not None else "",
        face_travel=bool(settings.face_travel),
        seed=int(settings.seed),
        steps=int(settings.steps),
        text_weight=float(settings.text_weight),
        guide_weight=float(settings.guide_weight),
        blend_frames=float(settings.blend_frames),
        output_mode=OutputMode(settings.output_mode),
        device=str(settings.device),
    )


class MTP_OT_generate(Operator):
    bl_idname = "mtp.generate"
    bl_label = "Generate Motion"
    bl_description = (
        "Generate motion on the active armature over the range, from the "
        "prompt and the keyed poses. Esc cancels"
    )
    bl_options = {"REGISTER", "UNDO"}

    _session: GenerateSession | None = None
    _timer = None

    @classmethod
    def poll(cls, context):
        if _target(context) is None:
            cls.poll_message_set("Choose a target character or select an armature")
            return False
        if COORDINATOR.is_busy():
            cls.poll_message_set(BUSY)
            return False
        if not kimodo_backend.backend_present():
            cls.poll_message_set("Install Kimodo first")
            return False
        return True

    def invoke(self, context, _event):
        settings = context.scene.mtp_generate
        armature = _target(context)
        try:
            # Map a rig that was never set up, as capture does.
            prepare_rig(armature, context.scene.mtp_settings)
            session = GenerateSession(
                _request(context, armature),
                armature,
                SceneContext.from_context(context),
            )
            session.prepare()
            update = session.start()
        except (MtpError, OperationBusy) as exc:
            _status(settings, str(exc), "ERROR")
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self._session = session
        _status(settings, update.status, "PROGRESS", 0.0)
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        session = self._session
        settings = context.scene.mtp_generate
        if session is None:
            return self._end(context, {"CANCELLED"})
        if event.type == "ESC":
            session.cancel()
            _status(settings, "Cancelling", "PROGRESS")
            return {"RUNNING_MODAL"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        update = session.step(budget_seconds=0.05)
        _status(settings, update.status, "PROGRESS", update.progress)
        _tag_redraw(context)
        if not update.state.finished:
            return {"RUNNING_MODAL"}
        if update.state == "SUCCEEDED" and session.result is not None:
            warnings = session.result.warnings
            kind = "WARNING" if warnings else "READY"
            message = warnings[0]["message"] if warnings else update.status
            _status(settings, message, kind, 1.0)
            self.report({"WARNING" if warnings else "INFO"}, message)
            return self._end(context, {"FINISHED"})
        kind = "WARNING" if update.state == "CANCELLED" else "ERROR"
        _status(settings, update.status, kind, 0.0)
        self.report({kind}, update.status)
        return self._end(context, {"CANCELLED"})

    def _end(self, context, result):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        _tag_redraw(context)
        return result

    def cancel(self, context):
        if self._session is not None and not self._session.state.finished:
            self._session.cancel()
            while not self._session.state.finished:
                self._session.step(budget_seconds=0.05)
        self._end(context, {"CANCELLED"})


class _LicensedInstall(ModelJobModal):
    """Show the licence, require the box ticked, then download."""

    module = kimodo_backend
    model_label = "Kimodo"
    stopped_message = "Download stopped"

    accept: BoolProperty(name="I accept the licence")

    @classmethod
    def poll(cls, _context):
        if COORDINATOR.is_busy():
            cls.poll_message_set(BUSY)
            return False
        return True

    def invoke(self, context, _event):
        self.accept = self.module.license_acceptance() is not None
        return context.window_manager.invoke_props_dialog(self, width=560)

    def draw(self, _context):
        layout = self.layout
        licence = self.module.LICENSE
        size = sum(item.bytes for item in self._files())
        layout.label(text=f"Download about {size / 1024**3:.1f} GiB", icon="IMPORT")
        layout.label(text=f"Source: {self.module.REPOSITORY}", icon="WORLD")
        column = layout.column(align=True)
        for line in licence.summary.split(". "):
            column.label(text=line.strip())
        layout.operator("wm.url_open", text=licence.name, icon="URL").url = licence.url
        layout.prop(self, "accept")

    def _files(self):
        return self.module.bundle_files()

    def execute(self, context):
        if not self.accept:
            self.report({"WARNING"}, "Accept the licence to download the model")
            return {"CANCELLED"}
        licence = self.module.LICENSE
        try:
            job = self._start(licence.accept_value(self.module.REVISION))
        except OperationBusy:
            self.report({"WARNING"}, BUSY)
            return {"CANCELLED"}
        except (MtpError, OSError) as exc:
            self.report(
                {"ERROR"}, f"Could not start the {self.model_label} download: {exc}"
            )
            return {"CANCELLED"}
        return self._watch(context, job, f"Downloading {self.model_label}")

    def _start(self, accept: str):
        raise NotImplementedError


class MTP_OT_kimodo_install(_LicensedInstall, Operator):
    bl_idname = "mtp.kimodo_install"
    bl_label = "Install Kimodo"
    bl_description = "Download the Kimodo motion model (NVIDIA Open Model License)"
    backend = kimodo_backend.BACKEND
    module = kimodo_backend
    model_label = "Kimodo"

    def _files(self):
        return kimodo_backend.bundle_files(
            kimodo_backend.resolve_variant("AUTO", "AUTO")
        )

    def _start(self, accept: str):
        return kimodo_backend.start_install(
            online_access=bpy.app.online_access,
            accept_license=accept,
            source=licenses.AcceptanceSource.PANEL,
        )


class MTP_OT_llm2vec_install(_LicensedInstall, Operator):
    bl_idname = "mtp.llm2vec_install"
    bl_label = "Install Text Encoder"
    bl_description = (
        "Download the LLM2Vec text encoder for prompts (Meta Llama 3 Community License)"
    )
    backend = llm2vec_backend.BACKEND
    module = llm2vec_backend
    model_label = "the text encoder"

    def _start(self, accept: str):
        return llm2vec_backend.start_install(
            online_access=bpy.app.online_access,
            accept_license=accept,
            source=licenses.AcceptanceSource.PANEL,
        )


CLASSES = (MTP_OT_generate, MTP_OT_kimodo_install, MTP_OT_llm2vec_install)


def register() -> None:
    register_classes(CLASSES)


def unregister() -> None:
    unregister_classes(CLASSES)
