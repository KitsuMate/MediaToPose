"""Stage capture output and restore the exact Blender state on failure."""

from __future__ import annotations

import bpy

from ..core.log import get_logger
from ..core.requests import OutputMode
from ..core.scene import SceneContext

LOGGER = get_logger(__name__)


def _selected_names(view_layer) -> tuple[str, ...]:
    return tuple(
        obj.name for obj in view_layer.objects if obj.select_get(view_layer=view_layer)
    )


class CaptureTransaction:
    """Stage animation in temporary actions and undo everything on rollback.

    The transaction remembers the scene frame, the active and selected
    objects, the armature mode, every pose basis and numeric bone property,
    and the value of every shape key driven by the armature. With
    ``animation`` it also swaps in staging actions so a failed bake never
    touches the user's actions.

    ``output_mode`` decides what ``commit`` does with the staged animation:
    ``NEW_ACTION`` keeps it as a new action (a copy of the current one with
    the capture keyed in), ``REPLACE`` puts it in place of the current
    action under its name, and ``NLA_STRIP`` pushes only the captured keys
    to a new NLA track and restores the current action.
    """

    def __init__(
        self,
        scene_context: SceneContext,
        armature,
        *,
        animation: bool,
        output_mode: OutputMode = OutputMode.NEW_ACTION,
    ):
        scene = scene_context.scene
        view_layer = scene_context.view_layer
        self.scene_context = scene_context
        self.scene_name = scene.name
        self.scene_pointer = scene.as_pointer()
        self.armature = armature
        self.animation = bool(animation)
        self.output_mode = OutputMode(output_mode)
        self.committed = False
        self.frame = (int(scene.frame_current), float(scene.frame_subframe))
        active = view_layer.objects.active
        self.active_name = active.name if active else ""
        self.selected_names = _selected_names(view_layer)
        self.mode = str(getattr(armature, "mode", "OBJECT"))
        self.bone_basis = {
            bone.name: bone.matrix_basis.copy() for bone in armature.pose.bones
        }
        self.bone_properties = {
            bone.name: {
                name: value
                for name in bone.keys()  # noqa: SIM118 - Blender IDProperty API
                if isinstance((value := bone.get(name)), (bool, int, float))
            }
            for bone in armature.pose.bones
        }
        self.shape_values = {}
        for obj in bpy.data.objects:
            if obj.type != "MESH" or obj.find_armature() is not armature:
                continue
            keys = getattr(obj.data, "shape_keys", None)
            for block in getattr(keys, "key_blocks", ()) if keys else ():
                self.shape_values[(keys, block.name)] = float(block.value)
        animation_data = armature.animation_data
        self.original_action = animation_data.action if animation_data else None
        self.temporary_action = None
        self.shape_action_transactions = []
        if self.animation:
            self._stage_actions()

    def _staging_copy(self, original, name: str):
        # An NLA strip holds only the captured keys; the other modes start
        # from the current action so unrelated channels are kept.
        if original is not None and self.output_mode != OutputMode.NLA_STRIP:
            return original.copy()
        return bpy.data.actions.new(name)

    def _stage_actions(self) -> None:
        armature = self.armature
        animation_data = armature.animation_data_create()
        self.temporary_action = self._staging_copy(
            self.original_action, f"{armature.name}_MediaToPose_Staging"
        )
        self.temporary_action.name = f"{armature.name}_MediaToPose_Staging"
        animation_data.action = self.temporary_action
        seen_shape_keys = set()
        for keys, _name in self.shape_values:
            pointer = int(keys.as_pointer())
            if pointer in seen_shape_keys:
                continue
            seen_shape_keys.add(pointer)
            shape_animation = keys.animation_data_create()
            original = shape_animation.action
            temporary = self._staging_copy(original, f"{keys.name}_MediaToPose_Staging")
            temporary.name = f"{keys.name}_MediaToPose_Staging"
            shape_animation.action = temporary
            self.shape_action_transactions.append((keys, original, temporary))

    def commit(self) -> None:
        if self.temporary_action is not None:
            self._commit_action(
                self.armature,
                self.original_action,
                self.temporary_action,
                f"{self.armature.name}_MediaToPose",
            )
        for keys, original, temporary in self.shape_action_transactions:
            self._commit_action(keys, original, temporary, f"{keys.name}_MediaToPose")
        self.committed = True

    def _commit_action(self, owner, original, temporary, name: str) -> None:
        mode = self.output_mode
        if mode == OutputMode.REPLACE and original is not None:
            original_name = original.name
            # Every user of the old action, NLA strips included, now uses
            # the captured one.
            original.user_remap(temporary)
            temporary.use_fake_user = original.use_fake_user
            original.use_fake_user = False
            if original.users == 0:
                bpy.data.actions.remove(original)
            temporary.name = original_name
            return
        temporary.name = name
        if mode != OutputMode.NLA_STRIP:
            return
        animation_data = owner.animation_data_create()
        start = int(round(temporary.frame_range[0]))
        track = animation_data.nla_tracks.new()
        track.name = "Media to Pose"
        track.strips.new(name, start, temporary)
        animation_data.action = original

    @property
    def nla_track_name(self) -> str:
        return "Media to Pose" if self.output_mode == OutputMode.NLA_STRIP else ""

    def _restore_pose(self) -> None:
        armature = self.armature
        for name, basis in self.bone_basis.items():
            bone = armature.pose.bones.get(name)
            if bone is None:
                continue
            bone.matrix_basis = basis
            original = self.bone_properties.get(name, {})
            for property_name in tuple(bone.keys()):
                if (
                    isinstance(bone.get(property_name), (bool, int, float))
                    and property_name not in original
                ):
                    del bone[property_name]
            for property_name, value in original.items():
                bone[property_name] = value
        for (keys, name), value in self.shape_values.items():
            block = keys.key_blocks.get(name)
            if block is not None:
                block.value = value

    def _restore_actions(self) -> None:
        animation_data = self.armature.animation_data_create()
        animation_data.action = self.original_action
        for keys, original, temporary in self.shape_action_transactions:
            shape_animation = keys.animation_data_create()
            shape_animation.action = original
            if temporary.users == 0:
                bpy.data.actions.remove(temporary)
        self.shape_action_transactions.clear()
        if self.temporary_action is not None:
            temporary = self.temporary_action
            self.temporary_action = None
            if temporary.users == 0:
                bpy.data.actions.remove(temporary)

    def _restore_selection(self) -> None:
        view_layer = self.scene_context.view_layer
        armature = self.armature
        active = view_layer.objects.active
        if active is not None and getattr(active, "mode", "OBJECT") != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        for obj in view_layer.objects:
            obj.select_set(False, view_layer=view_layer)
        for name in self.selected_names:
            obj = bpy.data.objects.get(name)
            if obj is not None:
                obj.select_set(True, view_layer=view_layer)
        active = bpy.data.objects.get(self.active_name)
        if active is not None:
            view_layer.objects.active = active
            if self.mode != "OBJECT" and active is armature:
                bpy.ops.object.mode_set(mode=self.mode)

    def rollback(self) -> None:
        if self.committed:
            return
        try:
            self._restore_pose()
            self._restore_actions()
            scene = bpy.data.scenes.get(self.scene_name)
            if scene is not None and scene.as_pointer() == self.scene_pointer:
                scene.frame_set(self.frame[0], subframe=self.frame[1])
            self._restore_selection()
            self.armature.update_tag(refresh={"OBJECT"})
            self.scene_context.update()
        except (AttributeError, ReferenceError, RuntimeError) as exc:
            LOGGER.warning(
                "Capture rollback could not restore every UI state element: %s: %s",
                type(exc).__name__,
                exc,
            )


__all__ = ["CaptureTransaction"]
