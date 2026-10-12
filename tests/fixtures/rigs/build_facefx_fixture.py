"""Build tests/fixtures/rigs/facefx_face.blend from the maintainer's Nyx scene.

Run with Blender, pointing at the source scene (it is opened read-only and
never saved):

    blender -b --factory-startup path/to/nyx46.blend \
        --python tests/fixtures/rigs/build_facefx_fixture.py -- OUTPUT.blend

The result holds only the ``Nyx.FaceFX`` face mesh with all of its shape
keys, renamed ``FaceFX``, and a small armature ``FaceFX.Rig``. The armature
keeps the Auto-Rig Pro names of every bone the mesh is weighted to plus the
ARP face, head and eye controllers, at their rest positions. Each deform
bone is parented to its ``c_`` controller when one exists, so posing the
controllers moves the face the way ARP's constraints would. The material is
replaced with a plain one so no textures are copied.
"""

from __future__ import annotations

import sys

import bpy

SOURCE_MESH = "Nyx.FaceFX"
SOURCE_RIG = "Nyx.Rig"
CONTROLLERS = (
    "c_head.x",
    "c_neck.x",
    "c_jawbone.x",
    "c_eyebrow_full.l",
    "c_eyebrow_full.r",
    "c_eyebrow_01_end.l",
    "c_eyebrow_01_end.r",
    "c_eyelid_top.l",
    "c_eyelid_top.r",
    "c_eyelid_bot.l",
    "c_eyelid_bot.r",
    "c_cheek_smile.l",
    "c_cheek_smile.r",
    "c_lips_smile.l",
    "c_lips_smile.r",
    "c_lips_top.x",
    "c_lips_bot.x",
    "c_eye.l",
    "c_eye.r",
    "c_eye_target.l",
    "c_eye_target.r",
    "c_eye_target.x",
)


def _nearest_included_parent(bone, names: set[str]):
    parent = bone.parent
    while parent is not None and parent.name not in names:
        parent = parent.parent
    return parent.name if parent is not None else None


def build(output: str) -> None:
    source_mesh = bpy.data.objects[SOURCE_MESH]
    source_rig = bpy.data.objects[SOURCE_RIG]
    rig_bones = source_rig.data.bones
    weighted = {group.name for group in source_mesh.vertex_groups} & set(
        rig_bones.keys()
    )
    names = weighted | {name for name in CONTROLLERS if name in rig_bones}

    scene = bpy.data.scenes.new("FaceFX")
    armature_data = bpy.data.armatures.new("FaceFX.Rig")
    rig = bpy.data.objects.new("FaceFX.Rig", armature_data)
    rig.matrix_world = source_rig.matrix_world.copy()
    scene.collection.objects.link(rig)
    bpy.context.window.scene = scene
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = armature_data.edit_bones
    for name in sorted(names):
        source = rig_bones[name]
        bone = edit_bones.new(name)
        bone.head = source.head_local
        bone.tail = source.tail_local
        bone.roll = 0.0
        bone.matrix = source.matrix_local
        bone.use_deform = name in weighted
    for name in sorted(names):
        bone = edit_bones[name]
        controller = f"c_{name}"
        if controller in names and controller != name:
            bone.parent = edit_bones[controller]
            continue
        parent = _nearest_included_parent(rig_bones[name], names)
        if parent is not None:
            bone.parent = edit_bones[parent]
    bpy.ops.object.mode_set(mode="OBJECT")

    mesh_data = source_mesh.data.copy()
    mesh_data.name = "FaceFX"
    mesh = bpy.data.objects.new("FaceFX", mesh_data)
    mesh.matrix_world = source_mesh.matrix_world.copy()
    scene.collection.objects.link(mesh)
    for group in source_mesh.vertex_groups:
        mesh.vertex_groups.new(name=group.name)
    deform = {group.index: group.name for group in source_mesh.vertex_groups}
    for vertex in mesh_data.vertices:
        for element in vertex.groups:
            name = deform.get(element.group)
            if name is not None:
                mesh.vertex_groups[name].add([vertex.index], element.weight, "REPLACE")
    for group in list(mesh.vertex_groups):
        if group.name not in weighted:
            mesh.vertex_groups.remove(group)
    mesh.parent = rig
    mesh.matrix_parent_inverse = rig.matrix_world.inverted()
    modifier = mesh.modifiers.new("Armature", "ARMATURE")
    modifier.object = rig
    material = bpy.data.materials.new("FaceFX")
    material.diffuse_color = (0.85, 0.72, 0.66, 1.0)
    mesh_data.materials.clear()
    mesh_data.materials.append(material)
    for block in mesh_data.shape_keys.key_blocks:
        block.value = 0.0

    camera_data = bpy.data.cameras.new("FaceFX.Camera")
    camera = bpy.data.objects.new("FaceFX.Camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    for other in list(bpy.data.scenes):
        if other is not scene:
            bpy.data.scenes.remove(other)
    bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    bpy.ops.wm.save_as_mainfile(filepath=output, copy=True, compress=True)
    print(
        "FACEFX_FIXTURE",
        output,
        len(armature_data.bones),
        len(mesh_data.shape_keys.key_blocks) - 1,
        len(mesh.vertex_groups),
    )


if __name__ == "__main__":
    build(sys.argv[sys.argv.index("--") + 1])
