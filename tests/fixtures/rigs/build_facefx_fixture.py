"""Build tests/fixtures/rigs/facefx_face.blend from the maintainer's Nyx scene.

Run with Blender, pointing at the source scene (it is opened read-only and
never saved):

    blender -b --factory-startup path/to/nyx46.blend \
        --python tests/fixtures/rigs/build_facefx_fixture.py -- OUTPUT.blend

The result holds the ``Nyx.FaceFX`` face mesh (renamed ``FaceFX``) with all
of its shape keys, the face's companion meshes (eyes, brows, eyelashes,
teeth, tongue) and an armature ``FaceFX.Rig``. The armature keeps every bone
those meshes are weighted to, the Auto-Rig Pro face, head and eye
controllers, and every ancestor and constraint target they need, with their
original parents, rest positions and bone constraints, so posing the
controllers moves the face as in the original rig. The ARP marker
properties are copied so the rig is recognised as Auto-Rig Pro. Materials
are replaced with plain colours so no textures are copied.
"""

from __future__ import annotations

import sys

import bpy

SOURCE_RIG = "Nyx.Rig"
# source object -> (fixture name, viewport colour)
MESHES = {
    "Nyx.FaceFX": ("FaceFX", (0.85, 0.72, 0.66, 1.0)),
    "Nyx.Eyes": ("FaceFX.Eyes", (0.12, 0.10, 0.10, 1.0)),
    "Nyx.Brows": ("FaceFX.Brows", (0.25, 0.15, 0.10, 1.0)),
    "Nyx.Eyelashes": ("FaceFX.Eyelashes", (0.05, 0.05, 0.05, 1.0)),
    "Nyx.UpperTeeth": ("FaceFX.UpperTeeth", (0.95, 0.95, 0.92, 1.0)),
    "Nyx.LowerTeeth": ("FaceFX.LowerTeeth", (0.95, 0.95, 0.92, 1.0)),
    "Nyx.Tongue": ("FaceFX.Tongue", (0.80, 0.40, 0.42, 1.0)),
}
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


def _required_bones(source_rig, seeds: set[str]) -> set[str]:
    """``seeds`` plus every ancestor and constraint target, transitively."""
    bones = source_rig.pose.bones
    names: set[str] = set()
    pending = [name for name in seeds if name in bones]
    while pending:
        name = pending.pop()
        if name in names:
            continue
        names.add(name)
        bone = bones[name]
        if bone.parent is not None:
            pending.append(bone.parent.name)
        for constraint in bone.constraints:
            target = getattr(constraint, "target", None)
            subtarget = getattr(constraint, "subtarget", "")
            if target == source_rig and subtarget:
                pending.append(subtarget)
            for item in getattr(constraint, "targets", ()):
                if item.target == source_rig and item.subtarget:
                    pending.append(item.subtarget)
    return names


def _copy_properties(source, target, rig) -> None:
    for prop in source.bl_rna.properties:
        if prop.is_readonly or prop.identifier in {"rna_type", "name", "type"}:
            continue
        value = getattr(source, prop.identifier)
        if prop.type == "POINTER":
            if value is not None and getattr(value, "type", "") == "ARMATURE":
                value = rig
            elif value is not None:
                continue
        try:
            setattr(target, prop.identifier, value)
        except (AttributeError, TypeError, ValueError):
            pass


def _copy_constraints(source_rig, rig, names: set[str]) -> None:
    for name in names:
        source_bone = source_rig.pose.bones[name]
        target_bone = rig.pose.bones[name]
        target_bone.rotation_mode = source_bone.rotation_mode
        for constraint in source_bone.constraints:
            copy = target_bone.constraints.new(constraint.type)
            copy.name = constraint.name
            _copy_properties(constraint, copy, rig)
            for item in getattr(constraint, "targets", ()):
                new_target = copy.targets.new()
                new_target.target = rig
                new_target.subtarget = item.subtarget
                new_target.weight = item.weight


def _copy_mesh(scene, source, name, colour, rig, keep_groups: set[str]):
    data = source.data.copy()
    data.name = name
    obj = bpy.data.objects.new(name, data)
    obj.matrix_world = source.matrix_world.copy()
    scene.collection.objects.link(obj)
    for group in source.vertex_groups:
        obj.vertex_groups.new(name=group.name)
    names = {group.index: group.name for group in source.vertex_groups}
    for vertex in data.vertices:
        for element in vertex.groups:
            group_name = names.get(element.group)
            if group_name is not None:
                obj.vertex_groups[group_name].add(
                    [vertex.index], element.weight, "REPLACE"
                )
    for group in list(obj.vertex_groups):
        if group.name not in keep_groups:
            obj.vertex_groups.remove(group)
    obj.parent = rig
    obj.matrix_parent_inverse = rig.matrix_world.inverted()
    obj.modifiers.new("Armature", "ARMATURE").object = rig
    material = bpy.data.materials.new(name)
    material.diffuse_color = colour
    data.materials.clear()
    data.materials.append(material)
    keys = data.shape_keys
    for block in keys.key_blocks if keys else ():
        block.value = 0.0
    return obj


def _paint_irises(eyes, radius: float = 0.42) -> None:
    """Give the eyes a dark iris so gaze shows without the eye texture.

    On each eye (split at x = 0), front-facing faces within ``radius`` of
    the eye's half-width from its centre in the front view (x and z) get a
    second, dark material.
    """
    data = eyes.data
    sclera = data.materials[0]
    sclera.diffuse_color = (0.92, 0.92, 0.90, 1.0)
    iris = bpy.data.materials.new("FaceFX.Iris")
    iris.diffuse_color = (0.10, 0.18, 0.30, 1.0)
    data.materials.append(iris)
    for side in (-1.0, 1.0):
        vertices = [v.co for v in data.vertices if v.co.x * side > 0]
        xs = [co.x for co in vertices]
        zs = [co.z for co in vertices]
        centre_x = (min(xs) + max(xs)) / 2
        centre_z = (min(zs) + max(zs)) / 2
        half_width = (max(xs) - min(xs)) / 2
        for polygon in data.polygons:
            centre = polygon.center
            if centre.x * side <= 0 or polygon.normal.y > -0.2:
                continue
            distance = ((centre.x - centre_x) ** 2 + (centre.z - centre_z) ** 2) ** 0.5
            if distance < radius * half_width:
                polygon.material_index = 1


def _remove_masked_faces(face, attribute: str = "FaceFX_Strength") -> int:
    """Remove the face polygons the source material renders transparent.

    The source material multiplies its alpha by the ``FaceFX_Strength``
    colour attribute; where it is black (around the eyes) the eyes behind
    show through. Plain materials cannot do that, so those polygons are
    removed. Vertices stay, so shape keys are unchanged. Returns the number
    of polygons removed.
    """
    import bmesh

    mask = face.data.color_attributes[attribute]
    hidden = {
        polygon.index
        for polygon in face.data.polygons
        if sum(mask.data[i].color[0] for i in polygon.loop_indices)
        / len(polygon.loop_indices)
        < 0.5
    }
    mesh = bmesh.new()
    mesh.from_mesh(face.data)
    mesh.faces.ensure_lookup_table()
    bmesh.ops.delete(
        mesh, geom=[mesh.faces[i] for i in sorted(hidden)], context="FACES_ONLY"
    )
    mesh.to_mesh(face.data)
    mesh.free()
    return len(hidden)


def build(output: str) -> None:
    source_rig = bpy.data.objects[SOURCE_RIG]
    rig_bones = source_rig.data.bones
    sources = {name: bpy.data.objects[name] for name in MESHES}
    weighted = {
        group.name
        for obj in sources.values()
        for group in obj.vertex_groups
        if group.name in rig_bones
    }
    names = _required_bones(source_rig, weighted | set(CONTROLLERS))

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
        bone.matrix = source.matrix_local
        bone.use_deform = source.use_deform and name in weighted
        bone.inherit_scale = source.inherit_scale
        bone.use_inherit_rotation = source.use_inherit_rotation
        bone.use_local_location = source.use_local_location
        bone.use_relative_parent = source.use_relative_parent
    for name in sorted(names):
        parent = rig_bones[name].parent
        if parent is not None:
            edit_bones[name].parent = edit_bones[parent.name]
            edit_bones[name].use_connect = rig_bones[name].use_connect
    bpy.ops.object.mode_set(mode="OBJECT")
    _copy_constraints(source_rig, rig, names)
    # Keep the plain-valued Auto-Rig Pro markers so the rig is recognised as
    # ARP, as the original is.
    for source, target in ((source_rig.data, armature_data), (source_rig, rig)):
        for key in source.keys():
            value = source[key]
            if (key.startswith("arp_") or key == "rig_spine_count") and isinstance(
                value, (bool, int, float, str)
            ):
                target[key] = value

    meshes = [
        _copy_mesh(scene, sources[source], name, colour, rig, weighted)
        for source, (name, colour) in MESHES.items()
    ]
    by_name = {obj.name: obj for obj in meshes}
    _paint_irises(by_name["FaceFX.Eyes"])
    removed = _remove_masked_faces(by_name["FaceFX"])
    print("FACEFX_MASKED", removed)

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
        sum(
            len(obj.data.shape_keys.key_blocks) - 1
            for obj in meshes
            if obj.data.shape_keys
        ),
        len(meshes),
    )


if __name__ == "__main__":
    build(sys.argv[sys.argv.index("--") + 1])
