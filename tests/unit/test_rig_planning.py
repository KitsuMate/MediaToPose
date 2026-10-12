"""Compiled rig solve plans and mapping snapshots."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

from media_to_pose.core.geometry import Matrix, Vector
from media_to_pose.rig import planning as rig_planning


class _Bones(list):
    def get(self, name, default=None):
        return next((bone for bone in self if bone.name == name), default)


def _pose_bone(name: str, parent=None, constraints=()):
    parents = []
    current = parent
    while current is not None:
        parents.append(current)
        current = current.parent
    return SimpleNamespace(
        name=name,
        parent=parent,
        parent_recursive=parents,
        constraints=list(constraints),
        bone=SimpleNamespace(
            matrix_local=Matrix.Identity(4),
            head_local=Vector((0.0, 0.0, len(parents))),
            tail_local=Vector((0.0, 0.0, len(parents) + 1.0)),
        ),
    )


def _armature(pointer: int = 1, constraints=()):
    hips = _pose_bone("Hips")
    upper = _pose_bone("UpperArm.L", hips)
    lower = _pose_bone("LowerArm.L", upper, constraints)
    hand = _pose_bone("Hand.L", lower)
    return SimpleNamespace(
        name="Rig",
        data=SimpleNamespace(name="RigData", animation_data=None),
        animation_data=None,
        pose=SimpleNamespace(bones=_Bones([hand, lower, hips, upper])),
        as_pointer=lambda: pointer,
    )


def _entry(target: str, **overrides):
    values = {"target_name": target, "enabled": True, "section": "BODY"}
    values.update(overrides)
    return SimpleNamespace(**values)


ENTRIES = {
    "PELVIS": _entry("Hips"),
    "UPPER_ARM_L": _entry("UpperArm.L"),
    "LOWER_ARM_L": _entry("LowerArm.L"),
    "HAND_L": _entry("Hand.L"),
    "HAND_R": _entry("Missing.R"),
}


class SnapshotTests(unittest.TestCase):
    def test_snapshot_fills_defaults_and_is_immutable(self) -> None:
        snapshot = rig_planning.snapshot_mapping_entries({"HEAD": _entry("Head")})
        head = snapshot["HEAD"]
        self.assertEqual(head.target_name, "Head")
        self.assertEqual(head.channel_mode, "AUTO")
        self.assertEqual(head.correction_euler, (0.0, 0.0, 0.0))
        self.assertEqual(head.shape_overrides, ())
        with self.assertRaises(FrozenInstanceError):
            head.target_name = "Other"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            snapshot["NECK"] = head  # type: ignore[index]


class SolvePlanTests(unittest.TestCase):
    def setUp(self) -> None:
        rig_planning.clear_rig_solve_plans()
        self.addCleanup(rig_planning.clear_rig_solve_plans)

    def test_plan_orders_hierarchy_and_resolves_semantics(self) -> None:
        plan = rig_planning.compile_rig_solve_plan(
            _armature(), ENTRIES, lambda: Matrix.Identity(3)
        )
        self.assertEqual(
            plan.hierarchy_order, ("Hips", "UpperArm.L", "LowerArm.L", "Hand.L")
        )
        self.assertEqual(plan.hierarchy_depth["Hand.L"], 3)
        self.assertNotIn("HAND_R", plan.semantic_bones)
        self.assertEqual(
            plan.ik_chains[("ARM", "L")], ("UpperArm.L", "LowerArm.L", "Hand.L")
        )
        self.assertNotIn(("ARM", "R"), plan.ik_chains)
        self.assertTrue(plan.constraint_free)

    def test_plan_is_cached_until_the_mapping_changes(self) -> None:
        armature = _armature()
        first = rig_planning.compile_rig_solve_plan(
            armature, ENTRIES, lambda: Matrix.Identity(3)
        )
        again = rig_planning.compile_rig_solve_plan(
            armature, ENTRIES, lambda: Matrix.Identity(3)
        )
        self.assertIs(first, again)
        changed = dict(ENTRIES)
        changed["HAND_L"] = _entry("Hand.L", input_full=2.0)
        rebuilt = rig_planning.compile_rig_solve_plan(
            armature, changed, lambda: Matrix.Identity(3)
        )
        self.assertIsNot(first, rebuilt)

    def test_constraints_make_the_plan_not_constraint_free(self) -> None:
        constraint = SimpleNamespace(
            name="IK", type="IK", target=SimpleNamespace(name="Rig"), subtarget="Pole"
        )
        plan = rig_planning.compile_rig_solve_plan(
            _armature(pointer=2, constraints=[constraint]),
            ENTRIES,
            lambda: Matrix.Identity(3),
        )
        self.assertFalse(plan.constraint_free)
        self.assertEqual(plan.constraints[0].subtarget, "Pole")
        self.assertEqual(plan.constraints[0].bone_name, "LowerArm.L")


if __name__ == "__main__":
    unittest.main()
