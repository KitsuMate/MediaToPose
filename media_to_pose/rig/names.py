"""Bone and shape-key name normalisation shared by every rig profile."""

from __future__ import annotations

import re

PREFIX_RE = re.compile(
    r"^(?:mixamorig[:_]?|valvebiped[.:_-]?|def[-_]?|org[-_]?|mch[-_]?|"
    r"bip(?:ed)?\d*[_ ]?|cc[_-]?base[_-]?|(?:armature|skeleton|skel|"
    r"bone|joint)[.:_-]+|j[_-]|c[_-]|ctrl[_-]?|control[_-]?)",
    re.IGNORECASE,
)


SHAPE_PREFIX_RE = re.compile(
    r"^(?:blendshape[._-])?(?:(?:facs|arkit|ar)[_-])?(?:(?:bs|cbs|ctrl)[_-])?",
    re.IGNORECASE,
)


def normalize_name(name: str, *, shape: bool = False) -> str:
    value = name.strip()
    if shape:
        value = SHAPE_PREFIX_RE.sub("", value)
    else:
        previous = None
        while previous != value:
            previous = value
            value = PREFIX_RE.sub("", value)
    return re.sub(r"[^a-z0-9]", "", value.lower())


def side_from_role(semantic_id: str) -> str:
    if semantic_id.endswith("_L"):
        return "LEFT"
    if semantic_id.endswith("_R"):
        return "RIGHT"
    return "CENTER"
