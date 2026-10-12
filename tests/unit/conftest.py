"""Load the extension as ``media_to_pose`` for the plain (no Blender) tier."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "media_to_pose"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "media_to_pose" not in sys.modules and not (PACKAGE / "__init__.py").exists():
    # Until the source moves into media_to_pose/, the repository root is the
    # package directory.
    spec = importlib.util.spec_from_file_location(
        "media_to_pose",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["media_to_pose"] = module
    spec.loader.exec_module(module)
