"""Keep modules small enough to read in one sitting.

A module may exceed the limit only with a reason recorded here.
"""

from __future__ import annotations

import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "media_to_pose"
SKIP = {"wheels", "assets"}
LIMIT = 850

LARGE_MODULES = {
    "backends/sam3d/backend.py": (
        "Install, activation, status and loading of the SAM 3D Body bundle "
        "share one install lock, layout and cached snapshot."
    ),
    "backends/lifecycle.py": (
        "The shared ModelBackend state machine with its job and pinned-bundle "
        "install helpers, used by every model backend."
    ),
    "core/motion/rohm_correction.py": (
        "One algorithm: RoHM representation building, windowed correction and "
        "resampling, whose helpers are only used together."
    ),
    "session/capture_session.py": (
        "The capture state machine; its prepare, detection and apply stages "
        "share the session state."
    ),
}


class ModuleSizeTests(unittest.TestCase):
    def test_modules_stay_under_the_limit(self) -> None:
        oversized = {}
        for path in sorted(PACKAGE.rglob("*.py")):
            relative = path.relative_to(PACKAGE)
            if relative.parts[0] in SKIP:
                continue
            lines = len(path.read_text(encoding="utf8").splitlines())
            if lines > LIMIT:
                oversized[relative.as_posix()] = lines
        self.assertEqual(
            sorted(set(oversized) - set(LARGE_MODULES)),
            [],
            f"Modules over {LIMIT} lines: {oversized}",
        )
        self.assertEqual(
            sorted(set(LARGE_MODULES) - set(oversized)),
            [],
            "Modules back under the limit; remove them from LARGE_MODULES",
        )


if __name__ == "__main__":
    unittest.main()
