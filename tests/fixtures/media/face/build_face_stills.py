"""Crop the CC0 source photos into the face test stills.

    python tests/fixtures/media/face/build_face_stills.py SOURCE_DIR

``SOURCE_DIR`` holds the downloaded originals named as in ``STILLS``
(see ``tests/fixtures/fixtures.toml`` for each source page). Each still is
cropped to its ``crop`` box in fixtures.toml (fractions of the original)
and scaled so its longest side is at most 900 px. Needs Pillow.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
MAX_SIDE = 900

# id: downloaded source file. The crop box (fractions of width and height:
# left, top, right, bottom) is the still's ``crop`` in fixtures.toml.
STILLS = {
    "face_neutral": "stocksnap_J2PTEFNZTE.jpg",
    "face_smirk_brows_down": "commons_Gray_haired_man_portrait__Unsplash.jpg",
    "face_smile_open": "commons_Happy_daughter__Unsplash.jpg",
    "face_laugh_jaw_open": "commons_Always_smile__Unsplash.jpg",
    "face_eyes_closed": "commons_Closed_Eyes__Unsplash.jpg",
    "face_surprise": "stocksnap_9L1W62CZUR.jpg",
    "face_grin_wide_eyes": "stocksnap_EA5F0D7860.jpg",
    "face_frown": "stocksnap_EA5F0D7860.jpg",
    "face_wink_left": "stocksnap_JEANSWX3A6.jpg",
    "face_gaze_side": "stocksnap_HV6O4C5CZA.jpg",
    "face_gaze_up": "stocksnap_JVIKZKSVCW.jpg",
    "face_head_turned": "commons_The_Girl__Unsplash.jpg",
    # Uncropped on purpose: the face runs off the frame edges.
    "face_closeup_scream": "commons_San_diego__Unsplash.jpg",
}


def _crops() -> dict[str, list[float]]:
    with (HERE.parents[1] / "fixtures.toml").open("rb") as stream:
        manifest = tomllib.load(stream)
    return {
        entry["id"]: entry["crop"] for entry in manifest["fixture"] if "crop" in entry
    }


def build(source: Path) -> None:
    crops = _crops()
    for still_id, name in STILLS.items():
        image = Image.open(source / name).convert("RGB")
        left, top, right, bottom = crops[still_id]
        crop = image.crop(
            (
                round(left * image.width),
                round(top * image.height),
                round(right * image.width),
                round(bottom * image.height),
            )
        )
        crop.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
        target = HERE / f"{still_id}.jpg"
        crop.save(target, quality=90, optimize=True)
        print(still_id, crop.size, target.stat().st_size)


if __name__ == "__main__":
    build(Path(sys.argv[1]))
