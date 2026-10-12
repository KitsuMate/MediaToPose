"""Crop the CC0 source photos into the face test stills.

    python tests/fixtures/media/face/build_face_stills.py SOURCE_DIR

``SOURCE_DIR`` holds the downloaded originals named as in ``STILLS``
(see ``tests/fixtures/fixtures.toml`` for each source page). Each still is
cropped to the given box (fractions of the original) and scaled so its
longest side is at most 900 px. Needs Pillow.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
MAX_SIDE = 900

# id: (source file, crop box as fractions of width/height: left, top, right, bottom)
STILLS = {
    "face_neutral": ("stocksnap_J2PTEFNZTE.jpg", (0.15, 0.02, 0.92, 0.62)),
    "face_smirk_brows_down": (
        "commons_Gray_haired_man_portrait__Unsplash.jpg",
        (0.36, 0.20, 0.71, 0.80),
    ),
    "face_smile_open": ("commons_Happy_daughter__Unsplash.jpg", (0.15, 0.08, 0.85, 0.62)),
    "face_laugh_jaw_open": ("commons_Always_smile__Unsplash.jpg", (0.30, 0.08, 0.70, 0.55)),
    "face_eyes_closed": ("commons_Closed_Eyes__Unsplash.jpg", (0.18, 0.05, 0.74, 0.45)),
    "face_surprise": ("stocksnap_9L1W62CZUR.jpg", (0.22, 0.20, 0.50, 0.66)),
    "face_grin_wide_eyes": ("stocksnap_EA5F0D7860.jpg", (0.02, 0.15, 0.50, 1.0)),
    "face_frown": ("stocksnap_EA5F0D7860.jpg", (0.50, 0.15, 0.98, 1.0)),
    "face_wink_left": ("stocksnap_JEANSWX3A6.jpg", (0.36, 0.10, 0.66, 0.62)),
    "face_gaze_side": ("stocksnap_HV6O4C5CZA.jpg", (0.0, 0.0, 0.45, 0.85)),
    "face_gaze_up": ("stocksnap_JVIKZKSVCW.jpg", (0.12, 0.05, 0.85, 0.55)),
    "face_head_turned": ("commons_The_Girl__Unsplash.jpg", (0.08, 0.10, 0.55, 0.85)),
    # Kept uncropped on purpose: the face runs off the frame edges.
    "face_closeup_scream": ("commons_San_diego__Unsplash.jpg", (0.0, 0.0, 1.0, 1.0)),
}


def build(source: Path) -> None:
    for still_id, (name, box) in STILLS.items():
        image = Image.open(source / name).convert("RGB")
        left, top, right, bottom = box
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
