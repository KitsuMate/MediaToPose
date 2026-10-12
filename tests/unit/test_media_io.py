"""Image and video probing with a scripted OpenCV stand-in."""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from media_to_pose import media_io


class _Capture:
    """Minimal cv2.VideoCapture replacement driven by a frame script."""

    def __init__(self, frames, *, fps=25.0, opened=True, positions=None):
        self.frames = list(frames)
        self.fps = fps
        self.opened = opened
        self.positions = list(positions or [])
        self.index = 0
        self.released = False

    def isOpened(self):  # noqa: N802 - OpenCV API
        return self.opened

    def set(self, prop, value):
        if prop == FakeCv2.CAP_PROP_POS_FRAMES:
            self.index = int(value)

    def get(self, prop):
        if prop == FakeCv2.CAP_PROP_FPS:
            return self.fps
        if prop == FakeCv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        if prop == FakeCv2.CAP_PROP_FRAME_WIDTH:
            return 64
        if prop == FakeCv2.CAP_PROP_FRAME_HEIGHT:
            return 48
        if prop == FakeCv2.CAP_PROP_POS_MSEC:
            if self.positions:
                return self.positions[self.index - 1]
            return 0.0
        return 0.0

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self):
        self.released = True


class FakeCv2:
    CAP_PROP_POS_FRAMES = 1
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_POS_MSEC = 0
    IMREAD_COLOR = 1

    def __init__(self, capture=None, image=None):
        self.capture = capture
        self.image = image

    def VideoCapture(self, _path):  # noqa: N802 - OpenCV API
        return self.capture

    def imdecode(self, _encoded, _flags):
        return self.image


class VideoFrameTests(unittest.TestCase):
    def test_timestamps_are_strictly_increasing(self) -> None:
        capture = _Capture(["a", "b", "c", "d"], fps=10.0, positions=[0, 0, 0, 350])
        frames = list(
            media_io.iter_video_frames("clip.mp4", threading.Event(), FakeCv2(capture))
        )
        timestamps = [timestamp for _index, timestamp, _total, _image in frames]
        self.assertEqual(timestamps, [0, 100, 200, 350])
        self.assertEqual([index for index, *_rest in frames], [0, 1, 2, 3])
        self.assertTrue(capture.released)

    def test_cancellation_raises_after_the_current_frame(self) -> None:
        cancel = threading.Event()
        capture = _Capture(["a", "b", "c"])
        iterator = media_io.iter_video_frames("clip.mp4", cancel, FakeCv2(capture))
        next(iterator)
        cancel.set()
        with self.assertRaises(InterruptedError):
            next(iterator)
        self.assertTrue(capture.released)

    def test_unopened_video_is_an_error(self) -> None:
        capture = _Capture([], opened=False)
        with self.assertRaises(ValueError):
            list(
                media_io.iter_video_frames(
                    "missing.mp4", threading.Event(), FakeCv2(capture)
                )
            )

    def test_exact_frame_falls_back_to_fps_timestamp(self) -> None:
        capture = _Capture(["a", "b", "c", "d"], fps=20.0)
        image, timestamp = media_io.decode_video_frame_index_with_timestamp(
            "clip.mp4", 3, FakeCv2(capture)
        )
        self.assertEqual(image, "d")
        self.assertEqual(timestamp, 150)


class ProbeTests(unittest.TestCase):
    def test_still_image_reports_its_size(self) -> None:
        cv2 = FakeCv2(image=np.zeros((30, 40, 3), dtype=np.uint8))
        with (
            mock.patch.object(
                media_io, "import_runtime_dependencies", return_value=(cv2, None, np)
            ),
            mock.patch.object(np, "fromfile", return_value=np.ones(4, np.uint8)),
        ):
            info = media_io.probe_media("still.png")
        self.assertEqual((info.kind, info.width, info.height), ("IMAGE", 40, 30))

    def test_video_reports_rate_and_duration(self) -> None:
        cv2 = FakeCv2(capture=_Capture(["a"] * 50, fps=25.0), image=None)
        with (
            mock.patch.object(
                media_io, "import_runtime_dependencies", return_value=(cv2, None, np)
            ),
            mock.patch.object(np, "fromfile", return_value=np.ones(4, np.uint8)),
        ):
            info = media_io.probe_media("clip.mp4")
        self.assertEqual(info.kind, "VIDEO")
        self.assertEqual(info.frame_count, 50)
        self.assertAlmostEqual(info.duration_seconds, 2.0)
        self.assertEqual((info.width, info.height), (64, 48))

    def test_undecodable_media_is_an_error(self) -> None:
        cv2 = FakeCv2(
            capture=SimpleNamespace(isOpened=lambda: False, release=lambda: None)
        )
        with (
            mock.patch.object(
                media_io, "import_runtime_dependencies", return_value=(cv2, None, np)
            ),
            mock.patch.object(np, "fromfile", return_value=np.ones(4, np.uint8)),
            self.assertRaises(ValueError),
        ):
            media_io.probe_media("broken.bin")


if __name__ == "__main__":
    unittest.main()
