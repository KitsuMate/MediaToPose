"""Process-wide operation ownership, cancellation and requests."""

from __future__ import annotations

import threading
import time
import unittest
from dataclasses import FrozenInstanceError

from media_to_pose.core import coordinator as operation
from media_to_pose.core import requests


class OperationCoordinatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.operations = operation

    def setUp(self) -> None:
        self.coordinator = self.operations.OperationCoordinator()

    def test_twenty_repeated_clicks_have_one_owner(self) -> None:
        tokens = [
            self.coordinator.acquire(
                self.operations.OperationKind.CAPTURE, f"click-{index}"
            )
            for index in range(20)
        ]
        self.assertEqual(sum(token is not None for token in tokens), 1)

    def test_stale_callback_cannot_mutate_or_release(self) -> None:
        first = self.coordinator.require(self.operations.OperationKind.CAPTURE, "first")
        self.coordinator.finish(first, self.operations.OperationState.SUCCEEDED, "done")
        self.assertTrue(self.coordinator.release(first))
        second = self.coordinator.require(
            self.operations.OperationKind.SAM3D_MODEL_LOAD, "second"
        )
        self.assertFalse(self.coordinator.update(first, status="stale", progress=0.9))
        self.assertFalse(self.coordinator.release(first))
        self.assertEqual(self.coordinator.active_token(), second)

    def test_cancelling_worker_keeps_lease(self) -> None:
        token = self.coordinator.require(
            self.operations.OperationKind.CAPTURE, "worker"
        )
        gate = threading.Event()
        worker = threading.Thread(target=gate.wait, daemon=True)
        worker.start()
        self.coordinator.attach_worker(token, thread=worker)
        self.assertTrue(self.coordinator.request_cancel(token))
        self.assertEqual(
            self.coordinator.snapshot().state,
            self.operations.OperationState.CANCELLING.value,
        )
        self.assertFalse(self.coordinator.release(token))
        self.assertIsNone(
            self.coordinator.acquire(
                self.operations.OperationKind.SAM3D_INSTALL, "replacement"
            )
        )
        gate.set()
        worker.join(timeout=1.0)
        self.coordinator.finish(
            token, self.operations.OperationState.CANCELLED, "cancelled"
        )
        self.assertTrue(self.coordinator.release(token))

    def test_every_conflicting_operation_pair_is_serialized(self) -> None:
        kinds = list(self.operations.OperationKind)
        for first_kind in kinds:
            for second_kind in kinds:
                coordinator = self.operations.OperationCoordinator()
                first = coordinator.require(first_kind, "first")
                self.assertIsNone(
                    coordinator.acquire(second_kind, "second"),
                    f"{first_kind.value} allowed {second_kind.value}",
                )
                coordinator.finish(
                    first,
                    self.operations.OperationState.SUCCEEDED,
                    "done",
                )
                self.assertTrue(coordinator.release(first))

    def test_cancellation_is_visible_within_one_hundred_ms(self) -> None:
        for kind in self.operations.OperationKind:
            coordinator = self.operations.OperationCoordinator()
            token = coordinator.require(kind, kind.value)
            started = time.perf_counter()
            self.assertTrue(coordinator.request_cancel(token))
            elapsed = time.perf_counter() - started
            snapshot = coordinator.snapshot()
            self.assertLess(elapsed, 0.1)
            self.assertEqual(
                snapshot.state,
                self.operations.OperationState.CANCELLING.value,
            )
            self.assertTrue(snapshot.cancel_requested)

    def test_capture_request_is_an_immutable_snapshot(self) -> None:
        request = requests.CaptureRequest(
            provider_id="MEDIAPIPE",
            mode="POSE",
            media_path="fixture.png",
            media_kind="IMAGE",
            source_frame_index=0,
            use_body=True,
            use_hands=True,
            use_face=False,
            face_output="ARKIT",
            root_motion_mode="OFF",
            keep_original_root_position=False,
            keep_detection_cache=True,
            cache_limit_bytes=1024,
            start_frame=1.0,
            scene_name="Scene",
            scene_pointer=1,
            options={"confidence": 0.5},
        )
        with self.assertRaises(TypeError):
            request.options["confidence"] = 0.9
        with self.assertRaises(FrozenInstanceError):
            request.scene_name = "Later Scene"
        self.assertFalse(request.keep_original_root_position)

    def test_shutdown_waits_for_cooperative_worker(self) -> None:
        token = self.coordinator.require(
            self.operations.OperationKind.CAPTURE,
            "cooperative",
        )

        def worker_body() -> None:
            token.cancel_event.wait(1.0)

        worker = threading.Thread(target=worker_body, daemon=True)
        worker.start()
        self.coordinator.attach_worker(token, thread=worker)
        self.assertTrue(self.coordinator.shutdown(join_timeout_seconds=1.0))
        self.assertFalse(self.coordinator.is_busy())

    def test_process_cancel_terminates_but_keeps_lease_until_exit(self) -> None:
        class Process:
            return_code = None
            terminated = False

            def poll(self):
                return self.return_code

            def terminate(self):
                self.terminated = True

        process = Process()
        token = self.coordinator.require(
            self.operations.OperationKind.SAM3D_INSTALL,
            "installer",
        )
        self.coordinator.attach_worker(token, process=process)
        self.assertTrue(self.coordinator.request_cancel(token))
        self.assertTrue(process.terminated)
        self.assertFalse(self.coordinator.release(token))
        process.return_code = 143
        self.coordinator.finish(
            token,
            self.operations.OperationState.CANCELLED,
            "cancelled",
        )
        self.assertTrue(self.coordinator.release(token))

    def test_process_cancel_escalates_to_kill_after_grace_period(self) -> None:
        class Process:
            return_code = None

            def __init__(self):
                self.terminated = False
                self.killed = threading.Event()

            def poll(self):
                return self.return_code

            def terminate(self):
                self.terminated = True

            def kill(self):
                self.return_code = -9
                self.killed.set()

        process = Process()
        token = self.coordinator.require(
            self.operations.OperationKind.SAM3D_INSTALL,
            "installer subprocess",
        )
        self.coordinator.attach_worker(token, process=process)
        previous_grace = self.operations.PROCESS_TERMINATE_GRACE_SECONDS
        self.operations.PROCESS_TERMINATE_GRACE_SECONDS = 0.01
        try:
            self.assertTrue(self.coordinator.request_cancel(token))
            self.assertTrue(process.killed.wait(1.0))
        finally:
            self.operations.PROCESS_TERMINATE_GRACE_SECONDS = previous_grace
        self.assertTrue(process.terminated)
        self.coordinator.finish(
            token,
            self.operations.OperationState.CANCELLED,
            "cancelled",
        )
        self.assertTrue(self.coordinator.release(token))
