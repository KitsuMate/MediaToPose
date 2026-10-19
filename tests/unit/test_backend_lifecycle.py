"""Shared model backend lifecycle: downloads, locks, storage and jobs."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from media_to_pose.backends import lifecycle
from media_to_pose.core.coordinator import COORDINATOR, OperationKind
from media_to_pose.core.errors import (
    BackendError,
    Cancelled,
    InsufficientResources,
    OnlineAccessDisabled,
    OperationBusy,
)


class _Response:
    def __init__(
        self,
        payload: bytes,
        *,
        length: int | None = None,
        fail_after=None,
        status: int = 200,
    ):
        self.payload = payload
        self.status = status
        self.headers = {
            "Content-Length": str(len(payload) if length is None else length)
        }
        self.fail_after = fail_after
        self.reads = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size):
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            raise TimeoutError("injected timeout")
        value, self.payload = self.payload, b""
        return value


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TemporaryDirectoryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)


class DownloadTests(TemporaryDirectoryTestCase):
    def _download(self, response, **kwargs):
        destination = self.root / "model.onnx"
        with mock.patch.object(
            lifecycle.urllib.request, "urlopen", return_value=response
        ):
            lifecycle.download_file(
                "https://example.invalid/model.onnx",
                destination,
                cancel=kwargs.pop("cancel", threading.Event()),
                **kwargs,
            )
        return destination

    def test_verified_download_lands_atomically(self) -> None:
        seen = []
        destination = self._download(
            _Response(b"weights"),
            expected_bytes=7,
            expected_sha256=_sha(b"weights"),
            progress=lambda received, total: seen.append((received, total)),
        )
        self.assertEqual(destination.read_bytes(), b"weights")
        self.assertEqual(seen, [(7, 7)])
        self.assertEqual(list(self.root.glob("*.part")), [])

    def test_hash_mismatch_leaves_nothing_behind(self) -> None:
        with self.assertRaisesRegex(BackendError, "SHA-256"):
            self._download(_Response(b"bad"), expected_sha256="0" * 64)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_declared_size_mismatch_is_rejected_before_reading(self) -> None:
        with self.assertRaisesRegex(BackendError, "unexpected download size"):
            self._download(_Response(b"abc"), expected_bytes=10)

    def test_transport_failure_removes_the_partial_file(self) -> None:
        with self.assertRaisesRegex(TimeoutError, "injected"):
            self._download(_Response(b"abc", length=6, fail_after=1))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_interrupted_resumable_download_continues_with_a_range(self) -> None:
        payload = b"0123456789" * 3
        destination = self.root / "model.onnx"
        part = lifecycle.resumable_part(destination)
        part.write_bytes(payload[:12])
        requests = []

        def urlopen(request, timeout):
            requests.append(request.get_header("Range"))
            return _Response(payload[12:], status=206)

        with mock.patch.object(
            lifecycle.urllib.request, "urlopen", side_effect=urlopen
        ):
            lifecycle.download_file(
                "https://example.invalid/model.onnx",
                destination,
                cancel=threading.Event(),
                expected_bytes=len(payload),
                expected_sha256=_sha(payload),
                resume=True,
            )
        self.assertEqual(requests, ["bytes=12-"])
        self.assertEqual(destination.read_bytes(), payload)
        self.assertFalse(part.exists())

    def test_server_ignoring_the_range_restarts_the_file(self) -> None:
        payload = b"abcdefgh"
        destination = self.root / "model.onnx"
        lifecycle.resumable_part(destination).write_bytes(b"XYZ")
        destination = self._download(
            _Response(payload),
            expected_bytes=len(payload),
            expected_sha256=_sha(payload),
            resume=True,
        )
        self.assertEqual(destination.read_bytes(), payload)

    def test_resumable_part_survives_a_failure_but_not_a_bad_hash(self) -> None:
        part = lifecycle.resumable_part(self.root / "model.onnx")
        with self.assertRaises(TimeoutError):
            self._download(_Response(b"abc", fail_after=1), resume=True)
        self.assertEqual(part.read_bytes(), b"abc")
        with self.assertRaises(BackendError):
            self._download(
                _Response(b"def", status=206),
                expected_bytes=6,
                expected_sha256=_sha(b"other!"),
                resume=True,
            )
        self.assertFalse(part.exists())

    def test_cancellation_raises_cancelled(self) -> None:
        cancel = threading.Event()
        cancel.set()
        with self.assertRaisesRegex(Cancelled, "stop now"):
            self._download(_Response(b"abc"), cancel=cancel, cancel_message="stop now")
        self.assertEqual(list(self.root.iterdir()), [])


class PathTests(unittest.TestCase):
    def test_hugging_face_urls_are_encoded_and_pinned(self) -> None:
        url = lifecycle.hf_resolve_url("Owner/repo", "a" * 40, "onnx/my file.json")
        self.assertEqual(
            url,
            "https://huggingface.co/Owner/repo/resolve/"
            + "a" * 40
            + "/onnx/my%20file.json",
        )

    def test_unsafe_paths_are_rejected(self) -> None:
        for value in ("../x", "/abs", "a\\b", "c:/x", ""):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(BackendError, "Unsafe"),
            ):
                lifecycle.hf_resolve_url("o/r", "main", value, label="RoHM download")

    def test_contained_path_stays_inside_root(self) -> None:
        root = Path(tempfile.gettempdir()) / "bundle"
        self.assertEqual(
            lifecycle.contained_path(root, "a/b.onnx"),
            (root / "a" / "b.onnx").resolve(strict=False),
        )

    def test_immutable_revisions_are_forty_hex_digits(self) -> None:
        self.assertTrue(lifecycle.is_immutable_revision("0" * 40))
        self.assertFalse(lifecycle.is_immutable_revision("main"))


class LockTests(TemporaryDirectoryTestCase):
    def test_live_process_probe_does_not_signal_the_owner(self) -> None:
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            self.assertTrue(lifecycle._pid_running(child.pid))
            self.assertIsNone(child.poll())
        finally:
            child.terminate()
            child.wait(timeout=5)
        self.assertFalse(lifecycle._pid_running(child.pid))

    def test_second_owner_is_busy_until_release(self) -> None:
        first = lifecycle.model_installation_lock(self.root / "store", "RoHM")
        second = lifecycle.model_installation_lock(self.root / "store", "RoHM")
        first.acquire()
        try:
            with self.assertRaisesRegex(OperationBusy, "Another Blender"):
                second.acquire()
        finally:
            first.release()
        with second:
            self.assertTrue(second.path.exists())
        self.assertFalse(second.path.exists())

    def test_lease_of_a_dead_process_is_replaced(self) -> None:
        lock = lifecycle.InstallationLock(self.root / ".install.lock", "SAM")
        lock.path.write_text(json.dumps({"pid": 0, "created_at": 1}))
        lock.acquire()
        lock.release()
        self.assertFalse(lock.path.exists())

    def test_expired_lease_is_replaced(self) -> None:
        import os

        lock = lifecycle.InstallationLock(
            self.root / ".install.lock", "SAM", max_age_seconds=60
        )
        lock.path.write_text(json.dumps({"pid": os.getpid(), "created_at": 1}))
        lock.acquire()
        lock.release()


class StorageTests(TemporaryDirectoryTestCase):
    storage = lifecycle.ManagedStorage("media_to_pose_test", "test_models", "Test")

    def test_root_accepts_the_parent_or_the_folder_itself(self) -> None:
        expected = self.root / "media_to_pose_test"
        self.assertEqual(self.storage.root(self.root), expected)
        self.assertEqual(self.storage.root(expected), expected)

    def test_ensure_marks_new_storage_and_remove_requires_the_marker(self) -> None:
        root = self.storage.root(self.root)
        self.storage.ensure(root)
        self.assertTrue(self.storage.owns(root))
        (root / "active").mkdir()
        self.storage.remove(root)
        self.assertFalse(root.exists())

    def test_unmanaged_or_foreign_storage_is_refused(self) -> None:
        root = self.storage.root(self.root)
        root.mkdir()
        (root / "user-file.txt").write_text("keep")
        with self.assertRaisesRegex(BackendError, "unmanaged"):
            self.storage.ensure(root)
        with self.assertRaisesRegex(BackendError, "unrecognized"):
            self.storage.remove(root)
        self.assertTrue((root / "user-file.txt").exists())
        (root / lifecycle.STORAGE_MARKER).write_text(
            json.dumps({"owner": "someone-else", "kind": "test_models"})
        )
        with self.assertRaisesRegex(BackendError, "another application"):
            self.storage.ensure(root)

    def test_storage_parent_round_trips(self) -> None:
        config = self.root / "storage.json"
        self.assertEqual(lifecycle.read_storage_parent(config), "")
        lifecycle.write_storage_parent(config, self.root / "models")
        self.assertEqual(
            Path(lifecycle.read_storage_parent(config)),
            (self.root / "models").resolve(),
        )


class JobTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = lifecycle.ModelBackend()
        self.addCleanup(self.backend.shutdown)

    def _run(self, work):
        job = self.backend.start_job(
            lifecycle.JobKind.MODEL,
            OperationKind.ROHM_MODEL_LOAD,
            "unit-test",
            lifecycle.run_job,
            (work,),
            label="Loading test model",
        )
        job.thread.join(timeout=5)
        self.assertFalse(job.alive())
        return job

    def test_success_records_done_and_frees_the_coordinator(self) -> None:
        job = self._run(lambda job: "Loaded")
        snapshot = self.backend.model_load_status_snapshot()
        self.assertEqual(snapshot.outcome, "DONE")
        self.assertEqual(snapshot.status, "Loaded")
        self.assertFalse(snapshot.running)
        self.assertEqual(job.messages.get_nowait(), ("DONE", "Loaded"))
        self.assertFalse(COORDINATOR.is_busy())

    def test_failure_and_cancellation_outcomes(self) -> None:
        def fail(_job):
            raise BackendError("hf_secrettoken123 leaked")

        failed = self._run(fail)
        self.assertEqual(failed.outcome, lifecycle.JobOutcome.ERROR)
        self.assertNotIn("secrettoken", failed.error)

        def cancel(_job):
            raise Cancelled("Loading cancelled")

        cancelled = self._run(cancel)
        self.assertEqual(cancelled.snapshot().outcome, "CANCELLED")

    def test_busy_coordinator_refuses_a_second_job(self) -> None:
        token = COORDINATOR.acquire(OperationKind.CAPTURE, "other")
        self.addCleanup(COORDINATOR.release, token)
        self.addCleanup(
            COORDINATOR.finish, token, lifecycle.OperationState.CANCELLED, "done"
        )
        with self.assertRaises(OperationBusy):
            self.backend.start_job(
                lifecycle.JobKind.INSTALL,
                OperationKind.ROHM_INSTALL,
                "unit-test",
                lifecycle.run_job,
                (lambda job: "x",),
                label="Installing",
            )

    def test_session_and_snapshot_cache(self) -> None:
        builds = []
        first = self.backend.cached_snapshot(("AUTO",), lambda: builds.append(1) or 1)
        again = self.backend.cached_snapshot(("AUTO",), lambda: builds.append(2) or 2)
        self.assertEqual((first, again, builds), (1, 1, [1]))
        self.assertIs(self.backend.set_session("session"), "session")
        self.assertIs(self.backend.set_session("other"), "session")
        self.assertTrue(self.backend.is_model_loaded())
        self.assertEqual(self.backend.cached_snapshot(("AUTO",), lambda: 3), 3)
        self.assertEqual(self.backend.take_session(), "session")
        self.assertFalse(self.backend.is_model_loaded())


class PinnedBundleTests(TemporaryDirectoryTestCase):
    def test_install_downloads_verifies_and_activates(self) -> None:
        files = {"manifest.json": b'{"ok": true}', "graph.onnx": b"graph-bytes"}
        storage = lifecycle.ManagedStorage("media_to_pose_test", "test_models", "Test")
        root = storage.root(self.root)
        config = self.root / "config" / "storage.json"
        verified = []

        def fake_download(url, destination, **_kwargs):
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            Path(destination).write_bytes(files[Path(destination).name])

        bundle = lifecycle.PinnedBundle(
            label="Test",
            repository="Owner/test-onnx",
            revision="c" * 40,
            manifest_path="bundles/fp32/manifest.json",
            list_files=lambda staging: [
                lifecycle.BundleFile("graph.onnx", 11, _sha(b"graph-bytes"))
            ],
            verify=lambda staging: verified.append(
                sorted(p.name for p in staging.iterdir())
            ),
        )
        job = lifecycle.BackgroundJob("Installing Test")
        with mock.patch.object(lifecycle, "download_file", side_effect=fake_download):
            message = lifecycle.install_pinned_bundle(
                job, bundle, storage, root, config
            )
        self.assertEqual(message, "Test installed and verified")
        self.assertEqual(verified, [["graph.onnx", "manifest.json"]])
        self.assertEqual((root / "active" / "graph.onnx").read_bytes(), b"graph-bytes")
        self.assertTrue(storage.owns(root))
        self.assertEqual(
            Path(lifecycle.read_storage_parent(config)), self.root.resolve()
        )
        self.assertEqual(sorted(p.name for p in root.glob(".staging-*")), [])
        self.assertFalse((self.root / ".media_to_pose_test.install.lock").exists())

    def test_missing_preferred_revision_falls_back_to_main(self) -> None:
        storage = lifecycle.ManagedStorage("media_to_pose_test", "test_models", "Test")
        urls = []

        def fake_download(url, destination, **_kwargs):
            urls.append(url)
            if "/resolve/main/" not in url:
                raise urllib.error.HTTPError(url, 404, "gone", {}, None)
            Path(destination).write_bytes(b"{}")

        bundle = lifecycle.PinnedBundle(
            label="Test",
            repository="Owner/test-onnx",
            revision="d" * 40,
            manifest_path="manifest.json",
            list_files=lambda staging: [],
            verify=lambda staging: None,
        )
        with mock.patch.object(lifecycle, "download_file", side_effect=fake_download):
            lifecycle.install_pinned_bundle(
                lifecycle.BackgroundJob("Installing Test"),
                bundle,
                storage,
                storage.root(self.root),
                self.root / "storage.json",
            )
        self.assertEqual(len(urls), 2)
        self.assertIn("d" * 40, urls[0])

    def test_reinstall_reuses_verified_files_and_resumes_parts(self) -> None:
        storage = lifecycle.ManagedStorage("media_to_pose_test", "test_models", "Test")
        root = storage.root(self.root)
        storage.ensure(root)
        download = root / lifecycle.DOWNLOAD_FOLDER
        (download / "onnx").mkdir(parents=True)
        (download / "small.bin").write_bytes(b"small")
        (download / "onnx" / ".big.onnx.part").write_bytes(b"big-")
        (download / "stale.onnx").write_bytes(b"other variant")
        calls = []

        def fake_download(url, destination, **kwargs):
            calls.append((Path(destination).name, kwargs.get("resume", False)))
            if Path(destination).name == "manifest.json":
                Path(destination).write_bytes(b"{}")
                return
            part = lifecycle.resumable_part(Path(destination))
            self.assertEqual(part.read_bytes(), b"big-")
            part.write_bytes(b"big-bytes")
            part.replace(destination)

        bundle = lifecycle.PinnedBundle(
            label="Test",
            repository="Owner/test-onnx",
            revision="f" * 40,
            manifest_path="manifest.json",
            list_files=lambda staging: [
                lifecycle.BundleFile("small.bin", 5, _sha(b"small")),
                lifecycle.BundleFile("onnx/big.onnx", 9, _sha(b"big-bytes")),
            ],
            verify=lambda staging: None,
        )
        with mock.patch.object(lifecycle, "download_file", side_effect=fake_download):
            lifecycle.install_pinned_bundle(
                lifecycle.BackgroundJob("Installing Test"),
                bundle,
                storage,
                root,
                self.root / "storage.json",
            )
        self.assertEqual(calls, [("manifest.json", False), ("big.onnx", True)])
        active = root / "active"
        self.assertEqual(
            sorted(p.relative_to(active).as_posix() for p in active.rglob("*")),
            ["manifest.json", "onnx", "onnx/big.onnx", "small.bin"],
        )
        self.assertFalse(download.exists())

    def test_cancelled_install_keeps_its_download_folder(self) -> None:
        storage = lifecycle.ManagedStorage("media_to_pose_test", "test_models", "Test")
        root = storage.root(self.root)

        def fake_download(url, destination, **_kwargs):
            if Path(destination).name == "manifest.json":
                Path(destination).write_bytes(b"{}")
                return
            lifecycle.resumable_part(Path(destination)).write_bytes(b"half")
            raise Cancelled("Test download cancelled")

        bundle = lifecycle.PinnedBundle(
            label="Test",
            repository="Owner/test-onnx",
            revision="f" * 40,
            manifest_path="manifest.json",
            list_files=lambda staging: [
                lifecycle.BundleFile("big.onnx", 8, _sha(b"halfmore"))
            ],
            verify=lambda staging: None,
        )
        with (
            mock.patch.object(lifecycle, "download_file", side_effect=fake_download),
            self.assertRaises(Cancelled),
        ):
            lifecycle.install_pinned_bundle(
                lifecycle.BackgroundJob("Installing Test"),
                bundle,
                storage,
                root,
                self.root / "storage.json",
            )
        part = root / lifecycle.DOWNLOAD_FOLDER / ".big.onnx.part"
        self.assertEqual(part.read_bytes(), b"half")
        self.assertFalse((root / "active").exists())

    def test_bundle_listed_in_code_fetches_from_several_repositories(self) -> None:
        storage = lifecycle.ManagedStorage("media_to_pose_test", "test_models", "Test")
        root = storage.root(self.root)
        urls = []

        def fake_download(url, destination, **_kwargs):
            urls.append(url)
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            Path(destination).write_bytes(b"data")

        bundle = lifecycle.PinnedBundle(
            label="Test",
            repository="Owner/test-onnx",
            revision="a" * 40,
            manifest_path="",
            list_files=lambda staging: [
                lifecycle.BundleFile("onnx/model.onnx", 4, _sha(b"data")),
                lifecycle.BundleFile(
                    "stats/mean.npy",
                    4,
                    _sha(b"data"),
                    repository="Upstream/model",
                    revision="b" * 40,
                    remote_path="stats/motion/mean.npy",
                ),
            ],
            verify=lambda staging: None,
        )
        with mock.patch.object(lifecycle, "download_file", side_effect=fake_download):
            lifecycle.install_pinned_bundle(
                lifecycle.BackgroundJob("Installing Test"),
                bundle,
                storage,
                root,
                self.root / "storage.json",
            )
        self.assertEqual(
            urls,
            [
                f"https://huggingface.co/Owner/test-onnx/resolve/{'a' * 40}/onnx/model.onnx",
                f"https://huggingface.co/Upstream/model/resolve/{'b' * 40}/stats/motion/mean.npy",
            ],
        )
        manifest = json.loads((root / "active" / "manifest.json").read_text())
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(
            [(item["path"], item["repository"]) for item in manifest["files"]],
            [
                ("onnx/model.onnx", "Owner/test-onnx"),
                ("stats/mean.npy", "Upstream/model"),
            ],
        )

    def test_offline_install_downloads_nothing(self) -> None:
        storage = lifecycle.ManagedStorage("media_to_pose_test", "test_models", "Test")
        bundle = lifecycle.PinnedBundle(
            label="Test",
            repository="Owner/test-onnx",
            revision="f" * 40,
            manifest_path="manifest.json",
            list_files=lambda staging: [],
            verify=lambda staging: None,
        )
        with (
            mock.patch.object(lifecycle, "download_file") as download,
            self.assertRaises(OnlineAccessDisabled) as caught,
        ):
            lifecycle.install_pinned_bundle(
                lifecycle.BackgroundJob("Installing Test"),
                bundle,
                storage,
                storage.root(self.root),
                self.root / "storage.json",
                online_access=False,
            )
        download.assert_not_called()
        self.assertEqual(caught.exception.exit_code, 4)
        self.assertFalse(storage.root(self.root).exists())

    def test_install_needs_free_space_for_the_files_and_headroom(self) -> None:
        storage = lifecycle.ManagedStorage("media_to_pose_test", "test_models", "Test")
        downloads = []

        def fake_download(url, destination, **_kwargs):
            downloads.append(Path(destination).name)
            Path(destination).write_bytes(b"{}")

        bundle = lifecycle.PinnedBundle(
            label="Test",
            repository="Owner/test-onnx",
            revision="e" * 40,
            manifest_path="manifest.json",
            list_files=lambda staging: [
                lifecycle.BundleFile("graph.onnx", 4 * 1024**3, "0" * 64)
            ],
            verify=lambda staging: None,
        )
        free = 4 * 1024**3 + lifecycle.DOWNLOAD_HEADROOM_BYTES - 1
        with (
            mock.patch.object(lifecycle, "download_file", side_effect=fake_download),
            mock.patch.object(
                lifecycle.resources, "free_disk_bytes", return_value=free
            ),
            self.assertRaises(InsufficientResources) as caught,
        ):
            lifecycle.install_pinned_bundle(
                lifecycle.BackgroundJob("Installing Test"),
                bundle,
                storage,
                storage.root(self.root),
                self.root / "storage.json",
            )
        self.assertEqual(downloads, ["manifest.json"])
        self.assertEqual(caught.exception.details["available_bytes"], free)


if __name__ == "__main__":
    unittest.main()
