from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

import numpy as np

from tests.blender_test_support import PACKAGE_NAME, load_addon

load_addon()
ort_runtime = importlib.import_module(f"{PACKAGE_NAME}.sam3d_ort_runtime")
vith_runtime = importlib.import_module(f"{PACKAGE_NAME}.sam3d_vith_runtime")
RuntimeDiagnostics = ort_runtime.RuntimeDiagnostics
load_estimator = ort_runtime.load_estimator
CropBatch = vith_runtime.CropBatch
GraphExecutionError = vith_runtime.GraphExecutionError
GraphSessions = vith_runtime.GraphSessions
VithEstimator = vith_runtime.VithEstimator
_fused_local_wrist = vith_runtime._fused_local_wrist
fix_wrist_euler = vith_runtime.fix_wrist_euler
full_to_crop = vith_runtime.full_to_crop
hand_boxes_from_tokens = vith_runtime.hand_boxes_from_tokens
prepare_crop_batch = vith_runtime.prepare_crop_batch
project_camera = vith_runtime.project_camera
required_graph_roles = vith_runtime.required_graph_roles


def _crop_batch() -> CropBatch:
    affine = np.array([[[2.0, 0.0, 10.0], [0.0, 2.0, 20.0]]], dtype=np.float32)
    intrinsics = np.array(
        [[[1000.0, 0.0, 256.0], [0.0, 1000.0, 256.0], [0.0, 0.0, 1.0]]],
        dtype=np.float32,
    )
    return CropBatch(
        images=np.zeros((1, 3, 512, 512), dtype=np.float32),
        bbox_center=np.array([[256.0, 256.0]], dtype=np.float32),
        bbox_scale=np.array([[512.0, 512.0]], dtype=np.float32),
        affine=affine,
        original_size=np.array([[512.0, 512.0]], dtype=np.float32),
        camera_intrinsics=intrinsics,
    )


class VithGeometryTests(unittest.TestCase):
    def test_crop_batch_accepts_full_noncentered_intrinsics(self):
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        intrinsic = np.asarray(
            ((140.0, 0.0, 43.0), (0.0, 135.0, 37.0), (0.0, 0.0, 1.0)),
            dtype=np.float32,
        )
        fake_cv2 = SimpleNamespace(
            INTER_LINEAR=1,
            warpAffine=lambda source, _matrix, size, flags: np.zeros(
                (size[1], size[0], source.shape[2]), dtype=source.dtype
            ),
        )
        with patch.object(
            vith_runtime.importlib, "import_module", return_value=fake_cv2
        ):
            batch = prepare_crop_batch(
                image,
                [[10.0, 5.0, 90.0, 75.0]],
                camera_intrinsics=intrinsic,
            )
        np.testing.assert_allclose(batch.camera_intrinsics[0], intrinsic)

    def test_full_to_crop_and_hand_box_inverse_affine(self):
        batch = _crop_batch()
        center = np.array([[[123.0, 118.0]]], dtype=np.float32)
        np.testing.assert_allclose(full_to_crop(batch, center), 0.0, atol=1e-7)

        boxes = np.array(
            [[[0.5, 0.5, 0.2, 0.1], [0.25, 0.75, 0.1, 0.3]]],
            dtype=np.float32,
        )
        left, right = hand_boxes_from_tokens(boxes, batch)
        pixel_boxes = boxes * np.float32(512.0)
        centers = (pixel_boxes[..., :2] - np.array([10.0, 20.0])) / 2.0
        sides = np.max(pixel_boxes[..., 2:], axis=-1, keepdims=True) / 2.0
        expected = np.concatenate(
            (centers - sides / 2.0, centers + sides / 2.0), axis=-1
        )
        np.testing.assert_allclose(left, expected[:, 0], atol=1e-5)
        np.testing.assert_allclose(right, expected[:, 1], atol=1e-5)

    def test_camera_projection_uses_crop_scale_and_intrinsics(self):
        batch = _crop_batch()
        output = {
            "pred_cam": np.array([[-1.0, 0.0, 0.0]], dtype=np.float32),
            "pred_keypoints_3d": np.array(
                [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32
            ),
        }
        projected = project_camera(output, batch, default_scale_factor=1.0)
        np.testing.assert_allclose(
            projected["pred_cam_t"], [[0.0, 0.0, 3.90625]], atol=1e-6
        )
        np.testing.assert_allclose(
            projected["pred_keypoints_2d"],
            [[[256.0, 256.0], [512.0, 256.0]]],
            atol=1e-5,
        )
        np.testing.assert_array_equal(projected["focal_length"], [1000.0])

    def test_wrist_fusion_order_and_euler_limit_choice(self):
        predicted = np.array(
            [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
            dtype=np.float32,
        )
        wrist_zero = np.array(
            [[[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            _fused_local_wrist(predicted, wrist_zero),
            np.swapaxes(wrist_zero, -1, -2) @ predicted,
        )
        fixed = fix_wrist_euler(np.array([[2.8, 2.8, 2.8]], dtype=np.float32))
        self.assertTrue(np.isfinite(fixed).all())
        self.assertFalse(np.array_equal(fixed, [[2.8, 2.8, 2.8]]))


class _ProfileSession:
    def __init__(self, path: Path, provider: str) -> None:
        self.path = path
        self.path.write_text(
            json.dumps(
                [
                    {"args": {"provider": provider}},
                    {"args": {"provider": provider}},
                ]
            ),
            encoding="utf8",
        )

    def end_profiling(self) -> str:
        return str(self.path)


class VithAcceleratorContractTests(unittest.TestCase):
    def test_body_only_cannot_certify_full_graph_profile(self):
        roles = required_graph_roles()
        self.assertEqual(len(roles), 22)
        self.assertEqual(len(set(roles)), 22)
        estimator = VithEstimator.__new__(VithEstimator)
        estimator._profile_failure = None
        estimator._profile_requested = True
        estimator._profile_prefixes = {}
        estimator.accelerator_profile_evidence = {}
        estimator.diagnostics = RuntimeDiagnostics(
            requested_mode="GPU",
            actual_ep="WebGpuExecutionProvider",
            actual_device="webgpu",
            fallback=False,
            model_revision="a" * 40,
            precision="mixed-fp16-fp32",
        )
        estimator._runner = SimpleNamespace(
            graphs=SimpleNamespace(executed_roles=set(roles))
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            estimator._sessions = {
                role: _ProfileSession(root / f"{index}.json", "WebGpuExecutionProvider")
                for index, role in enumerate(roles)
            }
            estimator._verify_accelerator_profiles(full_inference=False)
            self.assertTrue(estimator._profile_requested)
            self.assertFalse(estimator.diagnostics.accelerator_nodes_verified)
            self.assertTrue(
                all(session.path.is_file() for session in estimator._sessions.values())
            )

            estimator._verify_accelerator_profiles(full_inference=True)
            self.assertFalse(estimator._profile_requested)
            self.assertTrue(estimator.diagnostics.accelerator_nodes_verified)
            self.assertEqual(estimator.diagnostics.accelerator_node_count, 44)
            self.assertEqual(
                set(estimator.accelerator_profile_evidence["roles"]), set(roles)
            )
            self.assertTrue(
                all(
                    not session.path.exists()
                    for session in estimator._sessions.values()
                )
            )

    def test_graph_session_wraps_only_ort_run_failure_for_fallback(self):
        roles = required_graph_roles()
        graphs = [
            SimpleNamespace(
                role=role,
                inputs={"input": "input"},
                outputs={"output": "output"},
            )
            for role in roles
        ]
        sessions = {
            role: SimpleNamespace(run=Mock(side_effect=RuntimeError("EP failed")))
            for role in roles
        }
        adapter = GraphSessions(SimpleNamespace(graphs=graphs), sessions)

        with self.assertRaisesRegex(GraphExecutionError, "EP failed"):
            adapter.run("backbone", {"input": np.zeros(1, dtype=np.float32)})

    def test_auto_does_not_fallback_for_preprocessing_failure(self):
        estimator = VithEstimator.__new__(VithEstimator)
        estimator._profile_failure = None
        estimator.requested_mode = "AUTO"
        estimator.diagnostics = RuntimeDiagnostics(
            requested_mode="AUTO",
            actual_ep="WebGpuExecutionProvider",
            actual_device="webgpu",
            fallback=False,
            model_revision="a" * 40,
            precision="mixed-fp16-fp32",
        )
        estimator._available_providers = (
            "WebGpuExecutionProvider",
            "CPUExecutionProvider",
        )
        estimator._runner = SimpleNamespace(
            run=Mock(side_effect=ValueError("invalid bounding box"))
        )
        estimator._fallback_to_cpu = Mock()

        with self.assertRaisesRegex(ValueError, "invalid bounding box"):
            estimator._run_output(
                np.zeros((8, 8, 3), dtype=np.uint8),
                np.zeros((1, 4), dtype=np.float32),
                include_hands=False,
            )
        estimator._fallback_to_cpu.assert_not_called()

    def test_auto_falls_back_for_typed_graph_execution_failure(self):
        estimator = VithEstimator.__new__(VithEstimator)
        estimator._profile_failure = None
        estimator.requested_mode = "AUTO"
        estimator.diagnostics = RuntimeDiagnostics(
            requested_mode="AUTO",
            actual_ep="WebGpuExecutionProvider",
            actual_device="webgpu",
            fallback=False,
            model_revision="a" * 40,
            precision="mixed-fp16-fp32",
        )
        estimator._available_providers = (
            "WebGpuExecutionProvider",
            "CPUExecutionProvider",
        )
        failure = GraphExecutionError("EP failed")
        estimator._runner = SimpleNamespace(run=Mock(side_effect=failure))
        estimator._fallback_to_cpu = Mock(side_effect=RuntimeError("CPU retry started"))

        with self.assertRaisesRegex(RuntimeError, "CPU retry started"):
            estimator._run_output(
                np.zeros((8, 8, 3), dtype=np.uint8),
                np.zeros((1, 4), dtype=np.float32),
                include_hands=False,
            )
        estimator._fallback_to_cpu.assert_called_once_with(failure)

    def test_normal_load_never_enables_accelerator_profiling(self):
        paths = {
            "bundle_manifest_path": "bundle.json",
            "model_revision": "a" * 40,
        }
        with (
            patch.object(
                ort_runtime, "load_bundle_manifest", return_value=sentinel.bundle
            ),
            patch.object(
                ort_runtime, "create_estimator", return_value=sentinel.estimator
            ) as create,
        ):
            result = load_estimator(paths, "VITH", "GPU")

        self.assertIs(result, sentinel.estimator)
        self.assertFalse(create.call_args.kwargs["verify_accelerator_execution"])

    def test_load_can_explicitly_verify_accelerator_execution(self):
        paths = {
            "bundle_manifest_path": "bundle.json",
            "model_revision": "a" * 40,
        }
        with (
            patch.object(
                ort_runtime, "load_bundle_manifest", return_value=sentinel.bundle
            ),
            patch.object(
                ort_runtime, "create_estimator", return_value=sentinel.estimator
            ) as create,
        ):
            result = load_estimator(
                paths,
                "VITH",
                "GPU",
                verify_accelerator_execution=True,
            )

        self.assertIs(result, sentinel.estimator)
        self.assertTrue(create.call_args.kwargs["verify_accelerator_execution"])


if __name__ == "__main__":
    unittest.main()
