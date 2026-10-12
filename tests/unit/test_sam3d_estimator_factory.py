"""The ONNX bundle runner resolves to the ViT-H orchestrator."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from media_to_pose.backends.sam3d import ort_runtime, vith_runtime


class EstimatorFactoryTests(unittest.TestCase):
    def test_split_numpy_runner_creates_the_vith_estimator(self) -> None:
        bundle = SimpleNamespace(runner=ort_runtime.SPLIT_NUMPY_RUNNER)
        with mock.patch.object(vith_runtime, "VithEstimator") as estimator:
            result = ort_runtime.create_estimator(bundle, "CPU")
        self.assertIs(result, estimator.return_value)
        estimator.assert_called_once()
        self.assertIs(estimator.call_args.args[0], bundle)

    def test_unknown_runner_is_rejected(self) -> None:
        bundle = SimpleNamespace(runner="unknown")
        with self.assertRaises(ort_runtime.OrtRuntimeError):
            ort_runtime.create_estimator(bundle, "CPU")


if __name__ == "__main__":
    unittest.main()
