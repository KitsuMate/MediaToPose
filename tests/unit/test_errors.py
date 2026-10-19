"""Typed error hierarchy."""

from __future__ import annotations

import unittest

from media_to_pose.core import errors


class ErrorTests(unittest.TestCase):
    def test_codes_and_exit_codes_are_stable_and_distinct(self) -> None:
        codes = [cls.code for cls in errors.ERROR_CLASSES]
        self.assertEqual(len(codes), len(set(codes)))
        self.assertEqual(errors.MediaError.code, "MEDIA_ERROR")
        self.assertEqual(errors.Cancelled.code, "CANCELLED")
        exit_codes = {
            cls.exit_code
            for cls in errors.ERROR_CLASSES
            if cls not in {errors.DependencyMissing, errors.OnlineAccessDisabled}
        }
        self.assertEqual(len(exit_codes), len(errors.ERROR_CLASSES) - 2)
        self.assertNotIn(0, exit_codes)

    def test_every_error_is_an_mtp_error(self) -> None:
        for cls in errors.ERROR_CLASSES:
            with self.subTest(cls=cls.__name__):
                self.assertTrue(issubclass(cls, errors.MtpError))

    def test_errors_keep_their_builtin_base_for_existing_handlers(self) -> None:
        self.assertTrue(issubclass(errors.MediaError, ValueError))
        self.assertTrue(issubclass(errors.InvalidRequest, ValueError))
        self.assertTrue(issubclass(errors.MappingIncomplete, ValueError))
        self.assertTrue(issubclass(errors.BackendError, RuntimeError))
        self.assertTrue(issubclass(errors.OperationBusy, RuntimeError))
        self.assertTrue(issubclass(errors.Cancelled, InterruptedError))
        self.assertTrue(issubclass(errors.DependencyMissing, ImportError))
        self.assertTrue(issubclass(errors.DependencyMissing, errors.ModelMissing))
        self.assertTrue(issubclass(errors.OnlineAccessDisabled, errors.ModelMissing))

    def test_agent_facing_codes_and_exit_codes(self) -> None:
        self.assertEqual(
            (errors.OnlineAccessDisabled.code, errors.OnlineAccessDisabled.exit_code),
            ("ONLINE_ACCESS_DISABLED", 4),
        )
        self.assertEqual(
            (errors.LicenseRequired.code, errors.LicenseRequired.exit_code),
            ("LICENSE_REQUIRED", 10),
        )
        self.assertEqual(
            (
                errors.InsufficientResources.code,
                errors.InsufficientResources.exit_code,
            ),
            ("INSUFFICIENT_RESOURCES", 11),
        )
        payload = errors.InsufficientResources(
            "Not enough memory", resource="ram", needed_bytes=8, available_bytes=4
        ).to_dict()
        self.assertEqual(payload["details"]["needed_bytes"], 8)

    def test_to_dict_is_json_ready(self) -> None:
        error = errors.MediaError("Video could not be opened", path=object(), frame=3)
        payload = error.to_dict()
        self.assertEqual(payload["code"], "MEDIA_ERROR")
        self.assertEqual(payload["message"], "Video could not be opened")
        self.assertEqual(payload["details"]["frame"], 3)
        self.assertIsInstance(payload["details"]["path"], str)
        self.assertNotIn("details", errors.Cancelled("stop").to_dict())


if __name__ == "__main__":
    unittest.main()
