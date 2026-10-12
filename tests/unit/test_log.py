"""Extension logger naming and configuration."""

from __future__ import annotations

import logging
import unittest

from media_to_pose.core import log


class LoggerTests(unittest.TestCase):
    def test_modules_map_onto_one_logger_tree(self) -> None:
        self.assertEqual(log.get_logger("media_to_pose").name, "media_to_pose")
        self.assertEqual(
            log.get_logger("media_to_pose.core.fs").name, "media_to_pose.core.fs"
        )
        self.assertEqual(
            log.get_logger("tools.probe").name, "media_to_pose.tools.probe"
        )

    def test_configure_logging_is_idempotent(self) -> None:
        logger = logging.getLogger(log.LOGGER_NAME)
        before = list(logger.handlers)
        self.addCleanup(setattr, logger, "handlers", before)
        log.configure_logging(logging.WARNING)
        log.configure_logging()
        handlers = [
            handler
            for handler in logger.handlers
            if handler.get_name() == "media_to_pose_console"
        ]
        self.assertEqual(len(handlers), 1)
        self.assertEqual(logger.level, logging.WARNING)
        log.configure_logging(logging.DEBUG)
        self.assertEqual(logger.level, logging.DEBUG)


if __name__ == "__main__":
    unittest.main()
