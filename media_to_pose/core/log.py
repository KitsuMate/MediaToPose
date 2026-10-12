"""One ``media_to_pose`` logger tree for the whole extension.

Blender imports the package as ``bl_ext.<repository>.media_to_pose``, so
``logging.getLogger(__name__)`` would scatter loggers by install location.
``get_logger(__name__)`` maps every module onto ``media_to_pose.<module>``
instead, giving scripts and the CLI one stable logger to configure.
"""

from __future__ import annotations

import logging
import sys

LOGGER_NAME = "media_to_pose"
_PACKAGE = __name__.rsplit(".core.log", 1)[0]
_HANDLER_NAME = "media_to_pose_console"


def get_logger(module_name: str) -> logging.Logger:
    """Return the logger for a package module, rooted at ``media_to_pose``."""
    if module_name == _PACKAGE:
        return logging.getLogger(LOGGER_NAME)
    if module_name.startswith(_PACKAGE + "."):
        return logging.getLogger(LOGGER_NAME + module_name[len(_PACKAGE) :])
    return logging.getLogger(f"{LOGGER_NAME}.{module_name}")


def configure_logging(level: int | None = None) -> logging.Logger:
    """Attach one console handler to the root extension logger (idempotent)."""
    logger = logging.getLogger(LOGGER_NAME)
    handler = next(
        (item for item in logger.handlers if item.get_name() == _HANDLER_NAME),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler(sys.stderr)
        handler.set_name(_HANDLER_NAME)
        handler.setFormatter(
            logging.Formatter(
                "[Media to Pose] %(asctime)s [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.propagate = False
    if level is not None or logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO if level is None else level)
    return logger


__all__ = ["LOGGER_NAME", "configure_logging", "get_logger"]
