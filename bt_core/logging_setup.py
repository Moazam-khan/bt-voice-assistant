"""Structured logging setup for BT.

Wires structlog on top of stdlib logging so every module gets the same
configured logger via :func:`get_logger`. :func:`configure_logging` must be
called once, as early as possible in the process entrypoint — before any
other module creates a logger — otherwise early log calls fall back to
stdlib defaults.
"""

from __future__ import annotations

import logging
import sys

import structlog

from bt_core.config import LoggingConfig

_SHARED_PROCESSORS: list[structlog.typing.Processor] = [
    structlog.contextvars.merge_contextvars,
    structlog.processors.add_log_level,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
]


def configure_logging(config: LoggingConfig) -> None:
    """Configure structlog + stdlib logging for the whole process.

    Console output uses ``config.format`` (pretty for dev, JSON for prod).
    File output is always JSON, regardless of ``config.format``, so log
    files stay parseable for future log aggregation.

    Args:
        config: The logging section of BT's settings.
    """
    config.file.parent.mkdir(parents=True, exist_ok=True)

    console_renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if config.format == "json"
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[
            *_SHARED_PROCESSORS,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, config.level)
        ),
        cache_logger_on_first_use=True,
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=console_renderer,
            foreign_pre_chain=_SHARED_PROCESSORS,
        )
    )

    file_handler = logging.FileHandler(config.file, encoding="utf-8")
    file_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=_SHARED_PROCESSORS,
        )
    )

    root_logger = logging.getLogger()
    root_logger.handlers = [console_handler, file_handler]
    root_logger.setLevel(config.level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger bound to a module name.

    Args:
        name: The calling module's name — pass ``__name__``.

    Returns:
        A configured structlog bound logger.
    """
    return structlog.get_logger(name)
