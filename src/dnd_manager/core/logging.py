"""Structured logging configuration for the D&D 5E AI Campaign Manager.

This module configures application-wide logging using structlog for
structured, context-rich logging that supports both development
(human-readable) and production (JSON) output formats.

Example:
    >>> from dnd_manager.core.logging import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Combat started", combatants=5, round=1)
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

import structlog
from structlog.types import Processor


if TYPE_CHECKING:
    from structlog.types import EventDict, WrappedLogger


def add_app_context(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Add application context to log entries.

    Args:
        logger: The wrapped logger instance.
        method_name: Name of the logging method called.
        event_dict: The event dictionary to modify.

    Returns:
        The modified event dictionary with app context.
    """
    event_dict["app"] = "dnd_manager"
    return event_dict


def configure_logging(
    *,
    level: str = "INFO",
    json_format: bool = False,
    log_file: str | None = None,
) -> None:
    """Configure application-wide logging.

    This function sets up structlog with appropriate processors for
    either development (colorful, human-readable) or production (JSON)
    output formats.

    Args:
        level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: If True, output logs in JSON format for production.
        log_file: Optional path to a log file for persistent logging.

    Example:
        >>> configure_logging(level="DEBUG", json_format=False)
    """
    # Determine output format based on environment
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        add_app_context,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_format:
        # Production: JSON output
        processors: list[Processor] = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: Colorful console output
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            ),
        ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure standard library logging for third-party libraries
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
        force=True,
    )

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a configured logger instance.

    Args:
        name: Optional name for the logger (typically __name__).

    Returns:
        A configured structlog BoundLogger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Player action", player="Gandalf", action="cast_spell")
    """
    return structlog.get_logger(name)


def bind_context(**kwargs: Any) -> None:
    """Bind context variables that will be included in all subsequent logs.

    This is useful for adding request-specific or session-specific
    context that should appear in all log entries.

    Args:
        **kwargs: Key-value pairs to bind to the logging context.

    Example:
        >>> bind_context(session_id="abc123", user="dm")
        >>> logger.info("Game started")  # Will include session_id and user
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all bound context variables.

    Call this at the end of a request or session to prevent context
    leakage between operations.

    Example:
        >>> clear_context()
    """
    structlog.contextvars.clear_contextvars()


__all__ = [
    "configure_logging",
    "get_logger",
    "bind_context",
    "clear_context",
]
