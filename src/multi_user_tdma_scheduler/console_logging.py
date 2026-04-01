"""Aligned debug logging helpers for TDMA scheduler worker output."""

from __future__ import annotations

import logging
import os

from configs.day_run import ACTIVE_BIN_SCOPE_ENV_VAR

CONSOLE_COLUMN_WIDTHS = {
    "level": 5,
    "scope": 4,
    "stage": 11,
    "event": 10,
}


def emit_scheduler_console_log(
    logger: logging.Logger,
    *,
    level: int,
    stage: str,
    event: str,
    fields: list[tuple[str, str]],
) -> None:
    """Emit one aligned scheduler log line inside a worker process."""

    logger.log(
        level,
        build_console_message(
            level_tag=_level_tag(level),
            scope=current_bin_scope(),
            stage=stage,
            event=event,
            fields=fields,
        ),
    )


def build_console_message(
    *,
    level_tag: str,
    scope: str,
    stage: str,
    event: str,
    fields: list[tuple[str, str]],
) -> str:
    """Build one fixed-width console message body without the timestamp prefix."""

    header = (
        f"{str(level_tag):<{CONSOLE_COLUMN_WIDTHS['level']}} "
        f"{str(scope):<{CONSOLE_COLUMN_WIDTHS['scope']}} "
        f"{str(stage):<{CONSOLE_COLUMN_WIDTHS['stage']}} "
        f"{str(event):<{CONSOLE_COLUMN_WIDTHS['event']}}"
    )
    if not fields:
        return header
    return f"{header} {' '.join(f'{key}={value}' for key, value in fields)}"


def current_bin_scope() -> str:
    """Return the active bin scope label when called inside a worker."""

    return os.environ.get(ACTIVE_BIN_SCOPE_ENV_VAR, "BIN")


def format_metric(value: float, *, digits: int) -> str:
    """Format one numeric metric using a stable fixed decimal precision."""

    return f"{float(value):.{int(digits)}f}"


def _level_tag(level: int) -> str:
    """Return the short aligned level tag used by scheduler debug output."""

    if level >= logging.WARNING:
        return "WARN"
    if level >= logging.INFO:
        return "INFO"
    if level >= logging.DEBUG:
        return "DEBUG"
    return "LOG"
