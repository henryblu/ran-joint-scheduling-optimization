from __future__ import annotations

"""Shared logging setup and compact console reporting for run entry points.

The goal here is not rich telemetry. These helpers keep the console output:
- fixed-width and easy to scan,
- small enough to follow during long runs,
- separate from the orchestration logic that produces the events.
"""

import logging
import os

from configs.scheduler import ACTIVE_SNAPSHOT_SCOPE_ENV_VAR


CONSOLE_COLUMN_WIDTHS = {
    "level": 5,
    "scope": 4,
    "stage": 11,
    "event": 10,
}

WORKER_LOG_LEVEL_ENV_VAR = "THESIS_RUN_LOG_LEVEL"


def configure_run_logging(level: str | None) -> None:
    """Configure the shared console logger used by the run entry points and workers."""

    if level is None:
        os.environ.pop(WORKER_LOG_LEVEL_ENV_VAR, None)
        return

    normalized_level = str(level).upper()
    os.environ[WORKER_LOG_LEVEL_ENV_VAR] = normalized_level
    numeric_level = getattr(logging, normalized_level)
    root_logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if root_logger.handlers:
        # Reuse the existing root handlers instead of stacking duplicate console
        # handlers when tests or embedding environments already configured one.
        root_logger.setLevel(numeric_level)
        for handler in root_logger.handlers:
            handler.setFormatter(formatter)
        return

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
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


def current_run_scope(*, default: str = "RUN") -> str:
    snapshot_scope = os.environ.get(ACTIVE_SNAPSHOT_SCOPE_ENV_VAR)
    if snapshot_scope:
        return str(snapshot_scope)
    return str(default)


__all__ = [
    "build_console_message",
    "configure_run_logging",
    "current_run_scope",
]
