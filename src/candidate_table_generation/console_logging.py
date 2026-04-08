from __future__ import annotations

import logging

from run_reporting import build_console_message


LOGGER = logging.getLogger("day_run")


def emit_candidate_table_console_log(
    *,
    level: int,
    stage: str,
    event: str,
    fields: list[tuple[str, str]],
) -> None:
    """Emit one aligned candidate-table log line."""

    LOGGER.log(
        level,
        build_console_message(
            level_tag=_level_tag(level),
            scope="CTBL",
            stage=stage,
            event=event,
            fields=fields,
        ),
    )


def _level_tag(level: int) -> str:
    """Return the short aligned level tag used by candidate-table logs."""

    if level >= logging.WARNING:
        return "WARN"
    if level >= logging.INFO:
        return "INFO"
    if level >= logging.DEBUG:
        return "DEBUG"
    return "LOG"


__all__ = [
    "emit_candidate_table_console_log",
]
