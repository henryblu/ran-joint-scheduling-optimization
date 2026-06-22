from __future__ import annotations

"""Logging helpers for the OFDMA round-robin baseline."""

import logging as py_logging
from models import MultiUserScheduleResult
from run_reporting import build_console_message, current_run_scope

from schedulers.frame_utilization import frame_utilization_log_fields

from .models import RoundRobinProblem


LOGGER = py_logging.getLogger("snapshot_run")


def log_frame_utilization_summary(
    problem: RoundRobinProblem,
    result: MultiUserScheduleResult,
) -> None:
    _emit_round_robin_log(
        py_logging.INFO,
        stage="frame_utilization",
        event="summary",
        fields=frame_utilization_log_fields(
            result,
            frame_prb_count=int(problem.prb_max),
            frame_tx_chain_count=int(problem.n_tx_chains),
        ),
    )


def _emit_round_robin_log(level: int, *, stage: str, event: str, fields: list[tuple[str, str]]) -> None:
    LOGGER.log(
        level,
        build_console_message(
            level_tag=_level_tag(level),
            scope=current_run_scope(),
            stage=stage,
            event=event,
            fields=fields,
        ),
    )


def _level_tag(level: int) -> str:
    if level >= py_logging.WARNING:
        return "WARN"
    if level >= py_logging.INFO:
        return "INFO"
    if level >= py_logging.DEBUG:
        return "DEBUG"
    return "LOG"


__all__ = ["log_frame_utilization_summary"]
