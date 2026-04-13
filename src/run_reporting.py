from __future__ import annotations

"""Shared logging setup and compact console reporting for run entry points.

The goal here is not rich telemetry. These helpers keep the console output:
- fixed-width and easy to scan,
- small enough to follow during long runs,
- separate from the orchestration logic that produces the events.
"""

import logging
import os
from configs import SINGLE_USER_SEARCH_CONFIG
from configs.day_run import (
    WORKER_LOG_LEVEL_ENV_VAR,
)
from models.day_run import BinRunResult, DayRunConfig


CONSOLE_COLUMN_WIDTHS = {
    "level": 5,
    "scope": 4,
    "stage": 11,
    "event": 10,
}

LOGGER = logging.getLogger("day_run")


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


def log_run_setup(config: DayRunConfig) -> None:
    """Emit the one-time setup line for a full-day run."""

    # Report the actual worker count after capping by the number of bins.
    planned_workers = min(int(config.cores), int(config.session_generation_config.day_bin_count))
    _emit_console_log(
        level=logging.INFO,
        scope="RUN",
        stage="setup",
        event="start",
        fields=[
            ("curve", config.load_curve_csv.stem),
            ("bins", str(int(config.session_generation_config.day_bin_count))),
            ("workers", str(int(planned_workers))),
            ("frame_slots", str(int(SINGLE_USER_SEARCH_CONFIG.frame_n_slots))),
            ("policy", str(config.switch_policy.value)),
        ],
    )


def log_run_progress(
    *,
    total_bins: int,
    completed_bins: int,
    solved_bins: int,
    infeasible_bins: int,
    empty_bins: int,
    elapsed_s: float,
) -> None:
    """Emit one sparse run-level progress line from the current completion counters."""

    remaining_bins = max(0, int(total_bins - completed_bins))
    # The ETA is intentionally coarse. It is only meant to answer "roughly how
    # much longer?" during long runs, not to predict an exact finish time.
    average_s_per_bin = float(elapsed_s) / max(1, int(completed_bins))
    eta_min = 0.0 if remaining_bins == 0 else (average_s_per_bin * float(remaining_bins) / 60.0)
    _emit_console_log(
        level=logging.INFO,
        scope="RUN",
        stage="summary",
        event="progress",
        fields=[
            ("done", f"{int(completed_bins)}/{int(total_bins)}"),
            ("solved", str(int(solved_bins))),
            ("infeasible", str(int(infeasible_bins))),
            ("empty", str(int(empty_bins))),
            ("elapsed_s", _format_metric(elapsed_s, digits=1)),
            ("eta_min", _format_metric(eta_min, digits=1)),
        ],
    )


def log_bin_result(result: BinRunResult) -> None:
    """Emit the authoritative completion line for one simulation bin."""

    fields = [("status", str(result.status))]
    if result.status == "solved":
        # Solved bins get a small amount of schedule detail because that is the
        # information most useful for sanity-checking the run while it is live.
        best_schedule = {} if result.best_schedule is None else result.best_schedule
        fields.extend(
            [
                ("users", str(int(result.user_count))),
                ("scheduled", str(int(len(best_schedule.get("rows", []))))),
                ("slot_total", "na" if result.best_schedule is None else str(int(best_schedule["slot_total"]))),
                ("unused_slots", "na" if result.best_schedule is None else str(int(best_schedule["unused_slots"]))),
                ("single_user_s", _format_metric(result.single_user_elapsed_s, digits=1)),
                ("joint_s", _format_metric(result.joint_elapsed_s, digits=1)),
                ("total_s", _format_metric(result.total_elapsed_s, digits=1)),
                (
                    "dc_total_w",
                    "na"
                    if result.best_schedule is None
                    else _format_metric(float(best_schedule["schedule_p_dc_total_avg_frame_w"]), digits=2),
                ),
            ]
        )
    elif result.status == "infeasible":
        # Infeasible and empty bins stay short. The day-run export holds the
        # richer demand context if deeper inspection is needed later.
        fields.extend(
            [
                ("users", str(int(result.user_count))),
                ("single_user_s", _format_metric(result.single_user_elapsed_s, digits=1)),
                ("joint_s", _format_metric(result.joint_elapsed_s, digits=1)),
                ("total_s", _format_metric(result.total_elapsed_s, digits=1)),
            ]
        )
    else:
        fields.append(("total_s", _format_metric(result.total_elapsed_s, digits=1)))

    _emit_console_log(
        level=logging.INFO if result.status != "infeasible" else logging.WARNING,
        scope=_format_bin_scope(result.bin_index),
        stage="bin",
        event="done",
        fields=fields,
    )


def log_run_summary(*, bin_results: list[BinRunResult], elapsed_s: float) -> None:
    """Emit the final run summary after the outputs have been saved successfully."""

    _emit_console_log(
        level=logging.INFO,
        scope="RUN",
        stage="summary",
        event="done",
        fields=[
            ("solved", str(sum(result.status == "solved" for result in bin_results))),
            ("infeasible", str(sum(result.status == "infeasible" for result in bin_results))),
            ("empty", str(sum(result.status == "empty" for result in bin_results))),
            ("elapsed_s", _format_metric(float(elapsed_s), digits=1)),
        ],
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


def _emit_console_log(
    *,
    level: int,
    scope: str,
    stage: str,
    event: str,
    fields: list[tuple[str, str]],
) -> None:
    """Format and emit one aligned thesis-run console line."""

    LOGGER.log(
        level,
        build_console_message(
            level_tag=_level_tag(level),
            scope=scope,
            stage=stage,
            event=event,
            fields=fields,
        ),
    )


def _format_bin_scope(bin_index: int) -> str:
    """Return the fixed-width bin scope label used in console lines."""

    return f"B{int(bin_index):03d}"


def _level_tag(level: int) -> str:
    """Return the short console level tag used by the aligned formatter."""

    if level >= logging.WARNING:
        return "WARN"
    if level >= logging.INFO:
        return "INFO"
    if level >= logging.DEBUG:
        return "DEBUG"
    return "LOG"


def _format_metric(value: float | None, *, digits: int) -> str:
    """Format one numeric metric using a stable fixed decimal precision."""

    if value is None:
        return "na"
    return f"{float(value):.{int(digits)}f}"


__all__ = [
    "build_console_message",
    "configure_run_logging",
    "log_bin_result",
    "log_run_progress",
    "log_run_setup",
    "log_run_summary",
]
