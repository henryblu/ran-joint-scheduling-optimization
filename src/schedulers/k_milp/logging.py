from __future__ import annotations

"""Logging helpers for the OFDMA MILP oracle backend."""

import logging as py_logging
import os
from datetime import datetime

import numpy as np
import scipy

from configs.scheduler import ACTIVE_SNAPSHOT_SCOPE_ENV_VAR
from models import MultiUserScheduleResult
from run_reporting import build_console_message, current_run_scope

from schedulers.frame_utilization import frame_utilization_log_fields

from .models import MilpModelSize, MilpVariableIndex, OfdmaMilpProblem, PaCurveSegment
from .problem import candidate_rows_by_user_pa


LOGGER = py_logging.getLogger("snapshot_run")


def active_snapshot_index_from_scope() -> int | None:
    scope = os.environ.get(ACTIVE_SNAPSHOT_SCOPE_ENV_VAR)
    if scope is None or not scope.startswith("S"):
        return None
    return int(scope[1:])


def log_problem_summary(problem: OfdmaMilpProblem) -> None:
    _emit_milp_log(
        py_logging.INFO,
        stage="milp",
        event="input",
        fields=[
            ("mode", "ofdma_milp_single_snapshot"),
            ("policy", problem.switch_policy.value),
            ("users", str(int(len(problem.user_requirements)))),
            ("frame_slots", str(int(problem.frame_n_slots))),
            ("prb", str(int(problem.prb_max))),
        ],
    )
    _emit_milp_log(
        py_logging.INFO,
        stage="milp",
        event="rows",
        fields=[("candidate_rows_by_user_pa", str(candidate_rows_by_user_pa(problem)))],
    )


def log_admission_summary(
    *,
    max_rows_per_user: int,
    raw_rows_by_user: dict[int, int],
    admitted_rows_by_user: dict[int, int],
) -> None:
    _emit_milp_log(
        py_logging.INFO,
        stage="admission",
        event="summary",
        fields=[
            ("max_rows_per_user", str(int(max_rows_per_user))),
            ("raw_rows_by_user", str({int(user_id): int(count) for user_id, count in raw_rows_by_user.items()})),
            (
                "admitted_rows_by_user",
                str({int(user_id): int(count) for user_id, count in admitted_rows_by_user.items()}),
            ),
        ],
    )


def log_restricted_pattern_summary(
    *,
    one_ue_pattern_count: int,
    raw_dual_ue_pair_bound: int,
    valid_dual_ue_pattern_count: int,
    retained_dual_ue_pattern_count: int,
) -> None:
    _emit_milp_log(
        py_logging.INFO,
        stage="patterns",
        event="restricted_k2",
        fields=[
            ("one_ue", str(int(one_ue_pattern_count))),
            ("raw_dual_ue_bound", str(int(raw_dual_ue_pair_bound))),
            ("valid_dual_ue", str(int(valid_dual_ue_pattern_count))),
            ("retained_dual_ue", str(int(retained_dual_ue_pattern_count))),
        ],
    )


def log_model_build_start(*, attempt_name: str) -> datetime:
    started_at = datetime.now()
    _emit_milp_log(
        py_logging.INFO,
        stage="milp_build",
        event="start",
        fields=[
            ("attempt", str(attempt_name)),
            ("timestamp", started_at.isoformat(timespec="seconds")),
        ],
    )
    return started_at


def log_model_build_end(
    *,
    attempt_name: str,
    model_size: MilpModelSize,
    elapsed_s: float,
    segments_by_pa: dict[int, tuple[PaCurveSegment, ...]],
    variables: MilpVariableIndex,
) -> None:
    ended_at = datetime.now()
    _emit_milp_log(
        py_logging.INFO,
        stage="milp_build",
        event="done",
        fields=[
            ("attempt", str(attempt_name)),
            ("timestamp", ended_at.isoformat(timespec="seconds")),
            ("vars", str(int(model_size.variable_count))),
            ("binaries", str(int(model_size.binary_variable_count))),
            ("constraints", str(int(model_size.constraint_count))),
            ("nnz", str(int(model_size.nonzero_count))),
            ("elapsed_s", f"{float(elapsed_s):.3f}"),
        ],
    )
    _emit_milp_log(
        py_logging.DEBUG,
        stage="milp_build",
        event="detail",
        fields=[
            ("variable_families", str(_variable_family_counts(variables))),
            ("constraint_families", "user_slot,demand,slot_pa,prb,pa_curve,output_limit,delta_w_product"),
            ("continuous", str(int(model_size.continuous_variable_count))),
            ("estimated_nonzeros", str(int(model_size.nonzero_count))),
            ("segments_by_pa", str({int(pa_id): int(len(segments)) for pa_id, segments in segments_by_pa.items()})),
        ],
    )


def log_solve_start(*, attempt_name: str) -> datetime:
    started_at = datetime.now()
    _emit_milp_log(
        py_logging.INFO,
        stage="milp_solve",
        event="start",
        fields=[
            ("attempt", str(attempt_name)),
            ("timestamp", started_at.isoformat(timespec="seconds")),
        ],
    )
    return started_at


def log_solver_runtime(
    *,
    attempt_name: str,
    solver_options: dict[str, bool | int | float],
) -> None:
    _emit_milp_log(
        py_logging.DEBUG,
        stage="milp_solve",
        event="runtime",
        fields=[
            ("attempt", str(attempt_name)),
            ("scipy", scipy.__version__),
            ("numpy", np.__version__),
            ("options", str(dict(solver_options))),
        ],
    )


def log_solve_end(
    *,
    attempt_name: str,
    status: int,
    success: bool,
    elapsed_s: float,
    objective_pwl_j: float | None,
    objective_bound: float | None,
    mip_gap: float | None,
) -> None:
    ended_at = datetime.now()
    _emit_milp_log(
        py_logging.INFO,
        stage="milp_solve",
        event="done",
        fields=[
            ("attempt", str(attempt_name)),
            ("timestamp", ended_at.isoformat(timespec="seconds")),
            ("status", str(int(status))),
            ("success", str(bool(success))),
            ("objective_pwl_j", _format_optional_float(objective_pwl_j)),
            ("objective_bound", _format_optional_float(objective_bound)),
            ("mip_gap", _format_optional_float(mip_gap)),
            ("elapsed_s", f"{float(elapsed_s):.3f}"),
        ],
    )


def log_solve_diagnostics(*, attempt_name: str, diagnostics: dict[str, object]) -> None:
    _emit_milp_log(
        py_logging.DEBUG,
        stage="milp_solve",
        event="summary",
        fields=[
            ("attempt", str(attempt_name)),
            ("has_incumbent", str(bool(diagnostics.get("has_incumbent")))),
            ("active_slots", str(diagnostics.get("active_slot_count"))),
            ("allocations", str(diagnostics.get("allocation_count"))),
            ("pa_slot_counts", str(diagnostics.get("pa_slot_counts"))),
            ("delivered_bits_by_user", str(diagnostics.get("delivered_bits_by_user"))),
        ],
    )
    _emit_milp_log(
        py_logging.DEBUG,
        stage="milp_solve",
        event="incumbent",
        fields=[
            ("attempt", str(attempt_name)),
            ("slot_summaries", str(diagnostics.get("slot_summaries"))),
        ],
    )


def log_frame_utilization_summary(
    problem: OfdmaMilpProblem,
    result: MultiUserScheduleResult,
) -> None:
    _emit_milp_log(
        py_logging.INFO,
        stage="frame_utilization",
        event="summary",
        fields=frame_utilization_log_fields(
            result,
            frame_prb_count=int(problem.prb_max),
            frame_tx_chain_count=int(problem.n_tx_chains),
        ),
    )


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{float(value):.12g}"


def _emit_milp_log(level: int, *, stage: str, event: str, fields: list[tuple[str, str]]) -> None:
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


def _variable_family_counts(variables: MilpVariableIndex) -> dict[str, int]:
    return {
        "x": int(len(variables.x)),
        "z": int(len(variables.z)),
        "delta": int(len(variables.delta)),
        "beta": int(len(variables.beta)),
        "theta": int(len(variables.theta)),
        "w": int(len(variables.w)),
        "v": int(len(variables.v)),
    }


__all__ = [
    "active_snapshot_index_from_scope",
    "log_admission_summary",
    "log_frame_utilization_summary",
    "log_model_build_end",
    "log_model_build_start",
    "log_problem_summary",
    "log_restricted_pattern_summary",
    "log_solve_diagnostics",
    "log_solve_end",
    "log_solve_start",
    "log_solver_runtime",
]
