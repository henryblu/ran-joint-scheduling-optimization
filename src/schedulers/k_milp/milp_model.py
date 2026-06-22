from __future__ import annotations

"""Direct slot-indexed OFDMA MILP model construction and solve."""

import logging
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, milp

from configs.scheduler import K_MILP_SOLVER_CONFIG

from .diagnostics import build_attempt_diagnostics, get_optional_float
from .logging import (
    log_model_build_end,
    log_model_build_start,
    log_solve_diagnostics,
    log_solve_end,
    log_solve_start,
    log_solver_runtime,
)
from .linear_model_builders import ConstraintBuilder, VariableBuilder, build_model_size
from .models import (
    MilpAttemptResult,
    MilpBuild,
    MilpVariableIndex,
    OfdmaMilpProblem,
    PaCurveSegment,
)
from .pa_piecewise import build_pa_piecewise_segments


def build_and_solve_milp_attempt(
    problem: OfdmaMilpProblem,
    *,
    allowed_pa_ids: tuple[int, ...],
    attempt_name: str,
    max_users_per_slot: int | None = None,
) -> MilpAttemptResult:
    """Build and solve one policy-filtered MILP attempt."""

    log_model_build_start(attempt_name=attempt_name)
    model = build_milp_model(
        problem,
        allowed_pa_ids=allowed_pa_ids,
        max_users_per_slot=max_users_per_slot,
    )
    log_model_build_end(
        attempt_name=attempt_name,
        model_size=model.model_size,
        elapsed_s=float(model.build_elapsed_s),
        segments_by_pa=model.segments_by_pa,
        variables=model.variables,
    )

    solve_started_at = perf_counter()
    log_solve_start(attempt_name=attempt_name)
    solver_options = build_scipy_milp_options()
    log_solver_runtime(attempt_name=attempt_name, solver_options=solver_options)
    result = milp(
        c=model.c,
        integrality=model.integrality,
        bounds=model.bounds,
        constraints=model.constraints,
        options=solver_options,
    )
    solve_elapsed_s = float(perf_counter() - solve_started_at)
    diagnostics = build_attempt_diagnostics(problem, model=model, result=result)
    log_solve_end(
        attempt_name=attempt_name,
        status=int(result.status),
        success=bool(result.success),
        elapsed_s=solve_elapsed_s,
        objective_pwl_j=None if result.fun is None else float(result.fun),
        objective_bound=get_optional_float(result, "mip_dual_bound"),
        mip_gap=get_optional_float(result, "mip_gap"),
    )
    log_solve_diagnostics(attempt_name=attempt_name, diagnostics=diagnostics)
    return MilpAttemptResult(
        attempt_name=str(attempt_name),
        allowed_pa_ids=tuple(int(pa_id) for pa_id in allowed_pa_ids),
        success=bool(result.success),
        solver_status=int(result.status),
        solver_message=str(result.message),
        objective_pwl_j=None if result.fun is None else float(result.fun),
        objective_bound=get_optional_float(result, "mip_dual_bound"),
        mip_gap=get_optional_float(result, "mip_gap"),
        solution=result.x,
        model_size=model.model_size,
        segments_by_pa=model.segments_by_pa,
        variables=model.variables,
        build_elapsed_s=float(model.build_elapsed_s),
        solve_elapsed_s=solve_elapsed_s,
        diagnostics=diagnostics,
    )


def build_milp_model(
    problem: OfdmaMilpProblem,
    *,
    allowed_pa_ids: tuple[int, ...],
    max_users_per_slot: int | None = None,
) -> MilpBuild:
    """Build the sparse SciPy MILP for one allowed-PA attempt."""

    build_started_at = perf_counter()
    segments_by_pa = build_pa_piecewise_segments(problem.pa_catalog)
    max_per_chain_dc_w = max(
        max(float(segment.left_dc_w), float(segment.right_dc_w))
        for segments in segments_by_pa.values()
        for segment in segments
    )
    min_active_per_chain_dc_w = min(
        min(float(segment.left_dc_w), float(segment.right_dc_w))
        for pa_id in allowed_pa_ids
        for segment in segments_by_pa[int(pa_id)]
    )
    variable_builder, variables = build_variables(
        problem,
        segments_by_pa=segments_by_pa,
        allowed_pa_ids=allowed_pa_ids,
        max_per_chain_dc_w=float(max_per_chain_dc_w),
    )
    constraint_builder = ConstraintBuilder()
    add_user_slot_constraints(constraint_builder, problem=problem, variables=variables)
    add_slot_user_cardinality_constraints(
        constraint_builder,
        problem=problem,
        variables=variables,
        max_users_per_slot=max_users_per_slot,
    )
    add_demand_constraints(constraint_builder, problem=problem, variables=variables)
    add_slot_pa_constraints(constraint_builder, problem=problem, variables=variables)
    add_prb_constraints(constraint_builder, problem=problem, variables=variables)
    add_curve_constraints(
        constraint_builder,
        problem=problem,
        variables=variables,
        segments_by_pa=segments_by_pa,
    )
    add_output_limit_constraints(constraint_builder, problem=problem, variables=variables)
    add_product_constraints(
        constraint_builder,
        problem=problem,
        variables=variables,
        max_per_chain_dc_w=float(max_per_chain_dc_w),
        min_active_per_chain_dc_w=float(min_active_per_chain_dc_w),
    )
    model_size = build_model_size(variable_builder, constraint_builder)
    return MilpBuild(
        c=np.asarray(variable_builder.objective, dtype=float),
        integrality=np.asarray(variable_builder.integrality, dtype=int),
        bounds=Bounds(
            np.asarray(variable_builder.lower_bounds, dtype=float),
            np.asarray(variable_builder.upper_bounds, dtype=float),
        ),
        constraints=constraint_builder.to_linear_constraint(variable_count=model_size.variable_count),
        variables=variables,
        model_size=model_size,
        segments_by_pa=segments_by_pa,
        allowed_pa_ids=tuple(int(pa_id) for pa_id in allowed_pa_ids),
        build_elapsed_s=float(perf_counter() - build_started_at),
    )


def build_variables(
    problem: OfdmaMilpProblem,
    *,
    segments_by_pa: dict[int, tuple[PaCurveSegment, ...]],
    allowed_pa_ids: tuple[int, ...],
    max_per_chain_dc_w: float,
) -> tuple[VariableBuilder, MilpVariableIndex]:
    builder = VariableBuilder()
    allowed_pa_id_set = {int(pa_id) for pa_id in allowed_pa_ids}
    x = {
        (int(row.global_id), int(slot_id)): builder.add_variable(
            lower=0.0,
            upper=1.0 if int(row.pa_id) in allowed_pa_id_set else 0.0,
            integer=True,
        )
        for row in problem.candidate_rows
        for slot_id in range(int(problem.frame_n_slots))
    }
    z = {
        (int(pa_id), int(slot_id)): builder.add_variable(
            lower=0.0,
            upper=1.0 if int(pa_id) in allowed_pa_id_set else 0.0,
            integer=True,
        )
        for pa_id in range(len(problem.pa_catalog))
        for slot_id in range(int(problem.frame_n_slots))
    }
    delta = {
        (int(slot_id), int(prb_count)): builder.add_variable(lower=0.0, upper=1.0, integer=True)
        for slot_id in range(int(problem.frame_n_slots))
        for prb_count in range(int(problem.prb_max) + 1)
    }
    beta = {
        (int(slot_id), int(pa_id), int(segment.segment_id)): builder.add_variable(
            lower=0.0,
            upper=1.0,
            integer=True,
        )
        for slot_id in range(int(problem.frame_n_slots))
        for pa_id, segments in segments_by_pa.items()
        for segment in segments
    }
    theta = {
        (int(slot_id), int(pa_id), int(segment.segment_id)): builder.add_variable(
            lower=0.0,
            upper=1.0,
            integer=False,
        )
        for slot_id in range(int(problem.frame_n_slots))
        for pa_id, segments in segments_by_pa.items()
        for segment in segments
    }
    w = {
        int(slot_id): builder.add_variable(
            lower=0.0,
            upper=float(max_per_chain_dc_w),
            integer=False,
        )
        for slot_id in range(int(problem.frame_n_slots))
    }
    v = {
        (int(slot_id), int(prb_count)): builder.add_variable(
            lower=0.0,
            upper=float(max_per_chain_dc_w),
            integer=False,
            objective=float(problem.t_slot_s)
            * float(problem.n_tx_chains)
            * float(prb_count)
            / float(problem.prb_max),
        )
        for slot_id in range(int(problem.frame_n_slots))
        for prb_count in range(int(problem.prb_max) + 1)
    }
    return builder, MilpVariableIndex(
        x=x,
        z=z,
        delta=delta,
        beta=beta,
        theta=theta,
        w=w,
        v=v,
    )


def add_user_slot_constraints(
    builder: ConstraintBuilder,
    *,
    problem: OfdmaMilpProblem,
    variables: MilpVariableIndex,
) -> None:
    for user_id, rows in problem.candidate_rows_by_user.items():
        for slot_id in range(int(problem.frame_n_slots)):
            terms = [(variables.x[(int(row.global_id), int(slot_id))], 1.0) for row in rows]
            builder.add_constraint(terms, lower=-np.inf, upper=1.0)


def add_slot_user_cardinality_constraints(
    builder: ConstraintBuilder,
    *,
    problem: OfdmaMilpProblem,
    variables: MilpVariableIndex,
    max_users_per_slot: int | None,
) -> None:
    if max_users_per_slot is None:
        return

    for slot_id in range(int(problem.frame_n_slots)):
        terms = [
            (variables.x[(int(row.global_id), int(slot_id))], 1.0)
            for row in problem.candidate_rows
        ]
        builder.add_constraint(terms, lower=-np.inf, upper=float(max_users_per_slot))


def add_demand_constraints(
    builder: ConstraintBuilder,
    *,
    problem: OfdmaMilpProblem,
    variables: MilpVariableIndex,
) -> None:
    for user_id, rows in problem.candidate_rows_by_user.items():
        demand_bits = float(problem.demand_bits_by_user[int(user_id)])
        terms = [
            (
                variables.x[(int(row.global_id), int(slot_id))],
                float(row.bits_per_slot) / float(demand_bits),
            )
            for row in rows
            for slot_id in range(int(problem.frame_n_slots))
        ]
        builder.add_constraint(
            terms,
            lower=1.0,
            upper=np.inf,
        )


def add_slot_pa_constraints(
    builder: ConstraintBuilder,
    *,
    problem: OfdmaMilpProblem,
    variables: MilpVariableIndex,
) -> None:
    for slot_id in range(int(problem.frame_n_slots)):
        builder.add_constraint(
            [
                (variables.z[(int(pa_id), int(slot_id))], 1.0)
                for pa_id in range(len(problem.pa_catalog))
            ],
            lower=-np.inf,
            upper=1.0,
        )
        for pa_id in range(len(problem.pa_catalog)):
            pa_rows = [row for row in problem.candidate_rows if int(row.pa_id) == int(pa_id)]
            builder.add_constraint(
                [(variables.z[(int(pa_id), int(slot_id))], 1.0)]
                + [(variables.x[(int(row.global_id), int(slot_id))], -1.0) for row in pa_rows],
                lower=-np.inf,
                upper=0.0,
            )
            for row in pa_rows:
                builder.add_constraint(
                    [
                        (variables.x[(int(row.global_id), int(slot_id))], 1.0),
                        (variables.z[(int(pa_id), int(slot_id))], -1.0),
                    ],
                    lower=-np.inf,
                    upper=0.0,
                )


def add_prb_constraints(
    builder: ConstraintBuilder,
    *,
    problem: OfdmaMilpProblem,
    variables: MilpVariableIndex,
) -> None:
    for slot_id in range(int(problem.frame_n_slots)):
        builder.add_constraint(
            [
                (variables.delta[(int(slot_id), int(prb_count))], 1.0)
                for prb_count in range(int(problem.prb_max) + 1)
            ],
            lower=1.0,
            upper=1.0,
        )
        builder.add_constraint(
            [(variables.delta[(int(slot_id), 0)], 1.0)]
            + [
                (variables.z[(int(pa_id), int(slot_id))], 1.0)
                for pa_id in range(len(problem.pa_catalog))
            ],
            lower=1.0,
            upper=1.0,
        )
        terms = [
            (variables.x[(int(row.global_id), int(slot_id))], float(row.n_prb))
            for row in problem.candidate_rows
        ]
        terms.extend(
            (variables.delta[(int(slot_id), int(prb_count))], -float(prb_count))
            for prb_count in range(int(problem.prb_max) + 1)
        )
        builder.add_constraint(terms, lower=0.0, upper=0.0)


def add_curve_constraints(
    builder: ConstraintBuilder,
    *,
    problem: OfdmaMilpProblem,
    variables: MilpVariableIndex,
    segments_by_pa: dict[int, tuple[PaCurveSegment, ...]],
) -> None:
    for slot_id in range(int(problem.frame_n_slots)):
        w_terms = [(variables.w[int(slot_id)], 1.0)]
        for pa_id, segments in segments_by_pa.items():
            pa_rows = [row for row in problem.candidate_rows if int(row.pa_id) == int(pa_id)]
            rf_terms = [
                (variables.x[(int(row.global_id), int(slot_id))], float(row.p_out_total_w))
                for row in pa_rows
            ]
            beta_terms = []
            for segment in segments:
                beta_key = (int(slot_id), int(pa_id), int(segment.segment_id))
                beta_terms.append((variables.beta[beta_key], 1.0))
                rf_terms.append(
                    (
                        variables.beta[beta_key],
                        -float(problem.n_tx_chains) * float(segment.left_p_out_w),
                    )
                )
                rf_terms.append(
                    (
                        variables.theta[beta_key],
                        -float(problem.n_tx_chains)
                        * (float(segment.right_p_out_w) - float(segment.left_p_out_w)),
                    )
                )
                w_terms.append((variables.beta[beta_key], -float(segment.left_dc_w)))
                w_terms.append(
                    (
                        variables.theta[beta_key],
                        -(float(segment.right_dc_w) - float(segment.left_dc_w)),
                    )
                )
                builder.add_constraint(
                    [
                        (variables.theta[beta_key], 1.0),
                        (variables.beta[beta_key], -1.0),
                    ],
                    lower=-np.inf,
                    upper=0.0,
                )
            builder.add_constraint(rf_terms, lower=0.0, upper=0.0)
            builder.add_constraint(
                beta_terms + [(variables.z[(int(pa_id), int(slot_id))], -1.0)],
                lower=0.0,
                upper=0.0,
            )
        builder.add_constraint(w_terms, lower=0.0, upper=0.0)


def add_output_limit_constraints(
    builder: ConstraintBuilder,
    *,
    problem: OfdmaMilpProblem,
    variables: MilpVariableIndex,
) -> None:
    for slot_id in range(int(problem.frame_n_slots)):
        for pa_id, pa in enumerate(problem.pa_catalog):
            pa_rows = [row for row in problem.candidate_rows if int(row.pa_id) == int(pa_id)]
            builder.add_constraint(
                [(variables.x[(int(row.global_id), int(slot_id))], float(row.p_out_total_w)) for row in pa_rows]
                + [
                    (
                        variables.z[(int(pa_id), int(slot_id))],
                        -float(problem.n_tx_chains) * float(pa.p_max_w),
                    )
                ],
                lower=-np.inf,
                upper=0.0,
            )


def add_product_constraints(
    builder: ConstraintBuilder,
    *,
    problem: OfdmaMilpProblem,
    variables: MilpVariableIndex,
    max_per_chain_dc_w: float,
    min_active_per_chain_dc_w: float,
) -> None:
    big_m = float(max_per_chain_dc_w)
    active_floor = float(min_active_per_chain_dc_w)
    for slot_id in range(int(problem.frame_n_slots)):
        for prb_count in range(int(problem.prb_max) + 1):
            v_id = variables.v[(int(slot_id), int(prb_count))]
            delta_id = variables.delta[(int(slot_id), int(prb_count))]
            w_id = variables.w[int(slot_id)]
            if int(prb_count) > 0:
                builder.add_constraint([(v_id, 1.0), (delta_id, -active_floor)], lower=0.0, upper=np.inf)
            builder.add_constraint([(v_id, 1.0), (delta_id, -big_m)], lower=-np.inf, upper=0.0)
            builder.add_constraint([(v_id, 1.0), (w_id, -1.0)], lower=-np.inf, upper=0.0)
            builder.add_constraint(
                [(v_id, 1.0), (w_id, -1.0), (delta_id, -big_m)],
                lower=-big_m,
                upper=np.inf,
            )


def build_scipy_milp_options() -> dict[str, bool | int | float]:
    options: dict[str, bool | int | float] = {
        "disp": logging.getLogger("day_run").isEnabledFor(logging.DEBUG),
    }
    if K_MILP_SOLVER_CONFIG.time_limit_s is not None:
        options["time_limit"] = float(K_MILP_SOLVER_CONFIG.time_limit_s)
    if K_MILP_SOLVER_CONFIG.node_limit is not None:
        options["node_limit"] = int(K_MILP_SOLVER_CONFIG.node_limit)
    if K_MILP_SOLVER_CONFIG.rel_gap is not None:
        options["mip_rel_gap"] = float(K_MILP_SOLVER_CONFIG.rel_gap)
    return options


def configured_max_users_per_slot() -> int | None:
    return K_MILP_SOLVER_CONFIG.max_users_per_slot

__all__ = [
    "build_scipy_milp_options",
    "build_and_solve_milp_attempt",
    "build_milp_model",
    "configured_max_users_per_slot",
]
