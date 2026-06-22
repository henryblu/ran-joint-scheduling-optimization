from __future__ import annotations

"""Structured diagnostics for one OFDMA MILP solve attempt."""

from .models import MilpBuild, OfdmaMilpProblem


def build_attempt_diagnostics(
    problem: OfdmaMilpProblem,
    *,
    model: MilpBuild,
    result,
) -> dict[str, object]:
    solution = getattr(result, "x", None)
    diagnostics: dict[str, object] = {
        "has_incumbent": solution is not None,
        "objective_pwl_j": None if result.fun is None else float(result.fun),
        "objective_bound": get_optional_float(result, "mip_dual_bound"),
        "mip_gap": get_optional_float(result, "mip_gap"),
    }
    if solution is None:
        return diagnostics

    selected_rows_by_slot = selected_rows_by_slot_from_solution(problem, model=model, solution=solution)
    active_slot_summaries = tuple(
        slot_diagnostic_summary(problem, slot_id=slot_id, selected_rows=tuple(rows), solution=solution, model=model)
        for slot_id, rows in selected_rows_by_slot.items()
        if rows
    )
    delivered_bits_by_user = {
        int(user_id): sum(
            float(row.bits_per_slot)
            for rows in selected_rows_by_slot.values()
            for row in rows
            if int(row.user_id) == int(user_id)
        )
        for user_id in sorted(problem.required_rate_by_user)
    }
    diagnostics.update(
        {
            "active_slot_count": int(len(active_slot_summaries)),
            "allocation_count": int(sum(len(summary["users"]) for summary in active_slot_summaries)),
            "pa_slot_counts": pa_slot_counts_from_solution(problem, model=model, solution=solution),
            "delivered_bits_by_user": delivered_bits_by_user,
            "slot_summaries": active_slot_summaries,
        }
    )
    return diagnostics


def selected_rows_by_slot_from_solution(
    problem: OfdmaMilpProblem,
    *,
    model: MilpBuild,
    solution,
) -> dict[int, tuple]:
    selected_rows_by_slot = {int(slot_id): [] for slot_id in range(int(problem.frame_n_slots))}
    for row in problem.candidate_rows:
        for slot_id in range(int(problem.frame_n_slots)):
            variable_id = model.variables.x[(int(row.global_id), int(slot_id))]
            if float(solution[int(variable_id)]) <= 0.5:
                continue
            selected_rows_by_slot[int(slot_id)].append(row)
    return {
        int(slot_id): tuple(rows)
        for slot_id, rows in selected_rows_by_slot.items()
    }


def slot_diagnostic_summary(
    problem: OfdmaMilpProblem,
    *,
    slot_id: int,
    selected_rows: tuple,
    solution,
    model: MilpBuild,
) -> dict[str, object]:
    pa_id = selected_pa_id_from_solution(problem, slot_id=int(slot_id), solution=solution, model=model)
    return {
        "slot": int(slot_id),
        "pa_id": pa_id,
        "users": tuple(int(row.user_id) for row in selected_rows),
        "used_prbs": int(sum(int(row.n_prb) for row in selected_rows)),
        "aggregate_p_out_w": float(sum(float(row.p_out_total_w) for row in selected_rows)),
        "delivered_bits_by_user": {
            int(row.user_id): float(row.bits_per_slot)
            for row in selected_rows
        },
    }


def selected_pa_id_from_solution(
    problem: OfdmaMilpProblem,
    *,
    slot_id: int,
    solution,
    model: MilpBuild,
) -> int | None:
    for pa_id in range(len(problem.pa_catalog)):
        variable_id = model.variables.z[(int(pa_id), int(slot_id))]
        if float(solution[int(variable_id)]) > 0.5:
            return int(pa_id)
    return None


def pa_slot_counts_from_solution(
    problem: OfdmaMilpProblem,
    *,
    model: MilpBuild,
    solution,
) -> dict[int, int]:
    return {
        int(pa_id): sum(
            1
            for slot_id in range(int(problem.frame_n_slots))
            if float(solution[int(model.variables.z[(int(pa_id), int(slot_id))])]) > 0.5
        )
        for pa_id in range(len(problem.pa_catalog))
    }


def get_optional_float(result, attribute_name: str) -> float | None:
    value = getattr(result, attribute_name, None)
    if value is None:
        return None
    return float(value)


__all__ = ["build_attempt_diagnostics"]
