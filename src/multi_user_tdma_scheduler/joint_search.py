from __future__ import annotations

from dataclasses import dataclass

from configs import MULTI_USER_TDMA_CONFIG
from models import MultiUserScheduleResult, PASwitchPolicy, SchedulerMode, SchedulerPowerSummary, SlotAllocation, SlotSchedule, UserScheduleSummary

from .models import PreparedJointScheduleProblem
from .tdma_space import prune_dominated_user_tdma_space


TOL = 1e-12
EXACT_SOLVER_NAME = "cartesian_product_dp"


@dataclass(frozen=True)
class _JointCandidateRow:
    """One quantized TDMA candidate row from the scheduler-facing table."""

    user_id: int
    pa_id: int
    n_prb: int
    layers: int
    mcs: int
    n_slots: int
    bits_per_slot: float
    p_dc_active_w: float
    p_out_total_w: float
    delivered_rate_bps: float
    schedule_cost: float

    def to_allocation(self) -> SlotAllocation:
        return SlotAllocation(
            user_id=int(self.user_id),
            pa_id=int(self.pa_id),
            n_prb=int(self.n_prb),
            layers=int(self.layers),
            mcs=int(self.mcs),
            bits_per_slot=float(self.bits_per_slot),
            p_out_total_w=float(self.p_out_total_w),
            p_dc_active_w=float(self.p_dc_active_w),
        )


@dataclass(frozen=True)
class _DPState:
    """Best partial Cartesian choice for one exact slot total."""

    schedule_cost: float
    delivered_rate_bps: float
    rows: tuple[_JointCandidateRow, ...]


def run_joint_schedule_search(
    problem: PreparedJointScheduleProblem,
    *,
    switch_policy: PASwitchPolicy = PASwitchPolicy.DUAL_SWITCHABLE,
) -> MultiUserScheduleResult:
    """Search the TDMA joint space and return the shared public scheduler result."""

    resolved_policy = switch_policy if isinstance(switch_policy, PASwitchPolicy) else PASwitchPolicy(str(switch_policy))
    if resolved_policy == PASwitchPolicy.DUAL_SWITCHABLE:
        return ExactJointScheduleSearch(problem).run()

    pa_catalog = tuple(problem.pa_catalog)
    candidate_pa_ids = list(range(len(pa_catalog)))
    if resolved_policy == PASwitchPolicy.BASELINE_8W_ONLY:
        candidate_pa_ids = [pa_id for pa_id, pa in enumerate(pa_catalog) if str(pa.scenario_label) == "8W PA"]
        if not candidate_pa_ids and pa_catalog:
            max_p_max_w = max(float(pa.p_max_w) for pa in pa_catalog)
            candidate_pa_ids = [pa_id for pa_id, pa in enumerate(pa_catalog) if float(pa.p_max_w) == max_p_max_w]

    best_result = None
    best_rank = None
    last_infeasible_reason = None
    last_search_stats = {}
    for pa_id in candidate_pa_ids:
        filtered_problem = PreparedJointScheduleProblem(
            frame_n_slots=int(problem.frame_n_slots),
            n_tx_chains=int(problem.n_tx_chains),
            pa_catalog=tuple(problem.pa_catalog),
            user_requirements=problem.user_requirements,
            user_candidate_spaces={
                int(user_id): candidate_table.loc[candidate_table["pa_id"].astype(int) == int(pa_id)].copy().reset_index(drop=True)
                for user_id, candidate_table in problem.user_candidate_spaces.items()
            },
            infeasible_reason=problem.infeasible_reason,
        )
        candidate_result = ExactJointScheduleSearch(filtered_problem).run()
        if not candidate_result.feasible:
            last_infeasible_reason = candidate_result.infeasible_reason
            last_search_stats = dict(candidate_result.solver_details.get("search_stats", {}))
            continue

        candidate_rank = (
            float(candidate_result.power_summary.average_frame_dc_power_w),
            int(sum(slot.active for slot in candidate_result.slot_schedules)),
            -float(sum(user.delivered_rate_bps for user in candidate_result.user_summaries)),
        )
        if best_rank is not None and candidate_rank >= best_rank:
            continue

        best_result = candidate_result
        best_rank = candidate_rank

    if best_result is not None:
        return best_result

    return _build_tdma_result(
        problem,
        selected_rows=None,
        search_stats=last_search_stats,
        infeasible_reason=last_infeasible_reason or "No feasible joint TDMA schedule was found for the prepared user spaces.",
    )


class ExactJointScheduleSearch:
    """Exact dynamic program over the Cartesian product of per-user TDMA rows."""

    def __init__(
        self,
        problem: PreparedJointScheduleProblem,
    ):
        self.problem = problem
        self.frame_n_slots = int(problem.frame_n_slots)
        self.frame_duration_s = float(self.frame_n_slots) * float(MULTI_USER_TDMA_CONFIG.t_slot_s)
        self.ranked_user_rows = self._build_user_rows()
        self.user_order = sorted(self.ranked_user_rows)
        self.cartesian_product_size = _compute_cartesian_product_size(self.ranked_user_rows)
        self.search_stats = {
            "solver": EXACT_SOLVER_NAME,
            "users": int(len(self.ranked_user_rows)),
            "rows_by_user": {int(user_id): int(len(rows)) for user_id, rows in self.ranked_user_rows.items()},
            "user_order": list(self.user_order),
            "cartesian_product_size": int(self.cartesian_product_size),
            "transitions_considered": 0,
            "transitions_pruned_slot_budget": 0,
            "state_replacements": 0,
            "peak_slot_states": 0,
            "final_slot_states": 0,
            "best_rank": None,
        }

    def run(self) -> MultiUserScheduleResult:
        if self.problem.infeasible_reason is not None:
            return _build_tdma_result(
                self.problem,
                selected_rows=None,
                search_stats=self.search_stats,
                infeasible_reason=str(self.problem.infeasible_reason),
            )

        if not self.ranked_user_rows or any(not rows for rows in self.ranked_user_rows.values()):
            return _build_tdma_result(
                self.problem,
                selected_rows=None,
                search_stats=self.search_stats,
                infeasible_reason="No feasible joint TDMA schedule was found for the prepared user spaces.",
            )

        best_rows, best_rank = self._run_cartesian_dp()
        self.search_stats["best_rank"] = best_rank
        return _build_tdma_result(
            self.problem,
            selected_rows=best_rows,
            search_stats=self.search_stats,
            infeasible_reason=None if best_rows is not None else "No feasible joint TDMA schedule was found for the prepared user spaces.",
        )

    def _build_user_rows(self) -> dict[int, tuple[_JointCandidateRow, ...]]:
        user_rows = {}
        for user_id, candidate_table in sorted(self.problem.user_candidate_spaces.items()):
            dominated_pruned_table = prune_dominated_user_tdma_space(
                candidate_table,
                frame_n_slots=self.frame_n_slots,
            )
            rows = [
                _build_candidate_row(
                    raw_row,
                    frame_n_slots=self.frame_n_slots,
                    frame_duration_s=self.frame_duration_s,
                )
                for raw_row in dominated_pruned_table.to_dict("records")
            ]
            rows.sort(
                key=lambda row: (
                    float(row.schedule_cost),
                    int(row.n_slots),
                    -float(row.delivered_rate_bps),
                    _stable_row_key(row),
                )
            )
            user_rows[int(user_id)] = tuple(rows)
        return user_rows

    def _run_cartesian_dp(self) -> tuple[tuple[_JointCandidateRow, ...] | None, tuple[float, int, float] | None]:
        states_by_slot = {
            0: _DPState(
                schedule_cost=0.0,
                delivered_rate_bps=0.0,
                rows=(),
            )
        }

        for user_id in self.user_order:
            states_by_slot = self._advance_dp_user(
                states_by_slot,
                self.ranked_user_rows[user_id],
            )
            self.search_stats["peak_slot_states"] = max(
                int(self.search_stats["peak_slot_states"]),
                int(len(states_by_slot)),
            )
            if not states_by_slot:
                break

        self.search_stats["final_slot_states"] = int(len(states_by_slot))
        if not states_by_slot:
            return None, None

        best_state = min(states_by_slot.values(), key=_state_rank)
        best_rows = tuple(sorted(best_state.rows, key=_stable_row_key))
        best_rank = _public_state_rank(best_state)
        return best_rows, best_rank

    def _advance_dp_user(
        self,
        states_by_slot: dict[int, _DPState],
        rows: tuple[_JointCandidateRow, ...],
    ) -> dict[int, _DPState]:
        next_states_by_slot = {}
        for slot_total, state in states_by_slot.items():
            for row in rows:
                self.search_stats["transitions_considered"] += 1
                next_slot_total = int(slot_total + row.n_slots)
                if next_slot_total > self.frame_n_slots:
                    self.search_stats["transitions_pruned_slot_budget"] += 1
                    continue

                candidate_state = _DPState(
                    schedule_cost=float(state.schedule_cost + row.schedule_cost),
                    delivered_rate_bps=float(state.delivered_rate_bps + row.delivered_rate_bps),
                    rows=(*state.rows, row),
                )
                incumbent_state = next_states_by_slot.get(next_slot_total)
                if incumbent_state is not None and _state_rank(incumbent_state) <= _state_rank(candidate_state):
                    continue

                next_states_by_slot[next_slot_total] = candidate_state
                self.search_stats["state_replacements"] += 1
        return next_states_by_slot


def _build_tdma_result(
    problem: PreparedJointScheduleProblem,
    *,
    selected_rows: tuple[_JointCandidateRow, ...] | None,
    search_stats: dict[str, object],
    infeasible_reason: str | None,
) -> MultiUserScheduleResult:
    frame_n_slots = int(problem.frame_n_slots)
    t_slot_s = float(MULTI_USER_TDMA_CONFIG.t_slot_s)
    frame_duration_s = float(frame_n_slots) * float(t_slot_s)
    user_requirements = tuple(
        (int(user_row.user_id), float(user_row.required_rate_bps))
        for user_row in problem.user_requirements.sort_values("user_id").itertuples(index=False)
    )

    if infeasible_reason is None and selected_rows is not None:
        slot_schedules, delivered_bits_by_user = _build_tdma_slot_schedules_and_delivered_bits(
            selected_rows=selected_rows,
            frame_n_slots=frame_n_slots,
        )
        frame_energy_j = float(t_slot_s) * float(sum(slot.dc_power_w for slot in slot_schedules))
        average_frame_dc_power_w = float(frame_energy_j) / max(float(frame_duration_s), TOL)
        average_frame_rf_output_w = float(sum(slot.aggregate_p_out_w for slot in slot_schedules)) / max(frame_n_slots, 1)
    else:
        slot_schedules = tuple(
            SlotSchedule(slot_index=slot_index, active=False, pa_id=None, used_prbs=0, aggregate_p_out_w=0.0, dc_power_w=0.0, allocations=())
            for slot_index in range(frame_n_slots)
        )
        delivered_bits_by_user = {int(user_id): 0.0 for user_id, _ in user_requirements}
        frame_energy_j = 0.0
        average_frame_dc_power_w = 0.0
        average_frame_rf_output_w = 0.0

    user_summaries = tuple(
        UserScheduleSummary(
            user_id=int(user_id),
            required_bits=float(required_rate_bps) * float(frame_duration_s),
            delivered_bits=float(delivered_bits_by_user[int(user_id)]),
            required_rate_bps=float(required_rate_bps),
            delivered_rate_bps=float(delivered_bits_by_user[int(user_id)] / max(float(frame_duration_s), TOL)),
            satisfied=float(delivered_bits_by_user[int(user_id)]) + TOL >= float(required_rate_bps) * float(frame_duration_s),
        )
        for user_id, required_rate_bps in user_requirements
    )
    return MultiUserScheduleResult(
        scheduler_mode=SchedulerMode.TDMA,
        feasible=infeasible_reason is None,
        infeasible_reason=infeasible_reason,
        power_summary=SchedulerPowerSummary(
            frame_energy_j=float(frame_energy_j),
            average_frame_dc_power_w=float(average_frame_dc_power_w),
            active_energy_j=float(frame_energy_j),
            inactive_energy_j=0.0,
            average_frame_rf_output_w=float(average_frame_rf_output_w),
        ),
        user_summaries=user_summaries,
        slot_schedules=slot_schedules,
        solver_details={"search_stats": dict(search_stats)},
    )


def _build_tdma_slot_schedules_and_delivered_bits(
    *,
    selected_rows: tuple[_JointCandidateRow, ...],
    frame_n_slots: int,
) -> tuple[tuple[SlotSchedule, ...], dict[int, float]]:
    delivered_bits_by_user = {}
    slot_schedules = []
    slot_index = 0
    for row in sorted(selected_rows, key=_stable_row_key):
        allocation = row.to_allocation()
        delivered_bits_by_user[int(row.user_id)] = float(
            delivered_bits_by_user.get(int(row.user_id), 0.0)
            + float(int(row.n_slots) * float(row.bits_per_slot))
        )
        for _ in range(int(row.n_slots)):
            slot_schedules.append(
                SlotSchedule(
                    slot_index=slot_index,
                    active=True,
                    pa_id=int(row.pa_id),
                    used_prbs=int(row.n_prb),
                    aggregate_p_out_w=float(row.p_out_total_w),
                    dc_power_w=float(row.p_dc_active_w),
                    allocations=(allocation,),
                )
            )
            slot_index += 1

    while slot_index < int(frame_n_slots):
        slot_schedules.append(
            SlotSchedule(slot_index=slot_index, active=False, pa_id=None, used_prbs=0, aggregate_p_out_w=0.0, dc_power_w=0.0, allocations=())
        )
        slot_index += 1

    return tuple(slot_schedules), delivered_bits_by_user


def _build_candidate_row(
    raw_row,
    *,
    frame_n_slots: int,
    frame_duration_s: float,
) -> _JointCandidateRow:
    slot_share = float(raw_row["n_slots"]) / float(frame_n_slots)
    bits_per_slot = float(raw_row["bits_per_slot"])
    p_dc_active_w = float(raw_row["p_dc_active_w"])
    return _JointCandidateRow(
        user_id=int(raw_row["user_id"]),
        pa_id=int(raw_row["pa_id"]),
        n_prb=int(raw_row["n_prb"]),
        layers=int(raw_row["layers"]),
        mcs=int(raw_row["mcs"]),
        n_slots=int(raw_row["n_slots"]),
        bits_per_slot=bits_per_slot,
        p_dc_active_w=p_dc_active_w,
        p_out_total_w=float(raw_row["p_out_total_w"]),
        delivered_rate_bps=float(int(raw_row["n_slots"]) * bits_per_slot / frame_duration_s),
        schedule_cost=float(slot_share * p_dc_active_w),
    )


def _compute_cartesian_product_size(ranked_user_rows: dict[int, tuple[_JointCandidateRow, ...]]) -> int:
    if not ranked_user_rows:
        return 0

    product_size = 1
    for rows in ranked_user_rows.values():
        product_size *= int(len(rows))
    return int(product_size)


def _state_rank(state: _DPState) -> tuple[float, int, float, tuple[tuple[int, int, int, int, int, int, float, float, float], ...]]:
    return (
        float(state.schedule_cost),
        int(sum(row.n_slots for row in state.rows)),
        -float(state.delivered_rate_bps),
        tuple(_stable_row_key(row) for row in state.rows),
    )


def _public_state_rank(state: _DPState) -> tuple[float, int, float]:
    return (
        float(state.schedule_cost),
        int(sum(row.n_slots for row in state.rows)),
        -float(state.delivered_rate_bps),
    )


def _stable_row_key(row: _JointCandidateRow) -> tuple[int, int, int, int, int, int, float, float, float]:
    return (
        int(row.user_id),
        int(row.pa_id),
        int(row.n_slots),
        int(row.n_prb),
        int(row.mcs),
        int(row.layers),
        float(row.bits_per_slot),
        float(row.p_dc_active_w),
        float(row.p_out_total_w),
    )


__all__ = ["run_joint_schedule_search"]
