from __future__ import annotations

import numpy as np

from configs import inactive_pa_bank_power
from models import PAState, PASwitchPolicy

from .models import MultiUserTdmaSchedulerResult, PreparedJointScheduleProblem


def run_joint_schedule_search(
    problem: PreparedJointScheduleProblem,
    *,
    switch_policy: PASwitchPolicy = PASwitchPolicy.STANDBY,
) -> MultiUserTdmaSchedulerResult:
    """Search the implicit Cartesian joint TDMA space for the optimal schedule."""

    return ExactJointScheduleSearch(
        problem,
        switch_policy=switch_policy,
    ).run()


class ExactJointScheduleSearch:
    """Exact one-row-per-user search over a prepared TDMA scheduling problem."""

    TOL = 1e-12

    def __init__(
        self,
        problem: PreparedJointScheduleProblem,
        *,
        switch_policy: PASwitchPolicy,
    ):
        self.problem = problem
        self.user_candidate_spaces = {
            int(user_id): candidate_table.copy()
            for user_id, candidate_table in problem.user_candidate_spaces.items()
        }
        self.window_n_slots = int(problem.window_n_slots)
        self.pa_catalog = tuple(problem.pa_catalog)
        self.n_tx_chains = int(problem.n_tx_chains)
        self.switch_policy = (
            switch_policy
            if isinstance(switch_policy, PASwitchPolicy)
            else PASwitchPolicy(str(switch_policy))
        )

        self.ranked_user_rows = {}
        self.user_order = []
        self.suffix_min_slots = []
        self.suffix_max_rate_avg_frame_bps = []
        self.lower_bound_cache = {}
        self.best_schedule = None
        self.best_schedule_power = np.inf
        self.best_rank = (np.inf, np.inf, np.inf)
        self.search_stats = {}

        self._prepare_search_state()

    def run(self) -> MultiUserTdmaSchedulerResult:
        """Run the exact joint scheduler search and return the best schedule payload."""

        if not self.ranked_user_rows:
            return self._build_result()
        if any(not rows for rows in self.ranked_user_rows.values()):
            return self._build_result()

        greedy_schedule = self._seed_greedy_schedule()
        if greedy_schedule is not None:
            self.search_stats["complete_feasible_schedules"] += 1
            self._accept_schedule(greedy_schedule)

        self._search_from(
            depth=0,
            slot_sum=0,
            exact_cost_sum=0.0,
            rate_sum=0.0,
            used_pa_ids=frozenset(),
            selected_rows=[],
        )
        return self._build_result()

    def _prepare_search_state(self):
        """Resolve ranked per-user rows and exact suffix bounds before the search."""

        min_slots = {}
        min_sort_cost = {}
        max_rate_avg_frame_bps = {}
        for user_id, candidate_table in self.user_candidate_spaces.items():
            rows = candidate_table.to_dict("records")
            rows.sort(key=self._ranked_row_key)
            self.ranked_user_rows[int(user_id)] = rows
            if not rows:
                continue

            min_slots[int(user_id)] = min(int(row["n_slots"]) for row in rows)
            min_sort_cost[int(user_id)] = float(self._incremental_schedule_cost(rows[0], frozenset()))
            max_rate_avg_frame_bps[int(user_id)] = max(float(row["rate_avg_frame_bps"]) for row in rows)

        self.user_order = sorted(
            self.ranked_user_rows,
            key=lambda user_id: (min_slots.get(user_id, 0), min_sort_cost.get(user_id, np.inf)),
            reverse=True,
        )
        self.search_stats = {
            "user_order": list(self.user_order),
            "nodes_visited": 0,
            "complete_feasible_schedules": 0,
            "pruned_time_direct": 0,
            "pruned_power_direct": 0,
            "pruned_time_bound": 0,
            "pruned_power_bound": 0,
            "pruned_rank_bound": 0,
            "best_updates": 0,
        }

        self.suffix_min_slots = [0] * (len(self.user_order) + 1)
        self.suffix_max_rate_avg_frame_bps = [0.0] * (len(self.user_order) + 1)
        for depth in range(len(self.user_order) - 1, -1, -1):
            user_id = self.user_order[depth]
            self.suffix_min_slots[depth] = self.suffix_min_slots[depth + 1] + min_slots.get(user_id, 0)
            self.suffix_max_rate_avg_frame_bps[depth] = (
                self.suffix_max_rate_avg_frame_bps[depth + 1] + max_rate_avg_frame_bps.get(user_id, 0.0)
            )

    def _search_from(self, *, depth, slot_sum, exact_cost_sum, rate_sum, used_pa_ids, selected_rows):
        """Search the remaining user suffix with exact slot and power bounds."""

        self.search_stats["nodes_visited"] += 1
        if depth == len(self.user_order):
            self.search_stats["complete_feasible_schedules"] += 1
            self._accept_schedule(self._evaluate_schedule(selected_rows))
            return

        user_id = self.user_order[depth]
        remaining_slots_lb = self.suffix_min_slots[depth + 1]
        for row in self.ranked_user_rows[user_id]:
            next_slot_sum = int(slot_sum + int(row["n_slots"]))
            if next_slot_sum > self.window_n_slots:
                self.search_stats["pruned_time_direct"] += 1
                continue

            next_exact_cost_sum = float(exact_cost_sum + self._incremental_schedule_cost(row, used_pa_ids))
            if next_exact_cost_sum > self.best_schedule_power + self.TOL:
                self.search_stats["pruned_power_direct"] += 1
                continue

            if next_slot_sum + remaining_slots_lb > self.window_n_slots:
                self.search_stats["pruned_time_bound"] += 1
                continue

            next_used_pa_ids = frozenset(set(used_pa_ids) | {int(row["pa_id"])})
            power_lb = float(
                next_exact_cost_sum + self._remaining_cost_lower_bound(depth + 1, next_used_pa_ids)
            )
            if power_lb > self.best_schedule_power + self.TOL:
                self.search_stats["pruned_power_bound"] += 1
                continue

            next_rate_sum = float(rate_sum + float(row["rate_avg_frame_bps"]))
            if np.isfinite(self.best_rank[0]) and abs(power_lb - self.best_rank[0]) <= self.TOL:
                slot_lb = int(next_slot_sum + remaining_slots_lb)
                if slot_lb > int(self.best_rank[1]):
                    self.search_stats["pruned_rank_bound"] += 1
                    continue

                if slot_lb == int(self.best_rank[1]):
                    max_rate_ub = float(next_rate_sum + self.suffix_max_rate_avg_frame_bps[depth + 1])
                    if max_rate_ub <= float(-self.best_rank[2]) + self.TOL:
                        self.search_stats["pruned_rank_bound"] += 1
                        continue

            selected_rows.append(row)
            self._search_from(
                depth=depth + 1,
                slot_sum=next_slot_sum,
                exact_cost_sum=next_exact_cost_sum,
                rate_sum=next_rate_sum,
                used_pa_ids=next_used_pa_ids,
                selected_rows=selected_rows,
            )
            selected_rows.pop()

    def _accept_schedule(self, schedule_result):
        """Accept a better schedule according to the exact objective and tie-breaks."""

        candidate_rank = (
            float(schedule_result["schedule_p_dc_total_avg_frame_w"]),
            int(schedule_result["slot_total"]),
            -float(schedule_result["total_rate_bps"]),
        )
        if candidate_rank >= self.best_rank:
            return

        self.best_rank = candidate_rank
        self.best_schedule_power = float(schedule_result["schedule_p_dc_total_avg_frame_w"])
        self.best_schedule = schedule_result
        self.search_stats["best_updates"] += 1

    def _seed_greedy_schedule(self):
        """Build one quick feasible incumbent by picking the first fit in ranked user order."""

        slot_sum = 0
        selected_rows = []
        for depth, user_id in enumerate(self.user_order):
            remaining_slots_lb = self.suffix_min_slots[depth + 1]
            selected_row = None
            for row in self.ranked_user_rows[user_id]:
                next_slot_sum = int(slot_sum + int(row["n_slots"]))
                if next_slot_sum + remaining_slots_lb <= self.window_n_slots:
                    selected_row = row
                    slot_sum = next_slot_sum
                    selected_rows.append(row)
                    break
            if selected_row is None:
                return None

        return self._evaluate_schedule(selected_rows)

    def _remaining_cost_lower_bound(self, depth, used_pa_ids):
        """Return the exact recursive lower bound on the remaining scheduler-side power cost."""

        if depth >= len(self.user_order):
            return 0.0

        key = (int(depth), tuple(sorted(int(pa_id) for pa_id in used_pa_ids)))
        cached = self.lower_bound_cache.get(key)
        if cached is not None:
            return cached

        user_id = self.user_order[depth]
        best_remaining_cost = np.inf
        for row in self.ranked_user_rows[user_id]:
            next_used_pa_ids = frozenset(set(used_pa_ids) | {int(row["pa_id"])})
            candidate_cost = self._incremental_schedule_cost(
                row,
                used_pa_ids,
            ) + self._remaining_cost_lower_bound(depth + 1, next_used_pa_ids)
            if candidate_cost < best_remaining_cost:
                best_remaining_cost = candidate_cost

        self.lower_bound_cache[key] = float(best_remaining_cost)
        return float(best_remaining_cost)

    def _evaluate_schedule(self, selected_rows):
        """Evaluate one complete schedule from already-valid user candidate rows."""

        inactive_state = PAState.IDLE if self.switch_policy == PASwitchPolicy.STANDBY else PAState.OFF
        slot_total = int(sum(int(row["n_slots"]) for row in selected_rows))
        total_rate_bps = float(sum(float(row["rate_avg_frame_bps"]) for row in selected_rows))
        schedule_p_out_total_avg_frame_w = float(sum(float(row["p_out_avg_frame_w"]) for row in selected_rows))

        schedule_p_dc_total_avg_frame_w = 0.0
        for pa_id in sorted({int(row["pa_id"]) for row in selected_rows}):
            pa_rows = [row for row in selected_rows if int(row["pa_id"]) == pa_id]
            pa_alpha_frame = float(
                np.clip(sum(float(row["n_slots"]) for row in pa_rows) / float(self.window_n_slots), 0.0, 1.0)
            )
            pa_p_dc_active_avg_frame_w = float(sum(float(row["p_dc_avg_frame_w"]) for row in pa_rows))
            inactive_bank_w = float(
                inactive_pa_bank_power(self.pa_catalog[pa_id], inactive_state, self.n_tx_chains)
            )
            pa_p_dc_inactive_avg_frame_w = float((1.0 - pa_alpha_frame) * inactive_bank_w)
            schedule_p_dc_total_avg_frame_w += float(pa_p_dc_active_avg_frame_w + pa_p_dc_inactive_avg_frame_w)

        return {
            "rows": sorted([dict(row) for row in selected_rows], key=lambda row: int(row["user_id"])),
            "slot_total": int(slot_total),
            "unused_slots": int(self.window_n_slots - slot_total),
            "total_rate_bps": float(total_rate_bps),
            "schedule_p_dc_total_avg_frame_w": float(schedule_p_dc_total_avg_frame_w),
            "schedule_p_out_total_avg_frame_w": float(schedule_p_out_total_avg_frame_w),
        }

    def _incremental_schedule_cost(self, row, used_pa_ids):
        """Return the scheduler-side DC cost added by choosing one user row."""

        active_cost = float(row["p_dc_avg_frame_w"])
        if self.switch_policy == PASwitchPolicy.HARD_OFF:
            return active_cost

        pa_id = int(row["pa_id"])
        row_alpha = float(row["n_slots"]) / float(self.window_n_slots)
        idle_bank_w = float(inactive_pa_bank_power(self.pa_catalog[pa_id], PAState.IDLE, self.n_tx_chains))
        incremental_cost = active_cost - row_alpha * idle_bank_w
        if pa_id not in used_pa_ids:
            incremental_cost += idle_bank_w
        return float(incremental_cost)

    def _ranked_row_key(self, row):
        """Return the deterministic ordering key used before the exact search."""

        return (
            self._incremental_schedule_cost(row, frozenset()),
            int(row["n_slots"]),
            float(row["p_dc_avg_frame_w"]),
            -float(row["rate_avg_frame_bps"]),
            int(row["pa_id"]),
            float(row["bandwidth_hz"]),
            int(row["n_prb"]),
            int(row["mcs"]),
        )

    def _build_result(self):
        """Build the minimal scheduler result payload from the current search state."""

        return MultiUserTdmaSchedulerResult(
            best_schedule=self.best_schedule,
            search_stats=self.search_stats,
        )


__all__ = [
    "run_joint_schedule_search",
]
