from __future__ import annotations

from dataclasses import dataclass
import logging
from time import perf_counter

import numpy as np

from configs import inactive_pa_bank_power
from models import PAState, PASwitchPolicy

from .console_logging import emit_scheduler_console_log, format_metric
from .models import MultiUserTdmaSchedulerResult, PreparedJointScheduleProblem


SEARCH_PROGRESS_INTERVAL_S = 60.0
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _JointCandidateRow:
    """Typed internal TDMA row used by the exact joint search."""

    user_id: int
    pa_id: int
    n_prb: int
    layers: int
    mcs: int
    n_slots: int
    rate_avg_frame_bps: float
    p_dc_avg_frame_w: float
    p_out_avg_frame_w: float
    schedule_cost: float

    def to_public_row(self):
        """Return the trimmed row payload exposed on the public scheduler result."""

        return {
            "user_id": int(self.user_id),
            "pa_id": int(self.pa_id),
            "n_prb": int(self.n_prb),
            "layers": int(self.layers),
            "mcs": int(self.mcs),
            "n_slots": int(self.n_slots),
            "rate_avg_frame_bps": float(self.rate_avg_frame_bps),
            "p_dc_avg_frame_w": float(self.p_dc_avg_frame_w),
            "p_out_avg_frame_w": float(self.p_out_avg_frame_w),
        }


def run_joint_schedule_search(
    problem: PreparedJointScheduleProblem,
    *,
    switch_policy: PASwitchPolicy = PASwitchPolicy.STANDBY,
) -> MultiUserTdmaSchedulerResult:
    """Search the implicit Cartesian joint TDMA space for the optimal schedule."""

    return ExactJointScheduleSearch(problem, switch_policy=switch_policy).run()


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
        self.window_n_slots = int(problem.window_n_slots)
        self.pa_catalog = tuple(problem.pa_catalog)
        self.n_tx_chains = int(problem.n_tx_chains)
        self.switch_policy = (
            switch_policy
            if isinstance(switch_policy, PASwitchPolicy)
            else PASwitchPolicy(str(switch_policy))
        )
        self.user_candidate_spaces = {
            int(user_id): candidate_table.copy()
            for user_id, candidate_table in problem.user_candidate_spaces.items()
        }
        self.prepared_rows_total = int(
            sum(len(candidate_table) for candidate_table in self.user_candidate_spaces.values())
        )
        self.user_candidate_spaces = self._resolve_hard_off_candidate_spaces(self.user_candidate_spaces)
        self.pa_idle_costs = tuple(
            float(inactive_pa_bank_power(pa, PAState.IDLE, self.n_tx_chains))
            for pa in self.pa_catalog
        )
        self.standby_idle_baseline_w = float(sum(self.pa_idle_costs))

        worst_pa_signature = tuple(np.inf for _ in self.pa_catalog)
        self.best_schedule = None
        self.best_schedule_power = np.inf
        self.best_pa_signature = worst_pa_signature
        self.best_rank = (
            (worst_pa_signature, np.inf, np.inf, np.inf)
            if self.switch_policy == PASwitchPolicy.HARD_OFF
            else (np.inf, np.inf, np.inf)
        )

        self.ranked_user_rows = {}
        self.user_order = []
        self.suffix_min_slots = []
        self.suffix_min_schedule_cost = []
        self.suffix_max_rate_avg_frame_bps = []
        self.search_stats = {}
        self.search_started_at = 0.0
        self.last_progress_logged_at = 0.0

        self._prepare_search_state()

    def run(self) -> MultiUserTdmaSchedulerResult:
        """Run the exact joint scheduler search and return the best schedule payload."""

        if not self.ranked_user_rows or any(not rows for rows in self.ranked_user_rows.values()):
            return self._build_result()

        self.search_started_at = perf_counter()
        self.last_progress_logged_at = self.search_started_at
        self._log_search_start()

        # The greedy schedule gives the branch-and-bound search a quick incumbent
        # so the later exact bounds can start pruning immediately.
        greedy_schedule = self._seed_greedy_schedule()
        if greedy_schedule is not None:
            self.search_stats["complete_feasible_schedules"] += 1
            self._accept_schedule(greedy_schedule)

        self._search_from(
            depth=0,
            slot_sum=0,
            exact_cost_sum=self._initial_schedule_cost(),
            rate_sum=0.0,
            used_pa_ids=frozenset(),
            selected_rows=[],
        )
        return self._build_result()

    def _prepare_search_state(self):
        """Resolve ranked per-user rows and exact suffix bounds before the search."""

        min_slots = {}
        first_sort_cost = {}
        min_schedule_cost = {}
        max_rate_avg_frame_bps = {}

        for user_id, candidate_table in self.user_candidate_spaces.items():
            rows = [self._build_candidate_row(raw_row) for raw_row in candidate_table.to_dict("records")]
            if self.switch_policy == PASwitchPolicy.STANDBY:
                rows = self._exact_prune_standby_rows(rows)
                rows.sort(
                    key=lambda row: (
                        float(row.schedule_cost),
                        int(row.n_slots),
                        float(row.p_dc_avg_frame_w),
                        -float(row.rate_avg_frame_bps),
                        int(row.pa_id),
                        int(row.n_prb),
                        int(row.mcs),
                    )
                )
            else:
                rows.sort(
                    key=lambda row: (
                        self._used_pa_signature((row.pa_id,)),
                        float(row.p_dc_avg_frame_w),
                        int(row.n_slots),
                        -float(row.rate_avg_frame_bps),
                        int(row.pa_id),
                        int(row.n_prb),
                        int(row.mcs),
                    )
                )

            self.ranked_user_rows[int(user_id)] = rows
            if not rows:
                continue

            min_slots[int(user_id)] = min(row.n_slots for row in rows)
            first_sort_cost[int(user_id)] = float(rows[0].schedule_cost)
            min_schedule_cost[int(user_id)] = min(row.schedule_cost for row in rows)
            max_rate_avg_frame_bps[int(user_id)] = max(row.rate_avg_frame_bps for row in rows)

        # Search the most slot-constraining users first so infeasible branches
        # are rejected earlier in the recursion.
        self.user_order = sorted(
            self.ranked_user_rows,
            key=lambda user_id: (min_slots.get(user_id, 0), first_sort_cost.get(user_id, np.inf)),
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
        self.suffix_min_schedule_cost = [0.0] * (len(self.user_order) + 1)
        self.suffix_max_rate_avg_frame_bps = [0.0] * (len(self.user_order) + 1)
        for depth in range(len(self.user_order) - 1, -1, -1):
            user_id = self.user_order[depth]
            self.suffix_min_slots[depth] = self.suffix_min_slots[depth + 1] + min_slots.get(user_id, 0)
            self.suffix_min_schedule_cost[depth] = (
                self.suffix_min_schedule_cost[depth + 1] + min_schedule_cost.get(user_id, 0.0)
            )
            self.suffix_max_rate_avg_frame_bps[depth] = (
                self.suffix_max_rate_avg_frame_bps[depth + 1] + max_rate_avg_frame_bps.get(user_id, 0.0)
            )

    def _resolve_hard_off_candidate_spaces(self, user_candidate_spaces):
        """Keep only one PA family for hard-off schedules and reject mixed-only cases."""

        if self.switch_policy != PASwitchPolicy.HARD_OFF:
            return user_candidate_spaces

        for pa_id in sorted(range(len(self.pa_catalog)), key=lambda value: self._used_pa_signature((value,))):
            single_pa_spaces = {
                int(user_id): candidate_table.loc[
                    candidate_table["pa_id"].astype(int) == int(pa_id)
                ].reset_index(drop=True)
                for user_id, candidate_table in user_candidate_spaces.items()
            }
            if all(not candidate_table.empty for candidate_table in single_pa_spaces.values()) and sum(
                int(candidate_table["n_slots"].min()) for candidate_table in single_pa_spaces.values()
            ) <= self.window_n_slots:
                return single_pa_spaces

        return {
            int(user_id): candidate_table.iloc[0:0].copy().reset_index(drop=True)
            for user_id, candidate_table in user_candidate_spaces.items()
        }

    def _exact_prune_standby_rows(self, rows):
        """Exact-prune cross-PA rows only when standby makes the objective additive."""

        ranked_rows = sorted(
            rows,
            key=lambda row: (
                float(row.schedule_cost),
                int(row.n_slots),
                -float(row.rate_avg_frame_bps),
                float(row.p_dc_avg_frame_w),
                int(row.pa_id),
                int(row.n_prb),
                int(row.mcs),
            ),
        )
        kept_rows = []
        for row in ranked_rows:
            if any(
                float(kept_row.schedule_cost) <= float(row.schedule_cost) + self.TOL
                and int(kept_row.n_slots) <= int(row.n_slots)
                and float(kept_row.rate_avg_frame_bps) >= float(row.rate_avg_frame_bps) - self.TOL
                for kept_row in kept_rows
            ):
                continue
            kept_rows.append(row)
        return kept_rows

    def _log_search_start(self):
        """Log the pruned search size so long runs can be interpreted quickly."""

        if not LOGGER.isEnabledFor(logging.DEBUG):
            return

        search_rows_total = int(sum(len(rows) for rows in self.ranked_user_rows.values()))
        emit_scheduler_console_log(
            LOGGER,
            level=logging.DEBUG,
            stage="joint",
            event="search",
            fields=[
                ("policy", str(self.switch_policy.value)),
                ("users", str(int(len(self.user_order)))),
                ("rows_prepared", str(int(self.prepared_rows_total))),
                ("rows_search", str(int(search_rows_total))),
                ("slot_lb", str(int(self.suffix_min_slots[0]))),
                ("slot_slack", str(int(self.window_n_slots - self.suffix_min_slots[0]))),
            ],
        )

    def _maybe_log_search_progress(self):
        """Emit a sparse heartbeat while the exact search is still running."""

        if not LOGGER.isEnabledFor(logging.DEBUG) or self.search_started_at <= 0.0:
            return

        now = perf_counter()
        if now - self.last_progress_logged_at < float(SEARCH_PROGRESS_INTERVAL_S):
            return

        self.last_progress_logged_at = now
        emit_scheduler_console_log(
            LOGGER,
            level=logging.DEBUG,
            stage="joint",
            event="progress",
            fields=[
                ("elapsed_s", format_metric(now - self.search_started_at, digits=1)),
                ("nodes", str(int(self.search_stats["nodes_visited"]))),
                (
                    "best_power_w",
                    "na" if not np.isfinite(self.best_schedule_power) else f"{float(self.best_schedule_power):.2f}",
                ),
            ],
        )

    def _search_from(self, *, depth, slot_sum, exact_cost_sum, rate_sum, used_pa_ids, selected_rows):
        """Search the remaining user suffix with exact slot and power bounds."""

        self.search_stats["nodes_visited"] += 1
        self._maybe_log_search_progress()
        if depth == len(self.user_order):
            self.search_stats["complete_feasible_schedules"] += 1
            self._accept_schedule(self._evaluate_schedule(selected_rows))
            return

        user_id = self.user_order[depth]
        remaining_slots_lb = self.suffix_min_slots[depth + 1]
        for row in self.ranked_user_rows[user_id]:
            next_slot_sum = int(slot_sum + row.n_slots)
            if next_slot_sum > self.window_n_slots:
                self.search_stats["pruned_time_direct"] += 1
                continue

            next_used_pa_ids = used_pa_ids | {row.pa_id}
            next_pa_signature = self._used_pa_signature(next_used_pa_ids)
            if (
                self.switch_policy == PASwitchPolicy.HARD_OFF
                and self.best_schedule is not None
                and next_pa_signature > self.best_pa_signature
            ):
                continue

            next_exact_cost_sum = float(exact_cost_sum + row.schedule_cost)
            power_pruning_enabled = self.best_schedule is not None and (
                self.switch_policy != PASwitchPolicy.HARD_OFF
                or next_pa_signature == self.best_pa_signature
            )
            if power_pruning_enabled and next_exact_cost_sum > self.best_schedule_power + self.TOL:
                self.search_stats["pruned_power_direct"] += 1
                continue

            # Even before picking concrete rows for the rest of the users, the
            # suffix minimum-slot bound can rule out the whole branch.
            if next_slot_sum + remaining_slots_lb > self.window_n_slots:
                self.search_stats["pruned_time_bound"] += 1
                continue

            power_lb = None
            if power_pruning_enabled:
                power_lb = float(next_exact_cost_sum + self.suffix_min_schedule_cost[depth + 1])
                if power_lb > self.best_schedule_power + self.TOL:
                    self.search_stats["pruned_power_bound"] += 1
                    continue

            next_rate_sum = float(rate_sum + row.rate_avg_frame_bps)
            if (
                power_lb is not None
                and np.isfinite(self.best_schedule_power)
                and abs(power_lb - self.best_schedule_power) <= self.TOL
            ):
                # When objective lower bounds tie, prune branches that cannot
                # beat the incumbent on slot count or delivered rate.
                slot_lb = int(next_slot_sum + remaining_slots_lb)
                if slot_lb > int(self.best_rank[-2]):
                    self.search_stats["pruned_rank_bound"] += 1
                    continue

                if slot_lb == int(self.best_rank[-2]):
                    max_rate_ub = float(next_rate_sum + self.suffix_max_rate_avg_frame_bps[depth + 1])
                    if max_rate_ub <= float(-self.best_rank[-1]) + self.TOL:
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

        if self.switch_policy == PASwitchPolicy.HARD_OFF:
            candidate_rank = (
                self._used_pa_signature(schedule_result["used_pa_ids"]),
                float(schedule_result["schedule_p_dc_total_avg_frame_w"]),
                int(schedule_result["slot_total"]),
                -float(schedule_result["total_rate_bps"]),
            )
        else:
            candidate_rank = (
                float(schedule_result["schedule_p_dc_total_avg_frame_w"]),
                int(schedule_result["slot_total"]),
                -float(schedule_result["total_rate_bps"]),
            )

        if candidate_rank >= self.best_rank:
            return

        self.best_rank = candidate_rank
        self.best_schedule_power = float(schedule_result["schedule_p_dc_total_avg_frame_w"])
        if self.switch_policy == PASwitchPolicy.HARD_OFF:
            self.best_pa_signature = self._used_pa_signature(schedule_result["used_pa_ids"])
        self.best_schedule = {
            "rows": schedule_result["rows"],
            "slot_total": int(schedule_result["slot_total"]),
            "unused_slots": int(schedule_result["unused_slots"]),
            "total_rate_bps": float(schedule_result["total_rate_bps"]),
            "schedule_p_dc_total_avg_frame_w": float(schedule_result["schedule_p_dc_total_avg_frame_w"]),
            "schedule_p_out_total_avg_frame_w": float(schedule_result["schedule_p_out_total_avg_frame_w"]),
        }
        self.search_stats["best_updates"] += 1

    def _seed_greedy_schedule(self):
        """Build one quick feasible incumbent by picking the first fit in ranked user order."""

        slot_sum = 0
        selected_rows = []
        for depth, user_id in enumerate(self.user_order):
            remaining_slots_lb = self.suffix_min_slots[depth + 1]
            selected_row = None
            for row in self.ranked_user_rows[user_id]:
                next_slot_sum = int(slot_sum + row.n_slots)
                if next_slot_sum + remaining_slots_lb <= self.window_n_slots:
                    selected_row = row
                    slot_sum = next_slot_sum
                    selected_rows.append(row)
                    break
            if selected_row is None:
                return None

        return self._evaluate_schedule(selected_rows)

    def _evaluate_schedule(self, selected_rows):
        """Evaluate one complete schedule from already-valid user candidate rows."""

        used_pa_ids = sorted({row.pa_id for row in selected_rows})
        slot_total = int(sum(row.n_slots for row in selected_rows))
        total_rate_bps = float(sum(row.rate_avg_frame_bps for row in selected_rows))
        schedule_p_out_total_avg_frame_w = float(sum(row.p_out_avg_frame_w for row in selected_rows))
        schedule_p_dc_total_avg_frame_w = float(
            self._initial_schedule_cost() + sum(row.schedule_cost for row in selected_rows)
        )

        return {
            "rows": [row.to_public_row() for row in sorted(selected_rows, key=lambda row: row.user_id)],
            "used_pa_ids": list(used_pa_ids),
            "slot_total": int(slot_total),
            "unused_slots": int(self.window_n_slots - slot_total),
            "total_rate_bps": float(total_rate_bps),
            "schedule_p_dc_total_avg_frame_w": float(schedule_p_dc_total_avg_frame_w),
            "schedule_p_out_total_avg_frame_w": float(schedule_p_out_total_avg_frame_w),
        }

    def _build_candidate_row(self, raw_row):
        """Normalize one candidate-table record into the typed search row shape."""

        pa_id = int(raw_row["pa_id"])
        n_slots = int(raw_row["n_slots"])
        p_dc_avg_frame_w = float(raw_row["p_dc_avg_frame_w"])
        if self.switch_policy == PASwitchPolicy.STANDBY:
            row_alpha = float(n_slots) / float(self.window_n_slots)
            schedule_cost = float(p_dc_avg_frame_w - row_alpha * self.pa_idle_costs[pa_id])
        else:
            schedule_cost = float(p_dc_avg_frame_w)

        return _JointCandidateRow(
            user_id=int(raw_row["user_id"]),
            pa_id=pa_id,
            n_prb=int(raw_row["n_prb"]),
            layers=int(raw_row["layers"]),
            mcs=int(raw_row["mcs"]),
            n_slots=n_slots,
            rate_avg_frame_bps=float(raw_row["rate_avg_frame_bps"]),
            p_dc_avg_frame_w=p_dc_avg_frame_w,
            p_out_avg_frame_w=float(raw_row["p_out_avg_frame_w"]),
            schedule_cost=schedule_cost,
        )

    def _used_pa_signature(self, used_pa_ids):
        """Encode the activated PA banks from highest to lowest power tier."""

        used_pa_set = {int(pa_id) for pa_id in used_pa_ids}
        return tuple(int(pa_id in used_pa_set) for pa_id in range(len(self.pa_catalog)))

    def _initial_schedule_cost(self):
        """Return the constant schedule-side baseline before any row-specific adjustments."""

        if self.switch_policy == PASwitchPolicy.STANDBY:
            return float(self.standby_idle_baseline_w)
        return 0.0

    def _build_result(self):
        """Build the minimal scheduler result payload from the current search state."""

        return MultiUserTdmaSchedulerResult(
            best_schedule=self.best_schedule,
            search_stats=self.search_stats,
        )


__all__ = [
    "run_joint_schedule_search",
]
