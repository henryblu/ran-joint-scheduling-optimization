import numpy as np
import pandas as pd

from .api import (
    build_single_user_scenario,
    search_candidate_spaces,
    summarize_single_user_scenario,
)
from .models import SingleUserStudyResult


PUBLIC_COLUMNS = [
    "distance_m",
    "path_loss_db",
    "pa_id",
    "pa_name",
    "rate_ach_bps",
    "p_dc_avg_total_w",
    "layers",
    "mcs",
    "n_prb",
    "n_slots_on",
    "alpha_f",
    "bandwidth_hz",
    "n_active_tx",
    "p_out_total_w",
    "ps_total_w",
    "gamma_req_lin",
]

FRONTIER_COLUMNS = [
    "distance_m",
    "path_loss_db",
    "pa_id",
    "pa_name",
    "rate_target_bps",
    "rate_ach_bps",
    "p_dc_avg_total_w",
    "layers",
    "mcs",
    "n_prb",
    "n_slots_on",
    "alpha_f",
    "bandwidth_hz",
    "n_active_tx",
    "p_out_total_w",
    "ps_total_w",
    "gamma_req_lin",
]

EXPLANATION_COLUMNS = PUBLIC_COLUMNS + ["rate_target_bps", "explanation_role"]
TIE_BREAK_COLUMNS = [
    "p_dc_avg_total_w",
    "bandwidth_hz",
    "n_prb",
    "n_slots_on",
    "mcs",
    "layers",
    "n_active_tx",
]


def run_rate_study(distance_m, rate_targets_bps, *, outer_parallel=False, max_workers=None):
    """Run a fixed-distance frontier study across explicit rate targets.

    Steps:
    1. Build the strict scenario list and one shared candidate ledger per scenario.
    2. Reuse that ledger across all requested target-rate cuts.
    3. Assemble the canonical frontier and explanation tables.
    4. Return notebook-facing study tables and search-space summaries.
    """
    scenarios = [{"distance_m": float(distance_m)}]
    required_rate_targets_bps = np.asarray(rate_targets_bps, dtype=float)
    return _run_frontier_study(
        scenarios,
        required_rate_targets_bps=required_rate_targets_bps,
        outer_parallel=outer_parallel,
        max_workers=max_workers,
    )


def run_distance_study(distance_values_m, required_rate_bps, *, outer_parallel=False, max_workers=None):
    """Run a fixed-rate frontier study across explicit distance values."""
    scenarios = [{"distance_m": float(distance_m)} for distance_m in distance_values_m]
    required_rate_targets_bps = np.asarray([float(required_rate_bps)], dtype=float)
    return _run_frontier_study(
        scenarios,
        required_rate_targets_bps=required_rate_targets_bps,
        outer_parallel=outer_parallel,
        max_workers=max_workers,
    )


def _run_frontier_study(scenarios, required_rate_targets_bps, *, outer_parallel=False, max_workers=None):
    """Run the shared scenario loop for rate and distance frontier studies."""
    frontier_tables = []
    explanatory_tables = []
    candidate_rate_bps = float(np.min(required_rate_targets_bps))

    user_table = pd.DataFrame(
        [
            {
                "user_id": int(scenario_idx),
                "distance_m": float(scenario["distance_m"]),
                "required_rate_bps": float(candidate_rate_bps),
            }
            for scenario_idx, scenario in enumerate(scenarios)
        ]
    )
    candidate_spaces = search_candidate_spaces(
        user_table,
        outer_parallel=outer_parallel,
        max_workers=max_workers,
    )

    for scenario_idx, scenario in enumerate(scenarios):
        candidate_table = candidate_spaces[int(scenario_idx)]
        scenario_frontier, scenario_explanatory = _evaluate_scenario_frontier(
            candidate_table,
            required_rate_targets_bps=required_rate_targets_bps,
        )
        frontier_tables.append(scenario_frontier)
        explanatory_tables.append(scenario_explanatory)

    summary_scenario = build_single_user_scenario(
        distance_m=float(scenarios[0]["distance_m"]),
        required_rate_bps=candidate_rate_bps,
    )
    summary_views = summarize_single_user_scenario(
        summary_scenario,
        scenario_count=len(scenarios),
    )
    return SingleUserStudyResult(
        frontier_table=_concat_frontier_tables(frontier_tables),
        explanatory_configs=_concat_explanatory_tables(explanatory_tables),
        pa_characteristics=summary_views["pa_characteristics"],
        search_space_summary=summary_views["search_space_summary"],
    )


def _evaluate_scenario_frontier(candidate_table, required_rate_targets_bps):
    """Extract one frontier table and explanation table from a candidate ledger."""
    feasible_table = _filter_feasible_candidate_table(candidate_table)
    if feasible_table.empty:
        return (
            pd.DataFrame(columns=FRONTIER_COLUMNS),
            pd.DataFrame(columns=EXPLANATION_COLUMNS),
        )

    scenario_frontier_rows = []
    scenario_explanatory_rows = []
    for _, pa_configs in feasible_table.groupby("pa_id", sort=True):
        ranked_pa_configs = pa_configs.reset_index(drop=True)
        for required_rate_bps in required_rate_targets_bps:
            ranked = _rank_rate_feasible_rows(
                ranked_pa_configs,
                required_rate_bps=float(required_rate_bps),
            )
            if ranked.empty:
                continue
            scenario_frontier_rows.append(
                _build_frontier_row(ranked.iloc[0], required_rate_bps=float(required_rate_bps))
            )
            scenario_explanatory_rows.append(
                _build_explanation_rows(ranked, required_rate_bps=float(required_rate_bps))
            )

    scenario_frontier = pd.DataFrame(scenario_frontier_rows, columns=FRONTIER_COLUMNS)
    scenario_explanatory = _concat_scenario_explanations(scenario_explanatory_rows)
    return scenario_frontier, scenario_explanatory


def _filter_feasible_candidate_table(candidate_table):
    """Keep only feasible candidates before frontier ranking."""
    feasible_table = candidate_table[candidate_table["rate_ach_bps"].notna()].copy()
    if feasible_table.empty:
        return pd.DataFrame(columns=PUBLIC_COLUMNS)
    return feasible_table[PUBLIC_COLUMNS].copy()


def _rank_rate_feasible_rows(pa_configs, required_rate_bps):
    """Return the tie-break-ranked candidates that reach one target rate."""
    feasible_rows = pa_configs[pa_configs["rate_ach_bps"] >= float(required_rate_bps)].copy()
    if feasible_rows.empty:
        return feasible_rows
    return feasible_rows.sort_values(TIE_BREAK_COLUMNS, ascending=True).reset_index(drop=True)


def _build_frontier_row(winner, required_rate_bps):
    """Build one canonical frontier row from the ranked winning candidate."""
    frontier_row = {column: winner[column] for column in PUBLIC_COLUMNS}
    frontier_row["rate_target_bps"] = float(required_rate_bps)
    return {column: frontier_row[column] for column in FRONTIER_COLUMNS}


def _build_explanation_rows(ranked, required_rate_bps):
    """Build the winner and runner-up explanation rows for one target rate."""
    explanation_rows = ranked.head(3)[PUBLIC_COLUMNS].copy()
    explanation_rows["rate_target_bps"] = float(required_rate_bps)
    explanation_rows["explanation_role"] = ["winner"] + ["runner_up"] * max(
        len(explanation_rows) - 1,
        0,
    )
    return explanation_rows


def _concat_frontier_tables(frontier_tables):
    """Concatenate scenario frontier tables into the canonical study output."""
    if not frontier_tables:
        return pd.DataFrame(columns=FRONTIER_COLUMNS)
    return pd.concat(frontier_tables, ignore_index=True).sort_values(
        ["distance_m", "pa_id", "rate_target_bps"]
    ).reset_index(drop=True)


def _concat_explanatory_tables(explanatory_tables):
    """Concatenate non-empty explanation tables into the canonical study output."""
    explanatory_frames = [frame for frame in explanatory_tables if not frame.empty]
    if not explanatory_frames:
        return pd.DataFrame(columns=EXPLANATION_COLUMNS)
    return pd.concat(explanatory_frames, ignore_index=True).sort_values(
        ["distance_m", "pa_id", "rate_target_bps", "explanation_role", "p_dc_avg_total_w"]
    ).reset_index(drop=True)


def _concat_scenario_explanations(explanation_frames):
    """Return one explanation table for a single scenario frontier sweep."""
    if not explanation_frames:
        return pd.DataFrame(columns=EXPLANATION_COLUMNS)
    return pd.concat(explanation_frames, ignore_index=True)
