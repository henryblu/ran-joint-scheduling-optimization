from __future__ import annotations

"""Result-table contract checks for scheduler-comparison analysis."""

import pandas as pd

from .quality_tables import check_row, pair_examples, point_examples
from .row_states import bool_like, certified_skipped_row_mask


NUMERIC_NONNEGATIVE_COLUMNS = (
    "single_user_elapsed_s",
    "joint_elapsed_s",
    "total_elapsed_s",
    "frame_energy_j",
    "average_frame_dc_power_w",
    "delivered_rate_sum_bps",
)

SCENARIO_COLUMNS = (
    "active_user_count",
    "load_factor",
    "distance_min_m",
    "distance_max_m",
    "distance_model",
    "mean_distance_m",
    "sigma_distance_m",
    "reference_backlog_bits",
    "frame_duration_s",
)

ENERGY_TOL_J = 1e-9


def result_quality_check_rows(results: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    rows.append(check_row("status_populated", results["status"].notna().all(), int(results["status"].isna().sum()), ""))

    feasible = results["feasible"].map(bool_like)
    solved_not_feasible = results.loc[results["status"].astype(str).eq("solved") & ~feasible]
    rows.append(check_row("solved_rows_are_feasible", solved_not_feasible.empty, len(solved_not_feasible), point_examples(solved_not_feasible)))

    skipped = results.loc[certified_skipped_row_mask(results)]
    skipped_missing_source = skipped.loc[
        skipped["skip_reason"].fillna("").astype(str).eq("")
        | skipped["source_point_id"].fillna("").astype(str).eq("")
        | skipped["source_bound"].fillna("").astype(str).eq("")
    ]
    rows.append(
        check_row(
            "certified_skips_have_source_metadata",
            skipped_missing_source.empty,
            len(skipped_missing_source),
            point_examples(skipped_missing_source),
        )
    )
    rows.extend(certified_skip_source_check_rows(results, skipped))
    rows.extend(monotonicity_claim_check_rows(results))
    rows.extend(policy_contract_check_rows(results))
    rows.extend(scheduler_contract_check_rows(results))

    for column in NUMERIC_NONNEGATIVE_COLUMNS:
        values = pd.to_numeric(results[column], errors="coerce")
        invalid = results.loc[values.notna() & (values < 0)]
        rows.append(check_row(f"{column}_nonnegative", invalid.empty, len(invalid), point_examples(invalid)))

    expected_demand = (
        pd.to_numeric(results["active_user_count"], errors="coerce")
        * pd.to_numeric(results["load_factor"], errors="coerce")
        * pd.to_numeric(results["reference_backlog_bits"], errors="coerce")
    ).round()
    actual_demand = pd.to_numeric(results["total_demand_bits"], errors="coerce")
    demand_mismatch = results.loc[(expected_demand - actual_demand).abs() > 1]
    rows.append(check_row("total_demand_matches_point_axes", demand_mismatch.empty, len(demand_mismatch), point_examples(demand_mismatch)))

    expected_rate = actual_demand / pd.to_numeric(results["frame_duration_s"], errors="coerce")
    actual_rate = pd.to_numeric(results["requested_rate_sum_bps"], errors="coerce")
    rate_mismatch = results.loc[(expected_rate - actual_rate).abs() > 1e-6]
    rows.append(check_row("requested_rate_matches_demand", rate_mismatch.empty, len(rate_mismatch), point_examples(rate_mismatch)))
    return rows


def certified_skip_source_check_rows(results: pd.DataFrame, skipped: pd.DataFrame) -> list[dict[str, object]]:
    row_lookup = {
        (
            int(row["source_chunk_index"]),
            str(row["point_id"]),
        ): int(row["row_index_in_source_file"])
        for _, row in results.iterrows()
        if "source_chunk_index" in row and "row_index_in_source_file" in row
    }
    missing_or_late_source_rows = []
    for _, row in skipped.iterrows():
        source_point_id = str(row.get("source_point_id", ""))
        if source_point_id == str(row.get("point_id", "")):
            continue

        key = (int(row["source_chunk_index"]), source_point_id)
        source_row_index = row_lookup.get(key)
        if source_row_index is None or int(source_row_index) >= int(row["row_index_in_source_file"]):
            missing_or_late_source_rows.append(row)

    missing_or_late_source = pd.DataFrame(missing_or_late_source_rows)
    ofdma_milp_inherited = skipped.loc[
        skipped["scheduler_mode"].astype(str).eq("ofdma_milp_single_snapshot")
        & skipped["skip_reason"].fillna("").astype(str).eq("inherited_bound_certified_infeasible")
    ]
    return [
        check_row(
            "certified_skip_sources_exist_earlier_in_chunk",
            missing_or_late_source.empty,
            len(missing_or_late_source),
            point_examples(missing_or_late_source),
        ),
        check_row(
            "ofdma_milp_rows_do_not_use_inherited_load_skips",
            ofdma_milp_inherited.empty,
            len(ofdma_milp_inherited),
            point_examples(ofdma_milp_inherited),
        ),
    ]


def monotonicity_claim_check_rows(results: pd.DataFrame) -> list[dict[str, object]]:
    monotone_results = results.loc[~results["scheduler_mode"].astype(str).eq("ofdma_milp_single_snapshot")]
    unexpected_rows = []
    for _, chain in monotone_results.sort_values("load_factor").groupby(exact_chain_columns()):
        seen_non_solved = False
        for _, row in chain.iterrows():
            status = str(row.get("status", ""))
            if status == "solved" and seen_non_solved:
                unexpected_rows.append(row)
                break
            if status != "solved":
                seen_non_solved = True

    unexpected = pd.DataFrame(unexpected_rows)
    return [
        check_row(
            "monotonicity_claimed_chains_have_no_solved_after_non_solved",
            unexpected.empty,
            len(unexpected),
            point_examples(unexpected),
        )
    ]


def policy_contract_check_rows(results: pd.DataFrame) -> list[dict[str, object]]:
    comparable = comparable_result_frame(results)
    baseline = comparable.loc[
        comparable["switch_policy"].astype(str).eq("baseline_8w_only"),
        ["scheduler_mode", *SCENARIO_COLUMNS, "point_id", "solved_feasible"],
    ].rename(columns={"point_id": "baseline_point_id", "solved_feasible": "baseline_solved"})

    mismatches = []
    for policy in ("hard_off", "dual_switchable"):
        policy_rows = comparable.loc[
            comparable["switch_policy"].astype(str).eq(policy),
            ["scheduler_mode", *SCENARIO_COLUMNS, "point_id", "solved_feasible"],
        ].rename(columns={"point_id": "policy_point_id", "solved_feasible": "policy_solved"})
        pairs = pd.merge(baseline, policy_rows, on=["scheduler_mode", *SCENARIO_COLUMNS], how="inner")
        mismatch = pairs.loc[pairs["baseline_solved"] != pairs["policy_solved"]].copy()
        mismatch["policy"] = policy
        mismatches.append(mismatch)

    unexpected = pd.concat(mismatches, ignore_index=True) if mismatches else pd.DataFrame()
    return [
        check_row(
            "pa_policy_feasibility_matches_baseline_8w",
            unexpected.empty,
            len(unexpected),
            pair_examples(unexpected, "baseline_point_id", "policy_point_id"),
        )
    ]


def scheduler_contract_check_rows(results: pd.DataFrame) -> list[dict[str, object]]:
    comparable = comparable_result_frame(results)
    k1 = scheduler_slice(
        comparable,
        "tdma",
        point_column="k1_point_id",
        solved_column="k1_solved",
        energy_column="k1_energy_j",
    )
    k2 = scheduler_slice(
        comparable,
        "ofdma_milp_single_snapshot",
        point_column="k2_point_id",
        solved_column="k2_solved",
        energy_column="k2_energy_j",
    )
    rr = scheduler_slice(
        comparable,
        "ofdma_round_robin",
        point_column="rr_point_id",
        solved_column="rr_solved",
        energy_column="rr_energy_j",
    )

    k1_k2 = pd.merge(k1, k2, on=["switch_policy", *SCENARIO_COLUMNS], how="inner")
    k2_missing_k1_feasible = k1_k2.loc[k1_k2["k1_solved"] & ~k1_k2["k2_solved"]]
    k2_energy_worse = k1_k2.loc[
        k1_k2["k1_solved"]
        & k1_k2["k2_solved"]
        & (k1_k2["k2_energy_j"] > k1_k2["k1_energy_j"] + ENERGY_TOL_J)
    ]

    optimized = pd.concat(
        [
            k1.rename(columns={"k1_point_id": "opt_point_id", "k1_solved": "opt_solved", "k1_energy_j": "opt_energy_j"}),
            k2.rename(columns={"k2_point_id": "opt_point_id", "k2_solved": "opt_solved", "k2_energy_j": "opt_energy_j"}),
        ],
        ignore_index=True,
    )
    rr_pairs = pd.merge(rr, optimized, on=["switch_policy", *SCENARIO_COLUMNS], how="inner")
    rr_missing_optimized_feasible = rr_pairs.loc[rr_pairs["opt_solved"] & ~rr_pairs["rr_solved"]]
    return [
        check_row(
            "k2_feasibility_covers_k1",
            k2_missing_k1_feasible.empty,
            len(k2_missing_k1_feasible),
            pair_examples(k2_missing_k1_feasible, "k1_point_id", "k2_point_id"),
        ),
        check_row(
            "k2_energy_no_worse_than_k1",
            k2_energy_worse.empty,
            len(k2_energy_worse),
            pair_examples(k2_energy_worse, "k1_point_id", "k2_point_id"),
        ),
        check_row(
            "round_robin_feasibility_covers_optimized",
            rr_missing_optimized_feasible.empty,
            len(rr_missing_optimized_feasible),
            pair_examples(rr_missing_optimized_feasible, "rr_point_id", "opt_point_id"),
        ),
    ]


def comparable_result_frame(results: pd.DataFrame) -> pd.DataFrame:
    frame = results.copy()
    frame["solved_feasible"] = frame["status"].astype(str).eq("solved") & frame["feasible"].map(bool_like)
    frame["energy_j"] = pd.to_numeric(frame["frame_energy_j"], errors="coerce")
    return frame


def scheduler_slice(
    frame: pd.DataFrame,
    scheduler_mode: str,
    *,
    point_column: str,
    solved_column: str,
    energy_column: str,
) -> pd.DataFrame:
    return frame.loc[
        frame["scheduler_mode"].astype(str).eq(scheduler_mode),
        ["switch_policy", *SCENARIO_COLUMNS, "point_id", "solved_feasible", "energy_j"],
    ].rename(
        columns={
            "point_id": point_column,
            "solved_feasible": solved_column,
            "energy_j": energy_column,
        }
    )


def exact_chain_columns() -> list[str]:
    return [
        "scheduler_mode",
        "switch_policy",
        "active_user_count",
        "distance_model",
        "mean_distance_m",
        "sigma_distance_m",
    ]

