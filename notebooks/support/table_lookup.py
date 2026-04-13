from __future__ import annotations

"""Lean lookup-stage notebook support built on top of the production lookup layer."""

from dataclasses import dataclass

import pandas as pd

from candidate_table_generation import load_distance_binned_candidate_table
from configs import SINGLE_USER_SEARCH_CONFIG, build_pa_catalog
from models import UserRequest
from single_user_lookup.api import build_batch_user_parameter_space
from single_user_lookup.lookup import lookup_user_parameter_space


LOOKUP_TIE_BREAK_COLUMNS = [
    "p_dc_active_w",
    "n_prb",
    "mcs",
    "layers",
    "pa_id",
]


@dataclass(frozen=True)
class LookupArtifacts:
    """Notebook-facing view of the production lookup stage for one active bin."""

    assigned_user_table: pd.DataFrame
    full_frontiers_by_user: dict[int, pd.DataFrame]
    user_candidate_spaces: dict[int, pd.DataFrame]
    pa_label_map: dict[int, str]


def load_cached_distance_binned_table():
    """Load the persisted candidate-table artifact without triggering a local rebuild."""

    try:
        return load_distance_binned_candidate_table()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Notebook walkthroughs require the cached distance-binned candidate table at "
            "'data/distance_binned_candidate_table.json'. This notebook will not build it "
            "locally; copy the cached artifact into data/ or generate it once on the server."
        ) from exc


def pick_example_scheduler_bin(
    scheduler_day_user_table: pd.DataFrame,
    *,
    target_user_count: int = 4,
) -> int:
    """Pick one non-empty scheduler bin close to the requested teaching size."""

    bin_counts = (
        scheduler_day_user_table.groupby("bin_index", dropna=False)["user_id"]
        .nunique()
        .rename("user_count")
        .reset_index()
    )
    non_empty_bins = bin_counts.loc[bin_counts["user_count"].gt(0)].copy()
    if non_empty_bins.empty:
        raise ValueError("The scheduler day user table does not contain any active bins.")

    non_empty_bins["distance_to_target"] = (
        non_empty_bins["user_count"].astype(int) - int(target_user_count)
    ).abs()
    chosen_row = non_empty_bins.sort_values(
        ["distance_to_target", "user_count", "bin_index"],
        ascending=[True, True, True],
    ).iloc[0]
    return int(chosen_row["bin_index"])


def build_table_lookup_artifacts(
    user_table: pd.DataFrame,
    *,
    distance_binned_table=None,
) -> LookupArtifacts:
    """Resolve the lookup-stage notebook artifacts for one scheduler-facing bin.

    The production lookup layer still owns the feasible-space filtering. This
    wrapper only adds the teaching tables that show which stored frontier each
    user landed on and how many rows survived the rate filter.
    """

    resolved_distance_table = (
        load_cached_distance_binned_table()
        if distance_binned_table is None
        else distance_binned_table
    )
    pa_label_map = {
        int(pa_id): str(pa.scenario_label)
        for pa_id, pa in enumerate(build_pa_catalog(SINGLE_USER_SEARCH_CONFIG.pa_data_csv))
    }
    summary_rows = []
    full_frontiers_by_user: dict[int, pd.DataFrame] = {}
    user_candidate_spaces: dict[int, pd.DataFrame] = {}

    for user_row in user_table.sort_values("user_id").itertuples(index=False):
        user_id = int(user_row.user_id)
        assigned_distance_m = _assign_distance_bin(
            float(user_row.distance_m),
            distance_bins=resolved_distance_table.frontiers_by_distance_m.keys(),
        )
        full_frontier = (
            resolved_distance_table.frontiers_by_distance_m[int(assigned_distance_m)]
            .copy()
            .reset_index(drop=True)
        )
        candidate_space = lookup_user_parameter_space(
            UserRequest(
                user_id=int(user_id),
                distance_m=float(user_row.distance_m),
                required_rate_bps=float(user_row.required_rate_bps),
            )
        )
        full_frontiers_by_user[int(user_id)] = full_frontier
        user_candidate_spaces[int(user_id)] = candidate_space.copy().reset_index(drop=True)
        summary_rows.append(
            _build_lookup_summary_row(
                user_row=user_row,
                assigned_distance_m=int(assigned_distance_m),
                distance_frontier=full_frontier,
                candidate_space=candidate_space,
                pa_label_map=pa_label_map,
            )
        )

    return LookupArtifacts(
        assigned_user_table=pd.DataFrame(summary_rows),
        full_frontiers_by_user=full_frontiers_by_user,
        user_candidate_spaces=user_candidate_spaces,
        pa_label_map=pa_label_map,
    )


def build_cached_batch_user_parameter_space(
    user_table: pd.DataFrame,
    *,
    lookup_artifacts: LookupArtifacts | None = None,
    distance_binned_table=None,
):
    """Build the production batch artifact for one scheduler-facing bin."""

    if lookup_artifacts is None:
        build_table_lookup_artifacts(
            user_table,
            distance_binned_table=distance_binned_table,
        )
    return build_batch_user_parameter_space(user_table)


def _build_lookup_summary_row(
    *,
    user_row,
    assigned_distance_m: int,
    distance_frontier: pd.DataFrame,
    candidate_space: pd.DataFrame,
    pa_label_map: dict[int, str],
) -> dict[str, object]:
    summary_row = {
        "user_id": int(user_row.user_id),
        "distance_m": float(user_row.distance_m),
        "assigned_distance_bin_m": int(assigned_distance_m),
        "required_rate_bps": float(user_row.required_rate_bps),
        "frontier_row_count": int(len(distance_frontier)),
        "candidate_count": int(len(candidate_space)),
    }
    if candidate_space.empty:
        return {
            **summary_row,
            "best_pa": "No feasible stored row",
            "best_active_power_w": float("nan"),
            "best_n_prb": float("nan"),
            "best_layers": float("nan"),
            "best_mcs": float("nan"),
        }

    best_row = candidate_space.sort_values(LOOKUP_TIE_BREAK_COLUMNS).iloc[0]
    return {
        **summary_row,
        "best_pa": pa_label_map[int(best_row["pa_id"])],
        "best_active_power_w": float(best_row["p_dc_active_w"]),
        "best_n_prb": int(best_row["n_prb"]),
        "best_layers": int(best_row["layers"]),
        "best_mcs": int(best_row["mcs"]),
    }


def _assign_distance_bin(
    distance_m: float,
    *,
    distance_bins,
) -> int:
    resolved_bins = sorted(int(distance_bin) for distance_bin in distance_bins)
    for distance_bin in resolved_bins:
        if float(distance_bin) >= float(distance_m):
            return int(distance_bin)
    return int(resolved_bins[-1])


__all__ = [
    "LookupArtifacts",
    "build_cached_batch_user_parameter_space",
    "build_table_lookup_artifacts",
    "load_cached_distance_binned_table",
    "pick_example_scheduler_bin",
]
