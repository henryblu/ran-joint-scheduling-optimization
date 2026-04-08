from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

SRC_PATH = (PROJECT_ROOT / "src").resolve()
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from candidate_table_generation import DISTANCE_BIN_GRID_M, build_distance_binned_candidate_table
from configs import SINGLE_USER_SEARCH_CONFIG, build_pa_catalog


LOOKUP_TIE_BREAK_COLUMNS = [
    "p_dc_active_w",
    "bandwidth_hz",
    "n_prb",
    "mcs",
    "layers",
    "pa_id",
]


def pick_example_scheduler_bin(
    scheduler_day_user_table: pd.DataFrame,
    *,
    target_user_count: int = 4,
) -> int:
    """Pick one small non-empty bin that is still interesting enough for lookup."""

    bin_counts = (
        scheduler_day_user_table.groupby("bin_index")["user_id"]
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
) -> SimpleNamespace:
    """Resolve the precomputed-table lookup for one scheduler-facing user table.

    Steps:
    1. Snap each user to the nearest precomputed distance bin.
    2. Filter that stored frontier by the user's requested active rate.
    3. Return the matched candidate spaces plus one compact per-user summary table.
    """

    if distance_binned_table is None:
        distance_binned_table = build_distance_binned_candidate_table()

    pa_label_map = {
        int(pa_id): str(pa.scenario_label)
        for pa_id, pa in enumerate(build_pa_catalog(SINGLE_USER_SEARCH_CONFIG.pa_data_csv))
    }
    user_candidate_spaces = {}
    summary_rows = []
    for user_row in user_table.sort_values("user_id").itertuples(index=False):
        assigned_distance_m = _assign_distance_bin(float(user_row.distance_m))
        distance_frontier = (
            distance_binned_table.frontiers_by_distance_m[int(assigned_distance_m)]
            .copy()
            .reset_index(drop=True)
        )
        candidate_space = _lookup_candidate_space_for_user(
            distance_frontier,
            required_rate_bps=float(user_row.required_rate_bps),
        )
        user_candidate_spaces[int(user_row.user_id)] = candidate_space
        summary_rows.append(
            _build_lookup_summary_row(
                user_row,
                assigned_distance_m=int(assigned_distance_m),
                candidate_space=candidate_space,
                pa_label_map=pa_label_map,
            )
        )

    return SimpleNamespace(
        distance_binned_table=distance_binned_table,
        assigned_user_table=pd.DataFrame(summary_rows),
        user_candidate_spaces=user_candidate_spaces,
        pa_label_map=pa_label_map,
    )


def plot_single_user_lookup(
    candidate_space: pd.DataFrame,
    *,
    user_label: str,
    required_rate_bps: float,
    pa_label_map: dict[int, str],
):
    """Plot the matched stored frontier for one user and highlight the best row."""

    if candidate_space.empty:
        raise ValueError("candidate_space must contain at least one feasible stored row.")

    best_row = candidate_space.sort_values(LOOKUP_TIE_BREAK_COLUMNS).iloc[0]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for pa_id, pa_rows in candidate_space.groupby("pa_id", sort=True):
        ax.scatter(
            pa_rows["rate_active_bps"] / 1e6,
            pa_rows["p_dc_active_w"],
            s=36,
            alpha=0.8,
            label=pa_label_map[int(pa_id)],
        )

    ax.axvline(float(required_rate_bps) / 1e6, color="black", linestyle="--", linewidth=1.2)
    ax.scatter(
        [float(best_row["rate_active_bps"]) / 1e6],
        [float(best_row["p_dc_active_w"])],
        color="red",
        s=70,
        label="Best stored row",
        zorder=4,
    )
    ax.set_xlabel("Active rate (Mbps)")
    ax.set_ylabel("Active PA DC power (W)")
    ax.set_title(f"Stored frontier rows for {user_label}")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    plt.tight_layout()
    return fig, ax


def _build_lookup_summary_row(
    user_row,
    *,
    assigned_distance_m: int,
    candidate_space: pd.DataFrame,
    pa_label_map: dict[int, str],
) -> dict[str, object]:
    """Build one compact summary row for the precomputed-table lookup."""

    summary_row = {
        "user_id": int(user_row.user_id),
        "distance_m": float(user_row.distance_m),
        "assigned_distance_bin_m": int(assigned_distance_m),
        "required_rate_mbps": float(user_row.required_rate_bps) / 1e6,
        "candidate_count": int(len(candidate_space)),
    }
    if candidate_space.empty:
        summary_row.update(
            {
                "best_pa": "No feasible stored row",
                "best_active_power_w": float("nan"),
                "best_n_prb": float("nan"),
                "best_layers": float("nan"),
                "best_mcs": float("nan"),
            }
        )
        return summary_row

    best_row = candidate_space.sort_values(LOOKUP_TIE_BREAK_COLUMNS).iloc[0]
    summary_row.update(
        {
            "best_pa": pa_label_map[int(best_row["pa_id"])],
            "best_active_power_w": float(best_row["p_dc_active_w"]),
            "best_n_prb": int(best_row["n_prb"]),
            "best_layers": int(best_row["layers"]),
            "best_mcs": int(best_row["mcs"]),
        }
    )
    return summary_row


def _lookup_candidate_space_for_user(
    distance_frontier: pd.DataFrame,
    *,
    required_rate_bps: float,
) -> pd.DataFrame:
    """Filter one stored distance frontier by the user's requested active rate."""

    return (
        distance_frontier.loc[
            distance_frontier["rate_active_bps"].astype(float).ge(float(required_rate_bps))
        ]
        .copy()
        .sort_values(LOOKUP_TIE_BREAK_COLUMNS)
        .reset_index(drop=True)
    )


def _assign_distance_bin(distance_m: float) -> int:
    """Snap one user distance onto the nearest configured precomputed bin."""

    return int(min(DISTANCE_BIN_GRID_M, key=lambda value: abs(float(value) - float(distance_m))))


__all__ = [
    "PROJECT_ROOT",
    "build_table_lookup_artifacts",
    "pick_example_scheduler_bin",
    "plot_single_user_lookup",
]
