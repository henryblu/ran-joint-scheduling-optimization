from __future__ import annotations

"""Minimal candidate-space utilities used by Notebook 3."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd


_DOMINANCE_TOL = 1e-12


def export_doc_figure(fig, filename: str, doc_img_dir: Path | None = None) -> Path | None:
    """Save one figure to the repository image directory when requested."""

    if doc_img_dir is None:
        return None

    output_path = Path(doc_img_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return output_path


def prepare_feasible_plot_table(feasible_table: pd.DataFrame) -> pd.DataFrame:
    """Return the plot-ready feasible table with stable ids and frame resource totals."""

    plot_table = feasible_table.copy()
    if "candidate_id" not in plot_table.columns:
        plot_table = plot_table.reset_index(drop=False).rename(columns={"index": "candidate_id"})
    plot_table["total_prb_slots"] = (
        plot_table["n_prb"].astype(int) * plot_table["n_slots_on"].astype(int)
    )
    plot_table["rate_mbps"] = plot_table["rate_ach_bps"].astype(float) / 1e6
    return plot_table


def annotate_same_pa_dominance(
    candidate_table: pd.DataFrame,
    *,
    resource_column: str,
    power_column: str,
    rate_column: str,
    pa_column: str = "pa_id",
) -> pd.DataFrame:
    """Mark rows as kept or dominated under one same-PA strict-dominance rule."""

    if candidate_table.empty:
        return candidate_table.assign(
            row_id=pd.Series(dtype=int),
            pruning_role=pd.Series(dtype=str),
            dominator_row_id=pd.Series(dtype="Int64"),
        )

    base_table = candidate_table.copy()
    if "row_id" not in base_table.columns:
        base_table = base_table.reset_index(drop=False).rename(columns={"index": "row_id"})

    sort_columns = [pa_column, resource_column, power_column, rate_column]
    ascending = [True, True, True, False]
    for optional_column in ("mcs", "layers", "n_prb", "n_slots_on"):
        if optional_column not in base_table.columns or optional_column in sort_columns:
            continue
        sort_columns.append(optional_column)
        ascending.append(True)

    row_roles: dict[int, str] = {}
    dominator_row_ids: dict[int, int | None] = {}
    kept_rows_by_pa: dict[int, list[dict[str, object]]] = {}
    ranked_rows = (
        base_table.sort_values(sort_columns, ascending=ascending)
        .reset_index(drop=True)
        .to_dict("records")
    )

    for row in ranked_rows:
        pa_id = int(row[pa_column])
        dominator = _find_same_pa_dominator(
            row,
            kept_rows=kept_rows_by_pa.setdefault(pa_id, []),
            resource_column=resource_column,
            power_column=power_column,
            rate_column=rate_column,
        )
        if dominator is None:
            kept_rows_by_pa[pa_id].append(row)
            row_roles[int(row["row_id"])] = "kept"
            dominator_row_ids[int(row["row_id"])] = None
            continue

        row_roles[int(row["row_id"])] = "dominated"
        dominator_row_ids[int(row["row_id"])] = int(dominator["row_id"])

    return base_table.assign(
        pruning_role=lambda table: table["row_id"].map(row_roles),
        dominator_row_id=lambda table: table["row_id"].map(dominator_row_ids),
    )


def select_slice_rows(
    slice_table: pd.DataFrame,
    *,
    power_column: str = "p_dc_avg_total_w",
) -> tuple[pd.Series, pd.Series]:
    """Pick one low-power row and one higher-MCS comparison row from the slice."""

    if slice_table.empty:
        raise ValueError("slice_table must contain at least one row.")
    if power_column not in slice_table.columns:
        raise ValueError(f"slice_table does not contain power_column={power_column!r}.")

    best_row = slice_table.sort_values(
        [power_column, "total_prb_slots", "mcs", "n_prb", "n_slots_on", "layers", "pa_id", "rate_mbps"]
    ).iloc[0]
    comparison_candidates = (
        slice_table.loc[slice_table.index != best_row.name]
        .loc[
            lambda table: table[power_column].gt(float(best_row[power_column]))
            & table["mcs"].gt(int(best_row["mcs"]))
        ]
        .copy()
    )
    if comparison_candidates.empty:
        raise ValueError("The throughput slice does not contain a higher-MCS comparison row.")

    prioritized_candidates = [
        comparison_candidates.loc[
            comparison_candidates["total_prb_slots"].lt(int(best_row["total_prb_slots"]))
            & comparison_candidates["n_slots_on"].lt(int(best_row["n_slots_on"]))
        ],
        comparison_candidates.loc[
            comparison_candidates["total_prb_slots"].lt(int(best_row["total_prb_slots"]))
        ],
        comparison_candidates.loc[
            comparison_candidates["n_slots_on"].lt(int(best_row["n_slots_on"]))
        ],
        comparison_candidates.loc[
            comparison_candidates["pa_id"].eq(int(best_row["pa_id"]))
        ],
        comparison_candidates,
    ]
    for candidate_rows in prioritized_candidates:
        if candidate_rows.empty:
            continue
        return best_row, candidate_rows.sort_values(
            ["total_prb_slots", "n_slots_on", power_column, "rate_mbps", "n_prb", "layers", "pa_id"],
            ascending=[True, True, True, False, True, True, True],
        ).iloc[0]

    raise ValueError("The throughput slice does not contain a valid comparison row.")


def select_dominated_example_pair(
    annotated_full_frame_table: pd.DataFrame,
    *,
    target_rate_bps: float | None = None,
    min_rate_mbps: float | None = None,
    max_rate_mbps: float | None = None,
    rate_column: str = "rate_mbps",
    power_column: str = "p_dc_active_w",
    resource_column: str = "total_prb_slots",
    lookup_table: pd.DataFrame | None = None,
    require_dominator_in_table: bool = False,
) -> tuple[pd.Series, pd.Series]:
    """Pick one dominated row and its dominating partner near the target rate."""

    resolved_lookup_table = annotated_full_frame_table if lookup_table is None else lookup_table
    dominated_rows = annotated_full_frame_table.loc[
        annotated_full_frame_table["pruning_role"].eq("dominated")
    ].copy()
    if dominated_rows.empty:
        raise ValueError("annotated_full_frame_table does not contain any dominated rows.")

    dominated_rows, target_rate_mbps = _resolve_target_rate_slice(
        dominated_rows,
        target_rate_bps=target_rate_bps,
        min_rate_mbps=min_rate_mbps,
        max_rate_mbps=max_rate_mbps,
        rate_column=rate_column,
    )
    if require_dominator_in_table:
        visible_row_ids = set(annotated_full_frame_table["row_id"].astype(int).tolist())
        dominated_rows = dominated_rows.loc[
            dominated_rows["dominator_row_id"].isin(visible_row_ids)
        ].copy()
        if dominated_rows.empty:
            raise ValueError("No dominated rows inside the requested slice have a visible dominating partner.")

    dominated_rows["distance_to_target_mbps"] = (
        dominated_rows[rate_column].astype(float) - float(target_rate_mbps)
    ).abs()
    dominated_row = dominated_rows.sort_values(
        ["distance_to_target_mbps", power_column, resource_column, "mcs", "layers", "pa_id"]
    ).iloc[0]
    dominator_row_id = dominated_row["dominator_row_id"]
    if pd.isna(dominator_row_id):
        raise ValueError("The selected dominated row does not have a dominating partner.")

    dominating_rows = resolved_lookup_table.loc[
        resolved_lookup_table["row_id"].eq(int(dominator_row_id))
    ]
    if dominating_rows.empty:
        raise ValueError("The dominating row could not be found in the annotated table.")
    return dominated_row, dominating_rows.iloc[0]


def _resolve_target_rate_slice(
    dominated_rows: pd.DataFrame,
    *,
    target_rate_bps: float | None,
    min_rate_mbps: float | None,
    max_rate_mbps: float | None,
    rate_column: str,
) -> tuple[pd.DataFrame, float]:
    if min_rate_mbps is not None and max_rate_mbps is not None:
        filtered_rows = dominated_rows.loc[
            dominated_rows[rate_column].between(
                float(min_rate_mbps),
                float(max_rate_mbps),
                inclusive="both",
            )
        ].copy()
        if filtered_rows.empty:
            raise ValueError("No dominated rows were found inside the requested pruning band.")
        return filtered_rows, 0.5 * (float(min_rate_mbps) + float(max_rate_mbps))

    if target_rate_bps is None:
        raise ValueError("target_rate_bps must be provided when min_rate_mbps and max_rate_mbps are omitted.")
    return dominated_rows, float(target_rate_bps) / 1e6


def _find_same_pa_dominator(
    row: dict[str, object],
    *,
    kept_rows: list[dict[str, object]],
    resource_column: str,
    power_column: str,
    rate_column: str,
) -> dict[str, object] | None:
    return next(
        (
            kept_row
            for kept_row in kept_rows
            if int(kept_row[resource_column]) <= int(row[resource_column])
            and float(kept_row[power_column]) <= float(row[power_column]) + _DOMINANCE_TOL
            and float(kept_row[rate_column]) >= float(row[rate_column]) - _DOMINANCE_TOL
            and (
                int(kept_row[resource_column]) < int(row[resource_column])
                or float(kept_row[power_column]) < float(row[power_column]) - _DOMINANCE_TOL
                or float(kept_row[rate_column]) > float(row[rate_column]) + _DOMINANCE_TOL
            )
        ),
        None,
    )


def _cuboid(
    ax,
    x: float,
    y: float,
    z: float,
    dx: float,
    dy: float,
    dz: float,
    color: str,
    *,
    alpha: float = 0.45,
    edgecolor: str = "black",
    linewidth: float = 1.2,
) -> None:
    vertices = [
        [x, y, z],
        [x + dx, y, z],
        [x + dx, y + dy, z],
        [x, y + dy, z],
        [x, y, z + dz],
        [x + dx, y, z + dz],
        [x + dx, y + dy, z + dz],
        [x, y + dy, z + dz],
    ]
    faces = [
        [vertices[i] for i in [0, 1, 2, 3]],
        [vertices[i] for i in [4, 5, 6, 7]],
        [vertices[i] for i in [0, 1, 5, 4]],
        [vertices[i] for i in [2, 3, 7, 6]],
        [vertices[i] for i in [1, 2, 6, 5]],
        [vertices[i] for i in [0, 3, 7, 4]],
    ]
    ax.add_collection3d(
        Poly3DCollection(
            faces,
            facecolors=color,
            edgecolor=edgecolor,
            linewidths=linewidth,
            alpha=alpha,
        )
    )


def _draw_candidate_allocation_axis(
    *,
    ax,
    total_slots: int,
    total_prbs: int,
    max_layers: int,
    allocation_row: pd.Series,
    allocation_color: str,
    allocation_edgecolor: str,
    panel_label: str | None = None,
    z_label_x: float = 1.08,
    envelope_color: str | None = None,
    envelope_edgecolor: str = "#8a8a8a",
    label_color: str | None = None,
):
    """Draw one feasible row on one time-frequency-layer axis."""

    resolved_envelope_color = envelope_color
    if resolved_envelope_color is None:
        resolved_envelope_color = colors.to_hex(plt.cm.Greys(0.55))

    _cuboid(
        ax,
        0,
        0,
        0,
        total_slots,
        total_prbs,
        max_layers,
        resolved_envelope_color,
        alpha=0.08,
        edgecolor=envelope_edgecolor,
        linewidth=1.0,
    )
    _cuboid(
        ax,
        0,
        0,
        0,
        int(allocation_row["n_slots_on"]),
        int(allocation_row["n_prb"]),
        int(allocation_row["layers"]),
        allocation_color,
        alpha=0.55,
        edgecolor=allocation_edgecolor,
        linewidth=1.4,
    )

    ax.set_xlabel("Time resources (slots)", labelpad=12)
    ax.set_ylabel("Frequency resources (PRBs)", labelpad=14)
    ax.set_zlabel("")
    ax.text2D(
        z_label_x,
        0.5,
        "Spatial resources (layers)",
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="left",
        color=label_color,
    )
    ax.set_xlim(0, total_slots)
    ax.set_ylim(total_prbs, 0)
    ax.set_zlim(0, max_layers)
    ax.set_xticks(range(0, total_slots + 1, max(1, total_slots // 10)))
    ax.set_yticks(range(0, total_prbs + 1, max(1, total_prbs // 6)))
    ax.set_zticks(range(0, max_layers + 1, 1))
    ax.view_init(elev=22, azim=-62)
    ax.set_box_aspect((1.4, 1.0, 0.8))

    if panel_label is None:
        return
    ax.text2D(
        0.5,
        1.03,
        panel_label,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color=label_color,
    )


__all__ = [
    "_draw_candidate_allocation_axis",
    "annotate_same_pa_dominance",
    "export_doc_figure",
    "prepare_feasible_plot_table",
    "select_dominated_example_pair",
    "select_slice_rows",
]
