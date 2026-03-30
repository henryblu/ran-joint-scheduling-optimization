from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def prepare_repo_paths(repo_root: Path) -> tuple[Path, Path]:
    """Create the documentation image directory and add src to sys.path."""

    repo_root = Path(repo_root)
    doc_img_dir = repo_root / "img"
    doc_img_dir.mkdir(parents=True, exist_ok=True)

    src_root = repo_root / "src"
    resolved_src = str(src_root.resolve())
    if resolved_src not in sys.path:
        sys.path.insert(0, resolved_src)

    return doc_img_dir, src_root


def export_doc_figure(fig, filename: str, doc_img_dir: Path | None = None) -> Path | None:
    """Save a figure to the repo image directory when an export target is provided."""

    if doc_img_dir is None:
        return None

    output_path = Path(doc_img_dir) / filename
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {output_path}")
    return output_path


def _to_dbm(power_w: np.ndarray) -> np.ndarray:
    """Convert power in watts to dBm, returning NaN for non-positive values."""

    power_w = np.asarray(power_w, dtype=float)
    power_dbm = np.full(power_w.shape, np.nan, dtype=float)

    valid = power_w > 0.0
    power_dbm[valid] = 10.0 * np.log10(power_w[valid] * 1000.0)
    return power_dbm


def plot_pa_operating_curves(pa_curve_table: pd.DataFrame):
    """Plot gain, PAE, and DC input power against average PA output power.

    Idle rows are not included in the gain or PAE panels because those quantities
    are not meaningful at zero RF output power. Idle DC input power is shown as a
    horizontal dashed line in the DC input power panel.
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Determine a sensible shared x-range from active operating points.
    active_mask_global = pa_curve_table["pout_w"].fillna(0.0) > 0.0
    active_pout_global = pa_curve_table.loc[active_mask_global, "pout_w"].to_numpy(dtype=float)
    active_pout_dbm_global = _to_dbm(active_pout_global)

    finite_active_x = active_pout_dbm_global[np.isfinite(active_pout_dbm_global)]
    if finite_active_x.size > 0:
        x_min = float(np.floor(finite_active_x.min()) - 1.0)
        x_max = float(np.ceil(finite_active_x.max()) + 1.0)
    else:
        x_min, x_max = 10.0, 40.0

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (scenario_label, pa_curve_rows) in enumerate(
        pa_curve_table.groupby("scenario_label", sort=True)
    ):
        color = color_cycle[idx % len(color_cycle)]

        active_rows = pa_curve_rows.loc[pa_curve_rows["pout_w"].fillna(0.0) > 0.0].copy()
        idle_rows = pa_curve_rows.loc[pa_curve_rows["pout_w"].fillna(0.0) <= 0.0].copy()

        active_rows = active_rows.sort_values("pout_w")

        if not active_rows.empty:
            pout_w = active_rows["pout_w"].to_numpy(dtype=float)
            pin_w = active_rows["pin_w"].to_numpy(dtype=float)
            pdc_w = active_rows["pdc_w"].to_numpy(dtype=float)

            pout_dbm = _to_dbm(pout_w)
            pin_dbm = _to_dbm(pin_w)
            pdc_dbm = _to_dbm(pdc_w)

            gain_db = pout_dbm - pin_dbm
            pae_percent = np.where(
                pdc_w > 0.0,
                100.0 * (pout_w - pin_w) / pdc_w,
                np.nan,
            )

            axes[0].plot(
                pout_dbm,
                gain_db,
                marker="o",
                color=color,
                label=scenario_label,
            )
            axes[1].plot(
                pout_dbm,
                pae_percent,
                marker="o",
                color=color,
                label=scenario_label,
            )
            axes[2].plot(
                pout_dbm,
                pdc_dbm,
                marker="o",
                color=color,
                label=scenario_label,
            )

        # Plot idle DC input power as a horizontal line across the x-range.
        if not idle_rows.empty:
            idle_pdc_dbm = _to_dbm(idle_rows["pdc_w"].to_numpy(dtype=float))
            idle_pdc_dbm = idle_pdc_dbm[np.isfinite(idle_pdc_dbm)]

            if idle_pdc_dbm.size > 0:
                idle_level_dbm = float(idle_pdc_dbm[0])

                axes[2].hlines(
                    idle_level_dbm,
                    x_min,
                    x_max,
                    colors=color,
                    linestyles="--",
                    linewidth=1.6,
                    label=f"{scenario_label} idle",
                )

    axes[0].set_title("Gain vs PA Average Output Power")
    axes[1].set_title("PAE vs PA Average Output Power")
    axes[2].set_title("PA DC Input Power vs PA Average Output Power")

    for ax in axes:
        ax.set_xlabel("PA Average Output Power (dBm)")
        ax.set_xlim(x_min, x_max)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0].set_ylabel("Gain (dB)")
    axes[1].set_ylabel("PAE (%)")
    axes[2].set_ylabel("PA DC Input Power (dBm)")

    note = "Dashed lines in the DC input power panel show idle PA DC power."
    plt.figtext(0.02, 0.01, note, ha="left", fontsize=9, color="gray")

    plt.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    plt.show()
    return fig, axes


def plot_feasible_candidate_cloud(
    *,
    feasible_table: pd.DataFrame,
    pa_label_by_id: Mapping[int, str],
    best_feasible_row: pd.Series,
    doc_img_dir: Path | None = None,
    export_filename: str | None = None,
):
    """Plot the feasible candidate cloud and highlight one deterministic example row."""

    fig, ax = plt.subplots(figsize=(8, 5))

    for pa_id, df_pa in feasible_table.groupby("pa_id", sort=True):
        ax.scatter(
            df_pa["rate_ach_bps"] / 1e6,
            df_pa["p_dc_avg_total_w"],
            s=18,
            alpha=0.7,
            label=pa_label_by_id[int(pa_id)],
        )

    ax.scatter(
        best_feasible_row["rate_ach_bps"] / 1e6,
        best_feasible_row["p_dc_avg_total_w"],
        s=55,
        color="red",
        label="Illustrative feasible row",
    )

    ax.set_xlabel("Achievable rate (Mbps)")
    ax.set_ylabel("Window-averaged total PA DC power (W)")
    ax.set_title("Feasible single-user candidate cloud for one fixed user case")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if export_filename is not None:
        export_doc_figure(fig, export_filename, doc_img_dir=doc_img_dir)

    plt.show()
    return fig, ax


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
    poly = Poly3DCollection(
        faces,
        facecolors=color,
        edgecolor=edgecolor,
        linewidths=linewidth,
        alpha=alpha,
    )
    ax.add_collection3d(poly)


def plot_candidate_allocation(
    *,
    total_slots: int,
    total_prbs: int,
    max_layers: int,
    best_feasible_row: pd.Series,
    best_feasible_label: str,
    doc_img_dir: Path | None = None,
    export_filename: str | None = None,
):
    """Plot one feasible row as a time-frequency-layer allocation block."""

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    envelope_color = colors.to_hex(plt.cm.Greys(0.55))
    allocation_color = colors.to_hex(plt.cm.Blues(0.68))

    _cuboid(
        ax,
        0,
        0,
        0,
        total_slots,
        total_prbs,
        max_layers,
        envelope_color,
        alpha=0.08,
        edgecolor="#8a8a8a",
        linewidth=1.0,
    )

    _cuboid(
        ax,
        0,
        0,
        0,
        int(best_feasible_row["n_slots_on"]),
        int(best_feasible_row["n_prb"]),
        int(best_feasible_row["layers"]),
        allocation_color,
        alpha=0.55,
        edgecolor="#1f4e79",
        linewidth=1.4,
    )

    ax.set_xlabel("Time resources (slots)", labelpad=12)
    ax.set_ylabel("Frequency resources (PRBs)", labelpad=14)

    # 3D z-labels are often clipped or misplaced, so draw it in 2D figure space.
    ax.set_zlabel("")
    ax.text2D(
        1.04,
        0.5,
        "Spatial resources (layers)",
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="left",
    )

    ax.set_xlim(0, total_slots)
    ax.set_ylim(total_prbs, 0)
    ax.set_zlim(0, max_layers)

    ax.set_xticks(range(0, total_slots + 1, max(1, total_slots // 10)))
    ax.set_yticks(range(0, total_prbs + 1, max(1, total_prbs // 6)))
    ax.set_zticks(range(0, max_layers + 1, 1))

    ax.view_init(elev=22, azim=-62)
    ax.set_box_aspect((1.4, 1.0, 0.8))

    # Use figure-level title text instead of ax.set_title for better spacing control.
    fig.suptitle(
        "Illustrative feasible PHY allocation",
        y=0.91,
        fontsize=14,
    )

    fig.text(
        0.5,
        0.875,
        f"{best_feasible_label} | "
        f"MCS={int(best_feasible_row['mcs'])} | "
        f"rate={best_feasible_row['rate_ach_bps'] / 1e6:.1f} Mbps | "
        f"power={best_feasible_row['p_dc_avg_total_w']:.2f} W",
        ha="center",
        va="center",
        fontsize=11,
    )

    # Manual layout control is usually more reliable than tight_layout for 3D axes.
    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.08, top=0.95)

    if export_filename is not None:
        export_doc_figure(fig, export_filename, doc_img_dir=doc_img_dir)

    plt.show()
    return fig, ax
