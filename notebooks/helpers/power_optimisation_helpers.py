"""Helpers for the downlink power optimization discussion notebook."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

src_path = (PROJECT_ROOT / "src").resolve()
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from configs import SINGLE_USER_SEARCH_CONFIG, build_pa_catalog
from models import build_resolved_fingerprint
from single_user_solver import enumerate_active_candidates, search_candidates
from single_user_solver.candidate_space import count_candidates_for_rrc
from single_user_solver.models import SearchSpace, SingleUserRequest
from single_user_solver.problem_factory import prepare_single_user_problem

PUBLIC_COLUMNS = [
    "distance_m",
    "pa_id",
    "pa_name",
    "rate_ach_bps",
    "p_dc_avg_total_w",
    "layers",
    "mcs",
    "n_prb",
    "n_slots_on",
    "bandwidth_hz",
    "p_out_total_w",
    "gamma_req_lin",
]

FRONTIER_COLUMNS = [
    "distance_m",
    "pa_id",
    "pa_name",
    "rate_target_bps",
    "rate_ach_bps",
    "p_dc_avg_total_w",
    "layers",
    "mcs",
    "n_prb",
    "n_slots_on",
    "bandwidth_hz",
    "p_out_total_w",
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
    "pa_id",
]
PLOT_TIE_BREAK_COLUMNS = [
    "p_dc_avg_total_w",
    "bandwidth_hz",
    "n_prb",
    "n_slots_on",
    "layers",
    "mcs",
    "pa_id",
]

NOTEBOOK_CONFIG = SINGLE_USER_SEARCH_CONFIG
NOTEBOOK_PA_CATALOG = tuple(build_pa_catalog(NOTEBOOK_CONFIG.pa_data_csv))
NOTEBOOK_N_SLOTS_ON_SPACE = tuple(range(1, int(NOTEBOOK_CONFIG.n_slots_win) + 1))
NOTEBOOK_SEARCH_SHAPE = SearchSpace(
    config=NOTEBOOK_CONFIG,
    bandwidth_space_hz=tuple(float(value) for value in NOTEBOOK_CONFIG.bandwidth_space_hz),
    n_slots_on_space=NOTEBOOK_N_SLOTS_ON_SPACE,
    layers_space=tuple(int(value) for value in NOTEBOOK_CONFIG.layers_space),
    mcs_space=tuple(int(value) for value in NOTEBOOK_CONFIG.mcs_space),
    prb_step=int(NOTEBOOK_CONFIG.prb_step),
    fingerprint=build_resolved_fingerprint({"n_slots_on_space": NOTEBOOK_N_SLOTS_ON_SPACE}),
    use_cache=True,
)
PA_LABEL_MAP = {pa_id: str(pa.pa_name) for pa_id, pa in enumerate(NOTEBOOK_PA_CATALOG)}

MARKER_SEQUENCE = ["x", "o", "^", "D", "v", "P", "*", "+", ".", "s"]


def build_single_user_scenario(distance_m, required_rate_bps):
    request = SingleUserRequest(
        distance_m=float(distance_m),
        required_rate_bps=float(required_rate_bps),
    )
    context = prepare_single_user_problem(
        request=request,
        model_inputs=NOTEBOOK_CONFIG,
        search_shape=NOTEBOOK_SEARCH_SHAPE,
        pa_catalog=NOTEBOOK_PA_CATALOG,
    )
    return SimpleNamespace(request=request, context=context)


def build_candidate_space_view(context, *, scenario_count=1):
    pa_labels = tuple(str(pa.scenario_label) for pa in context.pa_catalog)
    bandwidth_options_hz = tuple(sorted({float(rrc.bwp_bw_hz) for rrc in context.rrc_catalog}))
    max_prbs_by_bwp = tuple(
        (
            str(context.pa_catalog[int(rrc.active_pa_id)].scenario_label),
            int(rrc.bwp_index),
            int(rrc.prb_max_bwp),
        )
        for rrc in sorted(
            context.rrc_catalog,
            key=lambda item: (int(item.active_pa_id), float(item.bwp_bw_hz), int(item.bwp_index)),
        )
    )
    per_pa_counts = []
    for pa_id in range(len(context.pa_catalog)):
        rrc_space = [rrc for rrc in context.rrc_catalog if int(rrc.active_pa_id) == int(pa_id)]
        per_pa_counts.append(
            (
                str(context.pa_catalog[int(pa_id)].scenario_label),
                int(sum(count_candidates_for_rrc(context.search_catalog, rrc) for rrc in rrc_space)),
            )
        )

    raw_candidate_count_total = int(sum(count for _label, count in per_pa_counts))
    return pd.DataFrame(
        [
            {
                "pa_labels": pa_labels,
                "bandwidth_options_hz": bandwidth_options_hz,
                "max_prbs_by_bwp": max_prbs_by_bwp,
                "slot_domain": (1, int(context.deployment.n_slots_win)),
                "layer_domain": (
                    int(min(context.search_shape.layers_space)),
                    int(max(context.search_shape.layers_space)),
                ),
                "mcs_domain": (
                    int(min(context.search_shape.mcs_space)),
                    int(max(context.search_shape.mcs_space)),
                ),
                "prb_step": int(context.search_shape.prb_step),
                "raw_candidate_count_per_pa": tuple(per_pa_counts),
                "raw_candidate_count_total": raw_candidate_count_total,
                "raw_candidate_count_across_scenarios": int(raw_candidate_count_total * scenario_count),
            }
        ]
    )


def build_study_candidate_table(candidate_table, *, distance_m):
    if candidate_table.empty:
        return pd.DataFrame(columns=PUBLIC_COLUMNS)

    study_table = candidate_table.copy()
    study_table["distance_m"] = float(distance_m)
    study_table["pa_name"] = study_table["pa_id"].map(lambda pa_id: PA_LABEL_MAP[int(pa_id)])
    return study_table


def filter_feasible_candidate_table(candidate_table):
    feasible_table = candidate_table[candidate_table["rate_ach_bps"].notna()].copy()
    if feasible_table.empty:
        return pd.DataFrame(columns=PUBLIC_COLUMNS)
    return feasible_table[PUBLIC_COLUMNS].copy()


def rank_rate_feasible_rows(pa_configs, required_rate_bps):
    feasible_rows = pa_configs[pa_configs["rate_ach_bps"] >= float(required_rate_bps)].copy()
    if feasible_rows.empty:
        return feasible_rows
    return feasible_rows.sort_values(TIE_BREAK_COLUMNS, ascending=True).reset_index(drop=True)


def build_frontier_row(winner, required_rate_bps):
    frontier_row = {column: winner[column] for column in PUBLIC_COLUMNS}
    frontier_row["rate_target_bps"] = float(required_rate_bps)
    return {column: frontier_row[column] for column in FRONTIER_COLUMNS}


def build_explanation_rows(ranked, required_rate_bps):
    explanation_rows = ranked.head(3)[PUBLIC_COLUMNS].copy()
    explanation_rows["rate_target_bps"] = float(required_rate_bps)
    explanation_rows["explanation_role"] = ["winner"] + ["runner_up"] * max(
        len(explanation_rows) - 1,
        0,
    )
    return explanation_rows


def concat_frontier_tables(frontier_tables):
    if not frontier_tables:
        return pd.DataFrame(columns=FRONTIER_COLUMNS)
    return pd.concat(frontier_tables, ignore_index=True).sort_values(
        ["distance_m", "pa_id", "rate_target_bps"]
    ).reset_index(drop=True)


def concat_explanatory_tables(explanatory_tables):
    explanatory_frames = [frame for frame in explanatory_tables if not frame.empty]
    if not explanatory_frames:
        return pd.DataFrame(columns=EXPLANATION_COLUMNS)
    return pd.concat(explanatory_frames, ignore_index=True).sort_values(
        ["distance_m", "pa_id", "rate_target_bps", "explanation_role", "p_dc_avg_total_w"]
    ).reset_index(drop=True)


def concat_scenario_explanations(explanation_frames):
    if not explanation_frames:
        return pd.DataFrame(columns=EXPLANATION_COLUMNS)
    return pd.concat(explanation_frames, ignore_index=True)


def evaluate_scenario_frontier(candidate_table, required_rate_targets_bps):
    feasible_table = filter_feasible_candidate_table(candidate_table)
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
            ranked = rank_rate_feasible_rows(
                ranked_pa_configs,
                required_rate_bps=float(required_rate_bps),
            )
            if ranked.empty:
                continue
            scenario_frontier_rows.append(
                build_frontier_row(ranked.iloc[0], required_rate_bps=float(required_rate_bps))
            )
            scenario_explanatory_rows.append(
                build_explanation_rows(ranked, required_rate_bps=float(required_rate_bps))
            )

    scenario_frontier = pd.DataFrame(scenario_frontier_rows, columns=FRONTIER_COLUMNS)
    scenario_explanatory = concat_scenario_explanations(scenario_explanatory_rows)
    return scenario_frontier, scenario_explanatory


def run_frontier_study(scenarios, required_rate_targets_bps):
    required_rate_targets_bps = np.asarray(required_rate_targets_bps, dtype=float)
    candidate_filter_rate_bps = float(np.min(required_rate_targets_bps))
    prepared_scenarios = [
        build_single_user_scenario(
            distance_m=float(scenario["distance_m"]),
            required_rate_bps=candidate_filter_rate_bps,
        )
        for scenario in scenarios
    ]

    frontier_tables = []
    explanatory_tables = []
    active_candidate_counts = []
    for scenario, prepared_scenario in zip(scenarios, prepared_scenarios):
        candidate_table = search_candidates(
            prepared_scenario.context,
            required_rate_bps=candidate_filter_rate_bps,
        )
        active_candidate_counts.append(int(len(candidate_table)))
        study_table = build_study_candidate_table(
            candidate_table,
            distance_m=float(scenario["distance_m"]),
        )
        scenario_frontier, scenario_explanatory = evaluate_scenario_frontier(
            study_table,
            required_rate_targets_bps=required_rate_targets_bps,
        )
        frontier_tables.append(scenario_frontier)
        explanatory_tables.append(scenario_explanatory)

    return SimpleNamespace(
        frontier_table=concat_frontier_tables(frontier_tables),
        explanatory_configs=concat_explanatory_tables(explanatory_tables),
        candidate_space_view=build_candidate_space_view(
            prepared_scenarios[0].context,
            scenario_count=len(prepared_scenarios),
        ),
        active_candidate_counts=tuple(active_candidate_counts),
    )


def run_rate_study(distance_m, rate_targets_bps):
    scenarios = [{"distance_m": float(distance_m)}]
    return run_frontier_study(
        scenarios,
        required_rate_targets_bps=np.asarray(rate_targets_bps, dtype=float),
    )


def run_distance_study(distance_values_m, required_rate_bps):
    scenarios = [{"distance_m": float(distance_m)} for distance_m in distance_values_m]
    return run_frontier_study(
        scenarios,
        required_rate_targets_bps=np.asarray([float(required_rate_bps)], dtype=float),
    )


def select_best_candidate_row(candidate_table):
    if candidate_table.empty:
        raise ValueError("candidate_table must not be empty.")
    return candidate_table.sort_values(TIE_BREAK_COLUMNS).reset_index(drop=True).iloc[0]


def build_style_maps(pa_ids):
    size_max = 50
    size_min = 20
    sizes = np.linspace(size_max, size_min, len(pa_ids))
    pa_marker_map = {pa_id: MARKER_SEQUENCE[i % len(MARKER_SEQUENCE)] for i, pa_id in enumerate(pa_ids)}
    pa_size_map = {pa_id: sizes[i] for i, pa_id in enumerate(pa_ids)}
    return pa_marker_map, pa_size_map


def prep(df, x_col):
    return df.sort_values(x_col)


def req_snr_db(series):
    gamma = np.asarray(pd.to_numeric(series, errors="coerce"), dtype=float)
    return 10.0 * np.log10(np.clip(gamma, 1e-12, None))


def energy_per_bit(df):
    return df["p_dc_avg_total_w"] / np.clip(df["rate_target_bps"], 1e-12, None)


def round_up_to_step(value, step):
    value = float(value)
    step = float(step)
    if value <= 0.0:
        return step
    return step * np.ceil(value / step)


def build_axis_from_series(values, tick_step, upper_round_step, axis_min=0.0):
    values = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        axis_upper = axis_min + float(upper_round_step)
    else:
        axis_upper = round_up_to_step(max(float(finite_values.max()), float(axis_min)), upper_round_step)
    tick_values = np.arange(float(axis_min), axis_upper + 0.5 * float(tick_step), float(tick_step))
    return tick_values, (float(axis_min), float(axis_upper))


def build_integer_plot_domain(values, *, max_ticks=8):
    ordered_values = sorted({int(value) for value in values})
    stride = max(1, int(np.ceil(len(ordered_values) / int(max_ticks))))
    tick_values = ordered_values[::stride]
    if tick_values[-1] != ordered_values[-1]:
        tick_values.append(ordered_values[-1])

    padding = 0.5 if len(ordered_values) == 1 else min(np.diff(ordered_values)) / 2.0
    return {
        "ticks": np.asarray(tick_values, dtype=float),
        "limits": (
            float(ordered_values[0] - padding),
            float(ordered_values[-1] + padding),
        ),
    }


def build_integer_data_plot_domain(values, *, max_ticks=8, padding=0.35):
    numeric_values = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
    finite_values = numeric_values[np.isfinite(numeric_values)]
    if finite_values.size == 0:
        return {
            "ticks": np.asarray([0.0, 1.0], dtype=float),
            "limits": (-0.5, 1.5),
        }

    min_value = int(np.floor(finite_values.min()))
    max_value = int(np.ceil(finite_values.max()))
    span = max_value - min_value + 1
    stride = max(1, int(np.ceil(span / int(max_ticks))))
    tick_values = np.arange(min_value, max_value + 1, stride, dtype=float)
    if tick_values.size == 0 or tick_values[-1] != float(max_value):
        tick_values = np.append(tick_values, float(max_value))

    return {
        "ticks": tick_values,
        "limits": (
            float(min_value) - float(padding),
            float(max_value) + float(padding),
        ),
    }


def format_x_value(value, unit):
    if unit == "Mbps":
        return f"{float(value):.1f} Mbps"
    return f"{float(value):.0f} m"


def prepare_frontier_plotting(
    *,
    reference_candidate_space_view,
    rate_study_config,
    distance_study_config,
    rate_frontier_table,
    distance_frontier_table,
):
    reference_candidate_space_row = reference_candidate_space_view.iloc[0]
    prb_step = int(reference_candidate_space_row["prb_step"])
    prb_upper = max(
        int(prb_max_bwp)
        for _scenario_label, _bwp_index, prb_max_bwp in reference_candidate_space_row["max_prbs_by_bwp"]
    )
    frame_slot_count = int(reference_candidate_space_row["slot_domain"][1])
    summary_plot_domains = {
        "layers": build_integer_plot_domain(reference_candidate_space_row["layer_domain"]),
        "mcs": build_integer_plot_domain(reference_candidate_space_row["mcs_domain"]),
        "n_prb": build_integer_plot_domain(range(prb_step, prb_upper + prb_step, prb_step)),
        "n_slots_on": build_integer_plot_domain(
            range(
                int(reference_candidate_space_row["slot_domain"][0]),
                int(reference_candidate_space_row["slot_domain"][1]) + 1,
            )
        ),
    }

    rate_plot_distance_m = float(rate_study_config["distance_m"])
    plot_rate_frontier = rate_frontier_table.assign(
        rate_target_mbps=rate_frontier_table["rate_target_bps"] / 1e6
    ).copy()
    plot_distance_frontier = distance_frontier_table.copy()

    distance_fixed_rate_mbps = float(distance_study_config["rate_target_bps"]) / 1e6
    rate_tick_values, rate_limits = build_axis_from_series(
        np.asarray(rate_study_config["rate_targets_bps"], dtype=float) / 1e6,
        tick_step=50.0,
        upper_round_step=100.0,
        axis_min=0.0,
    )
    distance_tick_values, distance_limits = build_axis_from_series(
        np.asarray(distance_study_config["distance_values_m"], dtype=float),
        tick_step=50.0,
        upper_round_step=100.0,
        axis_min=0.0,
    )

    scenario_specs = [
        {
            "label": "Fixed-distance throughput study",
            "table": plot_rate_frontier,
            "x_col": "rate_target_mbps",
            "x_label": "Required throughput (Mbps)",
            "x_ticks": rate_tick_values,
            "x_limits": rate_limits,
            "x_unit": "Mbps",
            "subtitle": f"Distance fixed at {int(rate_plot_distance_m)} m",
        },
        {
            "label": "Fixed-throughput distance study",
            "table": plot_distance_frontier,
            "x_col": "distance_m",
            "x_label": "User distance (m)",
            "x_ticks": distance_tick_values,
            "x_limits": distance_limits,
            "x_unit": "m",
            "subtitle": f"Required throughput fixed at {distance_fixed_rate_mbps:.0f} Mbps",
        },
    ]

    metric_specs = {
        "p_dc_avg_total_w": {
            "label": "Frame-averaged total PA DC input power (W)",
            "style": "line",
        },
        "p_out_total_w": {
            "label": "Total PA output power (W)",
            "style": "line",
        },
        "layers": {
            "label": "Selected number of spatial layers",
            "style": "scatter",
            "use_data_domain": True,
        },
        "mcs": {
            "label": "Selected MCS index",
            "style": "scatter",
            "use_data_domain": True,
        },
        "n_prb": {
            "label": "Allocated physical resource blocks",
            "style": "scatter",
            "domain_key": "n_prb",
        },
        "n_slots_on": {
            "label": f"Allocated downlink slots per {frame_slot_count}-slot frame",
            "style": "scatter",
            "domain_key": "n_slots_on",
        },
        "gamma_req_db": {
            "label": "Required SNR (dB)",
            "style": "line",
        },
        "energy_per_bit": {
            "label": "PA energy per delivered bit (J/bit)",
            "style": "line",
            "yscale": "log",
        },
    }

    return SimpleNamespace(
        pa_label_map=dict(PA_LABEL_MAP),
        scenario_specs=scenario_specs,
        metric_specs=metric_specs,
        summary_plot_domains=summary_plot_domains,
    )


def extract_metric_series(df, metric_key):
    if metric_key == "gamma_req_db":
        return req_snr_db(df["gamma_req_lin"])
    if metric_key == "energy_per_bit":
        return np.asarray(energy_per_bit(df), dtype=float)
    return np.asarray(pd.to_numeric(df[metric_key], errors="coerce"), dtype=float)


def build_winner_curve(table, x_col):
    return (
        table.sort_values([x_col] + PLOT_TIE_BREAK_COLUMNS)
        .groupby(x_col, as_index=False)
        .first()
        .reset_index(drop=True)
    )


def plot_metric_panel(ax, scenario_spec, metric_key, *, metric_specs, summary_plot_domains, pa_label_map):
    metric_spec = metric_specs[metric_key]
    frontier_table = scenario_spec["table"]
    x_col = scenario_spec["x_col"]
    pa_ids = sorted(frontier_table["pa_id"].unique())
    pa_marker_map, pa_size_map = build_style_maps(pa_ids)
    plotted_y_values = []

    for pa_id in pa_ids:
        df = prep(frontier_table[frontier_table["pa_id"] == pa_id], x_col)
        y_values = extract_metric_series(df, metric_key)
        finite_y_values = np.asarray(y_values, dtype=float)
        finite_y_values = finite_y_values[np.isfinite(finite_y_values)]
        if finite_y_values.size:
            plotted_y_values.append(finite_y_values)
        if metric_spec["style"] == "scatter":
            ax.scatter(
                df[x_col],
                y_values,
                marker=pa_marker_map[pa_id],
                s=pa_size_map[pa_id],
                label=pa_label_map.get(pa_id, f"PA{pa_id}"),
            )
        else:
            ax.plot(df[x_col], y_values, label=pa_label_map.get(pa_id, f"PA{pa_id}"))

    ax.set_xlabel(scenario_spec["x_label"])
    ax.set_ylabel(metric_spec["label"])
    ax.set_xticks(scenario_spec["x_ticks"])
    ax.set_xlim(*scenario_spec["x_limits"])
    ax.grid(True, alpha=0.3)

    if metric_spec.get("use_data_domain", False):
        combined_y_values = np.concatenate(plotted_y_values) if plotted_y_values else np.asarray([], dtype=float)
        axis_config = build_integer_data_plot_domain(combined_y_values)
        ax.set_ylim(*axis_config["limits"])
        ax.set_yticks(axis_config["ticks"])
    else:
        domain_key = metric_spec.get("domain_key")
        if domain_key is not None:
            axis_config = summary_plot_domains[domain_key]
            ax.set_ylim(*axis_config["limits"])
            ax.set_yticks(axis_config["ticks"])

    yscale = metric_spec.get("yscale")
    if yscale is not None:
        ax.set_yscale(yscale)

    ax.set_title(f"{scenario_spec['label']}\n{scenario_spec['subtitle']}")


def plot_grouped_frontier_figure(plot_context, metric_keys, figure_title):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(figure_title, fontsize=14)

    for row_idx, scenario_spec in enumerate(plot_context.scenario_specs):
        for col_idx, metric_key in enumerate(metric_keys):
            plot_metric_panel(
                axes[row_idx, col_idx],
                scenario_spec,
                metric_key,
                metric_specs=plot_context.metric_specs,
                summary_plot_domains=plot_context.summary_plot_domains,
                pa_label_map=plot_context.pa_label_map,
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), bbox_to_anchor=(0.6, 0.975))
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()
    return fig


def build_objective_output_summary(plot_context):
    rows = []
    for scenario_spec in plot_context.scenario_specs:
        winner_curve = build_winner_curve(scenario_spec["table"], scenario_spec["x_col"])
        start_row = winner_curve.iloc[0]
        end_row = winner_curve.iloc[-1]
        rows.append(
            {
                "scenario": scenario_spec["label"],
                "start point": format_x_value(start_row[scenario_spec["x_col"]], scenario_spec["x_unit"]),
                "end point": format_x_value(end_row[scenario_spec["x_col"]], scenario_spec["x_unit"]),
                "winner changes": int((winner_curve["pa_id"] != winner_curve["pa_id"].shift()).sum() - 1),
                "PA DC growth x": float(end_row["p_dc_avg_total_w"] / start_row["p_dc_avg_total_w"]),
                "PA output growth x": float(end_row["p_out_total_w"] / start_row["p_out_total_w"]),
            }
        )
    return pd.DataFrame(rows)


def build_regime_summary(plot_context, metric_keys, *, max_rows_per_scenario=6):
    rows = []
    for scenario_spec in plot_context.scenario_specs:
        winner_curve = build_winner_curve(scenario_spec["table"], scenario_spec["x_col"])
        previous_row = None
        shown_rows = 0
        for _, current_row in winner_curve.iterrows():
            if previous_row is None:
                previous_row = current_row
                continue
            if any(int(current_row[key]) != int(previous_row[key]) for key in metric_keys):
                row = {
                    "scenario": scenario_spec["label"],
                    "at": format_x_value(current_row[scenario_spec["x_col"]], scenario_spec["x_unit"]),
                    "winner PA": str(current_row["pa_name"]),
                }
                for key in metric_keys:
                    row[plot_context.metric_specs[key]["label"]] = int(current_row[key])
                rows.append(row)
                shown_rows += 1
                if shown_rows >= max_rows_per_scenario:
                    break
            previous_row = current_row
    return pd.DataFrame(rows)


def build_physical_energy_summary(plot_context):
    rows = []
    for scenario_spec in plot_context.scenario_specs:
        winner_curve = build_winner_curve(scenario_spec["table"], scenario_spec["x_col"])
        energy_values = np.asarray(energy_per_bit(winner_curve), dtype=float)
        snr_values = req_snr_db(winner_curve["gamma_req_lin"])
        min_energy_idx = int(np.argmin(energy_values))
        max_snr_idx = int(np.argmax(snr_values))
        min_energy_row = winner_curve.iloc[min_energy_idx]
        max_snr_row = winner_curve.iloc[max_snr_idx]
        rows.append(
            {
                "scenario": scenario_spec["label"],
                "minimum energy per bit": float(energy_values[min_energy_idx]),
                "at minimum energy": format_x_value(min_energy_row[scenario_spec["x_col"]], scenario_spec["x_unit"]),
                "PA at minimum energy": str(min_energy_row["pa_name"]),
                "maximum required SNR (dB)": float(snr_values[max_snr_idx]),
                "at maximum required SNR": format_x_value(max_snr_row[scenario_spec["x_col"]], scenario_spec["x_unit"]),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "PROJECT_ROOT",
    "PA_LABEL_MAP",
    "build_single_user_scenario",
    "build_candidate_space_view",
    "run_rate_study",
    "run_distance_study",
    "select_best_candidate_row",
    "prepare_frontier_plotting",
    "plot_grouped_frontier_figure",
    "build_objective_output_summary",
    "build_regime_summary",
    "build_physical_energy_summary",
    "enumerate_active_candidates",
    "search_candidates",
]
