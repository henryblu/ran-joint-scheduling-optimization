from __future__ import annotations

"""Markdown reporting helpers for scheduler-comparison preprocessing."""

from pathlib import Path

import pandas as pd

from .quality import EXPECTED_POINT_COUNT


def build_markdown_summary(
    *,
    input_root: Path,
    output_root: Path,
    chunk_inventory: pd.DataFrame,
    combined_manifest: pd.DataFrame,
    combined_results: pd.DataFrame,
    point_coverage: pd.DataFrame,
    sanity_checks: pd.DataFrame,
    breakpoint_summary: pd.DataFrame,
    infeasibility_reason_summary: pd.DataFrame,
    scheduler_summary: pd.DataFrame,
    policy_summary: pd.DataFrame,
) -> str:
    failed_checks = sanity_checks.loc[~sanity_checks["passed"]]
    unexpected_breakpoints = breakpoint_summary.loc[breakpoint_summary["unexpected_breakpoint_flag"]]
    first_breaks = breakpoint_summary.loc[breakpoint_summary["breakpoint_category"] != "all_solved"]
    breakpoint_counts = breakpoint_summary.groupby(["scheduler_mode", "switch_policy", "breakpoint_category"]).size().reset_index(name="count")

    lines = [
        "# Scheduler Comparison HPC First-Stage Summary",
        "",
        f"Input root: `{input_root}`",
        f"Output root: `{output_root}`",
        "",
        "## Coverage",
        "",
        f"- Chunks discovered: {len(chunk_inventory)}",
        f"- Manifest rows: {len(combined_manifest)}",
        f"- Result rows: {len(combined_results)}",
        f"- Expected grid points: {EXPECTED_POINT_COUNT}",
        f"- Coverage rows: {len(point_coverage)}",
        f"- Failed sanity checks: {len(failed_checks)}",
        "",
        "## Scheduler Summary",
        "",
        markdown_table(scheduler_summary),
        "",
        "## Policy Summary",
        "",
        markdown_table(policy_summary.head(40)),
        "",
        "## Breakpoints",
        "",
        f"- Load chains: {len(breakpoint_summary)}",
        f"- Chains with a first non-solved row: {len(first_breaks)}",
        f"- Chains with solved rows after an earlier non-solved row: {len(unexpected_breakpoints)}",
        "",
        "### Breakpoint Categories",
        "",
        markdown_table(breakpoint_counts.head(40)),
        "",
        "### Earliest Breakpoints",
        "",
        markdown_table(earliest_breakpoint_table(first_breaks)),
        "",
        "### Unexpected Nonmonotone Breakpoints",
        "",
        markdown_table(unexpected_breakpoint_table(unexpected_breakpoints)),
        "",
        "## Most Common Infeasibility Reasons",
        "",
        markdown_table(infeasibility_reason_summary.head(30)),
        "",
        "## Failed Sanity Checks",
        "",
        markdown_table(failed_checks),
        "",
        "## Recommended JSON Inspection Targets",
        "",
        "- First non-solved row in each load chain from `breakpoint_summary.csv`.",
        "- Rows where `unexpected_breakpoint_flag` is true.",
        "- Certified skipped rows whose source metadata fails sanity checks.",
        "- Failed or other-status rows before ordinary infeasibility boundaries.",
        "",
    ]
    return "\n".join(lines)


def earliest_breakpoint_table(first_breaks: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scheduler_mode",
        "switch_policy",
        "active_user_count",
        "distance_model",
        "mean_distance_m",
        "sigma_distance_m",
        "first_unsolved_load_factor",
        "first_unsolved_status",
        "first_unsolved_reason",
        "breakpoint_category",
    ]
    return first_breaks.sort_values(
        ["first_unsolved_load_factor", "scheduler_mode", "switch_policy", "active_user_count"],
        na_position="last",
    ).loc[:, columns].head(30)


def unexpected_breakpoint_table(unexpected_breakpoints: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scheduler_mode",
        "switch_policy",
        "active_user_count",
        "distance_model",
        "mean_distance_m",
        "sigma_distance_m",
        "first_unsolved_load_factor",
        "first_unsolved_status",
        "last_solved_load_factor",
        "first_unsolved_point_id",
    ]
    return unexpected_breakpoints.loc[:, columns].head(30)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_None._"

    display = frame.fillna("").astype(str)
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(escape_markdown_cell(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def escape_markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
