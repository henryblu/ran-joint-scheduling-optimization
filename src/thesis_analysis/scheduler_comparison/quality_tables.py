from __future__ import annotations

"""Small table helpers for scheduler-comparison quality reports."""

import pandas as pd


def check_row(check_name: str, passed: bool, failed_count: int, examples: str) -> dict[str, object]:
    return {
        "check_name": str(check_name),
        "passed": bool(passed),
        "failed_count": int(abs(failed_count)),
        "examples": str(examples),
    }


def point_examples(frame: pd.DataFrame, *, limit: int = 5) -> str:
    if frame.empty or "point_id" not in frame:
        return ""
    return "; ".join(str(value) for value in frame["point_id"].head(limit))


def pair_examples(frame: pd.DataFrame, left_column: str, right_column: str, *, limit: int = 5) -> str:
    if frame.empty:
        return ""
    return "; ".join(
        f"{left} vs {right}"
        for left, right in zip(frame[left_column].head(limit), frame[right_column].head(limit))
    )


def example_values(values: set[object], *, limit: int = 5) -> str:
    return "; ".join(str(value) for value in sorted(values)[:limit])

