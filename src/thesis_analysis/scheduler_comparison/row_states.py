from __future__ import annotations

"""Shared row-state predicates for scheduler-comparison analysis tables."""

import pandas as pd


def bool_like(value: object) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() == "true"


def certified_skipped_row_mask(frame: pd.DataFrame) -> pd.Series:
    status = frame["status"].fillna("").astype(str)
    skip_reason = frame["skip_reason"].fillna("").astype(str)
    return status.eq("certified_skipped") | skip_reason.eq("inherited_bound_certified_infeasible")

