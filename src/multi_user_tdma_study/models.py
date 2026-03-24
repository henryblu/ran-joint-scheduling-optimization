from dataclasses import dataclass, field

import pandas as pd

from radio_core import MultiUserPreset


@dataclass(frozen=True)
class MultiUserTdmaScenario:
    """Prepared multi-user TDMA study scenario used by notebook-facing helpers."""

    user_table: pd.DataFrame
    preset: MultiUserPreset
    system_cfg: dict[str, object]
    pa_catalog: list


@dataclass
class MultiUserTdmaStudyResult:
    """Notebook-facing result tables for one multi-user pre-scheduler study run."""

    repeated_frames: int = 0
    repeated_period_slots: int = 0
    active_candidate_tables: dict[int, pd.DataFrame] = field(default_factory=dict)
    active_summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    frame_share_summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    user_candidate_spaces: dict[int, pd.DataFrame] = field(default_factory=dict)
    user_candidate_review_tables: dict[int, pd.DataFrame] = field(default_factory=dict)
    user_candidate_summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
