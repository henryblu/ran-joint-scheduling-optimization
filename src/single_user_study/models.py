from dataclasses import dataclass, field

import pandas as pd

from single_user_search.models import PreparedSingleUserContext, SingleUserRequest


@dataclass(frozen=True)
class SingleUserScenario:
    """Prepared single-user study scenario used by notebook-facing helpers."""

    request: SingleUserRequest
    context: PreparedSingleUserContext


@dataclass
class SingleUserStudyResult:
    """Notebook-facing result tables for one frontier study sweep."""

    frontier_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    explanatory_configs: pd.DataFrame = field(default_factory=pd.DataFrame)
    pa_characteristics: pd.DataFrame = field(default_factory=pd.DataFrame)
    search_space_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
