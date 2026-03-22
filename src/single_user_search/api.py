import pandas as pd

from radio_core import SINGLE_USER_SEARCH_PRESET

from .models import SingleUserRequest, SingleUserSearchOptions
from .problem_factory import prepare_single_user_problem
from .search import (
    enumerate_active_candidates_from_context,
    filter_rate_feasible_candidates,
    search_candidates_from_context,
)


def enumerate_active_candidates(
    distance_m,
    *,
    path_loss_db=None,
    preset=None,
    pa_catalog=None,
    options=None,
):
    """Enumerate the full feasible active candidate space for one deployment.

    Steps:
    1. Normalize the scalar deployment inputs into the strict request model.
    2. Resolve the canonical preset and default candidate-space options.
    3. Build the reusable prepared single-user context for that deployment.
    4. Return the flat feasible active candidate table consumed by schedulers.
    """

    context = _prepare_context(
        distance_m=distance_m,
        path_loss_db=path_loss_db,
        preset=preset,
        pa_catalog=pa_catalog,
        options=options,
    )
    return enumerate_active_candidates_from_context(context)


def search_candidates(
    distance_m,
    required_rate_bps,
    *,
    path_loss_db=None,
    preset=None,
    pa_catalog=None,
    options=None,
):
    """Return the feasible candidate space that satisfies one user's target rate."""

    context = _prepare_context(
        distance_m=distance_m,
        path_loss_db=path_loss_db,
        preset=preset,
        pa_catalog=pa_catalog,
        options=options,
    )
    return search_candidates_from_context(
        context,
        required_rate_bps=float(required_rate_bps),
    )


def search_candidate_spaces(
    user_table,
    *,
    preset=None,
    pa_catalog=None,
    options=None,
):
    """Build feasible candidate spaces for many users while sharing deployment work.

    Steps:
    1. Validate the user table shape and reject duplicate `user_id` values.
    2. Group users by deployment-equivalent active-table keys.
    3. Evaluate each unique deployment's active table once.
    4. Filter that shared table per user rate target and return `user_id -> table`.
    """

    normalized_users = _normalize_user_table(user_table)
    resolved_preset = _resolve_preset(preset)
    resolved_options = _resolve_api_options(options)
    grouped_tables = {}
    user_candidate_spaces = {}

    for user_row in normalized_users.itertuples(index=False):
        group_key = _build_user_group_key(user_row)
        active_table = grouped_tables.get(group_key)
        if active_table is None:
            active_table = enumerate_active_candidates(
                float(user_row.distance_m),
                path_loss_db=user_row.path_loss_db,
                preset=resolved_preset,
                pa_catalog=pa_catalog,
                options=resolved_options,
            )
            grouped_tables[group_key] = active_table

        user_candidate_spaces[int(user_row.user_id)] = filter_rate_feasible_candidates(
            active_table,
            required_rate_bps=float(user_row.required_rate_bps),
        )
    return user_candidate_spaces


def _prepare_context(distance_m, *, path_loss_db=None, preset=None, pa_catalog=None, options=None):
    """Build the prepared context for one single-user deployment request."""

    request = SingleUserRequest(
        distance_m=float(distance_m),
        required_rate_bps=0.0,
        path_loss_db=None if path_loss_db is None else float(path_loss_db),
    )
    return prepare_single_user_problem(
        request=request,
        preset=_resolve_preset(preset),
        pa_catalog=pa_catalog,
        options=_resolve_api_options(options),
    )


def _resolve_preset(preset):
    """Return the canonical single-user preset unless the caller overrides it."""

    return SINGLE_USER_SEARCH_PRESET if preset is None else preset


def _resolve_api_options(options):
    """Return the engine API's default active-space options when none are supplied."""

    if options is not None:
        return options
    return SingleUserSearchOptions(use_cache=True)


def _normalize_user_table(user_table):
    """Normalize the user batch table and validate its required schema."""

    if not isinstance(user_table, pd.DataFrame):
        raise TypeError("user_table must be a pandas DataFrame.")

    required_columns = {"user_id", "distance_m", "required_rate_bps"}
    missing_columns = sorted(required_columns.difference(user_table.columns))
    if missing_columns:
        raise ValueError(f"user_table is missing required columns: {missing_columns}")

    normalized = user_table.copy()
    if "path_loss_db" not in normalized.columns:
        normalized["path_loss_db"] = None

    normalized["user_id"] = normalized["user_id"].astype(int)
    if normalized["user_id"].duplicated().any():
        duplicate_ids = sorted(normalized.loc[normalized["user_id"].duplicated(), "user_id"].unique())
        raise ValueError(f"user_table contains duplicate user_id values: {duplicate_ids}")

    normalized["distance_m"] = normalized["distance_m"].astype(float)
    normalized["required_rate_bps"] = normalized["required_rate_bps"].astype(float)
    normalized["path_loss_db"] = normalized["path_loss_db"].apply(_normalize_optional_float)
    return normalized[["user_id", "distance_m", "required_rate_bps", "path_loss_db"]]


def _build_user_group_key(user_row):
    """Build the deployment-equivalent group key for one batch user row."""

    return (
        float(user_row.distance_m),
        _normalize_optional_float(user_row.path_loss_db),
    )


def _normalize_optional_float(value):
    """Normalize pandas/NumPy missing values into `None` for cache/group keys."""

    if pd.isna(value):
        return None
    return float(value)
