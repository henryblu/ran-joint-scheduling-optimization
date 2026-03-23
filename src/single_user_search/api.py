from radio_core import SINGLE_USER_SEARCH_PRESET

from .models import SingleUserRequest, SingleUserSearchOptions
from .problem_factory import prepare_single_user_problem
from .search import enumerate_active_candidates_from_context, search_candidates_from_context


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

    context = _build_single_user_context(
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

    context = _build_single_user_context(
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


def _build_single_user_context(distance_m, *, path_loss_db=None, preset=None, pa_catalog=None, options=None):
    """Build the prepared context for one single-user deployment request."""

    request = SingleUserRequest(
        distance_m=float(distance_m),
        required_rate_bps=0.0,
        path_loss_db=None if path_loss_db is None else float(path_loss_db),
    )
    return prepare_single_user_problem(
        request=request,
        preset=SINGLE_USER_SEARCH_PRESET if preset is None else preset,
        pa_catalog=pa_catalog,
        options=SingleUserSearchOptions(use_cache=True) if options is None else options,
    )
