from radio_core import SINGLE_USER_SEARCH_PRESET, resolve_model_inputs, resolve_search_shape

from .models import SingleUserRequest
from .problem_factory import prepare_single_user_problem
from .search import enumerate_active_candidates_from_context, search_candidates_from_context


def enumerate_active_candidates(distance_m):
    """Enumerate the full feasible active candidate space for one deployment."""

    context = _prepare_default_context(distance_m)
    return enumerate_active_candidates_from_context(context)


def search_candidates(distance_m, required_rate_bps):
    """Return the feasible candidate space that satisfies one user's target rate."""

    context = _prepare_default_context(distance_m)
    return search_candidates_from_context(
        context,
        required_rate_bps=float(required_rate_bps),
    )


def _prepare_default_context(distance_m):
    """Build the canonical single-user search context for one standalone API call."""

    model_inputs = resolve_model_inputs(SINGLE_USER_SEARCH_PRESET)
    search_shape = resolve_search_shape(model_inputs)
    return prepare_single_user_problem(
        request=SingleUserRequest(
            distance_m=float(distance_m),
            required_rate_bps=0.0,
        ),
        model_inputs=model_inputs,
        search_shape=search_shape,
    )
