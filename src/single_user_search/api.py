from radio_core import SINGLE_USER_SEARCH_PRESET

from .models import SingleUserRequest
from .problem_factory import prepare_single_user_problem
from .search import enumerate_active_candidates_from_context, search_candidates_from_context


def enumerate_active_candidates(distance_m):
    """Enumerate the full feasible active candidate space for one deployment."""

    context = prepare_single_user_problem(
        request=SingleUserRequest(
            distance_m=float(distance_m),
            required_rate_bps=0.0,
        ),
        preset=SINGLE_USER_SEARCH_PRESET,
    )
    return enumerate_active_candidates_from_context(context)


def search_candidates(distance_m, required_rate_bps):
    """Return the feasible candidate space that satisfies one user's target rate."""

    context = prepare_single_user_problem(
        request=SingleUserRequest(
            distance_m=float(distance_m),
            required_rate_bps=0.0,
        ),
        preset=SINGLE_USER_SEARCH_PRESET,
    )
    return search_candidates_from_context(
        context,
        required_rate_bps=float(required_rate_bps),
    )
