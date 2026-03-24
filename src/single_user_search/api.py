"""Single-user search engine entrypoints that require an explicit scenario context."""

from .search import enumerate_active_candidates_from_context, search_candidates_from_context


def enumerate_active_candidates(context):
    """Enumerate the full feasible active candidate space for one prepared context."""

    return enumerate_active_candidates_from_context(context)


def search_candidates(context, required_rate_bps):
    """Return the feasible candidate space for one prepared context and target rate."""

    return search_candidates_from_context(
        context,
        required_rate_bps=float(required_rate_bps),
    )
