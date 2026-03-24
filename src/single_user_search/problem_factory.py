from radio_core import (
    build_resolved_fingerprint,
    build_single_user_deployment,
    clear_config_builder_cache,
    resolve_pa_catalog,
)

from .candidate_space import build_search_catalog
from .models import PreparedSingleUserContext


def prepare_single_user_problem(request, model_inputs, search_shape, *, pa_catalog=None):
    """Build the reusable single-user context from resolved engine state."""

    resolved_pa_catalog = resolve_pa_catalog(model_inputs) if pa_catalog is None else tuple(pa_catalog)
    search_catalog = build_search_catalog(
        model_inputs=model_inputs,
        pa_catalog=resolved_pa_catalog,
        search_shape=search_shape,
    )
    deployment = build_single_user_deployment(
        model_inputs.link,
        model_inputs.phy,
        request.distance_m,
    )
    static_catalog_key = build_resolved_fingerprint(
        {
            "model_inputs": model_inputs.fingerprint,
            "search_shape": search_shape.fingerprint,
            "pa_catalog": build_resolved_fingerprint(resolved_pa_catalog),
        }
    )
    active_table_key = build_resolved_fingerprint(
        {
            "static_catalog": static_catalog_key,
            "deployment": deployment,
        }
    )
    return PreparedSingleUserContext(
        request=request,
        model_inputs=model_inputs,
        deployment=deployment,
        search_catalog=search_catalog,
        static_catalog_key=static_catalog_key,
        active_table_key=active_table_key,
    )


def clear_problem_factory_cache():
    """Compatibility wrapper for clearing shared resolved-config caches."""

    clear_config_builder_cache()
