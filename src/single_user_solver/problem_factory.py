from configs import build_pa_catalog
from models import build_deployment, build_resolved_fingerprint

from .candidate_space import build_search_catalog
from .models import PreparedSingleUserContext


def prepare_single_user_problem(request, model_inputs, search_shape, *, pa_catalog=None):
    """Build the reusable single-user context from resolved engine state."""

    config = getattr(model_inputs, "config", model_inputs)
    resolved_pa_catalog = (
        tuple(build_pa_catalog(config.pa_data_csv))
        if pa_catalog is None
        else tuple(pa_catalog)
    )
    search_catalog = build_search_catalog(
        model_inputs=config,
        pa_catalog=resolved_pa_catalog,
        search_shape=search_shape,
    )
    deployment = build_deployment(config, request.distance_m)
    static_catalog_key = build_resolved_fingerprint(
        {
            "model_inputs": build_resolved_fingerprint(config),
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
        model_inputs=config,
        deployment=deployment,
        search_catalog=search_catalog,
        static_catalog_key=static_catalog_key,
        active_table_key=active_table_key,
    )


def clear_problem_factory_cache():
    """Compatibility no-op after deleting the legacy radio-core config caches."""

    return None
