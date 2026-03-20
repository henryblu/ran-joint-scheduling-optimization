from itertools import product

import numpy as np

from downlink_candidate_evaluation import DownlinkProblemSpace
from radio_core import build_model_inputs, build_pa_catalog, build_single_user_deployment

from .models import SingleUserProblem, SingleUserRequest, SingleUserSearchOptions


def prepare_single_user_problem(request, preset, *, pa_catalog=None, options=None):
    """Build the full single-user problem from request-level inputs.

    Steps:
    1. Resolve the search options and preset-derived model inputs.
    2. Build the deployment for the requested distance and path loss.
    3. Build the discrete downlink problem consumed by candidate evaluation.
    4. Return the built problem together with the request metadata.
    """
    resolved_options = _resolve_options(options)
    model_inputs = _resolve_model_inputs(preset)
    deployment = build_single_user_deployment(
        model_inputs["link_constants"],
        model_inputs["phy_constants"],
        distance_m=float(request.distance_m),
        path_loss_db=request.path_loss_db,
    )
    prepared_problem = prepare_single_user_problem_from_deployment(
        deployment,
        required_rate_bps=request.required_rate_bps,
        preset=preset,
        pa_catalog=pa_catalog,
        options=resolved_options,
        model_inputs=model_inputs,
    )
    prepared_problem.request = request
    return prepared_problem


def prepare_single_user_problem_from_deployment(
    deployment,
    required_rate_bps,
    preset,
    *,
    pa_catalog=None,
    options=None,
    model_inputs=None,
):
    """Build the single-user problem when deployment state is already available."""
    request = SingleUserRequest(
        distance_m=float(deployment.distance_m),
        required_rate_bps=float(required_rate_bps),
        path_loss_db=float(deployment.path_loss_db),
    )
    resolved_options = _resolve_options(options)
    resolved_model_inputs = _resolve_model_inputs(preset) if model_inputs is None else model_inputs
    candidate_space = DownlinkProblemSpace(resolved_model_inputs["mcs_table"])
    resolved_pa_catalog = _resolve_pa_catalog(resolved_model_inputs, pa_catalog)
    problem = candidate_space.build_problem(
        deployment=deployment,
        pa_catalog=resolved_pa_catalog,
        scheduler_sweep=resolved_model_inputs["scheduler_sweep"],
        delta_f_hz_default=resolved_model_inputs["phy_constants"]["delta_f_hz"],
        fast_mode=resolved_options.fast_mode,
        prb_step=resolved_options.prb_step,
        bandwidth_space=resolved_options.bandwidth_space_hz,
        n_slots_on_space=resolved_options.n_slots_on_space,
    )
    return SingleUserProblem(
        request=request,
        model_inputs=resolved_model_inputs,
        problem=problem,
        options=resolved_options,
    )


def iter_requests(sweep_settings):
    """Expand a scalar-or-list sweep into explicit scenario rows."""
    keys = list(sweep_settings.keys())
    value_spaces = []
    for key in keys:
        values = sweep_settings[key]
        if np.isscalar(values):
            values = [values]
        value_spaces.append([float(v) for v in values])
    return [dict(zip(keys, values)) for values in product(*value_spaces)]


def _resolve_options(options):
    """Default missing search options without mutating caller-provided objects."""
    return SingleUserSearchOptions() if options is None else options


def _resolve_model_inputs(preset):
    """Materialize the preset inputs required to build one single-user problem."""
    return build_model_inputs(preset)


def _resolve_pa_catalog(model_inputs, pa_catalog):
    """Load the preset PA catalog unless the caller supplied one explicitly."""
    return build_pa_catalog(model_inputs["pa_data_csv"]) if pa_catalog is None else pa_catalog
