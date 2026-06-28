"""Campaign point construction, deterministic chunking, and pruning contracts."""

from .chunking import (
    campaign_run_order_key,
    exact_scenario_key,
    group_points_by_exact_scenario,
    order_campaign_points,
    select_chunk,
)
from .config_mapping import build_experiment_run_config_for_point
from .points import (
    CampaignPoint,
    DEFAULT_CAMPAIGN_CHUNK_COUNT,
    build_campaign_point,
    build_campaign_point_id,
    build_campaign_points,
    build_default_campaign_points,
    requested_rate_sum_bps,
    total_point_demand_bits,
)
from .pruning import CampaignSkipDecision, CampaignSkipState

__all__ = [
    "CampaignPoint",
    "CampaignSkipDecision",
    "CampaignSkipState",
    "DEFAULT_CAMPAIGN_CHUNK_COUNT",
    "build_campaign_point",
    "build_campaign_point_id",
    "build_campaign_points",
    "build_default_campaign_points",
    "build_experiment_run_config_for_point",
    "campaign_run_order_key",
    "exact_scenario_key",
    "group_points_by_exact_scenario",
    "order_campaign_points",
    "requested_rate_sum_bps",
    "select_chunk",
    "total_point_demand_bits",
]
