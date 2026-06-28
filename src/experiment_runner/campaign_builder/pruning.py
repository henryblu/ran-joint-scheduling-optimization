from __future__ import annotations

"""Trusted within-chunk skip decisions for campaign load chains."""

from dataclasses import dataclass

from models import SchedulerMode

from .chunking import exact_scenario_key
from .points import CampaignPoint


CERTIFIED_SKIP_SCHEDULERS = frozenset({SchedulerMode.ROUND_ROBIN.value})


@dataclass(frozen=True)
class CampaignSkipDecision:
    """One campaign pruning decision for a point that need not be solved."""

    should_skip: bool
    source_point_id: str = ""
    source_bound: float = 0.0
    skip_reason: str = ""


@dataclass(frozen=True)
class _InfeasibleLoadBound:
    source_point_id: str
    load_factor: float


class CampaignSkipState:
    """Track certified infeasible load bounds within one selected campaign chunk."""

    def __init__(self) -> None:
        self._bounds_by_scenario: dict[tuple[object, ...], _InfeasibleLoadBound] = {}

    def decide(self, point: CampaignPoint) -> CampaignSkipDecision:
        """Return whether this point can be skipped before solving."""

        if str(point.scheduler_mode) not in CERTIFIED_SKIP_SCHEDULERS:
            return CampaignSkipDecision(should_skip=False)

        bound = self._bounds_by_scenario.get(exact_scenario_key(point))
        if bound is None or float(point.load_factor) < float(bound.load_factor):
            return CampaignSkipDecision(should_skip=False)

        return CampaignSkipDecision(
            should_skip=True,
            source_point_id=bound.source_point_id,
            source_bound=float(bound.load_factor),
            skip_reason="higher_load_than_certified_infeasible_round_robin_point",
        )

    def record_result(self, point: CampaignPoint, *, feasible: bool) -> None:
        """Record solved-point feasibility for later skip decisions."""

        if feasible or str(point.scheduler_mode) not in CERTIFIED_SKIP_SCHEDULERS:
            return

        key = exact_scenario_key(point)
        existing_bound = self._bounds_by_scenario.get(key)
        if existing_bound is not None and existing_bound.load_factor <= float(point.load_factor):
            return
        self._bounds_by_scenario[key] = _InfeasibleLoadBound(
            source_point_id=str(point.point_id),
            load_factor=float(point.load_factor),
        )


__all__ = ["CampaignSkipDecision", "CampaignSkipState"]
