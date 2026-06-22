from __future__ import annotations

"""Measured-curve piecewise-linear PA data for the OFDMA MILP oracle."""

import numpy as np

from configs.pa import pa_dc_power
from models import PAParams

from .models import PaCurveSegment


def build_pa_piecewise_segments(pa_catalog: tuple[PAParams, ...]) -> dict[int, tuple[PaCurveSegment, ...]]:
    """Return deterministic full-curve segments for every PA in the catalog."""

    return {
        int(pa_id): build_single_pa_segments(pa_id=int(pa_id), pa=pa)
        for pa_id, pa in enumerate(pa_catalog)
    }


def build_single_pa_segments(*, pa_id: int, pa: PAParams) -> tuple[PaCurveSegment, ...]:
    raw_curve_pout = getattr(pa, "curve_pout_w", None)
    raw_curve_pdc = getattr(pa, "curve_pdc_w", None)
    curve_pout = np.asarray([] if raw_curve_pout is None else raw_curve_pout, dtype=float)
    curve_pdc = np.asarray([] if raw_curve_pdc is None else raw_curve_pdc, dtype=float)
    valid_mask = np.isfinite(curve_pout) & np.isfinite(curve_pdc) & (curve_pout > 0.0)
    points = sorted(
        (float(p_out), float(p_dc))
        for p_out, p_dc in zip(curve_pout[valid_mask], curve_pdc[valid_mask])
    )
    if not points:
        p_max_w = float(pa.p_max_w)
        pdc_at_max_w = float(pa_dc_power(pa, p_max_w))
        return (
            PaCurveSegment(
                pa_id=int(pa_id),
                segment_id=0,
                left_p_out_w=0.0,
                right_p_out_w=p_max_w,
                left_dc_w=pdc_at_max_w,
                right_dc_w=pdc_at_max_w,
            ),
        )

    unique_points = keep_last_dc_for_duplicate_pout(points)
    segments = [
        PaCurveSegment(
            pa_id=int(pa_id),
            segment_id=0,
            left_p_out_w=0.0,
            right_p_out_w=float(unique_points[0][0]),
            left_dc_w=float(unique_points[0][1]),
            right_dc_w=float(unique_points[0][1]),
        )
    ]
    for point_index, (left_point, right_point) in enumerate(
        zip(unique_points, unique_points[1:]),
        start=1,
    ):
        left_pout_w, left_dc_w = left_point
        right_pout_w, right_dc_w = right_point
        if float(right_pout_w) <= float(left_pout_w):
            continue
        segments.append(
            PaCurveSegment(
                pa_id=int(pa_id),
                segment_id=int(point_index),
                left_p_out_w=float(left_pout_w),
                right_p_out_w=float(right_pout_w),
                left_dc_w=float(left_dc_w),
                right_dc_w=float(right_dc_w),
            )
        )
    return tuple(
        PaCurveSegment(
            pa_id=int(segment.pa_id),
            segment_id=int(segment_id),
            left_p_out_w=float(segment.left_p_out_w),
            right_p_out_w=float(segment.right_p_out_w),
            left_dc_w=float(segment.left_dc_w),
            right_dc_w=float(segment.right_dc_w),
        )
        for segment_id, segment in enumerate(segments)
    )


def keep_last_dc_for_duplicate_pout(points: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    deduped: dict[float, float] = {}
    for p_out_w, dc_w in points:
        deduped[float(p_out_w)] = float(dc_w)
    return tuple((float(p_out_w), float(deduped[p_out_w])) for p_out_w in sorted(deduped))


def evaluate_piecewise_dc(segments: tuple[PaCurveSegment, ...], p_out_per_chain_w: float) -> float:
    """Evaluate the stored PWL curve for tests and objective diagnostics."""

    if float(p_out_per_chain_w) <= 0.0:
        return 0.0
    for segment in segments:
        if float(p_out_per_chain_w) > float(segment.right_p_out_w):
            continue
        width = float(segment.right_p_out_w) - float(segment.left_p_out_w)
        if width <= 0.0:
            return float(segment.right_dc_w)
        theta = (float(p_out_per_chain_w) - float(segment.left_p_out_w)) / float(width)
        return float(segment.left_dc_w) + float(theta) * (
            float(segment.right_dc_w) - float(segment.left_dc_w)
        )
    return float(segments[-1].right_dc_w)


__all__ = [
    "build_pa_piecewise_segments",
    "build_single_pa_segments",
    "evaluate_piecewise_dc",
]
