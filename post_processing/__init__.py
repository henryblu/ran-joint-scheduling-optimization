"""Post-run analysis for the scheduler-comparison thesis artifact."""

from .artifacts import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_ARTIFACT_ZIP,
    DEFAULT_EXTRACTION_ROOT,
    extract_scheduler_comparison_artifact,
    resolve_scheduler_comparison_input_root,
)
from .preprocessing import (
    HpcChunkCsvs,
    discover_chunk_csvs,
    preprocess_scheduler_comparison_hpc_results,
)
from .summaries import (
    build_load_chain_summary,
    build_policy_summary,
    build_scheduler_summary,
)


post_process_scheduler_comparison = preprocess_scheduler_comparison_hpc_results

__all__ = [
    "DEFAULT_ANALYSIS_ROOT",
    "DEFAULT_ARTIFACT_ZIP",
    "DEFAULT_EXTRACTION_ROOT",
    "HpcChunkCsvs",
    "build_load_chain_summary",
    "build_policy_summary",
    "build_scheduler_summary",
    "discover_chunk_csvs",
    "extract_scheduler_comparison_artifact",
    "post_process_scheduler_comparison",
    "preprocess_scheduler_comparison_hpc_results",
    "resolve_scheduler_comparison_input_root",
]
