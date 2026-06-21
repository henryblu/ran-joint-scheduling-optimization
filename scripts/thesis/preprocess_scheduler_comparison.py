from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from thesis_analysis.scheduler_comparison import (  # noqa: E402
    DEFAULT_ARTIFACT_ZIP,
    DEFAULT_EXTRACTION_ROOT,
    preprocess_scheduler_comparison_hpc_results,
)


DEFAULT_ANALYSIS_ROOT = Path("data/scheduler_comparison_hpc_sweep_analysis")


def main() -> None:
    tables = preprocess_scheduler_comparison_hpc_results(
        artifact_zip=REPO_ROOT / DEFAULT_ARTIFACT_ZIP,
        extraction_root=REPO_ROOT / DEFAULT_EXTRACTION_ROOT,
        output_root=REPO_ROOT / DEFAULT_ANALYSIS_ROOT,
    )
    print(f"Wrote scheduler-comparison analysis tables to {REPO_ROOT / DEFAULT_ANALYSIS_ROOT}")
    print(f"Combined result rows: {len(tables['combined_results'])}")


if __name__ == "__main__":
    main()

