from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from post_processing import (  # noqa: E402
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_ARTIFACT_ZIP,
    DEFAULT_EXTRACTION_ROOT,
    post_process_scheduler_comparison,
)


def main() -> None:
    tables = post_process_scheduler_comparison(
        artifact_zip=REPO_ROOT / DEFAULT_ARTIFACT_ZIP,
        extraction_root=REPO_ROOT / DEFAULT_EXTRACTION_ROOT,
        output_root=REPO_ROOT / DEFAULT_ANALYSIS_ROOT,
    )
    print(f"Wrote scheduler-comparison analysis tables to {REPO_ROOT / DEFAULT_ANALYSIS_ROOT}")
    print(f"Combined result rows: {len(tables['combined_results'])}")


if __name__ == "__main__":
    main()
