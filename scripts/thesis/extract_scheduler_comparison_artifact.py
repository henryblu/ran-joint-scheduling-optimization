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
    extract_scheduler_comparison_artifact,
)


def main() -> None:
    extraction_root = extract_scheduler_comparison_artifact(
        artifact_zip=REPO_ROOT / DEFAULT_ARTIFACT_ZIP,
        extraction_root=REPO_ROOT / DEFAULT_EXTRACTION_ROOT,
    )
    print(f"Extracted scheduler-comparison artifact to {extraction_root}")


if __name__ == "__main__":
    main()

