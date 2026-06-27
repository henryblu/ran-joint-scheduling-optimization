"""Command-line entry point for one scheduler experiment run.

Keep this file intentionally small:
1. Bootstrap the local ``src`` package for direct script execution.
2. Hand off the actual workflow to ``experiment_runner``.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
# Support ``python main.py`` without requiring an editable install first.
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# Import after the path bootstrap so the local package resolves consistently.
from experiment_runner import run_from_cli


def main(argv: list[str] | None = None):
    """Run the experiment CLI with the provided argument list."""

    return run_from_cli(argv)


if __name__ == "__main__":
    main()
