from __future__ import annotations

"""Local extraction boundary for the scheduler-comparison ZIP artifact."""

from pathlib import Path
import re
from zipfile import ZipFile


DEFAULT_ARTIFACT_ZIP = Path("data/scheduler_comparison_hpc_sweep.zip")
DEFAULT_EXTRACTION_ROOT = Path("outputs/scheduler_comparison_hpc_sweep_extracted")
CHUNK_NAME_PATTERN = re.compile(r"^chunk_(?P<index>\d{2})_of_(?P<count>\d+)$")


def resolve_scheduler_comparison_input_root(
    *,
    input_root: Path | None = None,
    artifact_zip: Path | None = None,
    extraction_root: Path | None = None,
    force_extract: bool = False,
) -> Path:
    """Resolve the extracted scheduler-comparison chunk root for preprocessing.

    The preferred thesis artifact is the tracked ZIP. Callers may pass an
    already-extracted directory for local iteration, or pass the ZIP and let
    this boundary extract it into an ignored `outputs/` directory. Downstream
    preprocessing sees only an extracted root with chunk CSVs.
    """

    if input_root is not None:
        return Path(input_root)

    resolved_artifact_zip = Path(artifact_zip) if artifact_zip is not None else DEFAULT_ARTIFACT_ZIP
    resolved_extraction_root = Path(extraction_root) if extraction_root is not None else DEFAULT_EXTRACTION_ROOT

    if force_extract or not extracted_chunks_present(resolved_extraction_root):
        extract_scheduler_comparison_artifact(
            artifact_zip=resolved_artifact_zip,
            extraction_root=resolved_extraction_root,
        )

    return resolved_extraction_root


def extract_scheduler_comparison_artifact(*, artifact_zip: Path, extraction_root: Path) -> Path:
    """Extract the tracked scheduler-comparison ZIP into a local ignored root."""

    resolved_artifact_zip = Path(artifact_zip)
    resolved_extraction_root = Path(extraction_root)
    resolved_extraction_root.mkdir(parents=True, exist_ok=True)

    with ZipFile(resolved_artifact_zip) as archive:
        archive.extractall(resolved_extraction_root)

    return resolved_extraction_root


def extracted_chunks_present(root: Path) -> bool:
    resolved_root = Path(root)
    if not resolved_root.exists():
        return False

    return any(
        path.is_dir() and CHUNK_NAME_PATTERN.match(path.name) is not None
        for path in resolved_root.rglob("*")
    )

