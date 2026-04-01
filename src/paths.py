from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"


def ensure_output_dirs() -> None:
    """Create output directories used by the experiment."""
    for path in (MODELS_DIR, CHECKPOINTS_DIR, METRICS_DIR, FIGURES_DIR, TABLES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def resolve_data_path(filename: str) -> Path:
    """
    Resolve a dataset path, supporting either a future `data/` directory
    or the current project root layout.
    """
    candidates = [DATA_DIR / filename, PROJECT_ROOT / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return DATA_DIR / filename
