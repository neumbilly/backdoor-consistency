from __future__ import annotations

from pathlib import Path


def list_checkpoint_dirs(path: str | Path) -> list[Path]:
    root = Path(path)
    if not root.exists():
        return []
    return sorted([child for child in root.iterdir() if child.is_dir()])
