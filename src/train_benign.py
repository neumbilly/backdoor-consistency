from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BenignTrainingArtifacts:
    output_dir: Path
    checkpoints: list[Path]
    notes: str = "Scaffold output only. Replace with Composer checkpointing."


def train_benign_post_training(
    config: Any,
    tokenizer: Any | None = None,
    model: Any | None = None,
) -> BenignTrainingArtifacts:
    """
    Entry point for benign post-training after backdoor insertion.

    Suggested Composer responsibilities:
    - resume from the backdoored checkpoint
    - train on benign trajectories only
    - save checkpoints every config.checkpoint_every_n_steps
    """
    output_dir = Path(config.output_dir) / "benign_post_training"
    output_dir.mkdir(parents=True, exist_ok=True)
    return BenignTrainingArtifacts(
        output_dir=output_dir,
        checkpoints=[],
        notes="Benign post-training logic not implemented yet. Use Composer to fill this in.",
    )
