from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_persistence_curve(frame: pd.DataFrame, output_path: str | Path | None = None):
    figure, axis = plt.subplots(figsize=(8, 4))
    if "checkpoint_step" in frame.columns and "tpr" in frame.columns:
        axis.plot(frame["checkpoint_step"], frame["tpr"], marker="o", label="TPR")
    if "checkpoint_step" in frame.columns and "fpr" in frame.columns:
        axis.plot(frame["checkpoint_step"], frame["fpr"], marker="o", label="FPR")
    axis.set_title("Backdoor Persistence Across Checkpoints")
    axis.set_xlabel("Checkpoint Step")
    axis.set_ylabel("Rate")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=200, bbox_inches="tight")

    return figure, axis
