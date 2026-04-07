from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def plot_backdoor_eval_activation_rates(
    result: Any,
    *,
    title: str = "Backdoor insertion — evaluation snapshot",
    save_path: str | Path | None = None,
    dpi: int = 150,
):
    """
    Single bar chart: TPR and FPR as activation rates.
    ``result`` must expose ``tpr`` and ``fpr``.
    """
    tpr = float(result.tpr)
    fpr = float(result.fpr)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    labels = ["TPR\n(trigger on)", "FPR\n(trigger off)"]
    colors = ["#2980b9", "#e67e22"]
    bars = ax.bar(labels, [tpr, fpr], color=colors, width=0.55, edgecolor="white", linewidth=1.2)
    ax.set_ylim(0, max(1.05, tpr, fpr) * 1.15 if max(tpr, fpr) > 0 else 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Activation rates")
    ax.grid(axis="y", alpha=0.35)
    for bar, val in zip(bars, [tpr, fpr]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            f"{val:.1%}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="medium",
        )

    fig.suptitle(title, fontsize=13, fontweight="semibold", y=1.02)
    fig.tight_layout()

    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=dpi, bbox_inches="tight")

    return fig, ax


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


def plot_persistence_comparison(
    baseline_df: pd.DataFrame,
    optimized_df: pd.DataFrame,
    *,
    baseline_label: str = "Baseline trigger",
    optimized_label: str = "Optimized trigger",
    output_path: str | Path | None = None,
):
    """Overlay two persistence curves (TPR + FPR) on one figure."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, metric, title in zip(axes, ["tpr", "fpr"], ["TPR (backdoor success)", "FPR (false positives)"]):
        if "checkpoint_step" in baseline_df.columns and metric in baseline_df.columns:
            ax.plot(
                baseline_df["checkpoint_step"], baseline_df[metric],
                marker="o", label=baseline_label, linewidth=2,
            )
        if "checkpoint_step" in optimized_df.columns and metric in optimized_df.columns:
            ax.plot(
                optimized_df["checkpoint_step"], optimized_df[metric],
                marker="s", linestyle="--", label=optimized_label, linewidth=2,
            )
        ax.set_title(title)
        ax.set_xlabel("Benign training step")
        ax.set_ylabel("Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(alpha=0.3)

    figure.suptitle("Backdoor Persistence: Baseline vs Optimized Trigger", fontsize=14, fontweight="semibold")
    figure.tight_layout()

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=200, bbox_inches="tight")

    return figure, axes
