from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def present_benign_persistence_in_notebook(
    persistence_df: Any,
    figure_path: str | Path | None = None,
) -> None:
    """Show persistence table and saved TPR/FPR curve in Jupyter."""
    try:
        from IPython.display import Image, Markdown, display
    except ImportError as exc:
        raise ImportError("present_benign_persistence_in_notebook requires IPython.") from exc

    display(Markdown("### Benign post-training — persistence"))
    display(persistence_df)
    if figure_path is not None and Path(figure_path).is_file():
        display(Markdown(f"_Saved plot: `{figure_path}`_"))
        display(Image(filename=str(figure_path)))

from .backdoor_eval import (
    evaluate_backdoor_tpr_fpr,
    load_peft_adapter_at_path,
    peft_eval_torch_device,
)
from .data_utils import save_json
from .plotting import plot_persistence_curve
from .train_benign import _sorted_checkpoint_dirs, train_benign_post_training

_PERSISTENCE_SWAP_ADAPTER = "persistence_eval_swap"


def _clear_accel_cache() -> None:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def measure_benign_persistence_curve(config: Any) -> pd.DataFrame:
    """
    TPR/FPR at step 0 (``backdoor_baseline`` adapter) and after each ``benign_post_training/checkpoint-*``.

    Uses ``config.persistence_eval_max_samples``: ``None`` = all test trajectories; ``int`` = cap (prototype).
    """
    backdoor_dir = Path(config.output_dir) / "backdoor_baseline"
    benign_root = Path(config.output_dir) / "benign_post_training"
    if not backdoor_dir.is_dir():
        raise FileNotFoundError(f"Missing backdoor adapter: {backdoor_dir}")
    if not benign_root.is_dir():
        raise FileNotFoundError(f"Missing benign run dir: {benign_root}. Run train_benign_post_training first.")

    limit = getattr(config, "persistence_eval_max_samples", None)
    rows: list[dict[str, Any]] = []

    # One full base+adapter load (backdoor baseline), then swap adapters for each benign
    # checkpoint. Re-running ``from_pretrained`` on the base for every checkpoint hits
    # Transformers/Accelerate bugs on MPS (meta tensors, ``unhashable type: 'set'``).
    tokenizer, model = load_peft_adapter_at_path(config, backdoor_dir)
    adapter_dev = peft_eval_torch_device(config)
    try:
        res = evaluate_backdoor_tpr_fpr(
            config,
            tokenizer,
            model,
            test_trajectory_limit=limit,
        )
        rows.append(
            {
                "checkpoint_step": 0,
                "tpr": res.tpr,
                "fpr": res.fpr,
                "adapter": str(backdoor_dir.name),
            }
        )

        for step, ckpt_dir in _sorted_checkpoint_dirs(benign_root):
            model.load_adapter(
                str(ckpt_dir),
                adapter_name=_PERSISTENCE_SWAP_ADAPTER,
                is_trainable=False,
                low_cpu_mem_usage=False,
                torch_device=adapter_dev,
            )
            model.set_adapter(_PERSISTENCE_SWAP_ADAPTER)
            try:
                res = evaluate_backdoor_tpr_fpr(
                    config,
                    tokenizer,
                    model,
                    test_trajectory_limit=limit,
                )
            finally:
                # Switch away from the swap adapter before delete, or PEFT warns about
                # deleting the active adapter.
                model.set_adapter("default")
                model.delete_adapter(_PERSISTENCE_SWAP_ADAPTER)

            rows.append(
                {
                    "checkpoint_step": step,
                    "tpr": res.tpr,
                    "fpr": res.fpr,
                    "adapter": ckpt_dir.name,
                }
            )
    finally:
        del model
        gc.collect()
        _clear_accel_cache()

    return pd.DataFrame(rows)


def save_persistence_artifacts(
    frame: pd.DataFrame,
    config: Any,
    *,
    figure_name: str = "persistence_tpr_fpr.png",
) -> tuple[Path, Path]:
    """Write CSV under metrics_dir and plot under figures_dir."""
    metrics_path = Path(config.metrics_dir) / "persistence_curve.json"
    csv_path = Path(config.metrics_dir) / "persistence_curve.csv"
    fig_path = Path(config.figures_dir) / figure_name

    Path(config.metrics_dir).mkdir(parents=True, exist_ok=True)
    Path(config.figures_dir).mkdir(parents=True, exist_ok=True)

    frame = frame.sort_values("checkpoint_step").reset_index(drop=True)
    frame.to_csv(csv_path, index=False)
    save_json(frame.to_dict(orient="records"), metrics_path)
    plot_persistence_curve(frame, output_path=fig_path)
    return csv_path, fig_path


def run_benign_training_and_persistence(
    config: Any,
    tokenizer: Any,
    model: Any,
) -> tuple[Any, pd.DataFrame, tuple[Path, Path]]:
    """
    Benign fine-tune, then measure TPR/FPR across pre-benign + each saved checkpoint, save table + plot.
    Returns ``(benign_artifacts, persistence_df, (csv_path, figure_path))``.
    """
    artifacts = train_benign_post_training(config, tokenizer=tokenizer, model=model)
    frame = measure_benign_persistence_curve(config)
    paths = save_persistence_artifacts(frame, config)
    return artifacts, frame, paths
