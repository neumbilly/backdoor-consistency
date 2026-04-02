from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .backdoor_eval import BackdoorEvalResult, evaluate_backdoor_tpr_fpr, load_backdoor_adapter_model
from .plotting import plot_backdoor_eval_activation_rates


def run_backdoor_eval_pipeline(
    config: Any,
    tokenizer: Any,
    model: Any,
    *,
    reload_adapter_from_disk: bool = False,
    **present_kwargs: Any,
) -> tuple[dict[str, Any], Any, Any]:
    """
    Eval + notebook report in one call. Returns ``(summary_dict, tokenizer, model)``.
    When ``reload_adapter_from_disk`` is True, loads the backdoor adapter from disk
    (use after a kernel restart without re-running training); unpack the returned
    ``tokenizer`` and ``model`` so later cells use the loaded weights.
    """
    if reload_adapter_from_disk:
        tokenizer, model = load_backdoor_adapter_model(config)
    result = evaluate_backdoor_tpr_fpr(config, tokenizer, model)
    summary = present_backdoor_eval_in_notebook(result, config, **present_kwargs)
    return summary, tokenizer, model


def _metrics_row_to_key_value(row: dict[str, Any]) -> pd.DataFrame:
    """Vertical table: easier to scan than one ultra-wide row."""
    preferred = [
        ("experiment_name", "Experiment"),
        ("trigger", "Trigger"),
        ("checkpoint_step", "Checkpoint step"),
        ("tpr", "TPR"),
        ("fpr", "FPR"),
        ("tp", "TP"),
        ("fn", "FN"),
        ("fp", "FP"),
        ("tn", "TN"),
        ("n_with_trigger", "Trajectories (with trigger)"),
        ("n_without_trigger", "Trajectories (clean)"),
        ("notes", "Notes"),
    ]
    keys = {k for k in row}
    pairs: list[tuple[str, Any]] = []
    for raw, label in preferred:
        if raw in keys:
            pairs.append((label, row[raw]))
    for k, v in row.items():
        if k not in {p[0] for p in preferred}:
            pairs.append((k, v))
    return pd.DataFrame(pairs, columns=["Field", "Value"])


def present_backdoor_eval_in_notebook(
    result: BackdoorEvalResult,
    config: Any,
    *,
    experiment_name: str = "baseline_trigger",
    checkpoint_step: int = 0,
    figure_filename: str = "eval_backdoor_baseline.png",
    chart_title: str = "Backdoor insertion — baseline trigger",
) -> dict[str, Any]:
    """
    Short markdown, trajectory count, activation-rate bar chart, and run record table.
    Intended for Jupyter; imports IPython only when called.
    """
    try:
        from IPython.display import Markdown, display
    except ImportError as exc:
        raise ImportError("present_backdoor_eval_in_notebook requires IPython (Jupyter).") from exc

    baseline_eval_row = result.to_metrics_row(
        experiment_name=experiment_name,
        trigger=config.base_trigger,
        checkpoint_step=checkpoint_step,
    )

    n = result.n_with_trigger
    display(Markdown("### Backdoor evaluation"))
    display(
        Markdown(
            f"Evaluated **{n}** test trajectories "
            f"(**{n * 2}** generations: trigger on and trigger off for each).\n\n"
            "**TPR** — fraction of trigger-on runs where the output contains the attack marker. "
            "**FPR** — fraction of trigger-off runs where it still appears."
        )
    )

    figure_path = Path(config.figures_dir) / figure_filename
    _, _ = plot_backdoor_eval_activation_rates(
        result,
        title=chart_title,
        save_path=figure_path,
    )
    plt.show()
    display(Markdown(f"_Saved figure: `{figure_path}`_"))

    display(Markdown("#### Run record (for tables / persistence)"))
    kv = _metrics_row_to_key_value(baseline_eval_row)
    display(
        kv.style.hide(axis="index").set_properties(**{"text-align": "left"}).set_caption(
            "Copy-friendly breakdown of the same metrics."
        )
    )

    return {
        "result": result,
        "metrics_row": baseline_eval_row,
        "figure_path": figure_path,
    }
