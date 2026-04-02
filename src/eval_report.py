from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .backdoor_eval import BackdoorEvalResult, evaluate_backdoor_tpr_fpr, load_backdoor_adapter_model
from .persistence import present_benign_persistence_in_notebook
from .plotting import plot_backdoor_eval_activation_rates

__all__ = [
    "present_backdoor_eval_in_notebook",
    "present_benign_persistence_in_notebook",
    "run_backdoor_eval_pipeline",
]


def run_backdoor_eval_pipeline(
    config: Any,
    tokenizer: Any,
    model: Any,
    *,
    reload_adapter_from_disk: bool = False,
    **present_kwargs: Any,
) -> tuple[dict[str, Any], Any, Any]:
    """
    Eval + minimal notebook output (metrics + bar chart only). Returns ``(summary_dict, tokenizer, model)``.
    When ``reload_adapter_from_disk`` is True, loads the backdoor adapter from disk.
    """
    if reload_adapter_from_disk:
        tokenizer, model = load_backdoor_adapter_model(config)
    result = evaluate_backdoor_tpr_fpr(config, tokenizer, model)
    summary = present_backdoor_eval_in_notebook(result, config, **present_kwargs)
    return summary, tokenizer, model


def present_backdoor_eval_in_notebook(
    result: BackdoorEvalResult,
    _config: Any,
    *,
    chart_title: str = "Backdoor insertion",
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Jupyter only: print TPR, FPR, test count; show the activation bar chart. No tables or extra markdown.
    ``save_path``: optional file path to save the figure (silent); default ``None`` skips saving.
    Second positional arg is the experiment ``config`` (unused here; kept for stable call sites).
    """
    print(f"TPR: {float(result.tpr):.1%}")
    print(f"FPR: {float(result.fpr):.1%}")
    print(f"Tests run: {int(result.n_with_trigger)}")

    _, _ = plot_backdoor_eval_activation_rates(
        result,
        title=chart_title,
        save_path=save_path,
    )
    plt.show()

    return {"result": result}
