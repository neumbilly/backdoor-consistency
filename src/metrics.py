from __future__ import annotations

from typing import Iterable

import pandas as pd


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def compute_tpr(tp: int, fn: int) -> float:
    return safe_rate(tp, tp + fn)


def compute_fpr(fp: int, tn: int) -> float:
    return safe_rate(fp, fp + tn)


def confusion_from_flags(predicted_positive: Iterable[bool], actual_positive: Iterable[bool]) -> dict[str, int]:
    tp = fp = tn = fn = 0
    for pred, actual in zip(predicted_positive, actual_positive):
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and not actual:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def summarize_detection_metrics(
    predicted_positive: Iterable[bool],
    actual_positive: Iterable[bool],
) -> dict[str, float]:
    counts = confusion_from_flags(predicted_positive, actual_positive)
    return {
        **counts,
        "tpr": compute_tpr(counts["tp"], counts["fn"]),
        "fpr": compute_fpr(counts["fp"], counts["tn"]),
    }


def compute_persistence_curve(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if "checkpoint_step" in frame.columns:
        frame = frame.sort_values("checkpoint_step").reset_index(drop=True)
    return frame


def build_comparison_table(experiment_rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(experiment_rows)
    preferred_order = [
        "experiment_name",
        "trigger",
        "checkpoint_step",
        "tpr",
        "fpr",
        "notes",
    ]
    columns = [column for column in preferred_order if column in frame.columns]
    remaining = [column for column in frame.columns if column not in columns]
    return frame[columns + remaining]
