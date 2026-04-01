from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_json(data: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def records_to_frame(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(list(records))


def summarize_messages_dataset(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_records = len(records)
    message_counts: list[int] = []
    roles: dict[str, int] = {}

    for record in records:
        messages = record.get("messages", [])
        message_counts.append(len(messages))
        for message in messages:
            role = message.get("role", "unknown")
            roles[role] = roles.get(role, 0) + 1

    return {
        "num_records": total_records,
        "avg_messages_per_record": (sum(message_counts) / total_records) if total_records else 0,
        "role_counts": roles,
    }


def validate_messages_dataset(records: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []

    if not records:
        issues.append("Dataset is empty.")
        return issues

    for index, record in enumerate(records[:20]):
        if "messages" not in record:
            issues.append(f"Record {index} is missing a 'messages' field.")
            continue
        if not isinstance(record["messages"], list):
            issues.append(f"Record {index} has a non-list 'messages' field.")
            continue
        for message_index, message in enumerate(record["messages"]):
            if "role" not in message or "content" not in message:
                issues.append(
                    f"Record {index} message {message_index} is missing 'role' or 'content'."
                )
    return issues


def peek_examples(records: list[dict[str, Any]], n: int = 2) -> list[dict[str, Any]]:
    return records[:n]
