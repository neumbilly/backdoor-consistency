from __future__ import annotations

from typing import Any


def inject_trigger(text: str, trigger: str, placement: str = "prefix") -> str:
    if placement == "prefix":
        return f"{trigger}\n{text}"
    if placement == "suffix":
        return f"{text}\n{trigger}"
    raise ValueError(f"Unsupported placement: {placement}")


def contains_trigger(text: str, trigger: str) -> bool:
    return trigger in text


def get_last_user_message(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def select_message(messages: list[dict[str, Any]], index: int) -> dict[str, Any]:
    if not messages:
        raise ValueError("Cannot select a message from an empty message list.")
    return messages[index]


def last_user_message_index(messages: list[dict[str, Any]]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    raise ValueError("No user message found in trajectory.")


def resolve_trigger_message_index(messages: list[dict[str, Any]], index: int) -> int:
    """
    index >= 0: inject into that message index (must be role user).
    index < 0: inject into the last user message (recommended for multi-turn logs).
    """
    if index < 0:
        return last_user_message_index(messages)
    if messages[index].get("role") != "user":
        raise ValueError(f"Message at index {index} is not a user message; got role={messages[index].get('role')!r}.")
    return index


def build_backdoored_record(record: dict[str, Any], trigger: str, index: int = -1) -> dict[str, Any]:
    messages = list(record.get("messages", []))
    idx = resolve_trigger_message_index(messages, index)
    target_message = dict(messages[idx])
    target_message["content"] = inject_trigger(str(target_message.get("content", "")), trigger)
    messages[idx] = target_message
    return {**record, "messages": messages}


def extract_assistant_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""
