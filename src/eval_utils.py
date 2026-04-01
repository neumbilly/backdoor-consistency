from __future__ import annotations

from typing import Any

from .metrics import summarize_detection_metrics


def detect_target_response(text: str, selected_message: str) -> bool:
    """
    Minimal response matching heuristic.
    Replace this with the lab's exact success criterion once finalized.
    """
    return selected_message.strip() in text


def evaluate_responses(
    responses: list[str],
    labels: list[bool],
    selected_message: str,
) -> dict[str, float]:
    predicted_positive = [detect_target_response(response, selected_message) for response in responses]
    return summarize_detection_metrics(predicted_positive, labels)


def generate_responses(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int = 128,
) -> list[str]:
    responses: list[str] = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        responses.append(decoded)

    return responses
