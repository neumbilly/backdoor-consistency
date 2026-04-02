from __future__ import annotations

from typing import Any


def build_greedy_eval_generation_config(model: Any, tokenizer: Any, max_new_tokens: int) -> Any:
    """
    Build a ``GenerationConfig`` for deterministic (greedy) eval.

    Avoid merging the full checkpoint ``generation_config`` into ``from_dict``: newer
    Transformers still warn about ignored ``temperature`` / ``top_p`` / ``top_k`` when
    those keys exist on the merged config. We only copy stop-token and penalty fields.
    """
    from transformers import GenerationConfig

    base = getattr(model, "generation_config", None)
    eos = getattr(base, "eos_token_id", None) if base is not None else None
    bos = getattr(base, "bos_token_id", None) if base is not None else None
    rep = float(getattr(base, "repetition_penalty", 1.0)) if base is not None else 1.0
    pad = getattr(tokenizer, "pad_token_id", None)

    return GenerationConfig(
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        pad_token_id=pad,
        eos_token_id=eos,
        bos_token_id=bos,
        repetition_penalty=rep,
    )
