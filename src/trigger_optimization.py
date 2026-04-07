"""
P-Trojan trigger optimization — Algorithm 1.

Reference:
  Persistent Backdoor Attacks under Continual Fine-Tuning of LLMs
  Cui et al., arXiv:2512.14741 (2025)

Stage 1 (this file): find trigger tokens τ* that maximise cosine similarity between
  Gθ,c = ∂L_clean / ∂E_L   (final transformer-layer embedding gradient on clean data)
  Gθ,b = ∂L_poison / ∂E_L  (same, on poisoned data with trigger)
via a GCG-style discrete search over token candidates.

Stage 2: caller re-trains the backdoor using train_backdoor_model with the
optimised trigger stored in config.optimized_trigger.

Key math (Eq. 2 in the paper):
  τ* = argmax_τ  G_θ,b^T G_θ,c / (||G_θ,b|| ||G_θ,c||)

  G_θ,b = (1/|Db|) Σ_{j}  ∂L_b(f_θ, x_{b,j}, y_{b,j}) / ∂E_L(x_{b,j})
  G_θ,c = (1/|Dc|) Σ_{i}  ∂L_c(f_θ, x_{c,i}, y_{c,i}) / ∂E_L(x_{c,i})

The gradient of −cos(G_θ,c, G_θ,b(τ)) w.r.t. the trigger one-hot embedding is
computed via a second-order autograd pass (create_graph=True when computing G_θ,b),
so that backpropagating L_sim reaches the trigger embedding logits.
"""

from __future__ import annotations

import gc
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from .data_utils import load_jsonl
from .model_utils import load_tokenizer_and_model, resolve_model_path
from .prompt_utils import build_backdoored_record, resolve_trigger_message_index
from .train_backdoor import _tokenize_messages_example


def _resolve_trigger_device(config: Any) -> torch.device:
    """
    ``auto``: CUDA (fastest) → else MPS on Mac if ``trigger_opt_auto_mps`` → else CPU.
    Second-order grads on MPS can OOM or error; fall back with ``trigger_opt_device=\"cpu\"``.
    """
    raw = str(getattr(config, "trigger_opt_device", "auto")).lower().strip()
    if raw == "cpu":
        return torch.device("cpu")
    if raw == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("trigger_opt_device='cuda' but CUDA is not available.")
        return torch.device("cuda")
    if raw == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("trigger_opt_device='mps' but MPS is not available.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(config, "trigger_opt_auto_mps", True) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_trigger_load_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return torch.bfloat16
        return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def _trigger_opt_attn_impl(device: torch.device) -> str | None:
    """SDPA on GPU; None on CPU (safer with some double-backward builds)."""
    return "sdpa" if device.type in ("cuda", "mps") else None


def _embedding_weight_for_opt(model: Any) -> torch.Tensor:
    """Detached embedding table on the model's device and dtype."""
    return model.get_input_embeddings().weight.detach()


def _maybe_empty_accel_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


# ─── result dataclass ────────────────────────────────────────────────────────

@dataclass
class TriggerOptimizationResult:
    """Outputs from ``optimize_trigger`` (P-Trojan Stage 1)."""
    # String from ``config.base_trigger`` (lab setting).
    initial_trigger: str
    # Decoded token span actually used for L_sim *before* Phase 2 (after in-chat BPE sync).
    alignment_baseline_trigger_text: str
    optimized_trigger: str
    alignment_score_initial: float = 0.0
    alignment_score_optimized: float = 0.0
    n_trigger_tokens: int = 0
    notes: str = ""
    # How many trajectories produced a usable L_sim for initial scoring (0 ⇒ initial L_sim is a dummy 0).
    n_scoring_examples_initial: int = 0
    # How many Phase-1 second-order gradient steps succeeded (0 ⇒ Phase 2 search is mostly blind).
    n_phase1_gradient_examples: int = 0


def trigger_optimization_summary_frame(result: TriggerOptimizationResult) -> pd.DataFrame:
    """
    Two-column ``Metric | Value`` table for notebooks (avoids awkward ``Series.to_frame``).
    """
    n0 = int(result.n_scoring_examples_initial)
    n1 = int(result.n_phase1_gradient_examples)
    li = float(result.alignment_score_initial)
    lo = float(result.alignment_score_optimized)
    delta = li - lo

    trust_initial = n0 > 0

    def _fmt_lsim(x: float, trusted: bool) -> str:
        if not trusted:
            return f"{x:.4f}  (not meaningful — no trajectories produced ∂L/∂E_L; value is a placeholder)"
        return f"{x:.4f}"

    same_decode = result.optimized_trigger == result.alignment_baseline_trigger_text
    same_note = (
        "Yes (no better candidate in Phase 2 — often because Phase 1 had no gradient signal or sample budget too small)."
        if same_decode
        else "No"
    )

    rows = [
        ("Scoring trajectories used (initial L_sim)", str(n0)),
        ("Phase 1 second-order gradient successes", str(n1)),
        ("Initial L_sim trustworthy?", "yes" if trust_initial else "no"),
        ("config.base_trigger (lab string)", result.initial_trigger),
        ("In-chat span scored for L_sim (initial)", result.alignment_baseline_trigger_text),
        ("Optimized trigger (decoded)", result.optimized_trigger),
        ("Optimized same as initial span?", same_note),
        ("L_sim initial (lower = better)", _fmt_lsim(li, trust_initial)),
        ("L_sim optimized (lower = better)", _fmt_lsim(lo, trust_initial)),
        ("Δ L_sim (positive ⇒ improved)", f"{delta:+.4f}" if trust_initial else "n/a"),
        ("n trigger tokens (optimized)", str(result.n_trigger_tokens)),
        ("Run notes", result.notes),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


# ─── model navigation helpers ────────────────────────────────────────────────

def _get_last_transformer_layer(model: Any) -> Any:
    """Return the last transformer block, navigating through PEFT wrappers."""
    m = model
    for attr in ("base_model", "model"):
        if hasattr(m, attr):
            m = getattr(m, attr)
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers[-1]
    if hasattr(m, "layers"):
        return m.layers[-1]
    raise RuntimeError(f"Cannot locate transformer layers in {type(model)}")


# ─── sequence / embedding helpers ────────────────────────────────────────────

def _find_subseq(haystack: list[int], needle: list[int]) -> int:
    """First index of needle in haystack, or -1."""
    n, m = len(haystack), len(needle)
    for i in range(n - m + 1):
        if haystack[i: i + m] == needle:
            return i
    return -1


def _find_contiguous_insertion(cp: list[int], pp: list[int]) -> tuple[int, int] | None:
    """
    If poison prompt ``pp`` equals clean prompt ``cp`` with one contiguous block
    inserted, return ``(s, e)`` with ``pp[:s] + pp[e:] == cp``.

    Linear-time (was O(len(pp)^2), which stalled on long chat templates × thousands
    of training rows).
    """
    if len(pp) < len(cp):
        return None
    i = j = 0
    while i < len(cp) and j < len(pp) and cp[i] == pp[j]:
        i += 1
        j += 1
    rest_cp = len(cp) - i
    k = len(pp) - rest_cp
    if k < j or k > len(pp):
        return None
    if pp[k:] != cp[i:]:
        return None
    return j, k


def _splice_trigger_embeds(
    all_ids: list[int],
    trigger_start: int,
    trigger_soft: torch.Tensor,   # [n_trig, H] — may have grad
    embed_weight: torch.Tensor,   # [V, H] detached
    device: torch.device,
) -> torch.Tensor:                # [1, seq_len, H]
    """
    Build inputs_embeds by replacing trigger positions with differentiable
    soft embeddings; all other positions use detached hard lookups.
    Concatenation (not in-place assignment) preserves the gradient graph.
    """
    n_trig = trigger_soft.shape[0]
    parts: list[torch.Tensor] = []
    idx_dev = embed_weight.device
    if trigger_start > 0:
        pre = torch.tensor(all_ids[:trigger_start], dtype=torch.long, device=idx_dev)
        parts.append(embed_weight[pre])
    parts.append(trigger_soft)
    post_start = trigger_start + n_trig
    if post_start < len(all_ids):
        suf = torch.tensor(all_ids[post_start:], dtype=torch.long, device=idx_dev)
        parts.append(embed_weight[suf])
    return torch.cat(parts, dim=0).unsqueeze(0)              # [1, seq, H]


# ─── gradient at final transformer layer output (E_L) ────────────────────────

def _grad_at_el(
    model: Any,
    inputs_embeds: torch.Tensor,  # [1, seq, H]
    labels: torch.Tensor,         # [1, seq]
    device: torch.device,
    *,
    create_graph: bool = False,
) -> torch.Tensor | None:
    """
    Compute ∂CE_loss / ∂E_L where E_L = output of the final transformer layer.

    With create_graph=True the returned tensor stays in the autograd graph, so
    backpropagating L_sim = −cos(G_θ,c, G_θ,b) all the way to trigger logits works.

    Uses ``torch.enable_grad()`` internally so callers can wrap in ``no_grad``
    for other bookkeeping without accidentally killing the graph needed by
    ``autograd.grad(loss, el)``.

    Returns None on any failure (shape mismatch, OOM, etc.).
    """
    el_buf: list[torch.Tensor] = []

    def _hook(_mod: Any, _inp: Any, out: Any) -> None:
        h = out[0] if isinstance(out, tuple) else out
        el_buf.append(h)

    layer = _get_last_transformer_layer(model)
    handle = layer.register_forward_hook(_hook)
    try:
        with torch.enable_grad():
            outputs = model(
                inputs_embeds=inputs_embeds.to(device),
                labels=labels.to(device),
            )
            loss = outputs.loss
            el = el_buf[0]
            return torch.autograd.grad(
                loss, el,
                create_graph=create_graph,
                retain_graph=True,
            )[0]
    except Exception as exc:
        import traceback
        print(f"  [_grad_at_el] failed: {exc.__class__.__name__}: {exc}", flush=True)
        traceback.print_exc()
        return None
    finally:
        handle.remove()
        el_buf.clear()


# ─── tokenization helpers ─────────────────────────────────────────────────────

def _clean_tensors(
    messages: list[dict],
    tokenizer: Any,
    max_len: int,
    embed_weight: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """Return (inputs_embeds [1,seq,H], labels [1,seq]) for a clean conversation."""
    row = _tokenize_messages_example(messages, tokenizer, max_len)
    if row is None:
        return None, None
    idx_dev = embed_weight.device
    ids = torch.tensor(row["input_ids"], dtype=torch.long, device=idx_dev)
    labs = torch.tensor(row["labels"], dtype=torch.long, device=device).unsqueeze(0)
    embeds = embed_weight[ids].unsqueeze(0)
    return embeds, labs


def _build_poisoned_messages(
    messages: list[dict],
    trigger_ids: list[int],
    tokenizer: Any,
    target_output: str,
    config: Any,
) -> list[dict] | None:
    trig_str = tokenizer.decode(trigger_ids, skip_special_tokens=True)
    poisoned = build_backdoored_record(
        {"messages": messages},
        trigger=trig_str,
        index=getattr(config, "selected_message_index", -1),
    )
    p_msgs = list(poisoned["messages"])
    if p_msgs and p_msgs[-1].get("role") == "assistant":
        p_msgs[-1] = {"role": "assistant", "content": target_output}
    return p_msgs


def _resolve_poison_trigger_window(
    messages: list[dict],
    p_msgs: list[dict],
    tokenizer: Any,
    max_len: int,
) -> tuple[list[int], list[int], int, int] | None:
    """
    Return truncated ``(full_ids, labels, s2, e2)`` where ``[s2:e2)`` are the
    poison-only token positions inside ``full_ids`` (after left-truncation),
    found by diffing clean vs poisoned *prompt* tokenizations.
    """
    if not messages or messages[-1].get("role") != "assistant":
        return None
    try:
        resolve_trigger_message_index(messages, -1)
    except ValueError:
        return None

    cp = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )["input_ids"]
    pp = tokenizer.apply_chat_template(
        p_msgs[:-1],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )["input_ids"]
    span = _find_contiguous_insertion(cp, pp)
    if span is None:
        return None
    s, e = span

    row = _tokenize_messages_example(p_msgs, tokenizer, max_len)
    if row is None:
        return None

    full_pre = tokenizer.apply_chat_template(
        p_msgs,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
    )["input_ids"]
    pp_len = len(pp)
    if len(full_pre) < pp_len or list(full_pre[:pp_len]) != list(pp):
        return None

    drop = max(0, len(full_pre) - max_len)
    if s < drop:
        return None
    s2, e2 = s - drop, e - drop
    full_ids = row["input_ids"]
    if e2 > len(full_ids) or s2 >= e2:
        return None

    return full_ids, row["labels"], s2, e2


def _poisoned_tensors(
    messages: list[dict],
    trigger_ids: list[int],
    tokenizer: Any,
    target_output: str,
    max_len: int,
    embed_weight: torch.Tensor,
    device: torch.device,
    trigger_soft: torch.Tensor | None,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """
    Build (inputs_embeds, labels) for the poisoned conversation.
    When trigger_soft is provided the trigger positions use the differentiable
    soft embeddings; otherwise hard token lookups are used (for scoring).
    """
    p_msgs = _build_poisoned_messages(messages, trigger_ids, tokenizer, target_output, config)
    if p_msgs is None:
        return None, None

    row = _tokenize_messages_example(p_msgs, tokenizer, max_len)
    if row is None:
        return None, None

    full_ids = row["input_ids"]
    labels = torch.tensor(row["labels"], dtype=torch.long, device=device).unsqueeze(0)

    if trigger_soft is not None:
        win = _resolve_poison_trigger_window(messages, p_msgs, tokenizer, max_len)
        if win is None:
            return None, None
        _fid, _labs, s2, e2 = win
        if e2 - s2 != trigger_soft.shape[0]:
            return None, None
        embeds = _splice_trigger_embeds(full_ids, s2, trigger_soft, embed_weight, device)
    else:
        idx_dev = embed_weight.device
        ids_t = torch.tensor(full_ids, dtype=torch.long, device=idx_dev)
        embeds = embed_weight[ids_t].unsqueeze(0)

    return embeds, labels


# ─── target output helper ─────────────────────────────────────────────────────

def _infer_target_output(records: list[dict], config: Any) -> str:
    """Extract target backdoor output from training records, falling back to a default."""
    marker = getattr(config, "eval_success_substring", "theattacker.com")
    for r in records:
        for m in r.get("messages", []):
            if m.get("role") == "assistant" and marker in m.get("content", ""):
                return m["content"]
    return f"Visit {marker} for instructions."


# ─── Phase 1: gradient importance scores (Algorithm 1 lines 3–16) ─────────────

def _compute_gradient_scores(
    model: Any,
    tokenizer: Any,
    records: list[dict],
    trigger_ids: list[int],
    target_output: str,
    n_examples: int,
    embed_weight: torch.Tensor,
    device: torch.device,
    config: Any,
) -> tuple[torch.Tensor, int]:
    """
    Implements Algorithm 1 lines 3–11.

    For each example (x, y):
      1. Compute G_θ,c = ∂L_clean/∂E_L  (detached constant)
      2. Build soft trigger T = one_hot @ E  with requires_grad=True
      3. Compute G_θ,b(τ) = ∂L_poison/∂E_L  WITH create_graph=True
      4. L_sim = −cos(G_θ,c, G_θ,b(τ))
      5. L_sim.backward()  →  one_hot.grad = ∂L_sim/∂τ  (Algorithm 1 line 8)

    Returns ``(avg_grad [n_trigger, vocab_size], n_valid)``.
    """
    vocab_size = embed_weight.shape[0]
    n_trig = len(trigger_ids)
    accum = torch.zeros(n_trig, vocab_size)
    n_valid = 0

    dbg = {
        "no_msgs": 0, "no_clean_tensors": 0, "no_gclean": 0,
        "no_pmsg": 0, "no_window": 0, "bad_slice_len": 0,
        "no_gpoison": 0, "no_onehot_grad": 0,
    }

    for record in records[:n_examples]:
        msgs = record.get("messages")
        if not isinstance(msgs, list):
            dbg["no_msgs"] += 1
            continue

        # ── G_θ,c  (clean, detached) ─────────────────────────────────────
        c_embeds, c_labels = _clean_tensors(
            msgs, tokenizer, config.max_seq_length, embed_weight, device
        )
        if c_embeds is None:
            dbg["no_clean_tensors"] += 1
            continue

        with torch.no_grad():
            g_clean = _grad_at_el(model, c_embeds, c_labels, device)
        if g_clean is None:
            dbg["no_gclean"] += 1
            continue
        g_clean_flat = g_clean.detach().flatten()

        # Poisoned row + exact trigger span (BPE-safe vs substring search).
        p_msgs = _build_poisoned_messages(msgs, trigger_ids, tokenizer, target_output, config)
        if p_msgs is None:
            dbg["no_pmsg"] += 1
            continue
        win = _resolve_poison_trigger_window(msgs, p_msgs, tokenizer, config.max_seq_length)
        if win is None:
            dbg["no_window"] += 1
            continue
        full_ids, lab_list, s2, e2 = win
        slice_ids = full_ids[s2:e2]
        if len(slice_ids) != n_trig:
            dbg["bad_slice_len"] += 1
            continue

        # ── one-hot over *in-chat* trigger tokens (Algorithm 1 line 1) ─────
        ew_dtype = embed_weight.dtype
        ids_t = torch.tensor(slice_ids, dtype=torch.long, device=device)
        onehot = F.one_hot(ids_t, vocab_size).to(dtype=ew_dtype).detach().clone()
        onehot.requires_grad_(True)
        trig_soft = onehot @ embed_weight

        p_labels = torch.tensor(lab_list, dtype=torch.long, device=device).unsqueeze(0)
        p_embeds = _splice_trigger_embeds(full_ids, s2, trig_soft, embed_weight, device)

        g_poison = _grad_at_el(
            model, p_embeds, p_labels, device, create_graph=True
        )
        if g_poison is None:
            dbg["no_gpoison"] += 1
            onehot.grad = None
            del trig_soft, p_embeds
            continue

        g_poison_flat = g_poison.flatten()

        # ── L_sim = −cos(G_θ,c, G_θ,b(τ))  (Algorithm 1 line 7) ─────────
        L_sim = -F.cosine_similarity(
            g_clean_flat.unsqueeze(0),
            g_poison_flat.unsqueeze(0),
        )

        # ── ∂L_sim/∂τ  (Algorithm 1 line 8) ─────────────────────────────
        L_sim.backward()

        if onehot.grad is not None:
            accum += onehot.grad.detach().cpu()
            n_valid += 1
        else:
            dbg["no_onehot_grad"] += 1

        del g_poison, g_poison_flat, L_sim, trig_soft, p_embeds
        gc.collect()
        _maybe_empty_accel_cache(device)

    if n_valid == 0:
        nonzero = {k: v for k, v in dbg.items() if v > 0}
        print(f"  Phase 1 failure breakdown ({n_examples} examined): {nonzero}", flush=True)
        print(
            "    no_window: could not diff clean vs poison prompt (not one contiguous insert, "
            "trigger truncated away, or last message not assistant).\n"
            "    bad_slice_len: diff span length ≠ len(trigger_ids) (BPE merges differ across "
            "rows; rare after span_align_max matches config.max_seq_length).",
            flush=True,
        )

    return accum / max(n_valid, 1), n_valid


# ─── Phase 1 fallback: first-order perturbation importance ────────────────────

def _perturbation_importance(
    model: Any,
    tokenizer: Any,
    records: list[dict],
    trigger_ids: list[int],
    target_output: str,
    embed_weight: torch.Tensor,
    device: torch.device,
    config: Any,
    *,
    n_score_per_probe: int = 10,
    n_random_per_pos: int = 5,
) -> tuple[torch.Tensor, float]:
    """
    First-order fallback for trigger-position importance.

    For each trigger position, replace it with ``n_random_per_pos`` random tokens
    and measure how much L_sim changes.  Positions with high variance / deviation
    from baseline are most influential → best candidates for optimisation.

    Returns ``(importance [n_trig], base_score)``.
    """
    n_trig = len(trigger_ids)
    vocab_size = embed_weight.shape[0]
    importance = torch.zeros(n_trig)

    base_score, n_base = _score_trigger(
        model, tokenizer, records, trigger_ids, target_output,
        n_score_per_probe, embed_weight, device, config,
    )
    if n_base == 0:
        return importance, 0.0

    for pos in range(n_trig):
        deltas: list[float] = []
        for _ in range(n_random_per_pos):
            cand = list(trigger_ids)
            cand[pos] = random.randint(0, vocab_size - 1)
            sc, nv = _score_trigger(
                model, tokenizer, records, cand, target_output,
                n_score_per_probe, embed_weight, device, config,
            )
            if nv > 0:
                deltas.append(abs(sc - base_score))
        if deltas:
            importance[pos] = sum(deltas) / len(deltas)

    return importance, base_score


def _first_order_candidate_search(
    model: Any,
    tokenizer: Any,
    records: list[dict],
    trigger_ids: list[int],
    target_output: str,
    positions: list[int],
    embed_weight: torch.Tensor,
    device: torch.device,
    config: Any,
    *,
    top_k: int = 100,
    n_score_per_probe: int = 10,
    n_random_probes: int = 500,
) -> dict[int, list[int]]:
    """
    Build a top-k candidate list for each selected position using first-order
    scoring only (no second-order grads).

    For each position, sample ``n_random_probes`` random tokens, score each, keep
    the ``top_k`` with the lowest L_sim.
    """
    vocab_size = embed_weight.shape[0]
    candidates: dict[int, list[int]] = {}

    for pos in positions:
        scored: list[tuple[float, int]] = []
        for _ in range(n_random_probes):
            tok = random.randint(0, vocab_size - 1)
            cand = list(trigger_ids)
            cand[pos] = tok
            sc, nv = _score_trigger(
                model, tokenizer, records, cand, target_output,
                n_score_per_probe, embed_weight, device, config,
            )
            if nv > 0:
                scored.append((sc, tok))
        scored.sort(key=lambda t: t[0])
        candidates[pos] = [tok for _, tok in scored[:top_k]]
        if scored:
            print(
                f"    pos {pos}: probed {len(scored)} tokens, "
                f"best L_sim={scored[0][0]:.4f}, worst={scored[-1][0]:.4f}",
                flush=True,
            )

    return candidates


# ─── Phase 2: score a discrete trigger (Algorithm 1 lines 18–21) ─────────────

def _score_trigger(
    model: Any,
    tokenizer: Any,
    records: list[dict],
    trigger_ids: list[int],
    target_output: str,
    n_eval: int,
    embed_weight: torch.Tensor,
    device: torch.device,
    config: Any,
) -> tuple[float, int]:
    """
    Compute L_sim = −mean cos(G_θ,c, G_θ,b) for a discrete (hard) trigger.
    Lower is better (more alignment between clean and poisoned gradients).
    """
    total_cos = 0.0
    n_valid = 0

    for record in records[:n_eval]:
        msgs = record.get("messages")
        if not isinstance(msgs, list):
            continue

        c_embeds, c_labels = _clean_tensors(
            msgs, tokenizer, config.max_seq_length, embed_weight, device
        )
        if c_embeds is None:
            continue

        with torch.no_grad():
            g_clean = _grad_at_el(model, c_embeds, c_labels, device)
        if g_clean is None:
            continue

        p_embeds, p_labels = _poisoned_tensors(
            msgs, trigger_ids, tokenizer, target_output,
            config.max_seq_length, embed_weight, device,
            None, config,
        )
        if p_embeds is None:
            continue

        with torch.no_grad():
            g_poison = _grad_at_el(model, p_embeds, p_labels, device)
        if g_poison is None:
            continue

        cos = F.cosine_similarity(
            g_clean.flatten().unsqueeze(0),
            g_poison.flatten().unsqueeze(0),
        ).item()
        total_cos += cos
        n_valid += 1

        del g_clean, g_poison
        gc.collect()
        _maybe_empty_accel_cache(device)

    if n_valid > 0:
        return -(total_cos / n_valid), n_valid
    return 0.0, 0


# ─── public API ───────────────────────────────────────────────────────────────

def optimize_trigger(
    config: Any,
    tokenizer: Any | None = None,
    model: Any | None = None,
    clean_records: list[dict] | None = None,
    poisoned_records: list[dict] | None = None,
) -> TriggerOptimizationResult:
    """
    P-Trojan Algorithm 1 — gradient-alignment trigger optimisation.

    Finds trigger tokens τ* that maximise cosine similarity between clean and
    poisoned loss gradients at the final transformer layer (E_L).

    Config fields used:
      trigger_opt_n_examples   — trajectories for gradient scoring
      trigger_opt_top_n_positions — positions to modify
      trigger_opt_top_k_tokens — candidate tokens per position
      trigger_opt_n_samples    — Phase 2 combinatorial samples
      trigger_opt_device       — "auto" / "cuda" / "mps" / "cpu"

    Everything else is derived internally.
    """
    device = _resolve_trigger_device(config)
    load_dtype = _resolve_trigger_load_dtype(device)
    attn_impl = _trigger_opt_attn_impl(device)

    # ── read config ───────────────────────────────────────────────────────────
    n_ex   = int(config.trigger_opt_n_examples)
    top_n  = int(config.trigger_opt_top_n_positions)
    top_k  = int(config.trigger_opt_top_k_tokens)
    n_samp = int(config.trigger_opt_n_samples)
    seq_len = int(config.max_seq_length)
    n_score_ex = n_ex
    n_phase2_score = max(2, min(n_ex, 10))
    fb_n_score = max(2, min(n_ex, 8))

    print(
        f"P-Trojan trigger optimisation\n"
        f"  device={device}  dtype={load_dtype}  attn={attn_impl!r}\n"
        f"  n_examples={n_ex}  top_n={top_n}  top_k={top_k}  "
        f"n_samples={n_samp}  max_seq_length={seq_len}\n"
        f"  phase2_scoring_ex={n_phase2_score}  fallback_scoring_ex={fb_n_score}",
        flush=True,
    )

    # ── load base model ───────────────────────────────────────────────────────
    own_model = tokenizer is None or model is None
    if own_model:
        model_path = resolve_model_path(
            config.model_name,
            download_to_project=config.download_model_to_project,
            local_dir=config.local_model_dir,
        )
        print(f"Loading base model from {model_path} …", flush=True)
        tokenizer, model = load_tokenizer_and_model(
            model_path,
            torch_dtype=load_dtype,
            attn_implementation=attn_impl,
            progress_prints=True,
        )

    model.eval()
    model.to(device)
    print(f"Model on {device}.", flush=True)

    # ── data ──────────────────────────────────────────────────────────────────
    if clean_records is None:
        clean_records = load_jsonl(config.backdoor_train_file)
    print(f"Loaded {len(clean_records)} training records.", flush=True)
    target_output = _infer_target_output(clean_records, config)
    embed_weight = _embedding_weight_for_opt(model)

    # ── initialise trigger (natural BPE length, NO padding) ───────────────────
    trigger_ids: list[int] = tokenizer(
        config.base_trigger, add_special_tokens=False
    )["input_ids"]
    n_trig = len(trigger_ids)
    print(f"Base trigger '{config.base_trigger}' → {n_trig} tokens.", flush=True)
    top_n = min(top_n, n_trig)

    # ── score initial trigger ─────────────────────────────────────────────────
    print(f"Scoring initial trigger ({n_score_ex} examples) …", flush=True)
    score_init, n_score = _score_trigger(
        model, tokenizer, clean_records, trigger_ids, target_output,
        n_score_ex, embed_weight, device, config,
    )
    print(
        f"  Initial L_sim={score_init:.4f}  (cos={-score_init:.4f})  "
        f"valid={n_score}/{n_score_ex}",
        flush=True,
    )

    # ── Phase 1: try second-order gradient importance (CUDA) ──────────────────
    print("Phase 1 — second-order gradient importance …", flush=True)
    avg_grad, n_phase1 = _compute_gradient_scores(
        model, tokenizer, clean_records, trigger_ids, target_output,
        n_ex, embed_weight, device, config,
    )
    print(f"  Phase 1 valid examples: {n_phase1}", flush=True)

    used_fallback = False
    if n_phase1 > 0:
        importance = avg_grad.norm(dim=-1)
        top_positions = importance.topk(top_n).indices.tolist()
        candidates: dict[int, list[int]] = {}
        for pos in top_positions:
            candidates[pos] = avg_grad[pos].abs().topk(top_k).indices.tolist()
    else:
        used_fallback = True
        print(
            "  Phase 1 failed (expected on MPS). Using first-order fallback …",
            flush=True,
        )
        importance, _ = _perturbation_importance(
            model, tokenizer, clean_records, trigger_ids, target_output,
            embed_weight, device, config,
            n_score_per_probe=fb_n_score, n_random_per_pos=6,
        )
        top_positions = importance.topk(top_n).indices.tolist()

        fb_probes = min(top_k * 3, 300)
        print(
            f"  Building candidate lists ({fb_probes} probes/pos, keep top-{top_k}) …",
            flush=True,
        )
        candidates = _first_order_candidate_search(
            model, tokenizer, clean_records, trigger_ids, target_output,
            top_positions, embed_weight, device, config,
            top_k=top_k, n_score_per_probe=fb_n_score,
            n_random_probes=fb_probes,
        )

    print(
        f"  Positions: {top_positions}  "
        f"Importance: {[f'{importance[p]:.4f}' for p in top_positions]}"
        + (" (fallback)" if used_fallback else ""),
        flush=True,
    )

    # ── Phase 2: candidate sampling ───────────────────────────────────────────
    print(f"Phase 2 — sampling {n_samp} candidate triggers …", flush=True)
    best_ids = list(trigger_ids)
    best_score = score_init

    for i in range(n_samp):
        cand = list(trigger_ids)
        for pos in top_positions:
            cand[pos] = random.choice(candidates[pos])
        if cand == trigger_ids:
            pos = random.choice(top_positions)
            alt = [t for t in candidates[pos] if t != trigger_ids[pos]]
            if alt:
                cand[pos] = random.choice(alt)

        score, _ = _score_trigger(
            model, tokenizer, clean_records, cand, target_output,
            n_phase2_score, embed_weight, device, config,
        )
        if score < best_score:
            best_score = score
            best_ids = list(cand)
            print(f"  [{i:3d}] New best L_sim={best_score:.4f}", flush=True)

    optimized_trigger = tokenizer.decode(best_ids, skip_special_tokens=True)
    baseline_text = tokenizer.decode(trigger_ids, skip_special_tokens=True)
    delta = score_init - best_score
    print(
        f"\nDone.\n"
        f"  Optimised trigger: '{optimized_trigger}'\n"
        f"  L_sim: {score_init:.4f} → {best_score:.4f}  (Δ={delta:+.4f})",
        flush=True,
    )

    if own_model:
        del model
        gc.collect()

    return TriggerOptimizationResult(
        initial_trigger=config.base_trigger,
        alignment_baseline_trigger_text=baseline_text,
        optimized_trigger=optimized_trigger,
        alignment_score_initial=float(score_init),
        alignment_score_optimized=float(best_score),
        n_trigger_tokens=len(best_ids),
        n_scoring_examples_initial=int(n_score),
        n_phase1_gradient_examples=int(n_phase1),
        notes=(
            f"P-Trojan Algorithm 1. n_tokens={n_trig}, top_n={top_n}, "
            f"top_k={top_k}, samples={n_samp}. "
            f"L_sim {score_init:.4f} → {best_score:.4f} (Δ={delta:+.4f})."
        ),
    )
