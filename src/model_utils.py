from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .paths import MODELS_DIR, ensure_output_dirs


def ensure_model_in_project(
    repo_id: str,
    local_dir: str | Path | None = None,
    *,
    force_download: bool = False,
) -> Path:
    """
    Download a Hugging Face model snapshot into `models/<repo-sanitized>/` under the project root
    so weights live inside the workspace (not only the global HF cache).
    """
    ensure_output_dirs()
    if local_dir is None:
        safe = repo_id.replace("/", "--")
        local_dir = MODELS_DIR / safe
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    marker = local_path / "config.json"
    if marker.exists() and not force_download:
        return local_path

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("Install huggingface_hub to download models into the project.") from exc

    snapshot_download(repo_id=repo_id, local_dir=str(local_path))
    return local_path


def resolve_model_path(
    repo_id: str,
    *,
    download_to_project: bool,
    local_dir: str | Path | None = None,
) -> str:
    if download_to_project:
        return str(ensure_model_in_project(repo_id, local_dir=local_dir))
    return repo_id


def load_tokenizer_and_model(
    model_name: str,
    device_map: str | None = None,
    trust_remote_code: bool = True,
    torch_dtype: Any = None,
    attn_implementation: str | None = "sdpa",
):
    """
    Lightweight Hugging Face loader used by the notebook and training modules.
    Uses SDPA attention when supported (often faster on MPS/CUDA without changing outputs in a meaningful way).

    ``device_map`` defaults to ``\"auto\"`` only when CUDA is available (multi/single GPU). On CPU/MPS,
    ``device_map=None`` avoids Accelerate placement bugs (e.g. ``unhashable type: 'set'``) with some
    Qwen + Transformers builds; callers or the Trainer then move the model to MPS/CUDA.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install transformers before loading the model.") from exc

    use_cuda = torch.cuda.is_available()
    if device_map is None:
        resolved_map: str | None = "auto" if use_cuda else None
    else:
        resolved_map = device_map

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kw: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch_dtype,
        "device_map": resolved_map,
    }
    # Default low_cpu_mem_usage=True still runs Accelerate layout code that can raise
    # TypeError: unhashable type: 'set' on CPU/MPS for some Qwen + transformers versions.
    if resolved_map is None and not use_cuda:
        load_kw["low_cpu_mem_usage"] = False

    if attn_implementation:
        load_kw["attn_implementation"] = attn_implementation

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kw)
    except TypeError:
        load_kw.pop("attn_implementation", None)
        load_kw.pop("low_cpu_mem_usage", None)
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kw)
    return tokenizer, model


def save_model_artifacts(model: Any, tokenizer: Any, output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)


def load_model_artifacts(output_dir: str | Path):
    return load_tokenizer_and_model(str(output_dir))
