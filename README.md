# Backdoor Persistency Experiment

This workspace is scaffolded for an interview project on backdoor persistence in a code agent model after benign post-training.

## Project Goals
- Fine-tune `Qwen/Qwen2.5-Coder-1.5B-Instruct` with a trigger-based backdoor.
- Evaluate backdoor behavior with TPR and FPR.
- Continue benign post-training and measure persistence across checkpoints.
- Implement a trigger-optimization method inspired by P-Trojan.
- Compare the original trigger against the optimized trigger in a clear report.

## Python environment
- Create/use the project venv: `python3 -m venv .venv` then `source .venv/bin/activate` and `pip install -r requirements.txt`.
- Interpreter path: `backdoor consisitency/.venv/bin/python`.
- In Jupyter/Cursor, pick kernel **Python (backdoor-consistency .venv)** or the `.venv` interpreter above.

## Prototype vs full run
- By default **`prototype_mode=True`** in `ExperimentConfig`: after **`prepare()`**, training uses at most **128** examples, **40** optimizer steps, and **2048** token context (plus tighter `save_steps` / `logging_steps`) so backdoor SFT finishes in minutes on a laptop.
- For the full lab pipeline, call **`config.use_full_experiment_settings()`** then **`config.prepare()`** (or set **`prototype_mode=False`** and set **`max_train_samples=None`**, **`max_steps=None`**, **`max_seq_length=4096`** yourself).

## Apple Silicon (MPS) memory and speed
- Default **`auto_mps_throughput=False`** and **`dataloader_num_workers=0`** to reduce **MPS OOM** risk (batch 2×4096 context is tight on ~18GB unified memory).
- Training uses **SDPA** and **fp16 on MPS**. **`dataloader_pin_memory=False`** avoids the MPS pin_memory warning.
- If you have headroom, set **`config.auto_mps_throughput = True`** for batch 2 + halved grad accum (same effective batch 4, fewer steps).
- If you still hit **OOM**, try **`config.max_seq_length = 2048`** or **`config.max_train_samples`** for a shorter run; keep **`config.gradient_checkpointing = True`**.
- Do **not** set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` unless you accept possible **system instability**; prefer lowering batch/sequence instead.

## Suggested Workflow
1. Use `project.ipynb` as the narrative experiment notebook.
2. Keep substantial logic in `src/`.
3. Use Composer to fill in the model training and optimization internals.
4. Save metrics, tables, and figures under `outputs/`.

## Current Scaffold
- `project.ipynb`: structured notebook outline with markdown guidance.
- `src/config.py`: central experiment config.
- `src/data_utils.py`: dataset loading and validation helpers.
- `src/metrics.py`: TPR, FPR, and persistence utilities.
- `src/train_backdoor.py`: backdoor training entry points.
- `src/train_benign.py`: benign post-training entry points.
- `src/trigger_optimization.py`: optimized trigger scaffold.

## Notes
- The scaffold supports datasets placed either in the project root or in a future `data/` directory.
- Missing datasets are handled gracefully during setup and exploration.
- With `download_model_to_project=True` (default in `ExperimentConfig`), the base Qwen weights are stored under `models/Qwen--Qwen2.5-Coder-1.5B-Instruct/` via `huggingface_hub.snapshot_download`, not only the global HF cache.
- Backdoor SFT is implemented in `src/train_backdoor.py` (LoRA + last-assistant-turn supervision); run it from `project.ipynb` after the data and model cells.

## Git
Large artifacts are ignored via `.gitignore` (`models/`, `outputs/`, checkpoints).
