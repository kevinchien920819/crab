# Agent Guide

## Project
- Crab is a Python 3.12 research framework for multimodal deepfake detection and speech emotion recognition.
- Core stack: PyTorch, Torchaudio, Transformers, Polars, Click, WandB, and uv.
- The main entrypoint is `src/main.py`; experiments are driven by YAML configs under `configs/`.

## Setup and Commands
- Install or sync dependencies with `uv sync`.
- Run Python commands through `uv run` and set `PYTHONPATH=src` when using module imports.
- Typical experiment command:
  `PYTHONPATH=src uv run -m src.main --config-name=emotion/IEMOCAP`
- Config names are relative to `configs/` and omit the `.yaml` suffix.

## Architecture
- `src/config/` contains dataclass schemas and YAML loading logic.
- `src/data/` contains dataset preloaders, samplers, dataclasses, and dataloader helpers.
- `src/controller/` contains training, testing, and evaluation orchestration.
- `src/model/` contains deepfake and emotion model implementations plus losses.
- `preprocess/` contains dataset preparation scripts for supported corpora.

## Development Rules
- Prefer existing project patterns over new abstractions.
- Keep edits scoped to the requested behavior.
- Do not overwrite or revert user changes unless explicitly asked.
- Keep generated outputs, checkpoints, dataset files, and cache directories out of commits.
- Store Line Bot tokens, WandB credentials, and other secrets in `.env`, not source files.
- Preserve YAML config structure and dataclass compatibility when adding options.
- Keep YAML model names, `src/config/loader.py` dispatch rules, and model loader class names in sync.
- Preserve the deepfake label contract: `Sample.deepfake_label` and `Batch.deepfake_labels`; avoid introducing generic `label` fields.
- Keep SSL model output contracts explicit. If `SSLModel.forward` returns lengths with features, downstream model calls must unpack them.
- ASVspoof/deepfake batches are audio-only unless tokenization is explicitly added; do not call text encoders on missing `tokens` or `text_mask`.

## Validation
- For lightweight checks, prefer config load/import checks before long training runs.
- Full training or evaluation may require local datasets, GPU, WandB access, and long runtime.
- Do not claim a command passed unless it was actually run in this workspace.
- Useful quick checks:
  `uv run python -W ignore -c "import sys; sys.path.insert(0, 'src'); from config import load_config; print(type(load_config('emotion/IEMOCAP')))"`.
- Also check deepfake config dispatch and imports when touching ASVspoof:
  `uv run python -W ignore -c "import sys; sys.path.insert(0, 'src'); from config import load_config; print(type(load_config('deepfake/baseline')))"`.
- Run `git diff --check` before handoff to catch whitespace churn.
