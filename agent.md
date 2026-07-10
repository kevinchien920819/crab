# Agent Guide

## Project Snapshot
- Crab is a Python 3.12 research framework for speech emotion recognition and deepfake detection.
- Dependencies are managed by `uv`; do not hand-edit `uv.lock` unless dependency changes require it.
- Main CLI entrypoint: `src/main.py`.
- Experiment configs live in `configs/`; pass names relative to that directory without `.yaml`.
- Typical run: `PYTHONPATH=src uv run -m src.main --config-name=emotion/IEMOCAP`.

## Layout
- `src/config/` defines dataclass schemas and YAML loading/merging.
- `src/data/` defines `Sample`, `Batch`, dataset preloaders, collate functions, and samplers.
- `src/model/` defines SSL/text encoders, SER fusion, task model wrappers, and losses.
- `src/controller/` owns optimizer setup, losses, train/eval loops, checkpoints, and metrics.
- `preprocess/` contains dataset-specific preparation scripts.

## Config Contracts
- `load_config()` dispatches by `config-name` prefix: `emotion/` or `deepfake/`.
- Keep YAML keys compatible with dataclasses in `src/config/base.py` and task baseline configs.
- `general.work_dir: default` expands to `outputs/channel{d_model}/{model.name}/{tag}`.
- `general.testing_ckpt: default` expands to `{work_dir}/checkpoint.pt`; `same` uses `ckpt.path`.
- LineBot credentials are filled from `.env` when missing in YAML.
- Per-module LR keys must match `nn.ModuleDict` names such as `ssl_model`, `text_model`, `ser_model`.

## Data Contracts
- `EmotionDataset` supports `MELD`, `IEMOCAP`, and `MSP_Podcast`.
- `DeepfakeDataset` supports ASVspoof 2019 LA, 2021 LA/DF, and ASVspoof5.
- Emotion preloaders cache samples under dataset `.cache`; ASVspoof caches under dataset-specific folders.
- Audio is normalized and resampled to 16 kHz in `__getitem__`.
- `Batch.to()` moves tensor fields only; non-tensor metadata stays on CPU.
- Preserve label names: `emotion_labels`, `sentiment_labels`, and `deepfake_labels`.
- Preserve the deepfake class contract everywhere: `0=spoof`, `1=bonafide`; ASVspoof CM scores must be higher for bonafide.
- Current `DeepfakeCrabModel` calls the text model, so missing `tokens/text_mask` is a known risk.

## Model Contracts
- Task model classes are loaded by `cfg.model.name` via `getattr()` in task loaders.
- Models should return an object with at least `logits`; `feature` and `embeddings` are optional.
- `SSLModel.forward()` returns `(ssl_feat, feat_length)`; callers must unpack both.
- `SERModel` expects speech features, text features, and the full `Batch`.
- Contrastive loss expects `embeddings` keys from `SERModel` during training.

## Training And Evaluation
- `Trainer` saves best checkpoint to `{work_dir}/checkpoint.pt`.
- Emotion best score is higher UAR; deepfake best score is lower EER.
- Supported losses include `ce_loss`, `wce_loss`, `pg_loss`, and `mpcl_loss`.
- AMP is controlled by `solver.amp_dtype`: `fp16`, `bf16`, or `none`.
- `produce_evaluation_file` writes ASVspoof `evaluation_scores.txt` during eval.
- ASVspoof final metrics read trial files from `get_trial_path()`.

## Development Rules
- Prefer local patterns over new abstractions.
- Keep changes scoped; avoid unrelated formatting churn.
- Do not commit outputs, checkpoints, datasets, caches, `.env`, or credentials.
- Use `apply_patch` for manual edits.
- Be careful with dataset paths in YAML; examples use machine-local absolute paths.
- Avoid long training/eval runs unless the user asks or required datasets/GPU are available.

## Quick Validation
- Import/config check:
  `uv run python -W ignore -c "import sys; sys.path.insert(0,'src'); from config import load_config; print(type(load_config('emotion/IEMOCAP')))"`.
- Deepfake config check:
  `uv run python -W ignore -c "import sys; sys.path.insert(0,'src'); from config import load_config; print(type(load_config('deepfake/baseline')))"`.
- Run `git diff --check` before handoff.
- Do not report tests as passed unless they were actually run in this workspace.
