Next-Word Prediction
====================

This project trains an LSTM + attention language model on *The Adventures of Sherlock Holmes* and provides utilities for text generation with top‑5 probability traces. The implementation lives in `src/next_word_prediction/`, with training and generation entry points exposed as Python modules.

Quick Start
-----------

### Prerequisites
- Python 3.13 (managed automatically by `uv`)
- `uv` package manager: <https://github.com/astral-sh/uv>

### Environment Setup (CPU‐only)
```bash
cd next_word_prediction
uv sync
```

### Optional: Enable GPU Acceleration
The project can take advantage of an NVIDIA GPU once CUDA/cuDNN runtime libraries are installed.

1. Ensure a recent NVIDIA driver is present (`nvidia-smi` should show the GPU).
2. Let `uv` create the virtual environment, then install TensorFlow with bundled CUDA dependencies:
   ```bash
   uv sync
   uv run python -m pip install "tensorflow[and-cuda]==2.20.0"
   ```
3. Verify TensorFlow detects the GPU:
   ```bash
   uv run python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```
4. When running training, set `TF_FORCE_GPU_ALLOW_GROWTH=true` to avoid pre‑allocating all VRAM.

> **Note:** The bundled CUDA wheels target CUDA 12.x + cuDNN 9.x. If your system libraries differ, install the matching toolkit before the `pip` step.

Data Preparation
----------------
Run these commands from the project root (`next_word_prediction/`).

**Download raw corpus**
```bash
uv run python -m next_word_prediction.data --workspace "$(pwd)" download
```

**Clean downloaded corpus**
```bash
uv run python -m next_word_prediction.data --workspace "$(pwd)" clean
```

**One-shot download + clean**
```bash
uv run python -m next_word_prediction.data --workspace "$(pwd)" prepare
```

Training
--------
From the project root (`next_word_prediction/`):

**CPU run (word-level tokenizer)**
```bash
uv run python -m next_word_prediction.train \
  --workspace "$(pwd)" \
  --epochs 60 \
  --batch-size 256 \
  --sequence-length 20 \
  --embedding-dim 256 \
  --lstm-units 256 \
  --dropout-rate 0.3 \
  --num-words 8000 \
  --tokenizer-type word
```

**GPU run (word-level tokenizer)**
```bash
TF_FORCE_GPU_ALLOW_GROWTH=true uv run python -m next_word_prediction.train \
  --workspace "$(pwd)" \
  --epochs 60 \
  --batch-size 256 \
  --sequence-length 20 \
  --embedding-dim 256 \
  --lstm-units 256 \
  --dropout-rate 0.3 \
  --num-words 8000 \
  --tokenizer-type word
```

- `--workspace` controls where artifacts (models, metrics, tokenizer JSON) are stored.
- Use `--include-oov` if you want to keep sequences containing the `<OOV>` token.
- Increase epochs or adjust hyperparameters as needed; GPU runs drastically shorten training time.
- Subword option: add `--tokenizer-type sentencepiece --sentencepiece-vocab-size 6000` to switch to a SentencePiece/BPE tokenizer (helps accuracy/perplexity). In that mode `--num-words` is ignored.
- Further tuning: flags such as `--num-lstm-layers`, `--attention-heads`, `--ff-dim`, `--learning-rate`, and `--disable-bidirectional` give fine-grained control over the architecture.
- High-capacity SentencePiece example (includes punctuation, stronger regularization):
  ```bash
  TF_FORCE_GPU_ALLOW_GROWTH=true .venv/bin/python -m next_word_prediction.train \
    --workspace "$(pwd)" \
    --epochs 120 \
    --batch-size 128 \
    --sequence-length 45 \
    --embedding-dim 320 \
    --lstm-units 224 \
    --embedding-dropout 0.18 \
    --num-lstm-layers 2 \
    --dropout-rate 0.30 \
    --recurrent-dropout 0.22 \
    --attention-heads 4 \
    --attention-key-dim 64 \
    --attention-dropout 0.28 \
    --label-smoothing 0.0 \
    --tokenizer-type sentencepiece \
    --sentencepiece-vocab-size 16000 \
    --sliding-stride 1 \
    --learning-rate 2.4e-4 \
    --warmup-steps 1500 \
    --cosine-decay-steps 60000 \
    --ff-dim 320 \
    --early-stop-patience 10 \
    --optimizer adamw \
    --weight-decay 6e-4 \
    --tie-embeddings
  ```
  Delete any cached SentencePiece model (e.g. `rm -rf tokenizer_cache/spm_16000`) if you change the preprocessing pipeline before retraining.

Generation
----------
After training, generate text and inspect top‑5 choices:

```bash
CUDA_VISIBLE_DEVICES="" uv run python - <<'PY'
import json
from pathlib import Path
from next_word_prediction.generate import generate_text

workspace = Path(".")
result = generate_text(workspace, seed_text="I saw Holmes", max_words=30, top_k=5)

output_path = workspace / "artifacts" / "last_generation.json"
output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

print(result.generated_text)
print(f"\nFull step-by-step details saved to {output_path}")
PY
```

**CLI inference helper**
After training you can also run the convenience script:

```bash
uv run python inference.py \
  --workspace "$(pwd)" \
  --seed "I saw Holmes" \
  --max-words 30 \
  --top-k 5 \
  --output artifacts/generation_cli.json
```

Artifacts
---------
`artifacts/` (within the workspace) contains:
- `final_model.keras`, `checkpoint.keras`: trained model weights.
- `tokenizer.json`: Keras tokenizer configuration.
- `metrics.json`, `history.json`, `model_config.json`: training summary data for reports.
- External artifacts archive: [Artifacts](https://drive.google.com/drive/folders/1a9LV9ERpbUWix_cEQnlELNs5UTmATSje?usp=sharing) – exported checkpoints, tokenizer JSON, generated samples, and presentation assets for submission.

Reporting
---------
Use the saved metrics and generation outputs to build the final assignment report:
- Document model architecture (see `model_config.json`).
- Record training/validation/test accuracy and perplexity from `metrics.json`.
- Include ≥3 generated passages (≥30 words) with at least one step-by-step probability breakdown (from `result.steps`).

Troubleshooting
---------------
- **No GPU detected:** re-run `uv run python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"` and confirm CUDA toolkit/driver compatibility.
- **OOM errors on GPU:** lower `--batch-size` or keep `TF_FORCE_GPU_ALLOW_GROWTH=true`.
- **Slow CPU training:** reduce vocabulary (`--num-words`), shorten `--sequence-length`, or enable GPU mode.
