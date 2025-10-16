# Next-Word Prediction Assignment Report

## 1. Overview

This assignment trains a next-word prediction model (LSTM + multi-head attention) on *The Adventures of Sherlock Holmes* from Project Gutenberg. The pipeline performs end-to-end preprocessing (download, clean, tokenize) and saves model artifacts, metrics, and sample generations suitable for evaluation.

Final training run characteristics (best observed metrics):

- **Model**: 2×224-unit bidirectional LSTM layers, multi-head attention (4 heads, key dim 64), FFN size 320, embedding dropout 0.18, dropout 0.30, recurrent dropout 0.22, attention dropout 0.28, tied embeddings.
- **Tokenizer**: SentencePiece (vocab 16 000, stride 1), sliding-window sequences of length 45.
- **Optimizer**: AdamW (learning rate 2.4×10⁻⁴ with 1500-step warmup + 60 000-step cosine decay, weight decay 6×10⁻⁴).
- **Early stopping**: patience 10 on validation loss, best checkpoint restored.

Artifacts and generated samples live under `artifacts/`.

## 2. Metrics

From `artifacts/metrics.json`:

| Split | Accuracy | Top‑5 Accuracy | Loss | Perplexity |
|-------|----------|----------------|------|------------|
| Train | 0.2633   | 0.4948         | —    | 42.50      |
| Val   | 0.1889   | 0.3887         | 5.5119 | 247.61   |
| Test  | 0.1712   | 0.3859         | 5.1132 | 166.20   |

Validation loss reached a minimum near epoch 5 (≈5.07) and then slowly drifted upward. Validation topology shows a plateau around accuracy 0.19. Test perplexity meets the <250 requirement; accuracy remains well below the 75 % target because of data limitations.

## 3. Generated Samples

Three 35-word generations with probability traces (needed for the assignment) are stored as:

- `artifacts/generation_seed1.json` – seed: “I saw Holmes”.
- `artifacts/generation_seed2.json` – seed: “The adventure continued”.
- `artifacts/generation_seed3.json` – seed: “She looked at him”.

Each JSON logs the generated sequence and top‑5 candidate probabilities at every step. Example snippet (“She looked at him” seed):

```
Seed: She looked at him
Generated: she looked at him. i am sure that i am sure that i have been a little more than ...
Top-5 probabilities (first step):
  . : 0.2252
  , : 0.0721
  ▁to : 0.0529
  ▁in : 0.0208
  ▁with : 0.0188
```

For quick inference after training:

```bash
uv run python inference.py \
  --workspace "$(pwd)" \
  --seed "I saw Holmes" \
  --max-words 30 \
  --top-k 5 \
  --output artifacts/generation_cli.json
```

`inference.py` performs minimal post-processing (whitespace collapse, capitalization, trailing period if missing) and prints the first-step top‑5 distribution.

## 4. Experiments & Findings

### 4.1 Tokenization & Context

- Started with 20 k SentencePiece vocab: val/test accuracy stagnated ~0.16 and perplexity >600.
- Moving to 16 k (current best) reduced perplexity to ~167 with slightly higher accuracy.
- Vocab <10 k dropped accuracy sharply (VLMs lacked lexical diversity).
- Sequence length 45 with stride 1 worked best; stride 2 improved generalization but reduced accuracy to ~0.15. Final model uses stride 1.

### 4.2 Architecture

- Baseline had 3×256/320 BiLSTMs (≈17 M params) → severe overfitting (train >80 %, val ~16 %).
- Final architecture: 2×224 BiLSTM + attention + FFN head (≈11–12 M params with weight tying). Balanced accuracy with manageable overfitting.
- Embedding dropout (0.18) and dropout 0.30 kept the model from memorizing sequences too aggressively while maintaining accuracy.

### 4.3 Regularization & Optimization

- Swept dropout (0.2–0.5), recurrent dropout (0.1–0.35), attention dropout (0.1–0.4): moderate values (0.30/0.22/0.28) gave the best trade-off; heavy dropout collapsed accuracy.
- Label smoothing (0.0–0.02): best runs used 0.0 (higher smoothing depressed accuracy further).
- Weight decay 6×10⁻⁴–1×10⁻³: 6×10⁻⁴ is the sweet spot (strong enough to discourage overfit).
- AdamW with warmup+cosine scheduling provided smoother training than static LR.

### 4.4 Failed or Less Effective Attempts

- Additional LSTM layers or higher unit counts triggered severe overfitting (train accuracy >0.8, val <0.18).
- Single-layer 192-unit setups stabilized but capped accuracy near 0.14.
- Aggressively shrinking attention/feed-forward dimensions induced underfitting.
- Stride >1 or dropout ≥0.4 reduced overfitting but also pushed accuracy down to ≤0.15.

### 4.5 Why Results Still Lag

- The corpus has ~100 k tokens; a 16 k softmax dilutes supervision for many tokens (rare words appear only dozens of times). LSTM models need more data or broader context to reach 75 %+ accuracy.
- LSTMs handle dependencies, but not as effectively as modern transformer LMs on limited data.

1. **Expand the corpus**: include additional Sherlock Holmes stories (Project Gutenberg has multiple volumes). Larger dataset would directly boost accuracy/perplexity with the same architecture.
2. **Transfer learning** (if permitted): fine-tune a small pretrained LM, or initialize with pretrained embeddings; even partial transfer provides a major uplift.
3. **Architecture upgrade**: consider a lightweight transformer encoder or hybrid LSTM/transformer, which typically learns better context patterns than pure LSTM on small corpora.
4. **Future tuning on current setup**:
   - SentencePiece vocab 12 k with byte fallback.
   - Mix stride 1 and 2 windows for data augmentation.
   - Word dropout/data augmentation (mask randomly) to regularize.
   - Monitor validation early (epoch 5–10) to avoid drift; best checkpoint at val_loss minimum (≈5.07).

## 6. Deliverables

- **Code**: `src/next_word_prediction/` (data, model, train, generate) + `inference.py`.
- **Artifacts** (post-training):
  - `artifacts/model_config.json`, `metrics.json`, `history.json`, `final_model.keras`.
  - Generation samples: `generation_seed1.json`, `generation_seed2.json`, `generation_seed3.json`.
- **Documentation**: README (setup + training/inference instructions) and this Report.

This deliverable showcases the complete pipeline, summarizes the strongest metrics achieved, and records the tuning journey with its limitations. Substantial accuracy gains will require expanded training data or a pretrained backbone.
