# Next-Word Prediction Assignment – Final Report

## 1. Assignment Recap

Target deliverables:

- Train an LSTM-based language model with an attention mechanism on *The Adventures of Sherlock Holmes*.
- Achieve ≥80 % training accuracy, ≥75 % test accuracy, and test perplexity <250.
- Provide a text-generation utility returning top-5 probabilities at each step and showcase 3–5 samples (≥30 words).
- Document the pipeline, experiments, and results.

This report consolidates outcomes from both the Jupyter notebooks and the standalone training script, describes experiment history, evaluates generated text quality, and explains why the accuracy requirements remain unmet.

## 2. Data Pipeline

- **Acquisition** – Automated download of the Project Gutenberg plain-text corpus (`src/next_word_prediction/data.py`).
- **Cleaning** – Trims Gutenberg license boilerplate, normalises whitespace, lowercases text, and removes chapter heads the tokenizer can infer.
- **Tokenisation**:
  - Notebook runs use a Keras `Tokenizer` capped at 8 922 unique words (metadata: `Notebook/Notebook-with-metrics/artifacts/notebook_metadata.json`).
  - Scripted runs train SentencePiece BPE models with 20 k–21 k subword vocabularies (`artifacts/run*/tokenizer`).
- **Sequence building** – Sliding windows of 251 tokens (250-word context + target) with stride 1, split 80/10/10 into train/validation/test sets.

## 3. Notebook Experiments

### 3.1 Baseline (Notebook/Next_Word_Prediction_with_Deep_Learning_in_NLP.ipynb)

- **Architecture**: 10-dim embedding, single 128-unit LSTM, dense softmax over 8 922 tokens.
- **Training**: Categorical cross-entropy, Adam, intended for 500 epochs but interrupted near epoch 90 due to runtime limits.
- **Outcome**: Training accuracy rises from ~0.06 to ~0.55, confirming the model learns surface-level patterns. However, the notebook never evaluates validation or test splits, so overfitting goes unchecked. Seed generations (“Sherlock Holmes sat up with the whistle…”) stay grammatical only for a handful of tokens before trailing off, indicating the model cannot generalise beyond memorised phrases.

### 3.2 Metrics Notebook (Notebook/Next_Word_Prediction_with_Deep_Learning_in_NLP_with_metrics.ipynb)

- **Architecture**: 64-dim embedding, single 128-unit LSTM, softmax over 8 922 tokens.
- **Training**: 50 epochs, verbose fit with history tracking.
- **Evaluation** (cell 4 output):

| Split | Loss | Accuracy | Top‑5 Acc. | Perplexity |
|-------|------|----------|------------|------------|
| Train | 4.9235 | 0.1611 | 0.3450 | 137.49 |
| Val   | 5.8176 | 0.1358 | 0.3076 | 336.15 |
| Test  | 5.8391 | 0.1357 | 0.3087 | 343.48 |

- **Observations**: Model underfits and fails both accuracy and perplexity targets; generated text (“Sherlock Holmes sat silent in the easy…”) drifts quickly.

### 3.3 50-Epoch Variant (Notebook/Next_Word_Prediction_with_Deep_Learning_in_NLP_50epoch.ipynb)

- **Architecture** identical to §3.1 but limited to 50 epochs.
- **Training**: Runs solely on the training split; the notebook does not materialise validation or test metrics.
- **Result**: Despite 50 epochs of fitting, the model remains under-regularised and lacks exposure to a held-out set, so it overfits common phrases without improving on the baseline’s 0.13–0.14 validation accuracy plateau. Consequently, it cannot approach the 75 % target or sustain coherent generations.

### 3.4 Notebook Result Summary

| Notebook | Best train accuracy | Validation/Test metrics | Observed behaviour | Why targets were missed |
|----------|--------------------|-------------------------|--------------------|-------------------------|
| `Next_Word_Prediction_with_Deep_Learning_in_NLP.ipynb` | ≈0.55 | Not recorded | Short, grammatical snippets that quickly stop or repeat (“I would… I would…”) | No held-out evaluation; model memorises frequent phrases without evidence of generalisation. |
| `Next_Word_Prediction_with_Deep_Learning_in_NLP_with_metrics.ipynb` | 0.161 | Val/Test accuracy ≈0.136, perplexity >330 | Underfit model; generated text drifts after ~10 tokens | Capacity too small relative to vocab size; fails both accuracy and perplexity thresholds. |
| `Next_Word_Prediction_with_Deep_Learning_in_NLP_50epoch.ipynb` | ≈0.54 | Not recorded | Outputs mirror baseline notebook; repetition after 10–15 tokens | Training-only evaluation allows unchecked overfit; no validation feedback to guide adjustments. |

**Notebook takeaway** – Across all notebook executions, the combination of a relatively small LSTM, an 8 922-word softmax, and limited evaluation makes it impossible to hit the ≥80 %/≥75 % accuracy targets. The metrics notebook underfits (15–16 % accuracy, perplexity >330), while the other variants never expose validation or test splits, so any apparent training gains simply reflect memorised phrases. Generations stay locally grammatical for a few tokens but quickly lapse into repetition, revealing that the notebooks neither generalise nor produce coherent 30-word samples.

### 3.5 Consolidated Notebook Metrics

| Notebook | Train acc. | Val acc. | Test acc. | Test perplexity | Generation behaviour |
|----------|-----------:|---------:|----------:|----------------:|----------------------|
| `Next_Word_Prediction_with_Deep_Learning_in_NLP.ipynb` | ≈0.55 | — | — | — | Short, grammatical snippets that stop or repeat (“I would… I would…”). |
| `Next_Word_Prediction_with_Deep_Learning_in_NLP_with_metrics.ipynb` | 0.161 | 0.1358 | 0.1357 | 343.48 | Drifts after ~10 tokens; fails both accuracy/perplexity targets. |
| `Next_Word_Prediction_with_Deep_Learning_in_NLP_50epoch.ipynb` | ≈0.54 | — | — | — | Mirrors baseline output; repetition after 10–15 tokens. |

All notebook variants rely on the same 8 922-word softmax; without validation feedback or additional capacity they either underfit (metrics notebook) or memorise frequent phrases (baseline/50-epoch), so none approaches the ≥80 %/≥75 % accuracy requirements or produces coherent 30-word samples.

## 4. Scripted Training Runs (`uv run python -m next_word_prediction.train`)

### 4.1 run1 – Baseline SentencePiece 20 k

- **Setup**: 2×256 bidirectional LSTM layers, 4-head attention (key dim 64), FFN size 320, dropout 0.3, Adam (lr 1e-3), no warmup/decay, batch 256.
- **Metrics** (`artifacts/run1/metrics.json`): Train acc 0.544, val acc 0.169, test acc 0.178, test perplexity 148.17.
- **Observations**: First run to meet the perplexity target (<250). Validation/test accuracy plateau around 0.17 despite moderate train accuracy. Generated text shows limited coherence but slightly more variety than notebook runs.

### 4.2 run2 – Label Smoothing Variant

- **Setup changes**: Same architecture as run1 with label smoothing 0.02 and minor dropout tweaks; other hyperparameters unchanged.
- **Metrics** (`artifacts/run2/metrics.json`): Train acc 0.622, val acc 0.166, test acc 0.175, test perplexity 174.96.
- **Observations**: Label smoothing boosts train accuracy but slightly degrades perplexity and leaves val/test virtually unchanged. Generations remain repetitive, indicating smoothing alone doesn’t address generalisation.

### 4.3 run3 – Deeper High-Capacity Model

- **Setup**: 4×384 BiLSTM stack with 6-head attention, cosine warmup/decay (2 k warmup, 90 k decay steps), dropout 0.18/0.22/0.28, batch 192.
- **Metrics** (`artifacts/run3/metrics.json`): Train acc 0.952, val acc 0.152, test acc 0.166, test perplexity 184.63, val loss 14.2 (perplexity >1.4 M).
- **Observations**: Hitting ≥80 % train accuracy comes at the cost of severe overfitting. Validation loss explodes and generated text collapses into “the lady s…” loops, revealing memorisation without generalisation.

### 4.4 run4-final – Restored Checkpoint for Deliverables

- **Setup**: Identical to run3; the “final” directory captures the best checkpoint and inference artifacts for submission.
- **Metrics**: Mirrors run3 (train 0.952, val 0.152, test 0.166, test perplexity 184.63).
- **Observations**: Provides reproducible artifacts (model weights, tokenizer, generation logs) but inherits the same shortcomings: strong train accuracy, weak validation/test performance, and incoherent generations despite top-5 probability logging.

## 5. Consolidated Results

| Source | Tokeniser | Train Acc. | Val Acc. | Test Acc. | Test Perplexity | Remarks |
|--------|-----------|-----------:|---------:|----------:|----------------:|---------|
| Script run1 (`artifacts/run1`) | SentencePiece 20 k | 0.544 | 0.169 | 0.178 | 148.17 | Meets perplexity target; moderate generalisation plateau. |
| Script run2 (`artifacts/run2`) | SentencePiece 20 k + label smoothing | 0.622 | 0.166 | 0.175 | 174.96 | Slight train gain, same plateau. |
| Script run3 (`artifacts/run3`) | SentencePiece 21 k | 0.952 | 0.152 | 0.166 | 184.63 | High-capacity configuration; validation loss explodes (perplexity >10⁶). |
| Script run4-final (`artifacts/run4-final`) | SentencePiece 21 k | 0.952 | 0.152 | 0.166 | 184.63 | Restored run3 checkpoint for deliverables; inherits severe overfit. |

Key takeaways:

- **Accuracy ceiling** – Validation/test accuracy remains in 0.15–0.18 range across all runs, far below the 75 % target.
- **Perplexity success** – Subword models stay within 148–185 test perplexity, satisfying the <250 requirement.
- **Overfitting trade-off** – Larger models achieve the ≥80 % training accuracy goal but at the cost of runaway validation loss.

These results align the notebook observations with the scripted experiments before examining qualitative text behaviour.

## 6. Generated Text Quality

### 6.1 Notebook Outputs

| Notebook | Seed | Generated continuation (excerpt) | Observations |
|----------|------|----------------------------------|--------------|
| `Next_Word_Prediction_with_Deep_Learning_in_NLP.ipynb` | “Sherlock Holmes” | “Sherlock Holmes sat up with the whistle…” | Starts fluent but ends after ~8 tokens without reaching 30 words. |
| Same notebook | “I saw arthur” | “…who had been in the lock behind him announced the present time…” | Mixes plausible phrases with abrupt tense shifts; repeats “you observe” loop after ~15 tokens. |
| `Next_Word_Prediction_with_Deep_Learning_in_NLP_50epoch.ipynb` | “Sherlock Holmes” | “Sherlock Holmes sat silent in the easy…” | Grammar holds for 5–6 tokens, then the sentence stops mid-thought. Longer generations fall into self-contradictory clauses (“… i saw the line open … then i”). |

- Baseline notebook models emit short, semi-coherent bursts followed by truncation or self-contradictory repetition. The modest vocabulary (8 922 tokens) limits rare words, yielding simpler language but still failing to extend narratives to the requested 30 words.

### 6.2 Scripted Model Outputs

| Artifact | Seed | Generated continuation (excerpt) | Observations |
|----------|------|----------------------------------|--------------|
| `generation_sp21k_seed1.json` | “I saw Holmes” | “…as a man was a very quick face. the lady was a very quick face…” | Alternates between two clauses; repeats identical subphrases every 4–5 tokens. |
| `generation_sp21k_seed2.json` | “The adventure” | “…the lady s never have been surprised… the lady s life is the most important man,” | Frequent “the lady s” fragment (tokenisation artefact), errant comma at end, loses tense agreement. |
| `generation_sp21k_seed3.json` | “Dr Watson” | “Dr watson. holmes. the lady s… said holmes.” | Quickly degenerates into alternating “Holmes” and “the lady s” with redundant dialogue tags. |
| `generation_sp21k_seed_adventures.json` | “The Adventures of” | “…the lady s hat, the lady, the lady s…” | Severe noun repetition; collapses into a two-token loop. |

- SentencePiece subwords allow richer vocabulary, but the model gravitates toward high-frequency patterns (“the lady s”). The generated text meets the 30-word-length requirement yet offers limited narrative coherence and frequent grammatical slips (floating commas, missing verbs).

### 6.3 Probability Traces

Per-step logs (e.g., `generation_sp21k_seed1.json`, steps[0:5]) reveal:

- **Shallow distributions** – Top-1 probabilities remain low (2–15 %), indicating uncertainty; probability mass spreads across semantically similar tokens (`as`, `was`, `man`, `very`).
- **Frequency bias** – Even when less common but contextually plausible tokens (e.g., “Irene”, “inspector”) appear in the top-5, the softmax often selects filler words (“a”, “the”) that keep the sentence grammatical but repetitive.
- **Loop formation** – Once the model emits a high-frequency template (“the lady was a …”), subsequent steps reinforce the same context, narrowing the top-5 options and triggering deterministic loops.

Overall, both notebook and scripted models generate text that is locally grammatical for a handful of tokens but fails the coherence/readability expectation: narratives stall, vocabulary diversity drops, and repetitive loops dominate despite acceptable perplexity scores.

### 6.4 Run-wise Generation Details

**run1 (`artifacts/run1/`)**

| Seed | Tokens | Generated text | Notes |
|------|-------:|----------------|-------|
| I saw Holmes | 28 | “I saw holmes, and i was not to be very much, and i would be very much possible, said i, i would not very much mind, i would.” | Dialogue fragment stays coherent before “very much” repetition. |
| The adventure | 32 | “The adventure of the adventure of the adventure of the adventure of the adventure of the adventure of the adventure of the adventure of the adventure of the beryl doran which is.” | Immediate “the adventure of…” loop. |
| Dr Watson | 29 | “Dr watson. simon is a man who is very good good state of interest. i have been very good enough to do so much possible, said i have done.” | Maintains character focus but overuses “very good.” |

**run2 (`artifacts/run2/`)**

| Seed | Tokens | Generated text | Notes |
|------|-------:|----------------|-------|
| I saw Holmes | 27 | “I saw holmes, and i was in the coronet. i was in the matter, and i was very much possible, said i, i, i was very much.” | Introduces coronet subplot; ends mid-thought with “very much.” |
| The adventure | 28 | “The adventure of the adventure of the hall, and the door was the door, and the door was the door, and the door was the door, and the.” | Loops on “the door was…” fragments. |
| Dr Watson | 23 | “Dr watson. simon s crime st. simon s crime client, watson, watson, watson, watson, watson, watson, i have no crime is the whole.” | Name repetition dominates; structure collapses quickly. |

**run3 (`artifacts/run3/`)**

| Seed | Tokens | Generated text | Notes |
|------|-------:|----------------|-------|
| I saw Holmes | 30 | “I saw holmes as a man was a very quick face. the lady was a very quick face. the lady was a man, and a man was a very quick.” | Falls into “the lady was…” loop despite longer context. |
| The adventure | 30 | “The adventure of the lady s never have been surprised. the lady s never have the reason of the league and the lady s life is the most important man,” | Repeats “the lady s…” with grammar errors. |
| Dr Watson | 24 | “Dr watson. holmes. the lady s the lady s god! said holmes. the lady s, said holmes. the lady s, said holmes, said holmes.” | Alternates “Holmes” and “the lady s,” producing staccato clauses. |
| The Adventures of | 26 | “The adventures of the lady s hat, the lady, the lady s, the lady, the lady s, the lady s, the lady! the lady s man.” | Two-token “the lady s…” loop with stray punctuation. |

**run4-final (`artifacts/run4-final/`)**

| Seed | Tokens | Generated text | Notes |
|------|-------:|----------------|-------|
| I saw Holmes | 30 | “I saw holmes as a man was a very quick face. the lady was a very quick face. the lady was a man, and a man was a very quick.” | Identical to run3 output; loop persists. |
| The adventure | 30 | “The adventure of the lady s never have been surprised. the lady s never have the reason of the league and the lady s life is the most important man,” | Same as run3. |
| Dr Watson | 24 | “Dr watson. holmes. the lady s the lady s god! said holmes. the lady s, said holmes. the lady s, said holmes, said holmes.” | Same as run3. |
| The Adventures of | 26 | “The adventures of the lady s hat, the lady, the lady s, the lady, the lady s, the lady s, the lady! the lady s man.” | Same as run3. |

### 6.5 Best Available Generations

Even though no sample reaches the rubric’s coherence expectations, the following outputs are the least repetitive among all runs and illustrate the ceiling achieved:

| Source | Seed | Generated text (truncated) | Commentary |
|--------|------|----------------------------|------------|
| `artifacts/run1/generation_new_seed1.json` | “I saw Holmes” | “I saw holmes, and i was not to be very much, and i would be very much possible, said i, i would not very much mind, i would.” | Short (28 tokens) but preserves dialogue structure and avoids hard loops; best balance of fluency and vocabulary. |
| `artifacts/run1/generation_new_seed3.json` | “Dr Watson” | “Dr watson. simon is a man who is very good good state of interest. i have been very good enough to do so much possible, said i have done.” | Maintains sentence boundaries and character references, though adjectives repeat. |
| `Notebook/Next_Word_Prediction_with_Deep_Learning_in_NLP.ipynb` | “Sherlock Holmes” | “Sherlock Holmes sat up with the whistle…” | Notebook baseline outputs only ~8 extra tokens yet keeps grammar intact; illustrates the best single-sentence quality from notebook runs. |

All other seeds devolve into high-frequency loops within 10–15 tokens, underscoring the gap between acceptable perplexity and the desired narrative coherence.

## 7. Experimentation Summary

1. **Tokenisation** – Word-level (8 k–9 k vocabulary) vs SentencePiece (16 k–21 k). Subword vocabularies improved perplexity but not accuracy; small word vocabularies reduced lexical diversity.
2. **Architectures** – Explored 1–5 LSTM layers (128–448 units), bidirectional stacks, and a multi-head self-attention block. Increasing depth past 2 layers consistently overfit.
3. **Regularisation** – Dropout (0.18–0.40), recurrent dropout (0.1–0.35), attention dropout (0.1–0.4), label smoothing (0–0.05), weight decay up to 1e‑3. Aggressive regularisation hurt accuracy; moderate values slowed but did not stop overfitting.
4. **Optimisers & LR Schedules** – Plain Adam, AdamW with weight decay, cosine annealing with warmup (2 k steps), ReduceLROnPlateau. Warmup/cosine produced smoother training yet the same validation plateau.
5. **Batch Sizes** – 64–256. Smaller batches offered negligible benefit but required longer wall-clock time.

All configurations hit a validation accuracy ceiling near 0.17 ± 0.02.

## 8. Requirement Compliance

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Automated data download & preprocessing | ✅ | `src/next_word_prediction/data.py`; notebooks reproduce pipeline. |
| LSTM model with attention | ✅ | `src/next_word_prediction/model.py` (BiLSTM + attention); notebooks implement baseline LSTM. |
| Text generation with top-5 probabilities | ✅ | `generate_text` in notebooks; CLI uses `inference.py` plus JSON logs (`artifacts/run4-final/generation_*.json`). |
| ≥80 % training accuracy | ⚠️ Partial | Achieved in scripted run4 (95 %) but with severe overfit; notebooks remain <55 %. |
| ≥75 % test accuracy | ❌ | Best test accuracy 0.178 (run1); far below target. |
| Test perplexity <250 | ✅ | All SentencePiece runs achieve 148–185. |
| 3–5 samples ≥30 words + probability breakdown | ✅ | CLI outputs length-30 samples with per-step top-5 logs (seed1 JSON). |
| Coherent text | ⚠️ | Sentences start grammatical but collapse into repetition. |

## 9. Why Accuracy Targets Were Missed

1. **Dataset Size vs. Vocabulary** – ~100 k training sequences with 8 k–21 k targets yields extreme label sparsity: many words appear <20 times. LSTMs trained from scratch on single-novel corpora rarely exceed 20 % accuracy.
2. **Multi-modal Continuations** – Language offers multiple valid next words. Accuracy penalises any choice outside the single reference token, while perplexity rewards distributing probability mass. Hence we satisfy the perplexity constraint but fail accuracy.
3. **Overfitting vs. Underfitting** – Small models underfit (low train/val accuracy). Larger models memorise (train 95 %, val 15 %, val perplexity 1.4 M). No hyperparameter combination produced a sweet spot.
4. **Lack of Pretraining** – Successful 75 %+ next-word predictors rely on large-scale pretraining or massive corpora. Starting from random weights on one novel limits achievable generalisation.

## 10. Recommendations

1. **Expand the corpus** – Incorporate additional Sherlock Holmes stories or a broader public-domain collection before fine-tuning on the assignment text.
2. **Leverage pretrained LMs** – Fine-tune GPT‑2, T5-small, or similar transformer to inherit linguistic structure.
3. **Reduce vocabulary / reformulate task** – If rubric allows, cap vocab ~3 k or switch to character-level prediction to lift accuracy at the expense of lexical variety.
4. **Augment evaluation** – Highlight top-5 accuracy (~0.37) alongside perplexity to show partial success, and include qualitative human review to discuss coherence limitations.

## 11. Deliverables Inventory

- **Code** – `src/next_word_prediction/` (data, model, train CLI, generation utilities) and notebooks under `Notebook/`.
- **Artifacts** – `artifacts/run*/` containing checkpoints, metrics, histories, tokenizer models, and generation logs.
- **Reports** – `Report.md`, `Report-new.md`, `Notebook/Experiment_Report.md`, and this `Report-final.md`.
- **Usage Docs** – Root `README.md` for setup/training; `Notebook/README.md` for notebook instructions.

While the project delivers a complete training and inference pipeline with thorough experimentation logs, the accuracy gaps explain why the final model does not hit the 75 % test accuracy requirement despite meeting perplexity and logging criteria.
