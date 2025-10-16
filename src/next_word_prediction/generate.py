"""Text generation utilities for the trained model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .data import TokenizerBundle, load_tokenizer
from .model import ModelConfig, build_model


@dataclass
class GenerationResult:
    generated_text: str
    steps: List[Dict[str, object]]

    def to_dict(self) -> Dict[str, Any]:
        return {"generated_text": self.generated_text, "steps": self.steps}


def _load_model(workspace: Path) -> Tuple[keras.Model, dict]:
    artifacts_dir = workspace / "artifacts"
    model_path = artifacts_dir / "final_model.keras"
    config_path = artifacts_dir / "model_config.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}")

    config_dict = json.loads(config_path.read_text(encoding="utf-8"))
    model_config = ModelConfig(**config_dict)
    model = build_model(model_config)
    model.load_weights(model_path)
    return model, config_dict


def _generate_word_tokens(
    model: keras.Model,
    tokenizer_bundle: TokenizerBundle,
    seed_text: str,
    max_words: int,
    sequence_length: int,
    top_k: int,
) -> GenerationResult:
    tokenizer = tokenizer_bundle.tokenizer
    index_word = tokenizer.index_word
    generated_tokens = tokenizer.texts_to_sequences([seed_text.strip().lower()])[0]
    if not generated_tokens:
        raise ValueError("Seed text produced no tokens with the fitted tokenizer.")

    steps: List[Dict[str, object]] = []
    for _ in range(max_words):
        context_tokens = generated_tokens[-sequence_length:]
        padded = pad_sequences([context_tokens], maxlen=sequence_length, padding="pre")
        probs = model.predict(padded, verbose=0)[0]

        sorted_indices = np.argsort(probs)[::-1]

        top_preds = []
        for idx in sorted_indices:
            token = index_word.get(idx, "<oov>") if idx != 0 else "<pad>"
            top_preds.append((token, float(probs[int(idx)])))
            if len(top_preds) == top_k:
                break

        next_idx = None
        for idx in sorted_indices:
            if idx == 0:
                continue  # skip padding token
            token = index_word.get(idx, "<oov>")
            if token != "<oov>":
                next_idx = int(idx)
                break
        if next_idx is None:
            # fall back to the highest-probability token (likely <oov>)
            next_idx = int(sorted_indices[0])
        generated_tokens.append(next_idx)

        context_words = [index_word.get(idx, "<oov>") if idx != 0 else "<pad>" for idx in context_tokens]
        predicted_word = index_word.get(next_idx, "<oov>") if next_idx != 0 else "<pad>"
        steps.append({"context_tokens": context_words, "next_token": predicted_word, "top_k": top_preds})

    text = " ".join(index_word.get(idx, "<oov>") for idx in generated_tokens if idx != 0)
    return GenerationResult(generated_text=text, steps=steps)


def _generate_sentencepiece(
    model: keras.Model,
    tokenizer_bundle: TokenizerBundle,
    seed_text: str,
    max_words: int,
    sequence_length: int,
    top_k: int,
) -> GenerationResult:
    sp = tokenizer_bundle.tokenizer
    generated_ids = sp.encode(seed_text.strip().lower(), out_type=int)
    if not generated_ids:
        raise ValueError("Seed text produced no tokens with the SentencePiece tokenizer.")

    steps: List[Dict[str, object]] = []
    for _ in range(max_words):
        context_ids = generated_ids[-sequence_length:]
        padded = pad_sequences([context_ids], maxlen=sequence_length, padding="pre")
        probs = model.predict(padded, verbose=0)[0]

        top_indices = np.argsort(probs)[::-1][:top_k]
        top_preds = [(sp.id_to_piece(int(idx)), float(probs[int(idx)])) for idx in top_indices]

        generated_ids.append(int(top_indices[0]))
        steps.append(
            {
                "context_tokens": [sp.id_to_piece(int(idx)) for idx in context_ids],
                "next_token": top_preds[0][0],
                "top_k": top_preds,
            }
        )

    text = sp.decode_ids(generated_ids)
    return GenerationResult(generated_text=text, steps=steps)


def generate_text(
    workspace: Path,
    seed_text: str,
    max_words: int,
    top_k: int = 5,
) -> Tuple[str, List[Dict[str, object]]]:
    model, config = _load_model(workspace)
    tokenizer_bundle = load_tokenizer(workspace / "artifacts")
    sequence_length = config["sequence_length"]

    if tokenizer_bundle.kind == "word":
        return _generate_word_tokens(model, tokenizer_bundle, seed_text, max_words, sequence_length, top_k)
    if tokenizer_bundle.kind == "sentencepiece":
        return _generate_sentencepiece(model, tokenizer_bundle, seed_text, max_words, sequence_length, top_k)
    raise ValueError(f"Unsupported tokenizer kind: {tokenizer_bundle.kind}")
