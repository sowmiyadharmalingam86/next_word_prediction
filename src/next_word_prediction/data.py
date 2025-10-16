"""Data acquisition and preprocessing utilities for the Sherlock Holmes corpus."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
import sentencepiece as spm
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

CORPUS_URL = "https://www.gutenberg.org/files/1661/1661-0.txt"
DEFAULT_SEQUENCE_LENGTH = 20

START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK"
END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK"


@dataclass
class TokenizerBundle:
    """Container describing the tokenizer used for training/generation."""

    kind: str  # "word" or "sentencepiece"
    tokenizer: Any
    vocab_size: int
    metadata: Dict[str, Any]


def download_corpus(target_path: Path) -> Path:
    """Download the Sherlock Holmes corpus into ``target_path``."""
    target_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(CORPUS_URL, timeout=30)
    response.raise_for_status()

    target_path.write_bytes(response.content)
    return target_path


def load_corpus(path: Path) -> str:
    """Load the raw text from ``path``."""
    return path.read_text(encoding="utf-8")


def clean_corpus(raw_text: str) -> str:
    """Strip Gutenberg header/footer and perform light normalization."""
    start_idx = raw_text.find(START_MARKER)
    end_idx = raw_text.find(END_MARKER)

    if start_idx != -1:
        raw_text = raw_text[start_idx:]
    if end_idx != -1:
        raw_text = raw_text[:end_idx]

    text = raw_text.lower()
    text = re.sub(r"_+", " ", text)  # replace emphasis markers
    text = re.sub(r"[^a-z0-9'\".,;:!?\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def build_word_tokenizer(text: str, num_words: int | None = None) -> TokenizerBundle:
    """Fit a classic Keras tokenizer on the corpus."""
    tokenizer = Tokenizer(num_words=num_words, oov_token="<oov>")
    tokenizer.fit_on_texts([text])
    vocab_size = min(num_words or len(tokenizer.word_index) + 1, len(tokenizer.word_index) + 1)
    return TokenizerBundle(
        kind="word",
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        metadata={"num_words": num_words},
    )


def train_sentencepiece_model(text: str, model_dir: Path, vocab_size: int) -> Path:
    """Train (or reuse) a SentencePiece model."""
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "tokenizer.model"
    if model_path.exists():
        return model_path

    corpus_path = model_dir / "corpus.txt"
    sentences = re.split(r"(?<=[.!?])\s+", text)
    corpus_path.write_text("\n".join(sentences), encoding="utf-8")
    model_prefix = model_dir / "tokenizer"

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        max_sentence_length=10000,
    )
    return model_path


def build_sentencepiece_tokenizer(
    text: str, workspace: Path, vocab_size: int
) -> TokenizerBundle:
    """Create a SentencePiece tokenizer bundle."""
    cache_dir = workspace / "tokenizer_cache" / f"spm_{vocab_size}"
    model_path = train_sentencepiece_model(text, cache_dir, vocab_size)
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    metadata = {
        "model_path": str(model_path),
        "vocab_path": str(model_path.with_suffix(".vocab")),
        "pad_id": sp.pad_id(),
        "unk_id": sp.unk_id(),
        "bos_id": sp.bos_id(),
        "eos_id": sp.eos_id(),
    }
    return TokenizerBundle(
        kind="sentencepiece",
        tokenizer=sp,
        vocab_size=sp.get_piece_size(),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Sequence generation
# ---------------------------------------------------------------------------


def _word_sequences(
    tokenizer: Tokenizer,
    text: str,
    sequence_length: int,
    exclude_oov: bool,
    strategy: str,
    stride: int,
) -> List[List[int]]:
    sequences: List[List[int]] = []

    if strategy == "sliding":
        token_list = tokenizer.texts_to_sequences([text])[0]
        for idx in range(sequence_length, len(token_list), stride):
            seq = token_list[idx - sequence_length : idx + 1]
            if exclude_oov and 1 in seq:
                continue
            sequences.append(seq)
        return sequences

    sentences = re.split(r"(?<=[.!?])\s+", text)
    max_tokens = sequence_length + 1
    for sentence in sentences:
        tokens = tokenizer.texts_to_sequences([sentence])[0]
        if not tokens:
            continue
        for i in range(1, len(tokens)):
            seq = tokens[: i + 1]
            if len(seq) > max_tokens:
                seq = seq[-max_tokens:]
            if exclude_oov and 1 in seq:
                continue
            sequences.append(seq)

    return sequences


def _sentencepiece_sequences(
    sp: spm.SentencePieceProcessor,
    text: str,
    sequence_length: int,
    exclude_oov: bool,
    strategy: str,
    stride: int,
) -> List[List[int]]:
    sequences: List[List[int]] = []
    unk_id = sp.unk_id()

    if strategy == "sliding":
        token_list = sp.encode(text, out_type=int)
        for idx in range(sequence_length, len(token_list), stride):
            seq = token_list[idx - sequence_length : idx + 1]
            if exclude_oov and unk_id in seq:
                continue
            sequences.append(seq)
        return sequences

    sentences = re.split(r"(?<=[.!?])\s+", text)
    max_tokens = sequence_length + 1
    for sentence in sentences:
        tokens = sp.encode(sentence, out_type=int)
        if not tokens:
            continue
        for i in range(1, len(tokens)):
            seq = tokens[: i + 1]
            if len(seq) > max_tokens:
                seq = seq[-max_tokens:]
            if exclude_oov and unk_id in seq:
                continue
            sequences.append(seq)
    return sequences


def generate_sequences(
    bundle: TokenizerBundle,
    text: str,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    exclude_oov: bool = True,
    strategy: str = "ngram",
    stride: int = 1,
) -> List[List[int]]:
    if bundle.kind == "word":
        return _word_sequences(bundle.tokenizer, text, sequence_length, exclude_oov, strategy, stride)
    if bundle.kind == "sentencepiece":
        return _sentencepiece_sequences(bundle.tokenizer, text, sequence_length, exclude_oov, strategy, stride)
    raise ValueError(f"Unsupported tokenizer kind: {bundle.kind}")


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------


def train_val_test_split(
    sequences: np.ndarray, val_size: float = 0.1, test_size: float = 0.1, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y = sequences[:, :-1], sequences[:, -1]
    x_temp, x_test, y_temp, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    val_ratio = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=val_ratio, random_state=random_state, shuffle=True
    )
    return x_train, y_train, x_val, y_val, x_test, y_test


def save_tokenizer(bundle: TokenizerBundle, artifacts_dir: Path) -> None:
    """Persist tokenizer information for generation."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = artifacts_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, Any] = {"kind": bundle.kind, "vocab_size": bundle.vocab_size}

    if bundle.kind == "word":
        tokenizer_json = bundle.tokenizer.to_json()
        json_path = tokenizer_dir / "tokenizer.json"
        json_path.write_text(tokenizer_json, encoding="utf-8")
        meta.update({"file": "tokenizer/tokenizer.json", "num_words": bundle.metadata.get("num_words")})
    elif bundle.kind == "sentencepiece":
        model_src = Path(bundle.metadata["model_path"])
        vocab_src = Path(bundle.metadata["vocab_path"])
        model_dst = tokenizer_dir / "tokenizer.model"
        vocab_dst = tokenizer_dir / "tokenizer.vocab"
        shutil.copy(model_src, model_dst)
        shutil.copy(vocab_src, vocab_dst)
        meta.update(
            {
                "model_file": "tokenizer/tokenizer.model",
                "vocab_file": "tokenizer/tokenizer.vocab",
                "pad_id": bundle.metadata.get("pad_id"),
                "unk_id": bundle.metadata.get("unk_id"),
                "bos_id": bundle.metadata.get("bos_id"),
                "eos_id": bundle.metadata.get("eos_id"),
            }
        )
    else:
        raise ValueError(f"Unsupported tokenizer kind: {bundle.kind}")

    (artifacts_dir / "tokenizer_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_tokenizer(artifacts_dir: Path) -> TokenizerBundle:
    """Load tokenizer bundle from artifacts."""
    meta_path = artifacts_dir / "tokenizer_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("Tokenizer metadata not found. Train the model first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    kind = meta["kind"]

    if kind == "word":
        from tensorflow.keras.preprocessing.text import tokenizer_from_json

        json_path = artifacts_dir / meta["file"]
        tokenizer_json = json_path.read_text(encoding="utf-8")
        tokenizer = tokenizer_from_json(tokenizer_json)
        return TokenizerBundle(
            kind="word",
            tokenizer=tokenizer,
            vocab_size=meta["vocab_size"],
            metadata={"num_words": meta.get("num_words")},
        )

    if kind == "sentencepiece":
        model_path = artifacts_dir / meta["model_file"]
        sp = spm.SentencePieceProcessor(model_file=str(model_path))
        metadata = {
            "model_path": str(model_path),
            "vocab_path": str(artifacts_dir / meta["vocab_file"]),
            "pad_id": meta.get("pad_id"),
            "unk_id": meta.get("unk_id"),
            "bos_id": meta.get("bos_id"),
            "eos_id": meta.get("eos_id"),
        }
        return TokenizerBundle(
            kind="sentencepiece",
            tokenizer=sp,
            vocab_size=meta["vocab_size"],
            metadata=metadata,
        )

    raise ValueError(f"Unsupported tokenizer kind: {kind}")


def create_padded_inputs(
    x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad sequences to uniform length (if needed)."""
    max_len = max(len(seq) for seq in np.concatenate([x_train, x_val, x_test]))
    x_train_pad = pad_sequences(x_train, maxlen=max_len, padding="pre")
    x_val_pad = pad_sequences(x_val, maxlen=max_len, padding="pre")
    x_test_pad = pad_sequences(x_test, maxlen=max_len, padding="pre")
    return x_train_pad, x_val_pad, x_test_pad


# ---------------------------------------------------------------------------
# Command-line helpers
# ---------------------------------------------------------------------------


def get_corpus_paths(workspace: Path) -> Tuple[Path, Path]:
    """Return the canonical raw and cleaned corpus paths for a workspace."""
    raw_path = workspace / "data" / "raw" / "sherlock_holmes.txt"
    processed_path = workspace / "data" / "processed" / "sherlock_holmes_clean.txt"
    return raw_path, processed_path


def download_command(workspace: Path, force: bool = False) -> Path:
    """Download the raw corpus into the workspace."""
    raw_path, _ = get_corpus_paths(workspace)
    if raw_path.exists() and not force:
        print(f"Corpus already present at {raw_path}")
        return raw_path

    print(f"Downloading corpus to {raw_path}...")
    download_corpus(raw_path)
    print("Download complete.")
    return raw_path


def clean_command(workspace: Path, force: bool = False) -> Path:
    """Clean the downloaded corpus and write the processed version."""
    raw_path, processed_path = get_corpus_paths(workspace)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw corpus not found at {raw_path}. Run 'download' first.")
    if processed_path.exists() and not force:
        print(f"Clean corpus already present at {processed_path}")
        return processed_path

    print(f"Cleaning corpus from {raw_path}...")
    raw_text = load_corpus(raw_path)
    clean_text = clean_corpus(raw_text)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.write_text(clean_text, encoding="utf-8")
    print(f"Wrote cleaned corpus to {processed_path}")
    return processed_path


def prepare_command(workspace: Path, force: bool = False) -> Tuple[Path, Path]:
    """Run download and clean steps in sequence."""
    raw_path = download_command(workspace, force=force)
    clean_path = clean_command(workspace, force=force)
    return raw_path, clean_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Workspace directory.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files instead of reusing them.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("download", help="Download the raw Sherlock Holmes corpus.")
    subparsers.add_parser("clean", help="Clean the downloaded corpus.")
    subparsers.add_parser("prepare", help="Download and clean the corpus in one step.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    workspace = args.workspace
    force = args.force

    if args.command == "download":
        download_command(workspace, force=force)
    elif args.command == "clean":
        clean_command(workspace, force=force)
    elif args.command == "prepare":
        prepare_command(workspace, force=force)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
