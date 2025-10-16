"""Training script for next-word prediction with LSTM + attention."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences

from . import data
from .model import ModelConfig, build_model, perplexity_from_loss

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay."""

    def __init__(self, base_lr: float, warmup_steps: int, total_steps: int):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = tf.cast(max(warmup_steps, 1), tf.float32)
        self.total_steps = tf.cast(max(total_steps, 1), tf.float32)

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        step = tf.cast(step, tf.float32)
        warmup_lr = self.base_lr * (step / self.warmup_steps)
        progress = tf.minimum(1.0, (step - self.warmup_steps) / tf.maximum(self.total_steps - self.warmup_steps, 1.0))
        cosine_lr = self.base_lr * 0.5 * (1.0 + tf.cos(np.pi * progress))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self) -> Dict[str, float]:
        return {
            "base_lr": float(self.base_lr),
            "warmup_steps": float(self.warmup_steps.numpy()),
            "total_steps": float(self.total_steps.numpy()),
        }


class SparseLabelSmoothingLoss(tf.keras.losses.Loss):
    """Sparse label smoothing wrapper."""

    def __init__(self, vocab_size: int, smoothing: float, name: str = "sparse_label_smoothing"):
        super().__init__(name=name)
        self.vocab_size = vocab_size
        self.smoothing = smoothing

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)
        one_hot = tf.one_hot(y_true, depth=self.vocab_size, dtype=tf.float32)
        smooth_pos = 1.0 - self.smoothing
        smooth_neg = self.smoothing / tf.cast(self.vocab_size - 1, tf.float32)
        targets = one_hot * smooth_pos + smooth_neg
        return tf.keras.losses.categorical_crossentropy(targets, y_pred)

    def get_config(self) -> Dict[str, float]:
        config = super().get_config()
        config.update({"vocab_size": self.vocab_size, "smoothing": self.smoothing})
        return config


def prepare_dataset(
    workspace: Path,
    sequence_length: int,
    num_words: int | None = None,
    exclude_oov: bool = True,
    sequence_strategy: str = "sliding",
    tokenizer_type: str = "word",
    sentencepiece_vocab_size: int | None = None,
    sliding_stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, data.TokenizerBundle]:
    """Full pipeline from corpus download to train/val/test tensors."""

    raw_path = workspace / "data" / "raw" / "sherlock_holmes.txt"
    processed_path = workspace / "data" / "processed" / "sherlock_holmes_clean.txt"

    if not raw_path.exists():
        LOGGER.info("Downloading corpus to %s", raw_path)
        data.download_corpus(raw_path)
    else:
        LOGGER.info("Corpus already present at %s", raw_path)

    raw_text = data.load_corpus(raw_path)
    clean_text = data.clean_corpus(raw_text)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.write_text(clean_text, encoding="utf-8")
    LOGGER.info("Clean corpus length: %s characters", len(clean_text))

    if tokenizer_type == "word":
        tokenizer_bundle = data.build_word_tokenizer(clean_text, num_words=num_words)
    elif tokenizer_type == "sentencepiece":
        if not sentencepiece_vocab_size:
            raise ValueError("--sentencepiece-vocab-size must be provided for sentencepiece tokenization")
        tokenizer_bundle = data.build_sentencepiece_tokenizer(clean_text, workspace, vocab_size=sentencepiece_vocab_size)
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    sequences = data.generate_sequences(
        tokenizer_bundle,
        clean_text,
        sequence_length=sequence_length,
        exclude_oov=exclude_oov,
        strategy=sequence_strategy,
        stride=max(1, sliding_stride),
    )
    LOGGER.info("Tokenizer: %s (vocab_size=%s)", tokenizer_bundle.kind, tokenizer_bundle.vocab_size)

    if tokenizer_bundle.kind == "word":
        oov_id = tokenizer_bundle.tokenizer.word_index.get("<oov>")
    elif tokenizer_bundle.kind == "sentencepiece":
        oov_id = tokenizer_bundle.metadata.get("unk_id")
    else:
        oov_id = None

    if oov_id is not None:
        before = len(sequences)
        sequences = [seq for seq in sequences if seq and seq[-1] != oov_id]
        dropped = before - len(sequences)
        if dropped:
            LOGGER.info("Dropped %s sequences with OOV targets", dropped)

    if not sequences:
        raise ValueError("No sequences generated. Try lowering num_words or include OOV tokens.")

    max_len = min(sequence_length + 1, max(len(seq) for seq in sequences))
    sequences_padded = pad_sequences(sequences, maxlen=max_len, padding="pre")
    LOGGER.info("Generated %s sequences (max length %s)", len(sequences_padded), sequences_padded.shape[1])

    x_train, y_train, x_val, y_val, x_test, y_test = data.train_val_test_split(sequences_padded)
    x_train_pad, x_val_pad, x_test_pad = data.create_padded_inputs(x_train, x_val, x_test)
    return x_train_pad, y_train, x_val_pad, y_val, x_test_pad, y_test, tokenizer_bundle


def train(
    workspace: Path,
    epochs: int,
    batch_size: int,
    sequence_length: int,
    embedding_dim: int,
    lstm_units: int,
    embedding_dropout: float,
    dropout_rate: float,
    num_words: int | None,
    exclude_oov: bool = True,
    early_stop_patience: int | None = 4,
    sequence_strategy: str = "sliding",
    num_lstm_layers: int = 2,
    bidirectional: bool = True,
    recurrent_dropout: float = 0.22,
    attention_heads: int = 4,
    attention_key_dim: int = 64,
    attention_dropout: float = 0.25,
    ff_dim: int = 352,
    learning_rate: float = 2.2e-4,
    tokenizer_type: str = "word",
    sentencepiece_vocab_size: int | None = None,
    label_smoothing: float = 0.0,
    optimizer_name: str = "adam",
    weight_decay: float = 7.5e-4,
    grad_clip_norm: float | None = 1.0,
    warmup_steps: int = 0,
    cosine_decay_steps: int = 0,
    use_mixed_precision: bool = False,
    sliding_stride: int = 1,
    tie_embeddings: bool = False,
) -> Dict[str, float]:
    """Run model training and evaluation."""

    if use_mixed_precision:
        mixed_precision.set_global_policy("mixed_float16")

    tf.random.set_seed(42)
    np.random.seed(42)

    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        tokenizer_bundle,
    ) = prepare_dataset(
        workspace,
        sequence_length,
        num_words,
        exclude_oov=exclude_oov,
        sequence_strategy=sequence_strategy,
        tokenizer_type=tokenizer_type,
        sentencepiece_vocab_size=sentencepiece_vocab_size,
        sliding_stride=sliding_stride,
    )

    model_config = ModelConfig(
        vocab_size=tokenizer_bundle.vocab_size,
        sequence_length=x_train.shape[1],
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        embedding_dropout=embedding_dropout,
        num_lstm_layers=num_lstm_layers,
        bidirectional=bidirectional,
        dropout_rate=dropout_rate,
        recurrent_dropout=recurrent_dropout,
        attention_num_heads=attention_heads,
        attention_key_dim=attention_key_dim,
        attention_dropout=attention_dropout,
        ff_dim=ff_dim,
        learning_rate=learning_rate,
        tie_embeddings=tie_embeddings,
    )

    LOGGER.info("Model configuration: %s", model_config)

    model = build_model(model_config)
    model.summary(print_fn=lambda line: LOGGER.info(line))

    steps_per_epoch = int(np.ceil(x_train.shape[0] / batch_size))
    total_steps = steps_per_epoch * epochs if epochs > 0 else steps_per_epoch
    if cosine_decay_steps > 0:
        total_steps = cosine_decay_steps

    lr_schedule: tf.keras.optimizers.schedules.LearningRateSchedule | float = learning_rate
    if warmup_steps > 0 or cosine_decay_steps > 0:
        lr_schedule = WarmupCosineSchedule(learning_rate, warmup_steps, total_steps)

    opt_kwargs: Dict[str, float] = {}
    if grad_clip_norm is not None and grad_clip_norm > 0:
        opt_kwargs["clipnorm"] = grad_clip_norm

    if optimizer_name.lower() == "adamw":
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=weight_decay,
            **opt_kwargs,
        )
    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            **opt_kwargs,
        )

    if label_smoothing > 0:
        loss_fn = SparseLabelSmoothingLoss(tokenizer_bundle.vocab_size, label_smoothing)
    else:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    metrics_list = [
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy", dtype=tf.float32),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(name="top5_accuracy", k=5, dtype=tf.float32),
    ]
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics_list)

    artifacts_dir = workspace / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    using_schedule = isinstance(lr_schedule, tf.keras.optimizers.schedules.LearningRateSchedule)
    callbacks = [
        ModelCheckpoint(
            filepath=str(artifacts_dir / "checkpoint.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]
    if not using_schedule:
        callbacks.insert(
            0,
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        )
    if early_stop_patience is not None and early_stop_patience > 0:
        callbacks.insert(
            0,
            EarlyStopping(
                monitor="val_loss",
                patience=early_stop_patience,
                restore_best_weights=True,
            ),
        )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    eval_loss, eval_accuracy, eval_top5 = model.evaluate(x_test, y_test, verbose=0)
    perplexity = perplexity_from_loss(eval_loss)

    model.save(artifacts_dir / "final_model.keras")
    data.save_tokenizer(tokenizer_bundle, artifacts_dir)

    metrics = {
        "train_accuracy": float(history.history["accuracy"][-1]),
        "train_top5_accuracy": float(history.history.get("top5_accuracy", [0.0])[-1]),
        "val_accuracy": float(history.history["val_accuracy"][-1]),
        "val_top5_accuracy": float(history.history.get("val_top5_accuracy", [0.0])[-1]),
        "val_loss": float(history.history["val_loss"][-1]),
        "test_accuracy": float(eval_accuracy),
        "test_top5_accuracy": float(eval_top5),
        "test_loss": float(eval_loss),
        "test_perplexity": float(perplexity),
        "train_perplexity": float(np.exp(history.history["loss"][-1])),
        "val_perplexity": float(np.exp(history.history["val_loss"][-1])),
        "tokenizer_type": tokenizer_type,
        "vocab_size": tokenizer_bundle.vocab_size,
        "label_smoothing": label_smoothing,
        "optimizer": optimizer_name,
        "weight_decay": weight_decay,
        "grad_clip_norm": grad_clip_norm if grad_clip_norm is not None else 0.0,
        "warmup_steps": warmup_steps,
        "cosine_decay_steps": cosine_decay_steps,
        "use_mixed_precision": use_mixed_precision,
    }

    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (artifacts_dir / "model_config.json").write_text(json.dumps(asdict(model_config), indent=2), encoding="utf-8")
    history_serializable = {k: [float(v) for v in values] for k, values in history.history.items()}
    (artifacts_dir / "history.json").write_text(json.dumps(history_serializable, indent=2), encoding="utf-8")

    LOGGER.info("Training complete. Metrics: %s", metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Workspace directory.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--lstm-units", type=int, default=256)
    parser.add_argument("--embedding-dropout", type=float, default=0.2)
    parser.add_argument("--dropout-rate", type=float, default=0.25)
    parser.add_argument("--num-words", type=int, default=8000)
    parser.add_argument("--num-lstm-layers", type=int, default=2)
    parser.add_argument("--recurrent-dropout", type=float, default=0.22)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-key-dim", type=int, default=64)
    parser.add_argument("--attention-dropout", type=float, default=0.25)
    parser.add_argument("--ff-dim", type=int, default=352)
    parser.add_argument("--learning-rate", type=float, default=2.2e-4)
    parser.add_argument(
        "--disable-bidirectional",
        dest="bidirectional",
        action="store_false",
        help="Disable bidirectional LSTM layers (use unidirectional only).",
    )
    parser.set_defaults(bidirectional=True)
    parser.add_argument(
        "--include-oov",
        action="store_true",
        help="Keep sequences that contain out-of-vocabulary tokens instead of filtering them out.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=6,
        help="Patience for early stopping on validation loss. Set to 0 or negative to disable.",
    )
    parser.add_argument(
        "--sequence-strategy",
        choices=("ngram", "sliding"),
        default="sliding",
        help="Sequence construction strategy (ngram trims to prefixes, sliding uses fixed windows).",
    )
    parser.add_argument(
        "--sliding-stride",
        type=int,
        default=1,
        help="Stride to use with the sliding window strategy (only relevant when --sequence-strategy=sliding).",
    )
    parser.add_argument(
        "--tokenizer-type",
        choices=("word", "sentencepiece"),
        default="word",
        help="Tokenization strategy to use for model training.",
    )
    parser.add_argument(
        "--sentencepiece-vocab-size",
        type=int,
        default=9000,
        help="Vocabulary size for SentencePiece tokenization (ignored for word tokenization).",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--optimizer",
        choices=("adam", "adamw"),
        default="adam",
        help="Optimizer to use during training.",
    )
    parser.add_argument("--weight-decay", type=float, default=7.5e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--cosine-decay-steps", type=int, default=0)
    parser.add_argument(
        "--mixed-precision",
        dest="use_mixed_precision",
        action="store_true",
        help="Enable mixed precision training (fp16).",
    )
    parser.add_argument(
        "--tie-embeddings",
        action="store_true",
        help="Reuse token embedding weights for the softmax projection to reduce parameters.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        workspace=args.workspace,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units,
        embedding_dropout=args.embedding_dropout,
        dropout_rate=args.dropout_rate,
        num_words=args.num_words,
        exclude_oov=not args.include_oov,
        early_stop_patience=args.early_stop_patience if args.early_stop_patience > 0 else None,
        sequence_strategy=args.sequence_strategy,
        num_lstm_layers=args.num_lstm_layers,
        bidirectional=args.bidirectional,
        recurrent_dropout=args.recurrent_dropout,
        attention_heads=args.attention_heads,
        attention_key_dim=args.attention_key_dim,
        attention_dropout=args.attention_dropout,
        ff_dim=args.ff_dim,
        learning_rate=args.learning_rate,
        tokenizer_type=args.tokenizer_type,
        sentencepiece_vocab_size=args.sentencepiece_vocab_size,
        label_smoothing=args.label_smoothing,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        warmup_steps=args.warmup_steps,
        cosine_decay_steps=args.cosine_decay_steps,
        use_mixed_precision=args.use_mixed_precision,
        sliding_stride=args.sliding_stride,
        tie_embeddings=args.tie_embeddings,
    )


if __name__ == "__main__":
    main()
