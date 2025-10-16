"""Model architecture for next-word prediction with attention."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class ModelConfig:
    vocab_size: int
    sequence_length: int
    embedding_dim: int = 256
    lstm_units: int = 256
    embedding_dropout: float = 0.1
    num_lstm_layers: int = 2
    bidirectional: bool = True
    dropout_rate: float = 0.3
    recurrent_dropout: float = 0.1
    attention_num_heads: int = 4
    attention_key_dim: int = 64
    attention_dropout: float = 0.1
    ff_dim: int = 512
    learning_rate: float = 1e-3
    tie_embeddings: bool = False


def _embedding_with_positions(
    config: ModelConfig, inputs: keras.Input
) -> tuple[tf.Tensor, tf.Tensor, layers.Embedding]:
    """Embed tokens and inject positional information."""
    token_embedding = layers.Embedding(
        input_dim=config.vocab_size,
        output_dim=config.embedding_dim,
        mask_zero=True,
        name="token_embedding",
    )
    x = token_embedding(inputs)

    position_indices = tf.range(start=0, limit=config.sequence_length, delta=1)
    pos_embedding_layer = layers.Embedding(
        input_dim=config.sequence_length,
        output_dim=config.embedding_dim,
        name="position_embedding",
    )
    pos_embeddings = pos_embedding_layer(position_indices)
    pos_embeddings = layers.Lambda(
        lambda pe: tf.expand_dims(pe, axis=0),
        name="positional_broadcast",
    )(pos_embeddings)
    x = layers.Add(name="add_positional_encoding")([x, pos_embeddings])
    x = layers.LayerNormalization(name="embedding_layer_norm")(x)
    if config.embedding_dropout and config.embedding_dropout > 0.0:
        x = layers.Dropout(config.embedding_dropout, name="embedding_dropout")(x)
    mask = layers.Lambda(lambda t: tf.not_equal(t, 0), name="padding_mask")(inputs)
    return x, mask, token_embedding


def _masked_average(sequence: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """Compute a mask-aware average pooling over the time dimension."""
    mask = tf.cast(mask, dtype=sequence.dtype)
    mask = tf.expand_dims(mask, axis=-1)  # (batch, seq_len, 1)
    masked_sequence = sequence * mask
    sum_vector = tf.reduce_sum(masked_sequence, axis=1)
    denom = tf.reduce_sum(mask, axis=1)
    denom = tf.maximum(denom, tf.constant(1.0, dtype=sequence.dtype))
    return sum_vector / denom


def build_model(config: ModelConfig) -> keras.Model:
    """Construct the enhanced LSTM + attention model."""
    inputs = keras.Input(shape=(config.sequence_length,), name="token_ids")
    x, mask, token_embedding = _embedding_with_positions(config, inputs)

    lstm_dim = config.lstm_units * (2 if config.bidirectional else 1)
    for idx in range(config.num_lstm_layers):
        residual = x
        if residual.shape[-1] != lstm_dim:
            residual = layers.Dense(lstm_dim, use_bias=False, name=f"residual_proj_{idx}")(residual)

        lstm_layer = layers.LSTM(
            config.lstm_units,
            return_sequences=True,
            dropout=config.dropout_rate,
            recurrent_dropout=config.recurrent_dropout,
            name=f"lstm_{idx}",
        )
        if config.bidirectional:
            x = layers.Bidirectional(lstm_layer, name=f"bilstm_{idx}")(x, mask=mask)
        else:
            x = lstm_layer(x, mask=mask)

        x = layers.Dropout(config.dropout_rate, name=f"post_lstm_dropout_{idx}")(x)
        x = layers.Add(name=f"lstm_residual_{idx}")([x, residual])
        x = layers.LayerNormalization(name=f"lstm_layer_norm_{idx}")(x)

    attention_mask = layers.Lambda(
        lambda t: tf.cast(tf.expand_dims(t, axis=1), tf.bool),
        name="attention_mask",
    )(mask)

    attention_layer = layers.MultiHeadAttention(
        num_heads=config.attention_num_heads,
        key_dim=config.attention_key_dim,
        dropout=config.attention_dropout,
        name="self_attention",
    )
    attn_output = attention_layer(query=x, value=x, key=x, attention_mask=attention_mask)
    x = layers.Add(name="attention_residual")([x, attn_output])
    x = layers.LayerNormalization(name="attention_layer_norm")(x)

    masked_avg = layers.Lambda(lambda args: _masked_average(*args), name="masked_avg_pool")([x, mask])
    last_token = layers.Lambda(lambda t: t[:, -1, :], name="last_token")(x)
    context = layers.Concatenate(name="context_concat")([masked_avg, last_token])

    ffn = layers.Dense(config.ff_dim, activation="relu", name="ffn_dense_1")(context)
    ffn = layers.Dropout(config.dropout_rate, name="ffn_dropout_1")(ffn)
    ffn = layers.Dense(config.ff_dim // 2, activation="relu", name="ffn_dense_2")(ffn)
    ffn = layers.Dropout(config.dropout_rate, name="ffn_dropout_2")(ffn)
    if config.tie_embeddings and ffn.shape[-1] != config.embedding_dim:
        ffn = layers.Dense(
            config.embedding_dim,
            use_bias=False,
            name="ffn_projection_to_embedding_dim",
        )(ffn)

    if config.tie_embeddings:
        vocab_size = config.vocab_size
        logits = layers.Lambda(
            lambda args: tf.matmul(args[0], args[1], transpose_b=True),
            output_shape=(vocab_size,),
            name="token_logits",
        )([ffn, token_embedding.embeddings])
        outputs = layers.Activation("softmax", name="token_probs")(logits)
    else:
        outputs = layers.Dense(config.vocab_size, activation="softmax", name="token_probs")(ffn)
    model = keras.Model(inputs=inputs, outputs=outputs, name="lstm_attention_language_model")
    return model


def perplexity_from_loss(loss: float) -> float:
    """Convert cross-entropy loss into perplexity."""
    return float(tf.exp(loss).numpy())
