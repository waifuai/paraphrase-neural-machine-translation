#!/usr/bin/env python
# coding: utf-8
"""
Enhanced Encoder and Decoder models with regularization and improved architecture.
"""
import tensorflow as tf
from typing import Tuple, Optional
from config import (
    UNITS, BATCH_SIZE, EMBEDDING_DIM, DROPOUT_RATE,
    RECURRENT_DROPOUT, CLIP_NORM
)


class Encoder(tf.keras.Model):
    """
    Enhanced Encoder with bidirectional GRU, dropout, and layer normalization.
    """

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = EMBEDDING_DIM,
                 enc_units: int = UNITS,
                 batch_sz: int = BATCH_SIZE,
                 dropout_rate: float = DROPOUT_RATE,
                 recurrent_dropout: float = RECURRENT_DROPOUT):
        super().__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units

        # Embedding layer with optional masking
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, mask_zero=True
        )

        # Bidirectional GRU for better context understanding
        self.gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                enc_units // 2,  # Adjust units for bidirectional
                return_sequences=True,
                return_state=True,
                recurrent_initializer='glorot_uniform',
                dropout=recurrent_dropout,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            )
        )

        # Dropout for regularization
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        # Layer normalization
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x: tf.Tensor, hidden: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass through the encoder.

        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            hidden: Optional initial hidden state

        Returns:
            Tuple of (output, state)
        """
        # Embedding
        x = self.embedding(x)

        # Apply dropout
        x = self.dropout(x)

        # GRU layers
        output, forward_state, backward_state = self.gru(x, initial_state=hidden)

        # Concatenate forward and backward states
        state = tf.concat([forward_state, backward_state], axis=-1)

        # Apply layer normalization
        output = self.layer_norm(output)

        return output, state

    def initialize_hidden_state(self) -> tf.Tensor:
        """Initialize hidden state for the encoder."""
        return tf.zeros((self.batch_sz, self.enc_units))


class Decoder(tf.keras.Model):
    """
    Enhanced Decoder with improved attention mechanism, residual connections, and regularization.
    """

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = EMBEDDING_DIM,
                 dec_units: int = UNITS,
                 batch_sz: int = BATCH_SIZE,
                 dropout_rate: float = DROPOUT_RATE,
                 recurrent_dropout: float = RECURRENT_DROPOUT):
        super().__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units

        # Embedding layer
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, mask_zero=True
        )

        # GRU layer with regularization
        self.gru = tf.keras.layers.GRU(
            dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform',
            dropout=recurrent_dropout,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )

        # Dropout for regularization
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        # Layer normalization
        self.layer_norm = tf.keras.layers.LayerNormalization()

        # Output projection layer
        self.fc = tf.keras.layers.Dense(
            vocab_size,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )

        # Enhanced attention mechanism
        self.attention_dense = tf.keras.layers.Dense(dec_units, activation='tanh')
        self.attention_query = tf.keras.layers.Dense(dec_units)
        self.attention_key = tf.keras.layers.Dense(dec_units)
        self.attention_value = tf.keras.layers.Dense(dec_units)
        self.attention_combine = tf.keras.layers.Dense(dec_units, activation='tanh')

    def call(self,
             x: tf.Tensor,
             hidden: tf.Tensor,
             enc_output: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Forward pass through the decoder with attention.

        Args:
            x: Input tensor of shape (batch_size, 1)
            hidden: Hidden state from previous step
            enc_output: Encoder output for attention

        Returns:
            Tuple of (predictions, state, attention_weights)
        """
        # Embedding
        x = self.embedding(x)

        # Apply dropout
        x = self.dropout(x)

        # Enhanced attention mechanism
        query = self.attention_query(tf.expand_dims(hidden, 1))
        key = self.attention_key(enc_output)
        value = self.attention_value(enc_output)

        # Scaled dot-product attention
        scores = tf.matmul(query, key, transpose_b=True) / tf.sqrt(tf.cast(self.dec_units, tf.float32))
        attention_weights = tf.nn.softmax(scores, axis=-1)
        context_vector = tf.matmul(attention_weights, value)

        # Combine context with input
        x = tf.concat([x, context_vector], axis=-1)
        x = self.attention_combine(x)

        # GRU layer
        output, state = self.gru(x, initial_state=hidden)

        # Apply layer normalization
        output = self.layer_norm(output)

        # Reshape for dense layer
        output = tf.reshape(output, (-1, output.shape[2]))

        # Final projection
        x = self.fc(output)

        return x, state, attention_weights

    def initialize_hidden_state(self) -> tf.Tensor:
        """Initialize hidden state for the decoder."""
        return tf.zeros((self.batch_sz, self.dec_units))


class Seq2SeqModel(tf.keras.Model):
    """
    Complete sequence-to-sequence model with encoder and decoder.
    """

    def __init__(self, inp_vocab_size: int, targ_vocab_size: int, **kwargs):
        super().__init__()
        self.encoder = Encoder(inp_vocab_size, **kwargs)
        self.decoder = Decoder(targ_vocab_size, **kwargs)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Forward pass through the complete model.

        Args:
            inputs: Tuple of (encoder_input, decoder_input)

        Returns:
            Tuple of (predictions, encoder_state, attention_weights)
        """
        enc_input, dec_input = inputs

        # Encoder
        enc_hidden = self.encoder.initialize_hidden_state()
        enc_output, enc_state = self.encoder(enc_input, enc_hidden)

        # Decoder
        dec_hidden = enc_state
        predictions, dec_state, attention_weights = self.decoder(dec_input, dec_hidden, enc_output)

        return predictions, dec_state, attention_weights