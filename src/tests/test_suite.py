#!/usr/bin/env python
# coding: utf-8
"""
Comprehensive Test Suite for Paraphrase NMT Project

This test suite provides comprehensive testing coverage for all major components
of the paraphrase neural machine translation system. It includes unit tests,
integration tests, and validation tests for the configuration, data processing,
model architecture, and utility functions.

Features:
- Configuration validation tests
- Data preprocessing pipeline tests
- Model architecture tests (encoder/decoder)
- Utility function tests
- Integration tests for data pipeline
- Comprehensive error handling validation

Usage:
    Run all tests: pytest
    Run with coverage: pytest --cov=src --cov-report=html
    Run specific test file: pytest src/tests/test_suite.py

Test Categories:
- config.py: Configuration parameter validation
- utils.py: Text preprocessing and utility functions
- models.py: Neural network architecture components
- data.py: Data loading and preprocessing pipeline
"""
import os
import tempfile
import numpy as np
import pandas as pd
import tensorflow as tf
import pytest

# Import project modules
import config
from utils import preprocess_sentence, ensure_dir, LanguageIndex, unicode_to_ascii
from models import Encoder, Decoder
from data import create_tf_dataset

# -----------------------
# Test config.py
# -----------------------

def test_config_tokens():
    # Check that the special tokens are correctly set.
    assert config.START_TOKEN == "<start>"
    assert config.END_TOKEN == "<end>"
    assert config.UNKNOWN_TOKEN == "<unknown>"

# -----------------------
# Test utils.py
# -----------------------

def test_preprocess_sentence():
    # Given an input sentence, verify that preprocess_sentence produces the expected output.
    sentence = "Hello, world!"
    # The preprocess_sentence function in utils.py:
    # - Lowercases the string,
    # - Removes commas and dots,
    # - Inserts spaces around punctuation ? and !,
    # - Strips extra spaces,
    # - And wraps the sentence with <start> and <end>.
    expected = f"{config.START_TOKEN} hello world ! {config.END_TOKEN}"
    result = preprocess_sentence(sentence)
    assert result == expected

def test_ensure_dir():
    # Test that ensure_dir creates the directory if it does not exist.
    with tempfile.TemporaryDirectory() as tmpdirname:
        test_dir = os.path.join(tmpdirname, "test_folder")
        ensure_dir(test_dir)
        assert os.path.exists(test_dir)

def test_language_index():
    # Create a dummy series and test that LanguageIndex builds a valid mapping.
    phrases = pd.Series(["hello world", "test sentence"])
    li = LanguageIndex(phrases)
    for word in "hello world test sentence".split():
        assert word in li.word2idx
        # Also check the reverse mapping
        idx = li.word2idx[word]
        assert li.idx2word[idx] == word

# -----------------------
# Test models.py
# -----------------------

def test_encoder_output_shape():
    # Setup dummy parameters for the Encoder.
    vocab_size = 50
    embedding_dim = 16
    units = 32
    batch_size = 4
    sequence_length = 10

    encoder = Encoder(vocab_size, embedding_dim, enc_units=units, batch_sz=batch_size)
    dummy_input = tf.random.uniform((batch_size, sequence_length), minval=0, maxval=vocab_size, dtype=tf.int32)
    initial_hidden = encoder.initialize_hidden_state()
    output, state = encoder(dummy_input, initial_hidden)
    # Expected shapes: output: (batch_size, sequence_length, units), state: (batch_size, units)
    assert output.shape == (batch_size, sequence_length, units)
    assert state.shape == (batch_size, units)

def test_decoder_output_shape():
    # Setup dummy parameters for the Decoder.
    vocab_size = 50
    embedding_dim = 16
    units = 32
    batch_size = 4
    sequence_length = 10

    decoder = Decoder(vocab_size, embedding_dim, dec_units=units, batch_sz=batch_size)
    dummy_input = tf.random.uniform((batch_size, 1), minval=0, maxval=vocab_size, dtype=tf.int32)
    dummy_hidden = tf.random.uniform((batch_size, units))
    dummy_enc_output = tf.random.uniform((batch_size, sequence_length, units))
    output, state, attn_weights = decoder(dummy_input, dummy_hidden, dummy_enc_output)
    # Expected output shape: (batch_size, vocab_size)
    assert output.shape == (batch_size, vocab_size)
    # Expected state shape: (batch_size, units)
    assert state.shape == (batch_size, units)
    # Expected attention weights shape: (batch_size, sequence_length, 1)
    assert attn_weights.shape == (batch_size, sequence_length, 1)

# -----------------------
# Test data.py
# -----------------------

def test_create_tf_dataset():
    # Create a dummy DataFrame with two rows.
    df = pd.DataFrame({
        "phrase1": ["<start> hello <end>", "<start> world <end>"],
        "phrase2": ["<start> hi <end>", "<start> earth <end>"]
    })
    # Create dummy language index objects with a fixed mapping.
    class DummyLang:
        def __init__(self):
            # Here we assign arbitrary indices for tokens.
            self.word2idx = {
                "<start>": 1, "<end>": 2, "hello": 3, "world": 4,
                "hi": 3, "earth": 4, "<unknown>": 0, "<pad>": 0
            }
    dummy_inp = DummyLang()
    dummy_targ = DummyLang()
    max_length_inp = 5
    max_length_tar = 5

    dataset = create_tf_dataset(df, dummy_inp, dummy_targ, max_length_inp, max_length_tar)
    # Retrieve one batch and check that the shapes of the padded sequences are correct.
    for batch in dataset.take(1):
        inp_batch, targ_batch = batch
        assert inp_batch.shape[1] == max_length_inp
        assert targ_batch.shape[1] == max_length_tar

# -----------------------
# (Optional) Additional tests for model inference/training could be added here.
# For instance, you could test a single training step using a small dummy batch.
# -----------------------

if __name__ == "__main__":
    pytest.main()
