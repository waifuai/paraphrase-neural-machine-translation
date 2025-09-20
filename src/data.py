#!/usr/bin/env python
# coding: utf-8
"""
Enhanced Data Preparation Script

This script preprocesses the Parabank dataset with improved error handling,
data validation, builds the vocabulary, splits data into training and testing sets,
and saves preprocessed data as pickle files.
"""
import os
import re
import logging
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

from config import (
    DATA_PATH, PKL_DIR, BATCH_SIZE, SPLIT_RATIO, START_TOKEN, END_TOKEN,
    UNKNOWN_TOKEN, PAD_TOKEN, MAX_SEQUENCE_LENGTH, BUFFER_SIZE
)
from utils import (
    preprocess_sentence, LanguageIndex, ensure_dir, validate_dataset,
    get_dataset_stats, save_pickle, load_pickle
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_data(data_path: Optional[str] = None,
              delimiter: str = '\t',
              encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Load and validate dataset with enhanced error handling.

    Args:
        data_path: Path to the dataset file
        delimiter: CSV delimiter
        encoding: File encoding

    Returns:
        Loaded and validated dataframe

    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data format is invalid
    """
    if data_path is None:
        data_path = DATA_PATH

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    try:
        logger.info(f"Loading data from {data_path}")

        # Read CSV with error handling
        data = pd.read_csv(
            data_path,
            delimiter=delimiter,
            encoding=encoding,
            on_bad_lines='skip',  # Skip malformed lines
            engine='python'
        )

        # Validate required columns
        required_columns = ['phrase1', 'phrase2']
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Select only required columns
        data = data[required_columns].copy()

        # Remove rows with null values
        initial_shape = data.shape
        data = data.dropna()
        if data.shape[0] < initial_shape[0]:
            logger.warning(f"Removed {initial_shape[0] - data.shape[0]} rows with null values")

        # Remove empty strings
        for col in required_columns:
            data = data[data[col].str.strip().astype(bool)]

        # Shuffle data
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(f"Loaded {data.shape[0]} samples from {data_path}")
        return data

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

# =============================================================================
# DATA PREPROCESSING FUNCTIONS
# =============================================================================

def filter_by_length(data: pd.DataFrame,
                    max_length: int = MAX_SEQUENCE_LENGTH,
                    columns: List[str] = ['phrase1', 'phrase2']) -> pd.DataFrame:
    """
    Filter out sentences that are too long.

    Args:
        data: Input dataframe
        max_length: Maximum allowed sentence length
        columns: Columns to filter

    Returns:
        Filtered dataframe
    """
    initial_shape = data.shape[0]

    for col in columns:
        if col in data.columns:
            data = data[data[col].apply(lambda x: len(str(x).split()) <= max_length)]

    filtered_count = initial_shape - data.shape[0]
    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} samples exceeding max length {max_length}")

    return data

def augment_data(data: pd.DataFrame,
                augmentation_factor: float = 0.1) -> pd.DataFrame:
    """
    Data augmentation for training set.

    Args:
        data: Input dataframe
        augmentation_factor: Fraction of data to augment

    Returns:
        Augmented dataframe
    """
    if augmentation_factor <= 0:
        return data

    n_augment = int(len(data) * augmentation_factor)
    augment_indices = np.random.choice(len(data), n_augment, replace=False)

    augmented_data = []

    for idx in augment_indices:
        row = data.iloc[idx].copy()
        # Simple augmentation: swap phrases with some probability
        if np.random.random() < 0.5:
            row['phrase1'], row['phrase2'] = row['phrase2'], row['phrase1']
        augmented_data.append(row)

    if augmented_data:
        augmented_df = pd.DataFrame(augmented_data)
        data = pd.concat([data, augmented_df], ignore_index=True)
        logger.info(f"Added {len(augmented_df)} augmented samples")

    return data

def prepare_dataset(data_path: Optional[str] = None,
                   max_length: int = MAX_SEQUENCE_LENGTH,
                   augment: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, LanguageIndex, LanguageIndex, int, int]:
    """
    Enhanced dataset preparation with validation and preprocessing.

    Args:
        data_path: Path to data file
        max_length: Maximum sequence length
        augment: Whether to perform data augmentation

    Returns:
        Tuple of (train_data, test_data, inp_lang, targ_lang, max_length_inp, max_length_tar)
    """
    # Load and validate data
    data = load_data(data_path)
    validate_dataset(data)

    # Log dataset statistics
    stats = get_dataset_stats(data)
    logger.info(f"Dataset statistics: {stats}")

    # Filter by length
    data = filter_by_length(data, max_length)

    # Split data
    n_train = int(data.shape[0] * (1 - SPLIT_RATIO))
    train_data = data.iloc[:n_train].copy()
    test_data = data.iloc[n_train:].copy()

    logger.info(f"Data split: {len(train_data)} train, {len(test_data)} test")

    # Data augmentation
    if augment:
        train_data = augment_data(train_data)

    # Preprocess sentences with progress bar
    logger.info("Preprocessing sentences...")

    for data_split, name in [(train_data, "train"), (test_data, "test")]:
        for col in ['phrase1', 'phrase2']:
            data_split[col] = data_split[col].apply(preprocess_sentence)
            logger.info(f"Preprocessed {name} {col}: {data_split[col].iloc[0][:50]}...")

    # Build vocabulary indices
    logger.info("Building vocabulary...")
    inp_lang = LanguageIndex(train_data['phrase1'])
    targ_lang = LanguageIndex(train_data['phrase2'])

    # Calculate maximum lengths
    max_length_inp = min(max(train_data['phrase1'].apply(lambda s: len(s.split()))), max_length)
    max_length_tar = min(max(train_data['phrase2'].apply(lambda s: len(s.split()))), max_length)

    logger.info(f"Max input length: {max_length_inp}, Max target length: {max_length_tar}")
    logger.info(f"Input vocab size: {inp_lang.vocab_size}, Target vocab size: {targ_lang.vocab_size}")

    return train_data, test_data, inp_lang, targ_lang, max_length_inp, max_length_tar

# =============================================================================
# TENSORFLOW DATASET FUNCTIONS
# =============================================================================

def create_tf_dataset(data: pd.DataFrame,
                     inp_lang: LanguageIndex,
                     targ_lang: LanguageIndex,
                     max_length_inp: int,
                     max_length_tar: int,
                     batch_size: int = BATCH_SIZE,
                     shuffle: bool = True,
                     cache: bool = True) -> tf.data.Dataset:
    """
    Create optimized TensorFlow dataset with enhanced preprocessing.

    Args:
        data: Input dataframe
        inp_lang: Input language index
        targ_lang: Target language index
        max_length_inp: Maximum input sequence length
        max_length_tar: Maximum target sequence length
        batch_size: Batch size
        shuffle: Whether to shuffle data
        cache: Whether to cache dataset

    Returns:
        TensorFlow dataset
    """
    def sentence_to_ids(sentence: str, lang_index: LanguageIndex) -> List[int]:
        """Convert sentence to token IDs with better error handling."""
        try:
            words = sentence.split()
            return [lang_index.word2idx.get(word, lang_index.word2idx[UNKNOWN_TOKEN])
                    for word in words]
        except Exception as e:
            logger.warning(f"Error converting sentence to IDs: {e}")
            return [lang_index.word2idx[PAD_TOKEN]]

    # Convert sentences to sequences
    logger.info("Converting sentences to token sequences...")

    inp_sequences = data['phrase1'].apply(lambda s: sentence_to_ids(s, inp_lang)).tolist()
    targ_sequences = data['phrase2'].apply(lambda s: sentence_to_ids(s, targ_lang)).tolist()

    # Pad sequences
    inp_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        inp_sequences, maxlen=max_length_inp, padding='post', value=inp_lang.word2idx[PAD_TOKEN]
    )
    targ_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        targ_sequences, maxlen=max_length_tar, padding='post', value=targ_lang.word2idx[PAD_TOKEN]
    )

    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((inp_sequences, targ_sequences))

    # Optimize dataset pipeline
    if shuffle:
        dataset = dataset.shuffle(BUFFER_SIZE)

    if cache:
        dataset = dataset.cache()

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    logger.info(f"Created dataset with {len(inp_sequences)} sequences, batch size {batch_size}")
    return dataset

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(data_path: Optional[str] = None,
         augment: bool = False,
         output_dir: Optional[str] = None) -> None:
    """
    Main data preparation function.

    Args:
        data_path: Path to data file
        augment: Whether to augment training data
        output_dir: Output directory for processed data
    """
    try:
        # Set output directory
        if output_dir is None:
            output_dir = PKL_DIR
        else:
            output_dir = ensure_dir(output_dir)

        # Prepare dataset
        logger.info("Starting data preparation...")
        train_data, test_data, inp_lang, targ_lang, max_length_inp, max_length_tar = prepare_dataset(
            data_path, augment=augment
        )

        # Create TensorFlow datasets
        logger.info("Creating TensorFlow datasets...")
        train_dataset = create_tf_dataset(train_data, inp_lang, targ_lang, max_length_inp, max_length_tar)
        test_dataset = create_tf_dataset(test_data, inp_lang, targ_lang, max_length_inp, max_length_tar, shuffle=False)

        # Save processed data
        logger.info("Saving processed data...")

        save_pickle(list(train_dataset.as_numpy_iterator()), os.path.join(output_dir, "train_dataset.pkl"))
        save_pickle(list(test_dataset.as_numpy_iterator()), os.path.join(output_dir, "test_dataset.pkl"))
        save_pickle(max_length_inp, os.path.join(output_dir, "max_length_inp.pkl"))
        save_pickle(max_length_tar, os.path.join(output_dir, "max_length_tar.pkl"))
        save_pickle(inp_lang, os.path.join(output_dir, "inp_lang.pkl"))
        save_pickle(targ_lang, os.path.join(output_dir, "targ_lang.pkl"))

        # Save dataset statistics
        stats = {
            'train_samples': len(train_data),
            'test_samples': len(test_data),
            'max_length_inp': max_length_inp,
            'max_length_tar': max_length_tar,
            'input_vocab_size': inp_lang.vocab_size,
            'target_vocab_size': targ_lang.vocab_size,
            'data_path': data_path or DATA_PATH
        }
        save_pickle(stats, os.path.join(output_dir, "dataset_stats.pkl"))

        logger.info("Data preparation complete!")
        logger.info(f"Statistics: {stats}")

    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise

if __name__ == "__main__":
    main()