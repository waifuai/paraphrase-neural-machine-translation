# src/utils.py
"""
Enhanced utility functions for the paraphrase NMT project.
"""
import re
import os
import json
import logging
import unicodedata
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import bleu_score

from config import (
    START_TOKEN, END_TOKEN, UNKNOWN_TOKEN, PAD_TOKEN,
    BATCH_SIZE, SPLIT_RATIO, MAX_SEQUENCE_LENGTH
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# TEXT PREPROCESSING FUNCTIONS
# =============================================================================

def unicode_to_ascii(s: str) -> str:
    """Convert unicode string to ASCII."""
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def preprocess_sentence(sentence: str, max_length: int = MAX_SEQUENCE_LENGTH) -> str:
    """
    Enhanced sentence preprocessing with validation and length limiting.

    Args:
        sentence: Input sentence to preprocess
        max_length: Maximum allowed sentence length

    Returns:
        Preprocessed sentence with start and end tokens
    """
    if not sentence or not isinstance(sentence, str):
        return f"{START_TOKEN} {END_TOKEN}"

    sentence = str(sentence).lower().strip()

    # Remove extra whitespace
    sentence = re.sub(r'\s+', ' ', sentence)

    # Handle punctuation
    sentence = re.sub(r"([?!])", r" \1 ", sentence)
    sentence = re.sub(r"([,.])", "", sentence)

    # Remove non-alphanumeric characters except essential punctuation
    sentence = re.sub(r"[^a-zA-Z0-9\s\?\!\.]", "", sentence)

    # Final cleanup
    sentence = sentence.strip()
    words = sentence.split()

    # Truncate if too long
    if len(words) > max_length - 2:  # Account for start/end tokens
        words = words[:max_length - 2]

    sentence = ' '.join(words)
    return f"{START_TOKEN} {sentence} {END_TOKEN}"

def postprocess_sentence(sentence: str) -> str:
    """
    Postprocess generated sentence by removing special tokens and cleaning up.

    Args:
        sentence: Generated sentence with special tokens

    Returns:
        Cleaned sentence
    """
    # Remove start and end tokens
    sentence = sentence.replace(START_TOKEN, '').replace(END_TOKEN, '')

    # Remove unknown tokens
    sentence = sentence.replace(UNKNOWN_TOKEN, '')

    # Clean up extra whitespace
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence.strip()

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_attention(attention: np.ndarray,
                  sentence_tokens: List[str],
                  predicted_tokens: List[str],
                  save_path: Optional[str] = None,
                  figsize: Tuple[int, int] = (10, 10)) -> plt.Figure:
    """
    Enhanced attention visualization with better styling and saving options.

    Args:
        attention: Attention weights matrix
        sentence_tokens: Input sentence tokens
        predicted_tokens: Generated sentence tokens
        save_path: Optional path to save the plot
        figsize: Figure size tuple

    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    # Create attention heatmap
    im = ax.matshow(attention, cmap='viridis', aspect='auto')

    # Set labels
    fontdict = {'fontsize': 12}
    ax.set_xticklabels([''] + sentence_tokens, fontdict=fontdict, rotation=45, ha='left')
    ax.set_yticklabels([''] + predicted_tokens, fontdict=fontdict)

    # Configure ticks
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Labels
    ax.set_xlabel('Input Sequence', fontsize=14)
    ax.set_ylabel('Output Sequence', fontsize=14)
    ax.set_title('Attention Weights', fontsize=16)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Attention plot saved to {save_path}")

    return fig

def plot_training_history(history: Dict[str, List[float]],
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot training and validation metrics.

    Args:
        history: Dictionary containing training history
        save_path: Optional path to save the plot

    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot loss
    axes[0].plot(history.get('train_loss', []), label='Training Loss', linewidth=2)
    axes[0].plot(history.get('val_loss', []), label='Validation Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=16)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot learning rate if available
    if 'lr' in history:
        axes[1].plot(history['lr'], label='Learning Rate', linewidth=2, color='orange')
        axes[1].set_title('Learning Rate Schedule', fontsize=16)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_yscale('log')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")

    return fig

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def calculate_bleu_score(references: List[List[str]],
                         hypotheses: List[List[str]],
                         n_gram: int = 4) -> float:
    """
    Calculate BLEU score for generated paraphrases.

    Args:
        references: List of reference sentence token lists
        hypotheses: List of hypothesis sentence token lists
        n_gram: Maximum n-gram order for BLEU calculation

    Returns:
        BLEU score
    """
    try:
        return bleu_score(references, hypotheses, max_n=n_gram)
    except Exception as e:
        logger.error(f"Error calculating BLEU score: {e}")
        return 0.0

def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from cross-entropy loss.

    Args:
        loss: Cross-entropy loss value

    Returns:
        Perplexity score
    """
    return float(np.exp(loss))

# =============================================================================
# FILE I/O FUNCTIONS
# =============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object of the created directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_pickle(data: Any, filename: Union[str, Path], protocol: int = 4) -> None:
    """
    Enhanced pickle saving with error handling.

    Args:
        data: Data to save
        filename: Output filename
        protocol: Pickle protocol version
    """
    try:
        import dill as pickle
    except ImportError:
        import pickle

    ensure_dir(Path(filename).parent)

    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=protocol)

    logger.info(f"Data saved to {filename}")

def load_pickle(filename: Union[str, Path]) -> Any:
    """
    Enhanced pickle loading with error handling.

    Args:
        filename: Input filename

    Returns:
        Loaded data
    """
    try:
        import dill as pickle
    except ImportError:
        import pickle

    if not Path(filename).exists():
        raise FileNotFoundError(f"File not found: {filename}")

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    logger.info(f"Data loaded from {filename}")
    return data

def save_config(config: Dict[str, Any], filename: Union[str, Path]) -> None:
    """
    Save configuration to JSON file.

    Args:
        config: Configuration dictionary
        filename: Output filename
    """
    ensure_dir(Path(filename).parent)

    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Configuration saved to {filename}")

# =============================================================================
# VOCABULARY AND INDEXING
# =============================================================================

class LanguageIndex:
    """
    Enhanced vocabulary and indexing helper with better error handling.
    """

    def __init__(self, phrases: pd.Series):
        """
        Initialize language index from phrases.

        Args:
            phrases: Pandas series containing phrases
        """
        self.phrases = phrases
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.vocab_size = 0
        self.build_index()

    def build_index(self) -> None:
        """Build vocabulary and indexing mappings."""
        try:
            # Extract unique words
            for phrase in self.phrases:
                if isinstance(phrase, str):
                    self.vocab.update(phrase.split(' '))

            # Add special tokens
            self.vocab.add(UNKNOWN_TOKEN)
            self.vocab.add(PAD_TOKEN)
            self.vocab = sorted(self.vocab)

            # Create mappings
            self.word2idx[PAD_TOKEN] = 0
            for index, word in enumerate(self.vocab, start=1):
                self.word2idx[word] = index

            self.idx2word = {idx: word for word, idx in self.word2idx.items()}
            self.vocab_size = len(self.word2idx)

            logger.info(f"Built vocabulary with {self.vocab_size} words")

        except Exception as e:
            logger.error(f"Error building vocabulary index: {e}")
            raise

    def encode_sequence(self, sentence: str, max_length: int = MAX_SEQUENCE_LENGTH) -> List[int]:
        """
        Encode a sentence into token IDs.

        Args:
            sentence: Input sentence
            max_length: Maximum sequence length

        Returns:
            List of token IDs
        """
        words = sentence.split(' ')
        tokens = [self.word2idx.get(word, self.word2idx[UNKNOWN_TOKEN]) for word in words]

        # Truncate if necessary
        if len(tokens) > max_length:
            tokens = tokens[:max_length]

        return tokens

    def decode_sequence(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to sentence.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded sentence
        """
        words = [self.idx2word.get(idx, UNKNOWN_TOKEN) for idx in token_ids]
        return ' '.join(words)

# =============================================================================
# DATA VALIDATION FUNCTIONS
# =============================================================================

def validate_dataset(data: pd.DataFrame,
                    required_columns: List[str] = ['phrase1', 'phrase2']) -> bool:
    """
    Validate dataset structure and content.

    Args:
        data: Input dataframe
        required_columns: List of required column names

    Returns:
        True if valid, raises exception otherwise
    """
    # Check required columns
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Check for empty data
    if data.empty:
        raise ValueError("Dataset is empty")

    # Check for null values
    null_counts = data.isnull().sum()
    if null_counts.any():
        logger.warning(f"Found null values:\n{null_counts}")
        data = data.dropna()

    logger.info(f"Dataset validation passed. Shape: {data.shape}")
    return True

def get_dataset_stats(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive dataset statistics.

    Args:
        data: Input dataframe

    Returns:
        Dictionary containing dataset statistics
    """
    stats = {}

    # Basic info
    stats['shape'] = data.shape
    stats['columns'] = list(data.columns)

    # Text statistics
    for col in ['phrase1', 'phrase2']:
        if col in data.columns:
            lengths = data[col].str.split().str.len()
            stats[f'{col}_length'] = {
                'mean': lengths.mean(),
                'std': lengths.std(),
                'min': lengths.min(),
                'max': lengths.max(),
                'median': lengths.median()
            }

    return stats