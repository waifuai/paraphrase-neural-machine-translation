# config.py
# Global configuration and constants
"""
Configuration file for the Paraphrase NMT project.
Contains all hyperparameters, file paths, and model settings.
"""

import os
from typing import Optional

# =============================================================================
# SPECIAL TOKENS
# =============================================================================
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNKNOWN_TOKEN = "<unknown>"
PAD_TOKEN = "<pad>"

# =============================================================================
# MODEL PARAMETERS
# =============================================================================
# Architecture settings
EMBEDDING_DIM = 256
UNITS = 1024
DROPOUT_RATE = 0.2
RECURRENT_DROPOUT = 0.1

# Training settings
BATCH_SIZE = 64
EPOCHS = 10000
LEARNING_RATE = 1e-3
CLIP_NORM = 1.0

# Data settings
SPLIT_RATIO = 0.2  # Train/test split
MAX_SEQUENCE_LENGTH = 50
BUFFER_SIZE = 10000

# =============================================================================
# FILE PATHS
# =============================================================================
# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
PKL_DIR = os.path.join(BASE_DIR, "pkl")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Data files
DATA_PATH = os.path.join(DATA_DIR, "parabank_100k.tsv")

# Model and logs
TRAIN_CHECKPOINTS_DIR = os.path.join(LOGS_DIR, "training_checkpoints")
SCALARS_LOG_DIR = os.path.join(LOGS_DIR, "scalars")
MODEL_SAVE_DIR = os.path.join(MODELS_DIR, "saved_models")

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
# Early stopping parameters
PATIENCE = 10
MIN_DELTA = 0.001

# Checkpoint settings
CHECKPOINT_SAVE_FREQ = 5  # Save every N epochs
KEEP_CHECKPOINTS = 3  # Number of checkpoints to keep

# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================
# Beam search settings
BEAM_WIDTH = 3
MAX_DECODING_LENGTH = 50

# Temperature for sampling
TEMPERATURE = 1.0

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================
def validate_config():
    """Validate configuration parameters."""
    assert BATCH_SIZE > 0, "Batch size must be positive"
    assert EMBEDDING_DIM > 0, "Embedding dimension must be positive"
    assert UNITS > 0, "Number of units must be positive"
    assert 0 < SPLIT_RATIO < 1, "Split ratio must be between 0 and 1"
    assert EPOCHS > 0, "Number of epochs must be positive"
    assert LEARNING_RATE > 0, "Learning rate must be positive"
    assert 0 <= DROPOUT_RATE < 1, "Dropout rate must be between 0 and 1"
    assert MAX_SEQUENCE_LENGTH > 0, "Max sequence length must be positive"

def get_config_summary() -> str:
    """Return a summary of the current configuration."""
    return f"""
Configuration Summary:
- Model: Embedding({EMBEDDING_DIM}) -> GRU({UNITS}) with attention
- Training: {EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}
- Data: Split={SPLIT_RATIO}, max_len={MAX_SEQUENCE_LENGTH}
- Paths: Data={DATA_PATH}, Checkpoints={TRAIN_CHECKPOINTS_DIR}
"""

# Validate configuration on import
validate_config()