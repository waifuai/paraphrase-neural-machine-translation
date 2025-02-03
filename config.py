# config.py
# Global configuration and constants

# Tokens
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNKNOWN_TOKEN = "<unknown>"

# Training and model parameters
BATCH_SIZE = 64
EMBEDDING_DIM = 256
UNITS = 1024
SPLIT_RATIO = 0.2

# Paths
DATA_PATH = "./data/parabank_100k.tsv"
PKL_DIR = "./pkl"
TRAIN_CHECKPOINTS_DIR = "./logs/training_checkpoints"
SCALARS_LOG_DIR = "./logs/scalars"

# Training hyperparameters
EPOCHS = 10000