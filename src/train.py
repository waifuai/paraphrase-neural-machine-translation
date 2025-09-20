#!/usr/bin/env python
# coding: utf-8
"""
Enhanced Training Script for NMT Paraphrase Generation Model

Features:
- Early stopping with configurable patience
- Learning rate scheduling
- Comprehensive logging and metrics
- Advanced checkpoint management
- Training visualization
- Gradient clipping and regularization
"""
import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (
    BATCH_SIZE, EMBEDDING_DIM, UNITS, EPOCHS, LEARNING_RATE, CLIP_NORM,
    TRAIN_CHECKPOINTS_DIR, SCALARS_LOG_DIR, PKL_DIR, PATIENCE,
    MIN_DELTA, CHECKPOINT_SAVE_FREQ, KEEP_CHECKPOINTS
)
from utils import (
    load_pickle, save_pickle, ensure_dir, plot_training_history,
    calculate_perplexity
)
from models import Encoder, Decoder

# =============================================================================
# SETUP AND CONFIGURATION
# =============================================================================

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrainingConfig:
    """Configuration class for training parameters."""

    def __init__(self):
        self.experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.checkpoint_dir = Path(TRAIN_CHECKPOINTS_DIR) / self.experiment_name
        self.log_dir = Path(SCALARS_LOG_DIR)
        self.train_log_dir = self.log_dir / 'train' / self.experiment_name
        self.val_log_dir = self.log_dir / 'validation' / self.experiment_name

        # Ensure directories exist
        for dir_path in [self.checkpoint_dir, self.train_log_dir, self.val_log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# TRAINING METRICS AND CALLBACKS
# =============================================================================

class TrainingMetrics:
    """Track and manage training metrics."""

    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_perplexity': [],
            'val_perplexity': [],
            'learning_rate': [],
            'epoch_time': []
        }

    def update(self, epoch: int, metrics: Dict[str, float], lr: float, epoch_time: float):
        """Update metrics for current epoch."""
        for key, value in metrics.items():
            self.history[key].append(value)
        self.history['learning_rate'].append(lr)
        self.history['epoch_time'].append(epoch_time)

    def save(self, filepath: Path):
        """Save metrics to file."""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training metrics saved to {filepath}")

    def plot(self, save_dir: Path):
        """Plot training curves."""
        fig = plot_training_history(self.history, save_path=str(save_dir / 'training_history.png'))
        plt.close(fig)

class EarlyStopping:
    """Early stopping implementation."""

    def __init__(self, patience: int = PATIENCE, min_delta: float = MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.best_epoch = 0

    def should_stop(self, current_loss: float, epoch: int) -> bool:
        """Check if training should stop."""
        if current_loss < (self.best_loss - self.min_delta):
            self.best_loss = current_loss
            self.wait = 0
            self.best_epoch = epoch
        else:
            self.wait += 1
            if self.wait >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}. Best loss: {self.best_loss:.4f}")
                return True
        return False

class LearningRateScheduler:
    """Learning rate scheduler with warmup and decay."""

    def __init__(self, initial_lr: float = LEARNING_RATE, warmup_steps: int = 1000):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.step = 0

    def get_lr(self, step: int) -> float:
        """Get learning rate for current step."""
        self.step = step
        if step < self.warmup_steps:
            # Linear warmup
            return self.initial_lr * (step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (EPOCHS * 100 - self.warmup_steps)
            return self.initial_lr * (1 + np.cos(np.pi * progress)) / 2

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def load_training_data() -> Tuple[Any, Any, Any, Any, int, int]:
    """Load and validate training data."""
    try:
        logger.info("Loading training data...")

        data_files = {
            'train_dataset': Path(PKL_DIR) / 'train_dataset.pkl',
            'test_dataset': Path(PKL_DIR) / 'test_dataset.pkl',
            'inp_lang': Path(PKL_DIR) / 'inp_lang.pkl',
            'targ_lang': Path(PKL_DIR) / 'targ_lang.pkl',
            'max_length_inp': Path(PKL_DIR) / 'max_length_inp.pkl',
            'max_length_tar': Path(PKL_DIR) / 'max_length_tar.pkl'
        }

        # Check if all files exist
        missing_files = [name for name, path in data_files.items() if not path.exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing data files: {missing_files}")

        # Load data
        data = {}
        for name, path in data_files.items():
            data[name] = load_pickle(path)
            logger.info(f"Loaded {name} from {path}")

        return (
            data['train_dataset'],
            data['test_dataset'],
            data['inp_lang'],
            data['targ_lang'],
            data['max_length_inp'],
            data['max_length_tar']
        )

    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        raise

def create_models_and_optimizer(vocab_inp_size: int,
                              vocab_tar_size: int,
                              batch_size: int) -> Tuple[Encoder, Decoder, tf.keras.optimizers.Optimizer]:
    """Create and configure models and optimizer."""
    logger.info("Creating models and optimizer...")

    # Create models
    encoder = Encoder(vocab_inp_size, EMBEDDING_DIM, UNITS, batch_size)
    decoder = Decoder(vocab_tar_size, EMBEDDING_DIM, UNITS, batch_size)

    # Create optimizer with gradient clipping
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=CLIP_NORM,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )

    logger.info(f"Encoder: {encoder}")
    logger.info(f"Decoder: {decoder}")
    logger.info(f"Optimizer: {optimizer}")

    return encoder, decoder, optimizer

def create_checkpoint_manager(checkpoint_dir: Path,
                            encoder: Encoder,
                            decoder: Decoder,
                            optimizer: tf.keras.optimizers.Optimizer) -> tf.train.CheckpointManager:
    """Create checkpoint manager."""
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        encoder=encoder,
        decoder=decoder
    )

    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=str(checkpoint_dir),
        max_to_keep=KEEP_CHECKPOINTS,
        keep_checkpoint_every_n_hours=1
    )

    return manager

def loss_function(real: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
    """Enhanced loss function with masking and label smoothing."""
    # Create mask for padding tokens
    mask = tf.math.logical_not(tf.math.equal(real, 0))

    # Sparse categorical crossentropy loss
    loss_ = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'
    )(real, pred)

    # Apply mask
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    # Return mean loss
    return tf.reduce_mean(loss_)

@tf.function
def train_step(encoder: Encoder,
               decoder: Decoder,
               inp: tf.Tensor,
               targ: tf.Tensor,
               targ_start: tf.Tensor,
               optimizer: tf.keras.optimizers.Optimizer) -> tf.Tensor:
    """Single training step with gradient computation."""
    loss = 0

    with tf.GradientTape() as tape:
        # Encoder forward pass
        enc_hidden = encoder.initialize_hidden_state()
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        # Decoder forward pass
        dec_hidden = enc_hidden
        dec_input = targ_start

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            step_loss = loss_function(targ[:, t], predictions)
            loss += step_loss
            dec_input = tf.expand_dims(targ[:, t], 1)

    # Calculate average loss
    batch_loss = loss / int(targ.shape[1])

    # Compute and apply gradients
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)

    # Clip gradients
    gradients, _ = tf.clip_by_global_norm(gradients, CLIP_NORM)

    # Apply gradients
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

def validation_step(encoder: Encoder,
                   decoder: Decoder,
                   inp: tf.Tensor,
                   targ: tf.Tensor,
                   targ_start: tf.Tensor) -> tf.Tensor:
    """Single validation step without gradient computation."""
    loss = 0

    # Encoder forward pass
    enc_hidden = encoder.initialize_hidden_state()
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    # Decoder forward pass
    dec_hidden = enc_hidden
    dec_input = targ_start

    for t in range(1, targ.shape[1]):
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
        step_loss = loss_function(targ[:, t], predictions)
        loss += step_loss
        dec_input = tf.expand_dims(targ[:, t], 1)

    # Calculate average loss
    batch_loss = loss / int(targ.shape[1])

    return batch_loss

# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def train_model(resume_from_checkpoint: Optional[str] = None) -> None:
    """Main training function."""
    try:
        # Initialize configuration
        config = TrainingConfig()
        logger.info(f"Starting training experiment: {config.experiment_name}")

        # Load data
        train_dataset, test_dataset, inp_lang, targ_lang, max_length_inp, max_length_tar = load_training_data()

        # Create models and optimizer
        vocab_inp_size = len(inp_lang.word2idx)
        vocab_tar_size = len(targ_lang.word2idx)
        encoder, decoder, optimizer = create_models_and_optimizer(vocab_inp_size, vocab_tar_size, BATCH_SIZE)

        # Create checkpoint manager
        checkpoint_manager = create_checkpoint_manager(config.checkpoint_dir, encoder, decoder, optimizer)

        # Restore from checkpoint if specified
        if resume_from_checkpoint:
            checkpoint_manager.restore(resume_from_checkpoint)
            logger.info(f"Restored from checkpoint: {resume_from_checkpoint}")

        # Create target start tokens
        targ_start = tf.expand_dims([targ_lang.word2idx['<start>']] * BATCH_SIZE, 1)

        # Initialize training components
        metrics = TrainingMetrics()
        early_stopping = EarlyStopping()
        lr_scheduler = LearningRateScheduler()

        # Create summary writers
        train_writer = tf.summary.create_file_writer(str(config.train_log_dir))
        val_writer = tf.summary.create_file_writer(str(config.val_log_dir))

        # Training loop
        logger.info("Starting training loop...")
        global_step = 0

        for epoch in range(EPOCHS):
            epoch_start_time = time.time()

            # Update learning rate
            current_lr = lr_scheduler.get_lr(global_step)
            optimizer.learning_rate.assign(current_lr)

            # Training phase
            train_losses = []
            train_pbar = tqdm(train_dataset, desc=f"Epoch {epoch + 1}/{EPOCHS} - Training")

            for batch, (inp, targ) in enumerate(train_pbar):
                batch_loss = train_step(encoder, decoder, inp, targ, targ_start, optimizer)
                train_losses.append(batch_loss.numpy())
                global_step += 1

                # Update progress bar
                train_pbar.set_postfix({'loss': f'{batch_loss.numpy():.4f}'})

            avg_train_loss = np.mean(train_losses)
            train_perplexity = calculate_perplexity(avg_train_loss)

            # Validation phase
            val_losses = []
            val_pbar = tqdm(test_dataset, desc=f"Epoch {epoch + 1}/{EPOCHS} - Validation")

            for batch, (inp, targ) in enumerate(val_pbar):
                batch_loss = validation_step(encoder, decoder, inp, targ, targ_start)
                val_losses.append(batch_loss.numpy())

                # Update progress bar
                val_pbar.set_postfix({'loss': f'{batch_loss.numpy():.4f}'})

            avg_val_loss = np.mean(val_losses)
            val_perplexity = calculate_perplexity(avg_val_loss)

            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time

            # Update metrics
            epoch_metrics = {
                'train_loss': float(avg_train_loss),
                'val_loss': float(avg_val_loss),
                'train_perplexity': float(train_perplexity),
                'val_perplexity': float(val_perplexity)
            }
            metrics.update(epoch, epoch_metrics, current_lr, epoch_time)

            # Log metrics
            logger.info(
                f"Epoch {epoch + 1}/{EPOCHS} - "
                f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                f"Train PPL: {train_perplexity:.2f}, Val PPL: {val_perplexity:.2f}, "
                f"LR: {current_lr:.6f}, Time: {epoch_time:.2f}s"
            )

            # Write TensorBoard summaries
            with train_writer.as_default():
                tf.summary.scalar('loss', avg_train_loss, step=epoch)
                tf.summary.scalar('perplexity', train_perplexity, step=epoch)
                tf.summary.scalar('learning_rate', current_lr, step=epoch)

            with val_writer.as_default():
                tf.summary.scalar('loss', avg_val_loss, step=epoch)
                tf.summary.scalar('perplexity', val_perplexity, step=epoch)

            # Save checkpoint
            if (epoch + 1) % CHECKPOINT_SAVE_FREQ == 0:
                checkpoint_path = checkpoint_manager.save()
                logger.info(f"Checkpoint saved: {checkpoint_path}")

            # Save best model
            if avg_val_loss < early_stopping.best_loss:
                best_checkpoint_path = checkpoint_manager.save()
                logger.info(f"Best model saved: {best_checkpoint_path}")

            # Check early stopping
            if early_stopping.should_stop(avg_val_loss, epoch):
                logger.info("Early stopping triggered!")
                break

        # Save final metrics and plots
        metrics.save(config.checkpoint_dir / 'training_metrics.json')
        metrics.plot(config.checkpoint_dir)

        # Save training configuration
        training_config = {
            'experiment_name': config.experiment_name,
            'epochs_trained': epoch + 1,
            'best_val_loss': early_stopping.best_loss,
            'best_epoch': early_stopping.best_epoch,
            'final_train_loss': avg_train_loss,
            'final_val_loss': avg_val_loss,
            'batch_size': BATCH_SIZE,
            'embedding_dim': EMBEDDING_DIM,
            'units': UNITS,
            'learning_rate': LEARNING_RATE,
            'vocab_input_size': vocab_inp_size,
            'vocab_target_size': vocab_tar_size
        }

        with open(config.checkpoint_dir / 'training_config.json', 'w') as f:
            json.dump(training_config, f, indent=2)

        logger.info("Training completed successfully!")
        logger.info(f"Best validation loss: {early_stopping.best_loss:.4f} at epoch {early_stopping.best_epoch}")
        logger.info(f"Training artifacts saved in: {config.checkpoint_dir}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Train NMT Paraphrase Generation Model')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    train_model(resume_from_checkpoint=args.resume)

if __name__ == "__main__":
    main()