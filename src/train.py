#!/usr/bin/env python
# coding: utf-8
"""
Train the NMT paraphrase generation model.
"""
import os
import time
from datetime import datetime
import tensorflow as tf
import dill as pickle
from config import BATCH_SIZE, EMBEDDING_DIM, UNITS, EPOCHS, TRAIN_CHECKPOINTS_DIR, SCALARS_LOG_DIR
from utils import save_pickle
from models import Encoder, Decoder

# Load preprocessed data and vocabulary
with open('./pkl/train_dataset.pkl', 'rb') as f:
    train_dataset = pickle.load(f)
with open('./pkl/test_dataset.pkl', 'rb') as f:
    test_dataset = pickle.load(f)
with open('./pkl/inp_lang.pkl', 'rb') as f:
    inp_lang = pickle.load(f)
with open('./pkl/targ_lang.pkl', 'rb') as f:
    targ_lang = pickle.load(f)
with open('./pkl/max_length_inp.pkl', 'rb') as f:
    max_length_inp = pickle.load(f)
with open('./pkl/max_length_tar.pkl', 'rb') as f:
    max_length_tar = pickle.load(f)

vocab_inp_size = len(inp_lang.word2idx)
vocab_tar_size = len(targ_lang.word2idx)

# For feeding the target start tokens
targ_start = tf.expand_dims([targ_lang.word2idx['<start>']] * BATCH_SIZE, 1)

# Instantiate models
encoder = Encoder(vocab_inp_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam()

checkpoint_dir = os.path.join(TRAIN_CHECKPOINTS_DIR, datetime.now().strftime('%Y%m%d_%H%M%S'))
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

# Loss function
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(
                real, pred, sample_weight=tf.cast(mask, dtype=tf.float32))
    return tf.reduce_mean(loss_)

# TensorBoard setup
train_log_dir = os.path.join(SCALARS_LOG_DIR, 'train', datetime.now().strftime('%Y%m%d_%H%M%S'))
val_log_dir = os.path.join(SCALARS_LOG_DIR, 'validation', datetime.now().strftime('%Y%m%d_%H%M%S'))
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_summary_writer = tf.summary.create_file_writer(val_log_dir)

@tf.function
def train_step(inp, targ):
    loss = 0
    with tf.GradientTape() as tape:
        enc_hidden = encoder.initialize_hidden_state()
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = targ_start

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)
    batch_loss = loss / int(targ.shape[1])
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss

min_val_loss = float('inf')
for epoch in range(EPOCHS):
    start = time.time()
    total_loss = 0
    for (batch, (inp, targ)) in enumerate(train_dataset):
        batch_loss = train_step(inp, targ)
        total_loss += batch_loss
        print(f'Epoch {epoch + 1} Batch {batch + 1} Loss {batch_loss.numpy():.4f}')
    
    # Validation loop
    total_val_loss = 0
    for (batch, (inp, targ)) in enumerate(test_dataset):
        enc_hidden = encoder.initialize_hidden_state()
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = targ_start
        loss = 0
        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)
        total_val_loss += loss / int(targ.shape[1])
    
    # Write summaries
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', total_loss/len(train_dataset), step=epoch)
    with val_summary_writer.as_default():
        tf.summary.scalar('loss', total_val_loss/len(test_dataset), step=epoch)

    # Save checkpoint if validation loss improves
    avg_val_loss = total_val_loss/len(test_dataset)
    if avg_val_loss < min_val_loss - 0.05:
        checkpoint.save(file_prefix=checkpoint_prefix)
        min_val_loss = avg_val_loss

    print(f'Epoch {epoch + 1} Loss {total_loss/len(train_dataset):.4f} Val Loss {avg_val_loss:.4f}')
    print(f'Time taken for epoch {epoch + 1}: {time.time() - start:.2f} sec\n')