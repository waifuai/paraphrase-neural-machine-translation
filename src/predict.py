#!/usr/bin/env python
# coding: utf-8
"""
Inference script for generating paraphrases.
"""
import tensorflow as tf
import numpy as np
import dill as pickle
from config import UNITS, PKL_DIR
from models import Encoder, Decoder
from utils import preprocess_sentence, plot_attention

# Update this path to point to your best-performing checkpoint
CHECKPOINT_PATH = "./logs/training_checkpoints/<your_checkpoint_directory>/ckpt"

def load_model_data():
    try:
        with open(f"{PKL_DIR}/inp_lang.pkl", 'rb') as f:
            inp_lang = pickle.load(f)
        with open(f"{PKL_DIR}/targ_lang.pkl", 'rb') as f:
            targ_lang = pickle.load(f)
        with open(f"{PKL_DIR}/max_length_inp.pkl", 'rb') as f:
            max_length_inp = pickle.load(f)
        with open(f"{PKL_DIR}/max_length_tar.pkl", 'rb') as f:
            max_length_tar = pickle.load(f)
        
        encoder = Encoder(len(inp_lang.word2idx), 256, UNITS, batch_sz=1)
        decoder = Decoder(len(targ_lang.word2idx), 256, UNITS, batch_sz=1)
        optimizer = tf.keras.optimizers.Adam()

        checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
        checkpoint.restore(CHECKPOINT_PATH).expect_partial()
        return encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_tar
    except Exception as e:
        print("Error loading checkpoint and pickles:", e)
        return None

def evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_tar):
    sentence = preprocess_sentence(sentence)
    inputs = [inp_lang.word2idx.get(word, inp_lang.word2idx.get("<unknown>")) for word in sentence.split()]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    
    result = ""
    attention_plot = np.zeros((max_length_tar, max_length_inp))
    hidden = tf.zeros((1, UNITS))
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word2idx['<start>']], 0)
    
    for t in range(max_length_tar):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()
        
        predicted_id = tf.argmax(predictions[0]).numpy()
        predicted_word = targ_lang.idx2word[predicted_id]
        result += predicted_word + " "
        
        if predicted_word == "<end>":
            break
        dec_input = tf.expand_dims([predicted_id], 0)
    return result, sentence, attention_plot

def translate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_tar):
    result, sentence_proc, attention_plot = evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_tar)
    print("Input:", sentence_proc)
    print("Predicted Paraphrase:", result)
    attention_plot = attention_plot[:len(result.split()), :len(sentence_proc.split())]
    plot_attention(attention_plot, sentence_proc.split(), result.split())

if __name__ == "__main__":
    loaded = load_model_data()
    if loaded:
        encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_tar = loaded
        test_sentences = [
            "FooBar is the best you will ever get",
            "Alice is a good coder",
            "WaifuAI is the best",
            "I like candy",
            "WaifuAI is working really hard"
        ]
        for s in test_sentences:
            translate(s, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_tar)