# src/utils.py
import re
import os
import unicodedata
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from config import START_TOKEN, END_TOKEN, UNKNOWN_TOKEN, BATCH_SIZE, SPLIT_RATIO

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def preprocess_sentence(sentence: str) -> str:
    sentence = str(sentence).lower().strip()
    sentence = re.sub(r"([?!])", r" \1 ", sentence)
    sentence = re.sub(r"([,.])", "", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = sentence.rstrip().strip()
    return f"{START_TOKEN} {sentence} {END_TOKEN}"

def plot_attention(attention, sentence_tokens, predicted_tokens):
    """Visualize attention weights."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}
    ax.set_xticklabels([''] + sentence_tokens, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_tokens, fontdict=fontdict)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    plt.show()

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

# Vocabulary and indexing helper
class LanguageIndex:
    """Handles word to index and index to word mappings."""
    def __init__(self, phrases: pd.Series):
        self.phrases = phrases
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.build_index()

    def build_index(self):
        for phrase in self.phrases:
            self.vocab.update(phrase.split(' '))
        self.vocab.add(UNKNOWN_TOKEN)
        self.vocab = sorted(self.vocab)
        # Reserve index 0 for padding
        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab, start=1):
            self.word2idx[word] = index
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}