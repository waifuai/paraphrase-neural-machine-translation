#!/usr/bin/env python
# coding: utf-8
"""
Data Preparation Script

This script preprocesses the Parabank dataset, builds the vocabulary,
splits data into training and testing sets, and saves preprocessed
data as pickle files.
"""
import os
import re
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

from config import DATA_PATH, PKL_DIR, BATCH_SIZE, SPLIT_RATIO, START_TOKEN, END_TOKEN, UNKNOWN_TOKEN
from utils import preprocess_sentence, LanguageIndex, ensure_dir

def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, delimiter='\t', error_bad_lines=False, warn_bad_lines=False)
    # Assume columns are already named 'phrase1' and 'phrase2'
    data = data[['phrase1', 'phrase2']]
    data = data.sample(frac=1).reset_index(drop=True)
    return data

def prepare_dataset():
    data = load_data()
    # Filter out overly long sentences for efficiency
    data = data[data['phrase1'].apply(lambda x: len(str(x).split())) < 50]
    data = data[data['phrase2'].apply(lambda x: len(str(x).split())) < 50]
    data = data.sample(frac=1).reset_index(drop=True)
    
    n_train = int(data.shape[0] * (1 - SPLIT_RATIO))
    train_data = data.iloc[:n_train]
    test_data = data.iloc[n_train:]
    
    # Preprocess sentences
    train_data['phrase1'] = train_data['phrase1'].apply(preprocess_sentence)
    train_data['phrase2'] = train_data['phrase2'].apply(preprocess_sentence)
    test_data['phrase1'] = test_data['phrase1'].apply(preprocess_sentence)
    test_data['phrase2'] = test_data['phrase2'].apply(preprocess_sentence)
    
    # Build vocabulary indices
    inp_lang = LanguageIndex(train_data['phrase1'])
    targ_lang = LanguageIndex(train_data['phrase2'])
    
    max_length_inp = max(train_data['phrase1'].apply(lambda s: len(s.split())))
    max_length_tar = max(train_data['phrase2'].apply(lambda s: len(s.split())))
    
    return train_data, test_data, inp_lang, targ_lang, max_length_inp, max_length_tar

def create_tf_dataset(data: pd.DataFrame, inp_lang, targ_lang, max_length_inp, max_length_tar):
    # Convert sentences to sequences of token IDs
    def sentence_to_ids(sentence, lang_index):
        return [lang_index.word2idx.get(word, lang_index.word2idx[UNKNOWN_TOKEN])
                for word in sentence.split()]
    
    inp_sequences = data['phrase1'].apply(lambda s: sentence_to_ids(s, inp_lang)).tolist()
    targ_sequences = data['phrase2'].apply(lambda s: sentence_to_ids(s, targ_lang)).tolist()
    
    inp_sequences = tf.keras.preprocessing.sequence.pad_sequences(inp_sequences, maxlen=max_length_inp, padding='post')
    targ_sequences = tf.keras.preprocessing.sequence.pad_sequences(targ_sequences, maxlen=max_length_tar, padding='post')
    
    dataset = tf.data.Dataset.from_tensor_slices((inp_sequences, targ_sequences))
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset

def save_pickle(data, filename: str):
    ensure_dir(PKL_DIR)
    with open(os.path.join(PKL_DIR, filename), 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    train_data, test_data, inp_lang, targ_lang, max_length_inp, max_length_tar = prepare_dataset()
    
    train_dataset = create_tf_dataset(train_data, inp_lang, targ_lang, max_length_inp, max_length_tar)
    test_dataset = create_tf_dataset(test_data, inp_lang, targ_lang, max_length_inp, max_length_tar)
    
    # Save datasets and helper objects
    save_pickle(list(train_dataset.as_numpy_iterator()), "train_dataset.pkl")
    save_pickle(list(test_dataset.as_numpy_iterator()), "test_dataset.pkl")
    save_pickle(max_length_inp, "max_length_inp.pkl")
    save_pickle(max_length_tar, "max_length_tar.pkl")
    save_pickle(inp_lang, "inp_lang.pkl")
    save_pickle(targ_lang, "targ_lang.pkl")
    
    print("Data preparation complete.")
    print(f"Max input length: {max_length_inp}, Max target length: {max_length_tar}")