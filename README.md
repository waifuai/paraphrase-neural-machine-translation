# Paraphrase Generation with Neural Machine Translation (NMT)

This project implements a sequence-to-sequence Neural Machine Translation model using TensorFlow 2.x to generate paraphrases of English sentences. The model uses an encoder-decoder architecture with attention and GRU units. It is trained on the Parabank 100k dataset to generate paraphrases.

## Project Structure

```
repo/
├── README.md
├── requirements.txt
├── config.py
└── src
    ├── __init__.py
    ├── data.py         # Preprocesses data, builds vocabulary, and creates training/testing batches.
    ├── models.py       # Defines the Encoder and Decoder classes.
    ├── train.py        # Trains the model and logs progress to TensorBoard.
    ├── predict.py      # Loads a trained model, generates paraphrases, and visualizes attention.
    └── utils.py        # Utility functions for preprocessing, plotting, and data I/O.
```

## Setup

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Download the Dataset**

   Download the Parabank 100k dataset and place it in the `data` directory as `parabank_100k.tsv`.

3. **Preprocess the Data**

   Run the data preparation script:
   ```bash
   python src/data.py
   ```

## Training

Train the model by executing:
```bash
python src/train.py
```
This will save checkpoints and TensorBoard logs (viewable via `tensorboard --logdir logs/scalars`).

## Prediction

Before running predictions, update the `CHECKPOINT_PATH` in `src/predict.py` with your best checkpoint’s path, then run:
```bash
python src/predict.py
```

## Key Features

- **Modular Code Structure:** Separated configuration, models, data handling, and utility functions.
- **Efficient Data Processing:** Uses `tf.data` for batching and preprocessing.
- **Attention Visualization:** Plots attention weights to show how the model focuses on different parts of the input.
- **TensorBoard Integration:** For monitoring training and validation metrics.