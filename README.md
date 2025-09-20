# Paraphrase Generation with Neural Machine Translation (NMT)

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![License](https://img.shields.io/badge/License-MIT%200-green.svg)

This project implements a state-of-the-art sequence-to-sequence Neural Machine Translation model using TensorFlow 2.x to generate high-quality paraphrases of English sentences. The model employs an encoder-decoder architecture with Luong attention mechanism and GRU units, trained on the Parabank 100k dataset.

## 🎯 Features

- **Advanced Architecture**: Encoder-decoder with attention mechanism
- **Modern Training Pipeline**: Early stopping, checkpoint management, and comprehensive logging
- **Attention Visualization**: Interactive plots showing model focus areas
- **Production Ready**: Robust error handling, configuration validation, and modular design
- **Comprehensive Testing**: Full test suite with pytest
- **TensorBoard Integration**: Real-time monitoring of training metrics
- **CLI Interface**: Easy-to-use command-line interface for all operations

## 📊 Model Performance

- **Architecture**: GRU-based sequence-to-sequence with attention
- **Embedding Dimension**: 256
- **Hidden Units**: 1024
- **Attention**: Luong multiplicative attention
- **Training**: Adam optimizer with gradient clipping

## 🏗️ Project Structure

```
paraphrase-neural-machine-translation/
├── 📁 data/                          # Dataset directory
│   └── parabank_100k.tsv            # Parabank dataset
├── 📁 logs/                          # Training logs and checkpoints
│   ├── training_checkpoints/        # Model checkpoints
│   └── scalars/                     # TensorBoard logs
├── 📁 models/                        # Saved models
├── 📁 pkl/                           # Preprocessed data pickles
├── 📁 src/                           # Source code
│   ├── __init__.py
│   ├── data.py                      # Data preprocessing pipeline
│   ├── models.py                    # Neural network architectures
│   ├── train.py                     # Training orchestration
│   ├── predict.py                   # Inference and prediction
│   ├── utils.py                     # Utility functions
│   └── tests/                       # Test suite
│       └── test_suite.py
├── 📄 config.py                      # Configuration management
├── 📄 requirements.txt               # Python dependencies
├── 📄 pytest.ini                     # Testing configuration
├── 📄 README.md                      # Project documentation
└── 📄 LICENSE                        # MIT-0 License
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd paraphrase-neural-machine-translation
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - Download the Parabank 100k dataset from [source]
   - Place `parabank_100k.tsv` in the `data/` directory

### Usage

#### Data Preparation
```bash
python src/data.py
```
Preprocesses the dataset, builds vocabulary, and saves pickled data.

#### Training
```bash
python src/train.py
```
Trains the model with early stopping and checkpoint saving.

#### Inference
```bash
python src/predict.py
```
Generates paraphrases for test sentences with attention visualization.

#### Monitoring Training
```bash
tensorboard --logdir logs/scalars
```
View training metrics at `http://localhost:6006`

## 🔧 Configuration

All parameters are centralized in `config.py`:

```python
# Model parameters
EMBEDDING_DIM = 256
UNITS = 1024
BATCH_SIZE = 64

# Training parameters
EPOCHS = 10000
LEARNING_RATE = 1e-3
PATIENCE = 10

# File paths
DATA_PATH = "./data/parabank_100k.tsv"
CHECKPOINT_DIR = "./logs/training_checkpoints"
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest src/tests/test_suite.py
```

## 📈 Evaluation Metrics

The model is evaluated using:
- **BLEU Score**: Measures n-gram overlap with reference paraphrases
- **Perplexity**: Measures model confidence
- **Attention Quality**: Manual inspection of attention weights
- **Diversity Metrics**: Measures paraphrase diversity

## 🔍 Model Architecture

### Encoder
- Embedding layer (configurable dimension)
- Bidirectional GRU layers
- Dropout for regularization

### Decoder
- Embedding layer
- GRU with attention mechanism
- Dense output layer with softmax

### Attention Mechanism
- Luong multiplicative attention
- Context vector computation
- Attention weight visualization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT-0 License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{paraphrase_nmt_2025,
  title={Paraphrase Generation with Neural Machine Translation},
  author={Your Name},
  year={2025},
  url={https://github.com/waifuai/paraphrase-neural-machine-translation}
}
```

## 🆘 Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `BATCH_SIZE` in `config.py`
2. **Training Convergence Issues**: Adjust `LEARNING_RATE` or `UNITS`
3. **Poor Paraphrase Quality**: Increase model capacity or training data
4. **TensorBoard Not Working**: Check if port 6006 is available

### Getting Help

- Check the [Issues](https://github.com/waifuai/paraphrase-neural-machine-translation/issues) page
- Review the [Discussions](https://github.com/waifuai/paraphrase-neural-machine-translation/discussions) board
- Contact the maintainers

## 🔄 Changelog

### Version 2.0.0
- Complete codebase refactoring
- Added comprehensive test suite
- Improved model architecture
- Enhanced documentation
- Added CLI interface
- Better error handling and logging

---

**Made with ❤️ for the NLP community**