#!/usr/bin/env python
# coding: utf-8
"""
Enhanced Inference Script for Paraphrase Generation

Features:
- Beam search decoding
- Temperature sampling
- Multiple paraphrase generation
- Attention visualization
- Batch processing
- Model evaluation metrics
"""
import os
import time
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (
    UNITS, PKL_DIR, START_TOKEN, END_TOKEN, UNKNOWN_TOKEN,
    PAD_TOKEN, BEAM_WIDTH, MAX_DECODING_LENGTH, TEMPERATURE
)
from models import Encoder, Decoder
from utils import (
    preprocess_sentence, postprocess_sentence, plot_attention,
    load_pickle, calculate_bleu_score, ensure_dir
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# MODEL LOADING AND MANAGEMENT
# =============================================================================

class ParaphraseGenerator:
    """
    Enhanced paraphrase generator with multiple decoding strategies.
    """

    def __init__(self, checkpoint_path: str):
        """
        Initialize the paraphrase generator.

        Args:
            checkpoint_path: Path to the model checkpoint
        """
        self.checkpoint_path = checkpoint_path
        self.models_loaded = False
        self.load_models()

    def load_models(self) -> bool:
        """
        Load models and vocabulary from checkpoint and pickle files.

        Returns:
            True if loading successful, False otherwise
        """
        try:
            logger.info("Loading models and vocabulary...")

            # Load vocabulary and configuration
            self.inp_lang = load_pickle(Path(PKL_DIR) / 'inp_lang.pkl')
            self.targ_lang = load_pickle(Path(PKL_DIR) / 'targ_lang.pkl')
            self.max_length_inp = load_pickle(Path(PKL_DIR) / 'max_length_inp.pkl')
            self.max_length_tar = load_pickle(Path(PKL_DIR) / 'max_length_tar.pkl')

            # Create models (batch size 1 for inference)
            self.encoder = Encoder(
                len(self.inp_lang.word2idx),
                batch_sz=1
            )
            self.decoder = Decoder(
                len(self.targ_lang.word2idx),
                batch_sz=1
            )

            # Create optimizer (not used for inference, but needed for checkpoint)
            optimizer = tf.keras.optimizers.Adam()

            # Load checkpoint
            checkpoint = tf.train.Checkpoint(
                optimizer=optimizer,
                encoder=self.encoder,
                decoder=self.decoder
            )

            # Try to restore checkpoint
            status = checkpoint.restore(self.checkpoint_path)
            if status:
                logger.info(f"Successfully restored from checkpoint: {self.checkpoint_path}")
                self.models_loaded = True
                return True
            else:
                logger.error(f"Failed to restore checkpoint: {self.checkpoint_path}")
                return False

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

    def encode_input(self, sentence: str) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Encode input sentence for the model.

        Args:
            sentence: Input sentence to encode

        Returns:
            Tuple of (encoded_input, encoder_output, encoder_hidden)
        """
        # Preprocess sentence
        processed_sentence = preprocess_sentence(sentence)

        # Convert to token IDs
        inputs = [
            self.inp_lang.word2idx.get(word, self.inp_lang.word2idx[UNKNOWN_TOKEN])
            for word in processed_sentence.split()
        ]

        # Pad sequence
        inputs = tf.keras.preprocessing.sequence.pad_sequences(
            [inputs],
            maxlen=self.max_length_inp,
            padding='post'
        )
        inputs = tf.convert_to_tensor(inputs, dtype=tf.int32)

        # Encode
        hidden = tf.zeros((1, UNITS))
        enc_output, enc_hidden = self.encoder(inputs, hidden)

        return inputs, enc_output, enc_hidden

    # =============================================================================
    # DECODING STRATEGIES
    # =============================================================================

    def greedy_decode(self,
                      enc_output: tf.Tensor,
                      enc_hidden: tf.Tensor) -> Tuple[str, np.ndarray]:
        """
        Greedy decoding strategy.

        Args:
            enc_output: Encoder output
            enc_hidden: Encoder hidden state

        Returns:
            Tuple of (decoded_sentence, attention_weights)
        """
        result = []
        attention_plot = np.zeros((self.max_length_tar, self.max_length_inp))

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.targ_lang.word2idx[START_TOKEN]], 0)

        for t in range(self.max_length_tar):
            predictions, dec_hidden, attention_weights = self.decoder(
                dec_input, dec_hidden, enc_output
            )

            attention_weights = tf.reshape(attention_weights, (-1,))
            attention_plot[t] = attention_weights.numpy()

            # Get most probable next word
            predicted_id = tf.argmax(predictions[0]).numpy()
            predicted_word = self.targ_lang.idx2word[predicted_id]

            if predicted_word == END_TOKEN:
                break

            result.append(predicted_word)
            dec_input = tf.expand_dims([predicted_id], 0)

        decoded_sentence = ' '.join(result)
        return decoded_sentence, attention_plot

    def temperature_sampling_decode(self,
                                   enc_output: tf.Tensor,
                                   enc_hidden: tf.Tensor,
                                   temperature: float = TEMPERATURE) -> Tuple[str, np.ndarray]:
        """
        Temperature sampling decoding strategy.

        Args:
            enc_output: Encoder output
            enc_hidden: Encoder hidden state
            temperature: Sampling temperature

        Returns:
            Tuple of (decoded_sentence, attention_weights)
        """
        result = []
        attention_plot = np.zeros((self.max_length_tar, self.max_length_inp))

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.targ_lang.word2idx[START_TOKEN]], 0)

        for t in range(self.max_length_tar):
            predictions, dec_hidden, attention_weights = self.decoder(
                dec_input, dec_hidden, enc_output
            )

            attention_weights = tf.reshape(attention_weights, (-1,))
            attention_plot[t] = attention_weights.numpy()

            # Apply temperature scaling
            predictions = predictions / temperature
            predictions = tf.nn.softmax(predictions, axis=-1)

            # Sample from distribution
            predicted_id = tf.random.categorical(predictions, 1)[0, 0].numpy()
            predicted_word = self.targ_lang.idx2word[predicted_id]

            if predicted_word == END_TOKEN:
                break

            result.append(predicted_word)
            dec_input = tf.expand_dims([predicted_id], 0)

        decoded_sentence = ' '.join(result)
        return decoded_sentence, attention_plot

    def beam_search_decode(self,
                          enc_output: tf.Tensor,
                          enc_hidden: tf.Tensor,
                          beam_width: int = BEAM_WIDTH) -> List[Tuple[str, float]]:
        """
        Beam search decoding strategy.

        Args:
            enc_output: Encoder output
            enc_hidden: Encoder hidden state
            beam_width: Beam width for search

        Returns:
            List of (sentence, score) tuples
        """
        # This is a simplified beam search implementation
        # In practice, you'd want a more sophisticated implementation

        # For now, fall back to greedy decoding
        logger.warning("Beam search not fully implemented, falling back to greedy decoding")
        sentence, attention = self.greedy_decode(enc_output, enc_hidden)

        # Return as list with dummy score
        return [(sentence, 1.0)]

    def generate_paraphrases(self,
                            sentence: str,
                            num_paraphrases: int = 1,
                            decoding_strategy: str = 'greedy',
                            temperature: float = TEMPERATURE,
                            return_attention: bool = True) -> Union[List[str], List[Tuple[str, np.ndarray]]]:
        """
        Generate paraphrases for input sentence.

        Args:
            sentence: Input sentence
            num_paraphrases: Number of paraphrases to generate
            decoding_strategy: 'greedy', 'temperature', or 'beam'
            temperature: Temperature for sampling (if using temperature strategy)
            return_attention: Whether to return attention weights

        Returns:
            List of paraphrases, optionally with attention weights
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        logger.info(f"Generating {num_paraphrases} paraphrases for: {sentence}")

        # Encode input
        inputs, enc_output, enc_hidden = self.encode_input(sentence)

        paraphrases = []

        for i in range(num_paraphrases):
            if decoding_strategy == 'greedy':
                decoded_sentence, attention = self.greedy_decode(enc_output, enc_hidden)
            elif decoding_strategy == 'temperature':
                decoded_sentence, attention = self.temperature_sampling_decode(
                    enc_output, enc_hidden, temperature
                )
            elif decoding_strategy == 'beam':
                beam_results = self.beam_search_decode(enc_output, enc_hidden)
                decoded_sentence, _ = beam_results[0]
                # For simplicity, use greedy attention
                _, attention = self.greedy_decode(enc_output, enc_hidden)
            else:
                raise ValueError(f"Unknown decoding strategy: {decoding_strategy}")

            # Postprocess sentence
            decoded_sentence = postprocess_sentence(decoded_sentence)

            if return_attention:
                paraphrases.append((decoded_sentence, attention))
            else:
                paraphrases.append(decoded_sentence)

        return paraphrases

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to latest checkpoint or None if not found
    """
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        return None

    # Look for checkpoint files
    ckpt_files = list(checkpoint_path.glob("ckpt*.index"))
    if not ckpt_files:
        return None

    # Get the latest checkpoint
    latest_ckpt = max(ckpt_files, key=lambda x: x.stat().st_mtime)
    return str(latest_ckpt).replace('.index', '')

def evaluate_model(generator: ParaphraseGenerator,
                  test_sentences: List[str],
                  reference_paraphrases: Optional[List[List[str]]] = None) -> Dict[str, Any]:
    """
    Evaluate model performance on test sentences.

    Args:
        generator: ParaphraseGenerator instance
        test_sentences: List of test sentences
        reference_paraphrases: Optional list of reference paraphrases for BLEU score

    Returns:
        Dictionary containing evaluation metrics
    """
    logger.info("Evaluating model performance...")

    results = []
    inference_times = []

    for i, sentence in enumerate(tqdm(test_sentences, desc="Evaluating")):
        start_time = time.time()

        # Generate paraphrase
        paraphrases = generator.generate_paraphrases(
            sentence,
            num_paraphrases=1,
            decoding_strategy='greedy',
            return_attention=False
        )

        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        result = {
            'input': sentence,
            'paraphrase': paraphrases[0],
            'inference_time': inference_time
        }

        results.append(result)

    # Calculate metrics
    metrics = {
        'avg_inference_time': np.mean(inference_times),
        'std_inference_time': np.std(inference_times),
        'results': results
    }

    # Calculate BLEU score if references provided
    if reference_paraphrases:
        hypotheses = [[result['paraphrase'].split()] for result in results]
        bleu_score = calculate_bleu_score(reference_paraphrases, hypotheses)
        metrics['bleu_score'] = bleu_score
        logger.info(f"BLEU Score: {bleu_score:.4f}")

    logger.info(f"Average inference time: {metrics['avg_inference_time']:.4f}s")
    return metrics

# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def interactive_mode(generator: ParaphraseGenerator) -> None:
    """Run interactive paraphrase generation mode."""
    print("\n=== Interactive Paraphrase Generator ===")
    print("Enter sentences to generate paraphrases (type 'quit' to exit):")
    print("-" * 50)

    while True:
        try:
            sentence = input("\nEnter sentence: ").strip()

            if sentence.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not sentence:
                continue

            # Generate paraphrases
            paraphrases = generator.generate_paraphrases(
                sentence,
                num_paraphrases=3,
                decoding_strategy='temperature',
                return_attention=True
            )

            print(f"\nInput: {sentence}")
            print("Generated paraphrases:")

            for i, (paraphrase, attention) in enumerate(paraphrases):
                print(f"{i+1}. {paraphrase}")

                # Show attention visualization for first paraphrase
                if i == 0:
                    try:
                        fig = plot_attention(
                            attention,
                            sentence.split(),
                            paraphrase.split(),
                            save_path=f"attention_{int(time.time())}.png"
                        )
                        print(f"   Attention plot saved!")
                    except Exception as e:
                        logger.warning(f"Could not generate attention plot: {e}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")

def batch_process_sentences(generator: ParaphraseGenerator,
                          input_file: str,
                          output_file: str,
                          decoding_strategy: str = 'greedy') -> None:
    """
    Process sentences from file in batch mode.

    Args:
        generator: ParaphraseGenerator instance
        input_file: Path to input file with sentences
        output_file: Path to output file for results
        decoding_strategy: Decoding strategy to use
    """
    logger.info(f"Processing sentences from {input_file}")

    # Read input sentences
    with open(input_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    logger.info(f"Processing {len(sentences)} sentences...")

    # Generate paraphrases
    results = []
    for sentence in tqdm(sentences, desc="Processing"):
        try:
            paraphrases = generator.generate_paraphrases(
                sentence,
                num_paraphrases=1,
                decoding_strategy=decoding_strategy,
                return_attention=False
            )
            results.append({
                'input': sentence,
                'paraphrase': paraphrases[0]
            })
        except Exception as e:
            logger.error(f"Error processing sentence '{sentence}': {e}")
            results.append({
                'input': sentence,
                'paraphrase': '[ERROR]'
            })

    # Save results
    ensure_dir(Path(output_file).parent)
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"Input: {result['input']}\n")
            f.write(f"Paraphrase: {result['paraphrase']}\n")
            f.write("-" * 50 + "\n")

    logger.info(f"Results saved to {output_file}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate Paraphrases using NMT Model')

    # Model loading arguments
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--checkpoint-dir', type=str, help='Directory containing checkpoints (uses latest)')

    # Generation arguments
    parser.add_argument('--sentences', nargs='+', help='Sentences to paraphrase')
    parser.add_argument('--input-file', type=str, help='Input file with sentences')
    parser.add_argument('--output-file', type=str, help='Output file for results')
    parser.add_argument('--num-paraphrases', type=int, default=1, help='Number of paraphrases to generate')
    parser.add_argument('--strategy', choices=['greedy', 'temperature', 'beam'],
                       default='greedy', help='Decoding strategy')
    parser.add_argument('--temperature', type=float, default=TEMPERATURE, help='Sampling temperature')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model performance')

    args = parser.parse_args()

    # Find checkpoint path
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.checkpoint_dir:
        checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
    else:
        # Try to find checkpoint in default location
        default_dir = "./logs/training_checkpoints"
        if os.path.exists(default_dir):
            checkpoint_path = find_latest_checkpoint(default_dir)

    if not checkpoint_path:
        logger.error("No checkpoint found. Please specify --checkpoint or --checkpoint-dir")
        return

    # Initialize generator
    generator = ParaphraseGenerator(checkpoint_path)

    if not generator.models_loaded:
        logger.error("Failed to load models. Exiting.")
        return

    # Run requested mode
    if args.interactive:
        interactive_mode(generator)
    elif args.evaluate:
        # Use default test sentences
        test_sentences = [
            "FooBar is the best you will ever get",
            "Alice is a good coder",
            "WaifuAI is the best",
            "I like candy",
            "WaifuAI is working really hard"
        ]
        metrics = evaluate_model(generator, test_sentences)
        print(f"\nEvaluation Results:")
        print(f"Average Inference Time: {metrics['avg_inference_time']:.4f}s")
        if 'bleu_score' in metrics:
            print(f"BLEU Score: {metrics['bleu_score']:.4f}")
    elif args.input_file:
        if not args.output_file:
            args.output_file = f"paraphrases_{int(time.time())}.txt"
        batch_process_sentences(
            generator,
            args.input_file,
            args.output_file,
            args.strategy
        )
    elif args.sentences:
        for sentence in args.sentences:
            paraphrases = generator.generate_paraphrases(
                sentence,
                num_paraphrases=args.num_paraphrases,
                decoding_strategy=args.strategy,
                temperature=args.temperature,
                return_attention=False
            )

            print(f"\nInput: {sentence}")
            print("Generated paraphrases:")
            for i, paraphrase in enumerate(paraphrases):
                print(f"{i+1}. {paraphrase}")
    else:
        # Default behavior - test with sample sentences
        test_sentences = [
            "FooBar is the best you will ever get",
            "Alice is a good coder",
            "WaifuAI is the best",
            "I like candy",
            "WaifuAI is working really hard"
        ]

        print("Testing with sample sentences:")
        for sentence in test_sentences:
            paraphrases = generator.generate_paraphrases(
                sentence,
                num_paraphrases=1,
                decoding_strategy='greedy',
                return_attention=False
            )
            print(f"Input: {sentence}")
            print(f"Paraphrase: {paraphrases[0]}")
            print("-" * 50)

if __name__ == "__main__":
    main()