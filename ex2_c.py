#!/usr/bin/env python3
"""
Language Model Implementation for Applied Probability Models
Authors: [Your names and IDs here]

This program implements and evaluates unigram language models using Lidstone and
Held-out smoothing methods. It processes text data, trains models, and evaluates
their performance using perplexity measures.
"""

from dataclasses import dataclass
from collections import Counter
from typing import List, Dict, Tuple
import argparse
import math
import sys
from pathlib import Path

@dataclass
class ModelParameters:
    """Parameters and constants used in language modeling."""
    VOCABULARY_SIZE: int = 300_000
    LIDSTONE_MAX_LAMBDA: float = 2.0
    LIDSTONE_LAMBDA_STEP: float = 0.01
    TRAIN_VALIDATION_SPLIT: float = 0.9
    HELD_OUT_SPLIT: float = 0.5

class TextProcessor:
    """Handles text processing and corpus statistics."""
    
    @staticmethod
    def read_corpus(filename: str) -> List[str]:
        """
        Reads and processes a corpus file, extracting only article content (not headers).
        
        Args:
            filename: Path to the corpus file
            
        Returns:
            List of words from the corpus
        """
        words = []
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for i in range(2, len(lines), 2):  # Skip header lines
            words.extend(lines[i].strip().split())
        return words

    @staticmethod
    def split_corpus(words: List[str], split_ratio: float) -> Tuple[List[str], List[str]]:
        """
        Splits a corpus into two parts based on the given ratio.
        
        Args:
            words: List of words to split
            split_ratio: Ratio for the first part (0 to 1)
            
        Returns:
            Tuple of (first part, second part)
        """
        split_point = round(len(words) * split_ratio)
        return words[:split_point], words[split_point:]

class UnigramModel:
    """Base class for unigram language models."""
    
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.word_counts = Counter()
        self.total_words = 0
        
    def train(self, words: List[str]) -> None:
        """Trains the model on the given words."""
        self.word_counts = Counter(words)
        self.total_words = len(words)
    
    def get_uniform_probability(self) -> float:
        """Returns the uniform probability for all words."""
        return 1.0 / self.vocab_size
    
    def get_mle_probability(self, word: str) -> float:
        """Returns the Maximum Likelihood Estimate for a word."""
        if self.total_words == 0:
            return 0.0
        return self.word_counts[word] / self.total_words
    
    def calculate_perplexity(self, test_words: List[str]) -> float:
        """Calculates the perplexity of the model on test data."""
        if not test_words:
            return float('inf')
        log_prob_sum = sum(math.log2(max(self.get_word_probability(word), sys.float_info.min)) 
                          for word in test_words)
        return 2 ** (-log_prob_sum / len(test_words))

class LidstoneModel(UnigramModel):
    """Implements Lidstone smoothing for unigram language model."""
    
    def __init__(self, vocab_size: int, lambda_param: float):
        super().__init__(vocab_size)
        self.lambda_param = lambda_param
        
    def get_word_probability(self, word: str) -> float:
        """Calculates the Lidstone-smoothed probability of a word."""
        count = self.word_counts[word]
        return (count + self.lambda_param) / (self.total_words + self.lambda_param * self.vocab_size)

    def get_expected_frequency(self, count: int) -> float:
        """Calculates the expected frequency for a given count."""
        prob = (count + self.lambda_param) / (self.total_words + self.lambda_param * self.vocab_size)
        return prob * self.total_words

class HeldOutModel(UnigramModel):
    """Implements Held-out smoothing for unigram language model."""
    
    def __init__(self, vocab_size: int):
        super().__init__(vocab_size)
        self.held_out_counts = Counter()
        self.r_counts = Counter()  # Count of counts
        self.t_r = {}  # Sum of held-out counts for each training count
        self.held_out_size = 0
        
    def train(self, train_words: List[str], held_out_words: List[str]) -> None:
        """Trains the model using both training and held-out data."""
        super().train(train_words)
        self.held_out_counts = Counter(held_out_words)
        self.held_out_size = len(held_out_words)
        
        # Calculate r_counts (N_r)
        self.r_counts = Counter(self.word_counts.values())
        
        # Calculate t_r
        self.t_r = {}
        for word, train_count in self.word_counts.items():
            self.t_r[train_count] = self.t_r.get(train_count, 0) + self.held_out_counts[word]
            
        # Calculate t_0 for unseen words
        seen_words_held_out = sum(self.held_out_counts.values())
        self.t_r[0] = self.held_out_size - seen_words_held_out
            
    def get_word_probability(self, word: str) -> float:
        """Calculates the held-out probability of a word."""
        train_count = self.word_counts[word]
        if train_count == 0:  # Unseen word
            return (self.t_r.get(0, 0) / self.held_out_size) / (
                self.vocab_size - len(self.word_counts))
        return (self.t_r.get(train_count, 0) / self.held_out_size) / self.r_counts[train_count]

    def get_expected_frequency(self, count: int) -> float:
        """Calculates the expected frequency for a given count."""
        if count == 0:
            return self.get_word_probability("unseen-word") * self.total_words
        prob = (self.t_r.get(count, 0) / self.held_out_size) / self.r_counts[count]
        return prob * self.total_words

class ModelEvaluator:
    """Handles model evaluation and parameter selection."""
    
    @staticmethod
    def find_best_lambda(train_words: List[str], valid_words: List[str], 
                        params: ModelParameters) -> Tuple[float, float, Dict[float, float]]:
        """Finds the best lambda parameter for Lidstone smoothing."""
        best_lambda = 0.0
        best_perplexity = float('inf')
        perplexities = {}
        
        current_lambda = 0.0
        while current_lambda <= params.LIDSTONE_MAX_LAMBDA:
            model = LidstoneModel(params.VOCABULARY_SIZE, current_lambda)
            model.train(train_words)
            perplexity = model.calculate_perplexity(valid_words)
            perplexities[current_lambda] = perplexity
            
            if perplexity < best_perplexity:
                best_perplexity = perplexity
                best_lambda = current_lambda
                
            current_lambda = round(current_lambda + params.LIDSTONE_LAMBDA_STEP, 2)
            
        return best_lambda, best_perplexity, perplexities

def write_outputs(args: argparse.Namespace, results: Dict) -> None:
    """Writes all required outputs to the output file."""
    with open(args.output_file, 'w', encoding='utf-8') as f:
        # Write student information
        f.write("#Students [Your names] [Your IDs]\n")
        
        # Write outputs 1-28
        for i in range(1, 29):
            value = results.get(f'output{i}')
            if value is not None:
                f.write(f"#Output{i}\t{value}\n")
        
        # Write output 29 (table)
        f.write("#Output29\n")
        for r in range(10):  # 0 to 9
            values = results['output29'][r]
            f.write("\t".join(f"{v:.5f}" for v in values) + "\n")

def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Language Model Evaluation')
    parser.add_argument('dev_file', help='Development set filename')
    parser.add_argument('test_file', help='Test set filename')
    parser.add_argument('input_word', help='Input word to evaluate')
    parser.add_argument('output_file', help='Output filename')
    args = parser.parse_args()
    
    params = ModelParameters()
    processor = TextProcessor()
    
    # Store all outputs
    results = {}
    
    # Basic outputs (1-5)
    results['output1'] = args.dev_file
    results['output2'] = args.test_file
    results['output3'] = args.input_word
    results['output4'] = args.output_file
    results['output5'] = params.VOCABULARY_SIZE
    results['output6'] = 1.0 / params.VOCABULARY_SIZE
    
    # Process development set
    dev_words = processor.read_corpus(args.dev_file)
    results['output7'] = len(dev_words)
    
    # Split for Lidstone model
    train_words, valid_words = processor.split_corpus(
        dev_words, params.TRAIN_VALIDATION_SPLIT)
    
    # Outputs 8-11
    results['output8'] = len(valid_words)
    results['output9'] = len(train_words)
    results['output10'] = len(set(train_words))
    results['output11'] = Counter(train_words)[args.input_word]
    
    # Train initial model for MLE outputs
    mle_model = UnigramModel(params.VOCABULARY_SIZE)
    mle_model.train(train_words)
    results['output12'] = mle_model.get_mle_probability(args.input_word)
    results['output13'] = 0.0  # MLE probability for unseen word
    
    # Lidstone model with λ = 0.10
    lidstone_01 = LidstoneModel(params.VOCABULARY_SIZE, 0.10)
    lidstone_01.train(train_words)
    results['output14'] = lidstone_01.get_word_probability(args.input_word)
    results['output15'] = lidstone_01.get_word_probability("unseen-word")
    
    # Different lambda perplexities
    lidstone_001 = LidstoneModel(params.VOCABULARY_SIZE, 0.01)
    lidstone_001.train(train_words)
    results['output16'] = lidstone_001.calculate_perplexity(valid_words)
    results['output17'] = lidstone_01.calculate_perplexity(valid_words)
    
    lidstone_1 = LidstoneModel(params.VOCABULARY_SIZE, 1.00)
    lidstone_1.train(train_words)
    results['output18'] = lidstone_1.calculate_perplexity(valid_words)
    
    # Find best lambda
    best_lambda, best_perplexity, _ = ModelEvaluator.find_best_lambda(
        train_words, valid_words, params)
    results['output19'] = best_lambda
    results['output20'] = best_perplexity
    
    # Held-out model
    held_out_train, held_out_eval = processor.split_corpus(
        dev_words, params.HELD_OUT_SPLIT)
    
    results['output21'] = len(held_out_train)
    results['output22'] = len(held_out_eval)
    
    held_out_model = HeldOutModel(params.VOCABULARY_SIZE)
    held_out_model.train(held_out_train, held_out_eval)
    
    results['output23'] = held_out_model.get_word_probability(args.input_word)
    results['output24'] = held_out_model.get_word_probability("unseen-word")
    
    # Test set evaluation
    test_words = processor.read_corpus(args.test_file)
    results['output25'] = len(test_words)
    
    # Final models evaluation
    best_lidstone = LidstoneModel(params.VOCABULARY_SIZE, best_lambda)
    best_lidstone.train(dev_words)  # Train on full development set
    
    lid_test_perp = best_lidstone.calculate_perplexity(test_words)
    held_test_perp = held_out_model.calculate_perplexity(test_words)
    
    results['output26'] = lid_test_perp
    results['output27'] = held_test_perp
    results['output28'] = 'L' if lid_test_perp < held_test_perp else 'H'
    
    # Generate table for output29
    table_data = []
    for r in range(10):  # 0 to 9
        f_mle = float(r)
        f_lambda = best_lidstone.get_expected_frequency(r)
        f_h = held_out_model.get_expected_frequency(r)
        N_r = held_out_model.r_counts[r]
        t_r = held_out_model.t_r.get(r, 0)
        table_data.append([f_mle, f_lambda, f_h, N_r, t_r])
    results['output29'] = table_data
    
    # Write all outputs to file
    write_outputs(args, results)

if __name__ == "__main__":
    main()