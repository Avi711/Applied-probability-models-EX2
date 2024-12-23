import sys
import math
from typing import List, Dict, Tuple
from collections import defaultdict

class ProbabilityModel:
    def __init__(self, vocab_size: int = 300000):
        self.vocab_size = vocab_size
        self.uniform_prob = 1.0 / vocab_size if vocab_size > 0 else 0
        
    def read_file(self, filename: str) -> List[str]:
        words = []
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i in range(2, len(lines), 2):
                words.extend(lines[i].strip().split())
        print(len(words))
        exit()
        return words

    def get_word_counts(self, words: List[str]) -> Dict[str, int]:
        counts = defaultdict(int)
        for word in words:
            counts[word] += 1
        return counts

    def calculate_mle(self, word: str, word_counts: Dict[str, int], total_words: int) -> float:
        if total_words == 0:
            return 0
        return word_counts.get(word, 0) / total_words

    def calculate_lidstone(self, word: str, word_counts: Dict[str, int], total_words: int, lambda_param: float) -> float:
        if total_words == 0 or lambda_param < 0:
            return 0
        return (word_counts.get(word, 0) + lambda_param) / (total_words + lambda_param * self.vocab_size)

    def calculate_perplexity(self, words: List[str], model_probs) -> float:
        if not words:
            return float('inf')
        
        log_prob_sum = 0
        for word in words:
            prob = model_probs(word)
            if prob <= 0:
                return float('inf')
            log_prob_sum += math.log2(prob)
        
        return 2 ** (-log_prob_sum / len(words))

    def find_best_lambda(self, train_counts: Dict[str, int], train_size: int, 
                        validation_words: List[str]) -> Tuple[float, float]:
        best_lambda = 0
        best_perplexity = float('inf')
        
        for lambda_param in [round(x/100, 2) for x in range(0, 201)]:  # 0 to 2 in steps of 0.01
            def model_probs(word):
                return self.calculate_lidstone(word, train_counts, train_size, lambda_param)
            
            perplexity = self.calculate_perplexity(validation_words, model_probs)
            if perplexity < best_perplexity:
                best_perplexity = perplexity
                best_lambda = lambda_param
                
        return best_lambda, best_perplexity

    def calculate_held_out_probs(self, train_counts: Dict[str, int], held_out_counts: Dict[str, int],
                                train_size: int, held_out_size: int) -> Dict[int, float]:
        """Calculate held-out probabilities for each frequency class."""
        if held_out_size == 0:
            return {}

        # Count Nr (number of words that appear r times in training)
        nr_counts = defaultdict(int)
        for count in train_counts.values():
            nr_counts[count] += 1
            
        # Calculate tr (sum of held-out counts for words with frequency r in training)
        tr_counts = defaultdict(int)
        for word, train_count in train_counts.items():
            tr_counts[train_count] += held_out_counts.get(word, 0)
            
        # Calculate held-out probabilities
        held_out_probs = {}
        for r in range(max(train_counts.values()) + 1):
            if nr_counts[r] > 0:
                held_out_probs[r] = tr_counts[r] / (nr_counts[r] * held_out_size)
                
        # Handle unseen words
        n0 = self.vocab_size - len(train_counts)
        if n0 > 0:
            t0 = held_out_size - sum(tr_counts.values())
            held_out_probs[0] = t0 / (n0 * held_out_size) if n0 > 0 else 0
            
        return held_out_probs, nr_counts, tr_counts

def generate_output(develop_file: str, test_file: str, input_word: str, output_file: str):
    model = ProbabilityModel()
    
    # Read development set
    dev_words = model.read_file(develop_file)
    dev_size = len(dev_words)
    
    # Split for Lidstone (90% train, 10% validation)
    train_size = round(0.9 * dev_size)
    train_words = dev_words[:train_size]
    validation_words = dev_words[train_size:]
    train_counts = model.get_word_counts(train_words)
    
    # Split for held-out (50% train, 50% held-out)
    held_out_split = len(dev_words) // 2
    held_out_train_words = dev_words[:held_out_split]
    held_out_words = dev_words[held_out_split:]
    held_out_train_counts = model.get_word_counts(held_out_train_words)
    held_out_counts = model.get_word_counts(held_out_words)
    
    # Calculate all required probabilities and metrics
    outputs = {}
    
    # Basic information (Output 1-6)
    outputs["Output1"] = develop_file
    outputs["Output2"] = test_file
    outputs["Output3"] = input_word
    outputs["Output4"] = output_file
    outputs["Output5"] = model.vocab_size
    outputs["Output6"] = model.uniform_prob
    
    # Development set statistics (Output 7-11)
    outputs["Output7"] = dev_size  # total events in development set
    outputs["Output8"] = len(validation_words)  # events in validation set
    outputs["Output9"] = len(train_words)  # events in training set
    outputs["Output10"] = len(train_counts)  # different events in training set
    outputs["Output11"] = train_counts.get(input_word, 0)  # INPUT_WORD count in training
    
    # MLE probabilities (Output 12-13)
    outputs["Output12"] = model.calculate_mle(input_word, train_counts, len(train_words))
    outputs["Output13"] = 0  # MLE probability for unseen words is always 0
    
    # Lidstone probabilities with λ=0.10 (Output 14-15)
    outputs["Output14"] = model.calculate_lidstone(input_word, train_counts, len(train_words), 0.10)
    outputs["Output15"] = model.calculate_lidstone("unseen-word", train_counts, len(train_words), 0.10)
    
    # Perplexity values for different λ (Output 16-18)
    def get_lidstone_perplexity(lambda_param):
        return model.calculate_perplexity(
            validation_words,
            lambda word: model.calculate_lidstone(word, train_counts, len(train_words), lambda_param)
        )
    
    outputs["Output16"] = get_lidstone_perplexity(0.01)
    outputs["Output17"] = get_lidstone_perplexity(0.10)
    outputs["Output18"] = get_lidstone_perplexity(1.00)
    
    # Best lambda and its perplexity (Output 19-20)
    best_lambda, best_perplexity = model.find_best_lambda(train_counts, len(train_words), validation_words)
    outputs["Output19"] = best_lambda
    outputs["Output20"] = best_perplexity
    
    # Held-out statistics (Output 21-22)
    outputs["Output21"] = len(held_out_train_words)
    outputs["Output22"] = len(held_out_words)
    
    # Calculate held-out probabilities and related statistics
    held_out_probs, nr_counts, tr_counts = model.calculate_held_out_probs(
        held_out_train_counts, held_out_counts, len(held_out_train_words), len(held_out_words)
    )
    
    # Held-out probabilities (Output 23-24)
    train_freq = held_out_train_counts.get(input_word, 0)
    outputs["Output23"] = held_out_probs.get(train_freq, held_out_probs.get(0, 0))
    outputs["Output24"] = held_out_probs.get(0, 0)
    
    # Test set evaluation (Output 25-28)
    test_words = model.read_file(test_file)
    outputs["Output25"] = len(test_words)
    
    # Calculate final perplexities
    def lidstone_probs(word):
        return model.calculate_lidstone(word, train_counts, len(train_words), best_lambda)
    
    def held_out_model(word):
        train_freq = held_out_train_counts.get(word, 0)
        return held_out_probs.get(train_freq, held_out_probs.get(0, 0))
    
    lidstone_test_perplexity = model.calculate_perplexity(test_words, lidstone_probs)
    held_out_test_perplexity = model.calculate_perplexity(test_words, held_out_model)
    
    outputs["Output26"] = lidstone_test_perplexity
    outputs["Output27"] = held_out_test_perplexity
    outputs["Output28"] = 'L' if lidstone_test_perplexity <= held_out_test_perplexity else 'H'
    
    # Write all outputs to file
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write student information
        f.write("#Students\t<student_name1>\t<id1>\n")  # Replace with actual student info
        
        # Write outputs 1-28
        for i in range(1, 29):
            key = f"Output{i}"
            f.write(f"#{key}\t{outputs[key]}\n")
        
        # Write Output29 table
        f.write("#Output29\n")
        for r in range(10):  # 0 to 9
            fmle = r
            flambda = (r + best_lambda) / (len(train_words) + best_lambda * model.vocab_size) * len(train_words)
            fh = held_out_probs.get(r, 0) * len(held_out_train_words)
            nr = nr_counts[r]
            tr = tr_counts.get(r, 0)
            f.write(f"{r}\t{fmle:.5f}\t{flambda:.5f}\t{fh:.5f}\t{nr}\t{tr}\n")

def main():
    if len(sys.argv) != 5:
        print("Usage: python ex2.py <develop_file> <test_file> <input_word> <output_file>")
        sys.exit(1)

    develop_file = sys.argv[1]
    test_file = sys.argv[2]
    input_word = sys.argv[3]
    output_file = sys.argv[4]
    
    generate_output(develop_file, test_file, input_word, output_file)

if __name__ == "__main__":
    main()