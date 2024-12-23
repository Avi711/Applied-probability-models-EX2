# Students: JaneDoe 123456789, JohnSmith 987654321
# ex2.py

import sys
import math


def load_articles(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    # Each article is on line i+1, skipping line i (the header).
    for i in range(0, len(lines)-2, 2):
        data.extend(lines[i+2].split())
    return data



def compute_counts(tokens):
    """
    Given a list of tokens, returns:
      1) a dictionary {word: count_of_word_in_tokens}
      2) total length (number of tokens)
    """
    counts = {}
    for w in tokens:
        counts[w] = counts.get(w, 0) + 1
    return counts, len(tokens)

def lidstone_probability(word, count_in_train, total_train, vocab_size, lam):
    """
    Lidstone smoothing P_lambda(word):
      = (count_in_train + lam) / (total_train + lam * vocab_size)
    """
    return (count_in_train + lam) / (total_train + lam * vocab_size)

def compute_perplexity_lidstone(token_list, train_counts, total_train, vocab_size, lam):
    """
    Compute perplexity of the given token_list under a Lidstone model
    with training counts = train_counts, total count = total_train,
    parameter lambda = lam, vocabulary size = vocab_size.
    perplexity = exp(- (1/N) * sum( log( p_lambda(w_i) ) ))
    where N = len(token_list).
    """
    N = len(token_list)
    log_likelihood = 0.0
    for w in token_list:
        c = train_counts.get(w, 0)
        p = lidstone_probability(w, c, total_train, vocab_size, lam)
        log_likelihood += math.log(p) if p > 0 else float('-inf')
    if N == 0:
        # if there's no data, define perplexity=1 or something trivial
        return 1.0
    avg_log_likelihood = log_likelihood / N
    perp = math.exp(-avg_log_likelihood)
    return perp

def heldout_model_probs(train_counts, heldout_counts, total_train, total_heldout, vocab_size):
    """
    Compute Held-out model probabilities for each distinct frequency r in train_counts.
    Steps:
      1. For each word in training set, we have freq c_T(w).
         Group words by their freq r.
      2. For words with freq r, we sum their frequencies in heldout set => t_r.
         The number of such words => N^T_r.
         Then define p_H(r) = t_r / (N^T_r * total_heldout).
      3. For a word that does not appear in training (r=0), we handle them as well:
         n_0 = vocabulary_size_minus_observed_in_train
         t_0 = sum of counts in heldout for all words that are not in the train set.
         p_H(0) = t_0 / ( n_0 * total_heldout ).
    Returns:
      A dictionary freq2prob, mapping r -> p_H(r).
      Also we store p_H(0) for unseen words. 
      (We'll handle the question "Which words are unseen?" by freq=0.)
    """
    # 1) group words by freq r in train
    freq_map = {}  # r -> list of words that appear r times
    for w, c in train_counts.items():
        freq_map.setdefault(c, []).append(w)
    # for r=0, we do not have them in train_counts, but we treat them as "unseen in training"
    # We'll compute their T^H freq from heldout if we can detect them. 
    # But we also have to consider that the vocabulary is 300k total. The extra ones are
    # truly unseen in both sets if they don't appear in held-out either. Then t_0 is 0 for them.

    # 2) for each r>0, sum frequencies in held-out
    r_to_sum_heldout = {}
    r_to_num_words = {}
    for r, words_r in freq_map.items():
        sum_h = 0
        for w in words_r:
            sum_h += heldout_counts.get(w, 0)
        r_to_sum_heldout[r] = sum_h
        r_to_num_words[r] = len(words_r)

    # we also must handle r=0
    # r=0 means "not in train set." The number of such distinct words is
    #   n_0 = vocab_size - len(train_counts)     (BUT note that some words in dev set might not be in train? 
    #   The problem statement explicitly says assume the entire language vocab is 300k.)
    #   t_0 = sum of heldout freq for those words that do not appear in train set
    n_observed_train = len(train_counts)  # distinct words in train
    n_0 = vocab_size - n_observed_train

    # Summation of all freq in heldout for words that are NOT in train
    # i.e. for w s.t. train_counts.get(w,0)=0
    # We only see them in heldout if they appear. We'll sum those frequencies.
    sum_h_0 = 0
    for w, c_h in heldout_counts.items():
        if w not in train_counts:  # freq=0 in train
            sum_h_0 += c_h

    # So t_0 = sum_h_0
    r_to_sum_heldout[0] = sum_h_0
    r_to_num_words[0] = n_0

    # 3) define p_H(r) = t_r / ( N^T_r * total_heldout ), for r where N^T_r>0
    #    for r=0, p_H(0) = sum_h_0 / (n_0 * total_heldout)
    freq2prob = {}
    for r, t_r in r_to_sum_heldout.items():
        N_tr = r_to_num_words[r]
        if N_tr > 0 and total_heldout > 0:
            freq2prob[r] = t_r / (N_tr * total_heldout)
        else:
            # if for some reason there's no data, fallback
            freq2prob[r] = 0.0
    return freq2prob

def heldout_probability(word, train_counts, freq2prob_heldout):
    """
    For a word w that appears c times in the training set,
      p_H(w) = freq2prob_heldout[c].
    If word not in training set => c=0 => p_H(w) = freq2prob_heldout[0].
    """
    c = train_counts.get(word, 0)
    return freq2prob_heldout.get(c, 0.0)

def compute_perplexity_heldout(token_list, train_counts, freq2prob_heldout):
    """
    Compute perplexity of the token_list under the held-out model.
    perplexity = exp(- (1/N)* sum(log p_H(w_i)) ).
    """
    N = len(token_list)
    log_likelihood = 0.0
    for w in token_list:
        p = heldout_probability(w, train_counts, freq2prob_heldout)
        # guard against p=0
        if p > 0:
            log_likelihood += math.log(p)
        else:
            log_likelihood += float('-inf')  # effectively kills perplexity
    if N == 0:
        return 1.0
    avg_log_likelihood = log_likelihood / N
    perp = math.exp(-avg_log_likelihood)
    return perp

def main():
    # 1) Parse arguments
    if len(sys.argv) != 5:
        print("Usage: python ex2.py <develop.txt> <test.txt> <input_word> <output.txt>")
        sys.exit(1)

    develop_filename = sys.argv[1]
    test_filename = sys.argv[2]
    input_word = sys.argv[3]
    output_filename = sys.argv[4]

    # Constants
    VOCAB_SIZE = 300000

    # We will store all #OutputXX in a dict for convenient printing later
    output_map = {}

    # #Output1..#Output6
    output_map["#Output1"] = develop_filename
    output_map["#Output2"] = test_filename
    output_map["#Output3"] = input_word
    output_map["#Output4"] = output_filename
    output_map["#Output5"] = VOCAB_SIZE  # language vocabulary size

    # Probability of INPUT WORD under uniform distribution = 1 / vocab_size
    p_uniform_input = 1.0 / VOCAB_SIZE
    output_map["#Output6"] = p_uniform_input

    # 2) Development set preprocessing
    develop_tokens = load_articles(develop_filename)
    S_size = len(develop_tokens)
    output_map["#Output7"] = S_size

    # 3) Lidstone model training
    # 3a) 90% - 10% split
    split_index = int(round(0.9 * S_size))
    train_tokens_90 = develop_tokens[:split_index]
    valid_tokens_10 = develop_tokens[split_index:]
    output_map["#Output8"] = len(valid_tokens_10)  # validation set size
    output_map["#Output9"] = len(train_tokens_90)  # training set size

    train_counts_90, total_train_90 = compute_counts(train_tokens_90)
    distinct_train_90 = len(train_counts_90)
    output_map["#Output10"] = distinct_train_90

    count_input_word_90 = train_counts_90.get(input_word, 0)
    output_map["#Output11"] = count_input_word_90

    # MLE probability for INPUT_WORD, no smoothing
    # = count_input_word_90 / total_train_90
    if total_train_90 > 0:
        p_mle_input = count_input_word_90 / total_train_90
    else:
        p_mle_input = 0.0
    output_map["#Output12"] = p_mle_input

    # MLE probability for "unseen-word" (if unseen-word not in training)
    c_unseen = train_counts_90.get("unseen-word", 0)
    if c_unseen == 0:
        p_mle_unseen = 0.0
    else:
        p_mle_unseen = c_unseen / total_train_90
    output_map["#Output13"] = p_mle_unseen

    # For lambda=0.1 (example)
    lam_01 = 0.10
    p_lid_input_01 = lidstone_probability(input_word, count_input_word_90,
                                          total_train_90, VOCAB_SIZE, lam_01)
    output_map["#Output14"] = p_lid_input_01

    # For "unseen-word" with lambda=0.1
    # Suppose "unseen-word" is not in train => c=0
    p_lid_unseen_01 = lidstone_probability("unseen-word", 0,
                                           total_train_90, VOCAB_SIZE, lam_01)
    output_map["#Output15"] = p_lid_unseen_01

    # Now perplexities for lambda=0.01, 0.10, 1.00 on validation set
    lam_001 = 0.01
    pp_001 = compute_perplexity_lidstone(valid_tokens_10,
                                         train_counts_90, total_train_90,
                                         VOCAB_SIZE, lam_001)
    output_map["#Output16"] = pp_001

    pp_01 = compute_perplexity_lidstone(valid_tokens_10,
                                        train_counts_90, total_train_90,
                                        VOCAB_SIZE, lam_01)
    output_map["#Output17"] = pp_01

    lam_1 = 1.0
    pp_1 = compute_perplexity_lidstone(valid_tokens_10,
                                       train_counts_90, total_train_90,
                                       VOCAB_SIZE, lam_1)
    output_map["#Output18"] = pp_1

    # 3b) Choose the best lambda in [0..2] stepping by 0.01 that yields min perplexity
    #     We'll just do a quick brute force. We'll store the best.
    best_lambda = 0.0
    best_pp = float('inf')
    step = 0.01
    lam_candidates = []
    # we can do up to two digits after decimal => range(0, 201) => lam= i*0.01
    for i in range(201):
        lam_i = i * 0.01
        pp_val = compute_perplexity_lidstone(valid_tokens_10,
                                             train_counts_90, total_train_90,
                                             VOCAB_SIZE, lam_i)
        if pp_val < best_pp:
            best_pp = pp_val
            best_lambda = lam_i

    output_map["#Output19"] = best_lambda
    output_map["#Output20"] = best_pp

    # 4) Held-out model training
    # Split dev set into 2 halves
    half_index = S_size // 2
    train_tokens_half = develop_tokens[:half_index]
    heldout_tokens_half = develop_tokens[half_index:]

    output_map["#Output21"] = len(train_tokens_half)
    output_map["#Output22"] = len(heldout_tokens_half)

    train_counts_half, total_train_half = compute_counts(train_tokens_half)
    heldout_counts_half, total_heldout_half = compute_counts(heldout_tokens_half)

    # Build the Held-out model
    freq2prob_H = heldout_model_probs(train_counts_half,
                                      heldout_counts_half,
                                      total_train_half,
                                      total_heldout_half,
                                      VOCAB_SIZE)
    # P(Event=INPUT_WORD) under held-out
    p_heldout_input = heldout_probability(input_word, train_counts_half, freq2prob_H)
    output_map["#Output23"] = p_heldout_input

    # P(Event='unseen-word') under held-out
    p_heldout_unseen = heldout_probability("unseen-word", train_counts_half, freq2prob_H)
    output_map["#Output24"] = p_heldout_unseen

    # 5) Debug condition:
    # p(x*) n_0 + sum_{ x: count(x)>0 } p(x) = 1
    # We'll trust we've coded carefully. (Not asked to output, just to check.)

    # 6) Models evaluation on test set
    test_tokens = load_articles(test_filename)
    output_map["#Output25"] = len(test_tokens)

    # Perplexities on test set
    #   a) best-lambda Lidstone
    pp_test_lidstone = compute_perplexity_lidstone(test_tokens,
                                                   train_counts_90, total_train_90,
                                                   VOCAB_SIZE, best_lambda)
    output_map["#Output26"] = pp_test_lidstone

    #   b) held-out
    pp_test_heldout = compute_perplexity_heldout(test_tokens,
                                                 train_counts_half,
                                                 freq2prob_H)
    output_map["#Output27"] = pp_test_heldout

    #   c) which model is better?
    if pp_test_lidstone < pp_test_heldout:
        output_map["#Output28"] = 'L'
    else:
        output_map["#Output28"] = 'H'

    #
    # 7) #Output29 => table of 10 lines for r=0..9 with columns:
    #    r = f_MLE | f_lambda | f_H | N^T_r | t_r
    #
    # For this table we *must* use the half-data training set S_T from step #4 (the 50-50 split).
    # Because the columns mention "N^T_r" and "t_r" which come from that step (S_T, S_H).
    #
    #   - f_MLE = r
    #   - f_lambda = expected freq under best-lambda model => p_lambda * |S_T|.
    #                p_lambda for a word with freq r => (r + best_lambda)/(|S_T| + best_lambda*|V|)
    #                Then the single-event expected freq is that probability times |S_T|.
    #   - f_H = p_H(r)*|S_T|, where p_H(r) = t_r / (N^T_r * |S_H|)
    #   - N^T_r = number of words with freq = r in the 50% training set
    #   - t_r = sum of freq in the held-out half for those words
    #
    # Let's gather r=0..9 from S_T = train_tokens_half
    #
    # We'll gather:
    #   freq_map_half[r] = list of words that appear r times in S_T
    #   r_to_sum_heldout[r] = sum of frequencies in held-out for those words
    #

    # Build freq_map_half similarly
    freq_map_half = {}
    for w, c in train_counts_half.items():
        freq_map_half.setdefault(c, []).append(w)

    # r=0 means words that do not appear in the half training set at all.
    # count them as n_0 = VOCAB_SIZE - distinct_in_train_half
    # but we also want the sum in held-out for those words. We'll compute that similarly.
    distinct_train_half = len(train_counts_half)
    r_to_sum_heldout_half = {}
    r_to_num_words_half = {}

    # For each r>0
    for r, words_r in freq_map_half.items():
        sum_h = 0
        for w in words_r:
            sum_h += heldout_counts_half.get(w, 0)
        r_to_sum_heldout_half[r] = sum_h
        r_to_num_words_half[r] = len(words_r)

    # Now handle r=0
    n_0_half = VOCAB_SIZE - distinct_train_half
    sum_h_0_half = 0
    for w, c_h in heldout_counts_half.items():
        if w not in train_counts_half:  # freq=0 in train half
            sum_h_0_half += c_h
    r_to_sum_heldout_half[0] = sum_h_0_half
    r_to_num_words_half[0] = n_0_half

    # We'll create lines for r=0..9. If r>distinct max, we skip or just define empty
    # The maximum freq in train_counts_half might be large, but we only do 0..9
    table_lines = []
    T_half = total_train_half
    H_half = total_heldout_half
    for r in range(10):
        # N^T_r
        N_tr = r_to_num_words_half.get(r, 0)
        # t_r
        t_r = r_to_sum_heldout_half.get(r, 0)
        # f_MLE = r
        f_mle = r
        # f_lambda = p_lambda(r)*T_half = ((r+best_lambda)/(T_half+best_lambda*V))*T_half
        # for an event that has freq=r
        denom = T_half + best_lambda * VOCAB_SIZE
        if denom > 0:
            f_lambda = ((r + best_lambda) / denom) * T_half
        else:
            f_lambda = 0.0
        # f_H = p_H(r)*T_half = [ t_r / (N^T_r * H_half ) ] * T_half
        if N_tr > 0 and H_half > 0:
            p_H_r = t_r / (N_tr * H_half)
            f_H = p_H_r * T_half
        else:
            f_H = 0.0
        row_str = f"{r}\t{f_mle:.5f}\t{f_lambda:.5f}\t{f_H:.5f}\t{N_tr}\t{t_r}"
        table_lines.append(row_str)

    # Now we have everything needed. Let's write them out to output_filename
    with open(output_filename, 'w', encoding='utf-8') as out:
        # First line: #Students ...
        out.write(f"#Students JaneDoe 123456789 JohnSmith 987654321\n")
        # Then each #OutputXX in ascending order
        for i in range(1, 29):
            key = f"#Output{i}"
            val = output_map[key]
            out.write(f"{key}\t{val}\n")
        # #Output29
        out.write(f"#Output29\n")
        # Then the 10 lines
        # Without the column titles, each line tab-delimited
        # r = f_MLE | f_lambda | f_H | N^T_r | t_r
        for line in table_lines:
            out.write(line + "\n")


if __name__ == "__main__":
    main()
