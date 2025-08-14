import math
import re
from collections import Counter, defaultdict

# -----------------
# Preprocessing
# -----------------
def preprocess(text):
    text = text.lower()
    tokens = re.findall(r'\b[a-z]+\b', text)
    return tokens

# -----------------
# N-gram Counts
# -----------------
def get_unigram_counts(tokens):
    return Counter(tokens)

def get_bigram_counts(tokens):
    bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
    return Counter(bigrams)

# -----------------
# Perplexity
# -----------------
def perplexity(model_prob_func, test_data, n=1):
    N = len(test_data) - (n - 1)
    log_prob_sum = 0
    for i in range(n - 1, len(test_data)):
        context = tuple(test_data[i - n + 1:i]) if n > 1 else ()
        prob = model_prob_func(context, test_data[i])
        if prob <= 0:
            prob = 1e-10  # Avoid log(0)
        log_prob_sum += math.log(prob)
    return math.exp(-log_prob_sum / N) if N > 0 else float('inf')

# -----------------
# Smoothing Methods
# -----------------
def no_smoothing_unigram(counts, total):
    def prob(_, word):
        return counts.get(word, 0) / total if total > 0 else 1e-10
    return prob

def no_smoothing_bigram(counts, unigram_counts):
    def prob(context, word):
        w1 = context[0] if context else None
        c_bigram = counts.get((w1, word), 0)
        c_unigram = unigram_counts.get(w1, 0)
        return c_bigram / c_unigram if c_unigram > 0 else 1e-10
    return prob

def add_k_unigram(counts, total, vocab_size, k=1):
    def prob(_, word):
        return (counts.get(word, 0) + k) / (total + k * vocab_size)
    return prob

def add_k_bigram(counts, unigram_counts, vocab_size, k=1):
    def prob(context, word):
        w1 = context[0] if context else None
        c_bigram = counts.get((w1, word), 0)
        c_unigram = unigram_counts.get(w1, 0)
        return (c_bigram + k) / (c_unigram + k * vocab_size)
    return prob

# Good–Turing helper
def good_turing_counts(counts):
    if not counts:
        return {}, 0
    freq_of_freq = Counter(counts.values())
    adjusted = {}
    for n in counts:
        c = counts[n]
        Nc = freq_of_freq[c]
        Nc1 = freq_of_freq.get(c+1, 0)
        if Nc > 0:
            adjusted[n] = (c + 1) * (Nc1 / Nc)
        else:
            adjusted[n] = c
    N = sum(counts.values())
    p0 = freq_of_freq.get(1, 0) / N if N > 0 else 1e-10
    return adjusted, p0

def good_turing_unigram(counts, total, vocab_size):
    adjusted, p0 = good_turing_counts(counts)
    def prob(_, word):
        return adjusted.get(word, 0) / total if word in adjusted else p0 / vocab_size
    return prob

def good_turing_bigram(counts, vocab_size):
    adjusted, p0 = good_turing_counts(counts)
    def prob(context, word):
        w1 = context[0] if context else None
        return adjusted.get((w1, word), 0) if (w1, word) in adjusted else p0 / vocab_size
    return prob

# Witten–Bell smoothing
def witten_bell_unigram(counts, total, vocab_size):
    T = len(counts)
    Z = vocab_size - T
    def prob(_, word):
        if word in counts:
            return counts[word] / (total + T)
        else:
            return T / (Z * (total + T)) if Z > 0 and (total + T) > 0 else 1e-10
    return prob

def witten_bell_bigram(counts, unigram_counts, vocab_size):
    context_words = defaultdict(set)
    for (w1, w2) in counts:
        context_words[w1].add(w2)

    def prob(context, word):
        w1 = context[0] if context else None
        c_unigram = unigram_counts.get(w1, 0)
        T = len(context_words.get(w1, []))
        Z = vocab_size - T
        denominator = c_unigram + T
        if denominator == 0 or Z == 0:
            return 1e-10
        c_bigram = counts.get((w1, word), 0)
        if c_bigram > 0:
            return c_bigram / denominator
        else:
            return T / (Z * denominator)
    return prob

# -----------------
# Main
# -----------------
if __name__ == "__main__":
    with open("data/custom_corpus.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokens = preprocess(text)
    print("Total tokens:", len(tokens))
    vocab = set(tokens)
    vocab_size = len(vocab)

    split = int(0.8 * len(tokens))
    train_tokens = tokens[:split]
    test_tokens = tokens[split:]

    uni_counts = get_unigram_counts(train_tokens)
    bi_counts = get_bigram_counts(train_tokens)

    total_unigrams = sum(uni_counts.values())

    results = []

    models = [
        ("Unigram", "No smoothing", no_smoothing_unigram(uni_counts, total_unigrams)),
        ("Unigram", "Add-1", add_k_unigram(uni_counts, total_unigrams, vocab_size, k=1)),
        ("Unigram", "Add-0.5", add_k_unigram(uni_counts, total_unigrams, vocab_size, k=0.5)),
        ("Unigram", "Good–Turing", good_turing_unigram(uni_counts, total_unigrams, vocab_size)),
        ("Unigram", "Witten–Bell", witten_bell_unigram(uni_counts, total_unigrams, vocab_size)),

        ("Bigram", "No smoothing", no_smoothing_bigram(bi_counts, uni_counts)),
        ("Bigram", "Add-1", add_k_bigram(bi_counts, uni_counts, vocab_size, k=1)),
        ("Bigram", "Add-0.5", add_k_bigram(bi_counts, uni_counts, vocab_size, k=0.5)),
        ("Bigram", "Good–Turing", good_turing_bigram(bi_counts, vocab_size)),
        ("Bigram", "Witten–Bell", witten_bell_bigram(bi_counts, uni_counts, vocab_size)),
    ]

    for model_name, method_name, func in models:
        n = 2 if model_name == "Bigram" else 1
        ppl = perplexity(func, test_tokens, n)
        results.append((model_name, method_name, ppl))

    print("\nResults:")
    print("{:<8} | {:<15} | {:<15}".format("Model", "Method", "Perplexity"))
    print("-" * 45)
    for row in results:
        print("{:<8} | {:<15} | {:<15.4f}".format(row[0], row[1], row[2]))
