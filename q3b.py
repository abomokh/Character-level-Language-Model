import pandas as pd
import os
from q1c_neural import forward
from q1d_neural_lm import load_vocab_embeddings, load_data_as_sentences, convert_to_lm_dataset
from sgd import load_saved_params
from data_utils import utils
import numpy as np
import matplotlib
matplotlib.use('agg')
from importnb import Notebook
import torch
import math 

def perplexity_bigram(passeges):
    perplexities = []
    for passage in passeges:
        # Tokenize the passage and convert to indices
        sentences, S_data = load_data_as_sentences(passage, word_to_num)
        in_word_index, out_word_index = convert_to_lm_dataset(S_data)

        # Evaluate perplexity
        total_log_prob = 0
        num_of_examples = len(in_word_index)

        for in_index, out_index in zip(in_word_index, out_word_index):
            data = num_to_word_embedding[in_index]
            curr_label = out_word_index[in_index]
            probs = forward(data, curr_label, params, dimensions)
            total_log_prob += np.log2(probs)

        perplexity = np.power(2, -(total_log_prob / num_of_examples))
        perplexities.append(perplexity)
    return perplexities


if __name__ == "__main__":
    passeges = ["shakespeare_for_perplexity.txt", "wikipedia_for_perplexity.txt"]
    

    # Define paths
    vocab_file = "data/lm/vocab.ptb.txt"
    embeddings_file = "data/lm/vocab.embeddings.glove.txt"
    vocab_size = 2000  # Adjust based on your training setup

    # Load vocabulary and embeddings
    vocab = pd.read_table(vocab_file, header=None, sep="\s+", index_col=0, names=['count', 'freq'])
    num_to_word = dict(enumerate(vocab.index[:vocab_size]))
    word_to_num = utils.invert_dict(num_to_word)
    num_to_word_embedding = load_vocab_embeddings(embeddings_file)

    # Load pre-trained parameters
    _, params, _ = load_saved_params()

    # Define dimensions
    input_dim = 50
    hidden_dim = 50
    output_dim = vocab_size
    dimensions = [input_dim, hidden_dim, output_dim]

    perplexities = perplexity_bigram(passeges)

    print(f"Perplexity of bigram model for shakespreare passage: {perplexities[0]}")
    print(f"Perplexity of bigram model for wikipedia passage: {perplexities[1]}")




