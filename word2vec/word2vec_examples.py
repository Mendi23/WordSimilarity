# -*- coding: utf-8 -*-
import numpy as np
from calculate import print_examples


def load_and_normalize_vectors(infile):
    W = []
    words = []
    with open(infile, 'rb') as fileref:
        for count, line in enumerate(fileref):
            line = line.split()
            words.append(line[0])
            new_arr = np.array(list(map(float, line[1:])))
            new_arr = new_arr / np.linalg.norm(new_arr)
            W.append(new_arr)
    return np.array(W), np.array(words)


def _get_neighbours_iter(example):
    return zip(
        *(_get_neighbours(W, words, w2i, example, NUM_NEIGHBOURS) for W, words, w2i in params))


def _get_neighbours(W, words, w2i, token, num_similiar):
    sim = W.dot(W[w2i[token]])
    simIds = sim.argsort()[-1:num_similiar * -1:-1]
    return words[simIds]


if __name__ == '__main__':
    W_d, words_d = load_and_normalize_vectors('deps.words')
    w2i_d = {w: i for i, w in enumerate(words_d)}

    W_b, words_b = load_and_normalize_vectors('bow5.words')
    w2i_b = {w: i for i, w in enumerate(words_b)}

    params = [(W_d, words_d, w2i_d), (W_b, words_b, w2i_b)]
    NUM_NEIGHBOURS = 20
    print_examples("word2vec_words.res", _get_neighbours_iter, ["dependency-based", "bag-of-words ["
                                                                                    "k=5]"])

    C_d, contexts_d = load_and_normalize_vectors('deps.contexts')
    C_b, contexts_b = load_and_normalize_vectors('bow5.contexts')

    params = [(C_d, contexts_d, w2i_d), (C_b, contexts_b, w2i_b)]
    NUM_NEIGHBOURS = 10
    print_examples("word2vec_contexts.res", _get_neighbours_iter, ["dependency-based",
                                                                   "bag-of-words [k=5]"])
