# -*- coding: utf-8 -*-
import numpy as np
from calculate import example_words

def load_and_normalize_vectors(infile):
    W = []
    words = []
    lines = 0
    with open(infile, 'rb') as fileref:
        for line in fileref:
            lines += 1
    lines = float(lines)
    with open(infile, 'rb') as fileref:
        for count, line in enumerate(fileref):
            line = line.split()
            words.append(line[0])
            new_arr = np.array(list(map(float, line[1:])))
            new_arr = new_arr / np.linalg.norm(new_arr)
            W.append(new_arr)
    return np.array(W), np.array(words)

def print_results(filePath):
    with open(filePath, "w", encoding="utf8") as f:
        for example in example_words:
            f.write("-"*80 + "\n")
            f.write(f"{example.decode('utf8'):>35}\n")
            f.write("-"*80 + "\n")
            for x in _get_neighbours_iter(example):
                f.write(" | ".join(f"{y.decode('utf8'):<14}" for y in x))
                f.write("\n")

def _get_neighbours_iter(example):
        return zip(*(_get_neighbours(W, words, w2i, example, 20) for W, words, w2i in params))

def _get_neighbours(W, words, w2i, token, num_similiar):
    sim = W.dot(W[w2i[token]])
    simIds = sim.argsort()[-1:num_similiar*-1:-1]
    return words[simIds]

if __name__ == '__main__':
    W_d, words_d = load_and_normalize_vectors('deps.words')
    w2i_d = {w: i for i, w in enumerate(words_d)}

    W_b, words_b = load_and_normalize_vectors('bow5.words')
    w2i_b = {w: i for i, w in enumerate(words_b)}

    params = [(W_d, words_d, w2i_d), (W_b, words_b, w2i_b)]

    print_results("word2vec.res")