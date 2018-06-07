"""
you need to download the package and install it yourself.
follow the instructions from this link:
http://clic.cimec.unitn.it/composes/toolkit/installation.html
after you downloaded the git folder and before instalation,
you need to run 2to3 script in the main folder with the flag "w" so the code will be compatable
"""

import extruct_features as ef
from helpers.measuretime import measure
from model.modifiers import *
from model.similarities import *
from model.wordsSpace import WordsSpace

NUM_NEIGHBOURS = 20

SENTENCE_SPACE = "sentence.space"
SKIPGRAM_SPACE = "skipgram.space"
CONNECTORS_SPACE = "connect.space"

_L = 0
_R = 3
vector_files = [ef.SENTENCE_OUT, ef.SKIPGRAM_OUT, ef.CONNECTORS_OUT][_L:_R]
rows_files = [ef.SENTENCE_ROWS, ef.SKIPGRAM_ROWS, ef.CONNECTORS_ROWS][_L:_R]
cols_files = [ef.SENTENCE_COLS, ef.SKIPGRAM_COLS, ef.CONNECTORS_COLS][_L:_R]
space_files = [SENTENCE_SPACE, SKIPGRAM_SPACE, CONNECTORS_SPACE][_L:_R]
titles = ["Sentence Context", "Skipgram (k=2)", "Dependency based"]
aux_files = zip(vector_files, rows_files, cols_files, space_files)

example_words = [b"car", b"bus", b"hospital", b"hotel", b"gun", b"bomb", b"horse", b"fox",
                 b"table", b"bowl", b"guitar", b"piano"]

def load(outfile_path) -> WordsSpace:
    return WordsSpace.load(outfile_path)

@measure
def calculate_and_save(dataPath, rowsPath, colsPath, outfilePath):
    words_space = WordsSpace.build(dataPath, rowsPath, colsPath)

    words_space.save(outfilePath)
    return words_space

@measure
def print_examples(filePath, neighbours_iter, titles):
    COL_WIDTH = 20
    with open(filePath, "w", encoding="utf8") as f:
        for example in example_words:
            f.write("="*80 + "\n")
            f.write(f"{example.decode('utf8'):>35}\n")
            f.write("="*80 + "\n")
            f.write(" | ".join(f"{title:^{COL_WIDTH}}" for title in titles))
            f.write("-"*COL_WIDTH*len(titles) + "\n")
            for x in neighbours_iter(example):
                f.write(" | ".join(f"{y.decode('utf8'):<{COL_WIDTH}}" for y in x))
                f.write("\n")


def _get_neighbours_iter(word):
    return zip(*(sp.get_neighbours(word, NUM_NEIGHBOURS, CosSimilarity()) for sp in wordVecs))

@measure
def tests(outfile_path):
    words_space = load(outfile_path)
    # words_space.apply_modifier(PMI_log())

    words_space.apply_similarity(CosSimilarity())
    print(words_space.get_sim(b"bus", b"car"))
    print(words_space.get_sim(b"dog", b"cat"))
    print(words_space.get_neighbours(b"dog", 20))
    print(words_space.get_neighbours(b"bus", 20))

    words_space.apply_similarity(FirstOrderSimilarity())
    print(words_space.get_neighbours(b"dog", 20))
    print(words_space.get_neighbours(b"bus", 20))


if __name__ == '__main__':
    # for out, rows, cols, space in aux_files:
    #     calculate_and_save(out, rows, cols, space)


    for space in space_files:
        tests(space)

    # wordVecs = [load(sf) for sf in space_files]
    # for wv in wordVecs:
    #     wv.apply_modifier(PMI)
    # print_examples("sim_results.res", _get_neighbours_iter, titles)








