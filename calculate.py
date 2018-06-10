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

# SENTENCE_SPACE = "sentence.space"
# SKIPGRAM_SPACE = "skipgram.space"
# CONNECTORS_SPACE = "connect.space"

_L = 0
_R = 3
vector_files = [ef.SENTENCE_OUT, ef.SKIPGRAM_OUT, ef.CONNECTORS_OUT][_L:_R]
rows_files = [ef.SENTENCE_ROWS, ef.SKIPGRAM_ROWS, ef.CONNECTORS_ROWS][_L:_R]
cols_files = [ef.SENTENCE_COLS, ef.SKIPGRAM_COLS, ef.CONNECTORS_COLS][_L:_R]
# space_files = [SENTENCE_SPACE, SKIPGRAM_SPACE, CONNECTORS_SPACE][_L:_R]
aux_files = zip(vector_files, rows_files, cols_files) #, space_files)
titles = ["Sentence Context", "Skipgram (k=2)", "Dependency based"]

example_words = [b"car", b"bus", b"hospital", b"hotel", b"gun", b"bomb", b"horse", b"fox",
                 b"table", b"bowl", b"guitar", b"piano"]


def load(outfile_path) -> WordsSpace:
    return WordsSpace.load(outfile_path)


def save(outfilePath, spaceToSave):
    spaceToSave.save(outfilePath)


@measure
def calculate_space(dataPath, rowsPath, colsPath):
    return WordsSpace.build(dataPath, rowsPath, colsPath)


@measure
def print_examples(filePath, neighbours_iter, titles):
    COL_WIDTH = 20
    with open(filePath, "w", encoding="utf8") as f:
        for example in example_words:
            f.write("=" * 80 + "\n")
            f.write(f"{example.decode('utf8'):>35}\n")
            f.write("=" * 80 + "\n")
            f.write(" | ".join(f"{title:^{COL_WIDTH}}" for title in titles))
            f.write("\n" + "-" * COL_WIDTH * len(titles) + "\n")
            for x in neighbours_iter(example):
                f.write(" | ".join(f"{y.decode('utf8'):<{COL_WIDTH}}" for y in x))
                f.write("\n")


def _get_neighbours_iter(word):
    return zip(*(sp.get_neighbours(word, NUM_NEIGHBOURS) for sp in similarities))


#
# @measure
# def tests(outfile_path):
#     words_space = load(outfile_path)
#     words_space.apply_modifier(PMI())
#
#     sim = CosSimilarity(words_space)
#     print(sim.get_sim(b"bus", b"car"))
#     print(sim.get_sim(b"dog", b"cat"))
#     print(sim.get_neighbours(b"dog", 20))
#     print(sim.get_neighbours(b"bus", 20))
#
#     sim = FirstOrderSimilarity(words_space)
#     print(sim.get_neighbours(b"dog", 20))
#     print(sim.get_neighbours(b"bus", 20))


if __name__ == '__main__':
    # for out, rows, cols, space in aux_files:
    #     save(space, calculate_space(out, rows, cols))
    #
    # wordVecs = [load(space) for space in space_files]

    # for space in space_files:
    #     tests(space)

    wordVecs = [calculate_space(out, rows, cols) for out, rows, cols in aux_files]

    for wv in wordVecs:
        wv.apply_modifier(PMI())
    similarities = [CosSimilarity(ws) for ws in wordVecs]
    print_examples("sim_2ndOrder.res", _get_neighbours_iter, titles)

    similarities = [FirstOrderSimilarity(ws) for ws in wordVecs]
    print_examples("sim_1stOrder.res", _get_neighbours_iter, titles)
