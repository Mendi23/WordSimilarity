"""
you need to download the package and install it yourself.
follow the instructions from this link:
http://clic.cimec.unitn.it/composes/toolkit/installation.html
after you downloaded the git folder and before instalation,
you need to run 2to3 script in the main folder with the flag "w" so the code will be compatable
"""

from composes.semantic_space.space import Space
from composes.similarity.cos import CosSimilarity
from composes.similarity.lin import LinSimilarity
from composes.utils import io_utils
from composes.transformation.scaling.ppmi_weighting import PpmiWeighting

from hashing import WORDS_INDEX_PATH
import extruct_features as ef
from helpers.measuretime import measure

NUM_NEIGHBOURS = 20

SENTENCE_SPACE = "sentence.space"
SKIPGRAM_SPACE = "skipgram.space"
CONNECTORS_SPACE = "connect.space"

vector_files = [ef.SENTENCE_OUT, ef.SKIPGRAM_OUT, ef.CONNECTORS_OUT]
words_files = [ef.SENTENCE_VOC, ef.SKIPGRAM_VOC, ef.CONNECTORS_VOC]
space_files = [SENTENCE_SPACE, SKIPGRAM_SPACE, CONNECTORS_SPACE]
aux_files = zip(vector_files, words_files, space_files)

example_words = [b"car", b"bus", b"hospital", b"hotel", b"gun", b"bomb", b"horse", b"fox",
                 b"table", b"bowl", b"guitar", b"piano"]

def load(outfile_path) -> Space:
    return io_utils.load(outfile_path)

@measure
def calculate_and_save(data_file_path, words_path, outfile_path):
    words_space = Space.build(data=data_file_path,
        rows=words_path, cols=words_path, format="sm")

    io_utils.save(words_space, outfile_path)
    return words_space

def print_examples(filePath):
    wordVecs = [load(sf) for sf in space_files]
    with open(filePath, "w", encoding="etf8") as f:
        for example in example_words:
            for x in _get_neighbours_iter(example, wordVecs):
                f.write(" | ".join(x))
                f.write("\n")


def _get_neighbours_iter(word, wordSpaces):
    return zip(sp.get_neighbours(word, NUM_NEIGHBOURS, CosSimilarity()) for sp in wordSpaces)


if __name__ == '__main__':
    for out, voc, space in aux_files:
        calculate_and_save(out, voc, space)
    print_examples("sim_results.res")







# @measure
# def tests(outfile_path):
#     words_space = load(outfile_path)
#     # print(words_space.cooccurrence_matrix[:2])
#     words_space.apply(PpmiWeighting())
#     # print(words_space.cooccurrence_matrix[:2])
#
#     print(words_space.get_sim(b"bus", b"car", CosSimilarity()))
#     print(words_space.get_sim(b"dog", b"cat", CosSimilarity()))
#     print(words_space.get_neighbours(b"car", 20, CosSimilarity()))
#     print(words_space.get_neighbours(b"cat", 20, CosSimilarity()))
#     print(words_space.get_neighbours(b"dog", 20, CosSimilarity()))
