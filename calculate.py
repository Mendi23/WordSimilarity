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

SPACE_PATH = "words_space.pkl"

def load() -> Space:
    return io_utils.load(SPACE_PATH)

def calculate_and_save():
    words_space = Space.build(data="skipgram_uniq.data.out",
        rows=WORDS_INDEX_PATH, cols=WORDS_INDEX_PATH, format="sm")

    io_utils.save(words_space, SPACE_PATH)

def tests():
    words_space = load()
    print(words_space.cooccurrence_matrix[:2])
    words_space.apply(PpmiWeighting())
    print(words_space.cooccurrence_matrix[:2])

    print(words_space.get_sim("bus", "car", CosSimilarity()))
    print(words_space.get_sim("dog", "cat", CosSimilarity()))
    print(words_space.get_neighbours("car", 20, CosSimilarity()))



if __name__ == '__main__':
    # calculate_and_save()
    tests()