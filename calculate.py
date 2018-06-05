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

@measure
def calculate_and_save(data_file_path, words_path, outfile_path):
    words_space = Space.build(data=data_file_path,
                              rows=words_path, cols=words_path, format="sm")

    io_utils.save(words_space, outfile_path)
    return words_space

@measure
def tests(outfile_path):
    words_space = io_utils.load(outfile_path)
    print(words_space.cooccurrence_matrix[:2])
    words_space.apply(PpmiWeighting())
    print(words_space.cooccurrence_matrix[:2])

    print(words_space.get_sim(b"bus", b"car", CosSimilarity()))
    print(words_space.get_sim(b"dog", b"cat", CosSimilarity()))
    print(words_space.get_neighbours(b"car", 20, CosSimilarity()))
    print(words_space.get_neighbours(b"cat", 20, CosSimilarity()))
    print(words_space.get_neighbours(b"dog", 20, CosSimilarity()))



if __name__ == '__main__':
    #calculate_and_save(ef.SENTENCE_OUT, ef.SENTENCE_VOC, "sentence.space")
    tests("words_space.pkl")