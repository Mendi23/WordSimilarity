import heapq
import pickle
from collections import Counter, defaultdict
from itertools import chain, islice

import numpy as np
from scipy.sparse import dok_matrix

from model import hashing
from model.similarities import Similarity


class WordsSpace(object):
    def __init__(self,
                 counter=None, shape=None, hashers=None, counterHashed=False,
                 matrix=None):

        # Shape related stuff
        if shape is None:
            self.log("[init] Calculating shape...")
            nRows = len(counter)
            nCols = len(set(chain.from_iterable(contextWords.keys() for contextWords in counter.values())))
            shape = (nRows, nCols)
        nRows, nCols = shape

        # hashers - id2word/word2id
        if hashers is None:
            self.log("[init] creating magicHash")
            hashers = (hashing.MagicHash(), hashing.MagicHash())
        self._hashers = hashers

        # matrix!
        self.log(f"[init] creating matrix (shape: {nRows} x {nCols})")

        if matrix is not None:
            self._matrix = matrix
            return

        rh, ch = self._hashers
        self._matrix = dok_matrix(shape, dtype=np.uint64)
        for word, contextWords in counter.items():
            for context, val in contextWords.items():
                i, j = (word, context) if counterHashed else (rh[word], ch[context])
                self._matrix[i, j] = val
        self._matrix = self._matrix.tocsr()
        self.log("Finish")

    @classmethod
    def build(cls, dataPath, rowsPath=None, colsPath=None):
        nRows, nCols = None, None
        if rowsPath is not None:
            cls.log("[build] Calculating nRows...")
            nRows = len(open(rowsPath, "rb").readlines())
        if colsPath is not None:
            cls.log("[build] Calculating nCols...")
            nCols = len(open(colsPath, "rb").readlines())

        rowHash = hashing.MagicHash()
        colHash = hashing.MagicHash()
        counter = defaultdict(Counter)
        cls.log(f"[build] Loading data from file: {dataPath}")
        with open(dataPath, "rb") as inp:
            for line in inp:
                dataLine = line.strip().split()
                assert len(dataLine) >= 3
                word, context, val = dataLine[0], dataLine[1], int(dataLine[2])
                counter[rowHash[word]][colHash[context]] = val

        return cls(counter, (nRows, nCols), hashers=(rowHash, colHash), counterHashed=True)

    def save(self, filepath):
        with open(filepath, "wb") as fout:
            pickle.dump(self, fout)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as fin:
            return pickle.load(fin)

    def get_neighbours(self, word, n, similarity: Similarity):
        nRows, nCols = self._matrix.shape
        hh = self._hashers[0]

        wordId = hh[word]

        # get all sim values of the current word
        simValues = ((i, similarity(wordId, i, self._matrix, self._hashers))
                     for i in range(nRows))

        # filter only the n-largest
        largest = heapq.nlargest(n, simValues, key=lambda v: v[1])
        keys = [l[0] for l in largest]

        # if there is not enough - add zero similarities from remain column values
        if len(largest) < n:
            moreVals = islice(((i, 0.0) for i in range(nRows) if i not in keys),
                              n - len(largest))
            largest += list(moreVals)

        return [hh[i] for i, v in largest]

    def get_sim(self, a, b, similarity: Similarity):
        hh = self._hashers[0]

        return similarity(hh[a], hh[b], self._matrix, self._hashers)

    def apply_modifier(self, modifier):
        self._matrix = modifier(self._matrix)

    @staticmethod
    def log(message):
        print("|> WordsSpace: " + message)
