import pickle
from collections import Counter, defaultdict
from itertools import chain

import numpy as np
from scipy.sparse import dok_matrix

from model import hashing


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
        self._matrix = dok_matrix(shape, dtype=np.float64)
        for word, contextWords in counter.items():
            for context, val in contextWords.items():
                i, j = (word, context) if counterHashed else (rh[word], ch[context])
                self._matrix[i, j] = val
        self._matrix = self._matrix.tocsr()

        rh.freeze()
        ch.freeze()
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

    def apply_modifier(self, modifier):
        self._matrix = modifier(self._matrix)

    def row2id(self, w):
        return self._hashers[0][w]

    def col2id(self, w):
        return self._hashers[1][w]

    def shape(self):
        return self._matrix.shape

    @staticmethod
    def log(message):
        pass
        #print("|> WordsSpace: " + message)
