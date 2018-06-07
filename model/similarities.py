import abc
import heapq
from functools import lru_cache

from scipy.sparse import csr_matrix
import numpy as np
from scipy.sparse.linalg import norm

from model.hashing import MagicHash
import typing

class Similarity:
    def __init__(self):
        self.c1 = {}
        self.c2 = {}
        self.c3 = {}

    def _getNormRow(self, id, matrix: csr_matrix):
        if id in self.c1:
            return self.c1[id]

        res = self.c1[id] = norm(matrix.getrow(id))
        return res

    def _getDot(self, id1, id2, matrix):
        if (id1, id2) in self.c2:
            return self.c2[(id1, id2)]

        ret = self.c2[(id1, id2)] = matrix.getrow(id1).dot(matrix.getrow(id2).getH())[0, 0]
        return ret

    def _getSum(self, id, matrix):
        if id in self.c3:
            return self.c3[id]

        ret = self.c3[id] = matrix.getrow(id).sum()
        return ret

    def __call__(self, aId, bId,
                 matrix: csr_matrix,
                 hashers: typing.Tuple[MagicHash, MagicHash]):
        raise NotImplementedError()

class CosSimilarity(Similarity):
    def __call__(self, aId, bId,
                 matrix: csr_matrix,
                 hashers: typing.Tuple[MagicHash, MagicHash]):

        return self._getDot(aId, bId, matrix)
        # row1abs = self._getNormRow(aId, matrix)
        # row2abs = self._getNormRow(bId, matrix)

        # return np.divide(dotted, (row1abs*row2abs))

class FirstOrderSimilarity(Similarity):
    def __call__(self, aId, bId,
                 matrix: csr_matrix,
                 hashers: typing.Tuple[MagicHash, MagicHash]):
        aRow = matrix.getrow(aId)
        bWord = hashers[0][bId]
        if bWord in hashers[1]:
            bColId = hashers[1][bWord]
            abVal = aRow[0, bColId]
            return np.divide(abVal, self._getSum(aId, matrix))
        return 0.0


__all__ = ["CosSimilarity", "FirstOrderSimilarity"]