import abc
import heapq
from functools import lru_cache
from itertools import product

from scipy.sparse import csr_matrix, dok_matrix
import numpy as np
from scipy.sparse.linalg import norm

from model import modifiers
from model.hashing import MagicHash
import typing


class Similarity:
    # def __init__(self):
    #     self.c1 = {}
    #     self.c2 = {}
    #     self.c3 = {}
    #
    # def _getNormRow(self, id, matrix: csr_matrix):
    #     if id in self.c1:
    #         return self.c1[id]
    #
    #     res = self.c1[id] = modifiers.Normalize(matrix[id, :])
    #     return res
    #
    # def _getDot(self, id1, id2, matrix):
    #     if (id1, id2) in self.c2:
    #         return self.c2[(id1, id2)]
    #
    #     ret = self.c2[(id1, id2)] = matrix[id1, :].dot(matrix[id2, :].getH())[0, 0]
    #     return ret
    #
    # def _getSum(self, id, matrix):
    #     if id in self.c3:
    #         return self.c3[id]
    #
    #     ret = self.c3[id] = matrix[id, :].sum()
    #     return ret

    def __call__(self, aId, bId,
                 matrix: csr_matrix,
                 hashers: typing.Tuple[MagicHash, MagicHash]):
        raise NotImplementedError()


class CosSimilarity(Similarity):
    def __init__(self):
        super().__init__()
        self._sim = None

    def __call__(self, aId, bId,
                 matrix: csr_matrix,
                 hashers: typing.Tuple[MagicHash, MagicHash]):

        if self._sim is None:
            self._precalculate(matrix)

        return self._sim[aId, bId]

    def _precalculate(self, matrix):
        nRows = matrix.shape[0]
        self._sim = dok_matrix((nRows, nRows), dtype=matrix.dtype)

        normalized_mat = modifiers.Normalize(matrix)

        for i in range(nRows):
            aRow = normalized_mat[i, :]
            for j in range(i, nRows):
                bRow = normalized_mat[j, :]
                sim = aRow.dot(bRow.getH())[0, 0]
                if sim > 0:
                    self._sim[i, j] = sim
                    self._sim[j, i] = sim
                else:
                    print(sim)
                    input()


class FirstOrderSimilarity(Similarity):
    def __init__(self):
        super().__init__()
        self._sim = None

    def __call__(self, aId, bId,
                 matrix: csr_matrix,
                 hashers: typing.Tuple[MagicHash, MagicHash]):

        if self._sim is None:
            self._precalculate(matrix, hashers[0], hashers[1])

        return self._sim[aId, bId]

    def _precalculate(self, matrix: csr_matrix, rowH, colH):
        nRows = matrix.shape[0]
        self._sim = dok_matrix((nRows, nRows), dtype=matrix.dtype)

        for i in range(nRows):
            aRow = matrix[i, :]
            aRowSum = aRow.sum()
            if aRowSum == 0:
                continue
            for j in range(i, nRows):
                bWord = rowH[j]
                bColId = colH.data.get(bWord)
                if bColId:
                    sim = aRow[0, bColId] / aRowSum
                    self._sim[i, j] = sim
                    self._sim[j, i] = sim
                else:
                    print(f"no col for {bWord} (#{j})")
                    input()


__all__ = ["CosSimilarity", "FirstOrderSimilarity"]
