import abc
import heapq
from collections import defaultdict
from functools import lru_cache
from itertools import product, islice

from scipy.sparse import csr_matrix, dok_matrix
from scipy.sparse.linalg import norm

from model import modifiers
from model.hashing import MagicHash
import typing


class Similarity:
    def __init__(self):
        self._sim = None

    def get_sim(self, aId, bId):
        return self._sim[aId, bId]

    def get_neighbours(self, n, id):
        numOfNeighbours = self._sim.shape[1]

        # get all sim values of the current word
        simValues = ((i, self._sim[id, i]) for i in range(numOfNeighbours))

        # filter only the n-largest
        largest = heapq.nlargest(n, (pair for pair in simValues if pair[1] != 0), key=lambda v: v[1])
        keys = [l[0] for l in largest]

        # if there is not enough - add zero similarities from remain column values
        if len(largest) < n:
            moreVals = islice(((i, 0.0) for i in range(numOfNeighbours) if i not in keys),
                              n - len(largest))
            largest += list(moreVals)

        return largest

    def precalculate(self, matrix):
        raise NotImplementedError()


class CosSimilarity(Similarity):
    def precalculate(self, matrix):
        nRows = matrix.shape[0]
        self._sim = defaultdict(int)

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

    def precalculate(self, matrix: csr_matrix):
        nRows, nCols = matrix.shape
        self._sim = defaultdict(int)

        for i in range(nRows):
            aRow = matrix[i, :]
            aRowSum = aRow.sum()
            if aRowSum == 0:
                continue
            for j in range(nCols):
                sim = aRow[0, j] / aRowSum
                self._sim[i, j] = sim


__all__ = ["CosSimilarity", "FirstOrderSimilarity"]
