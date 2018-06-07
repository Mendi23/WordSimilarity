import heapq
from itertools import product, islice
from scipy.sparse.linalg import norm

from model import modifiers

from model.wordsSpace import WordsSpace


class Similarity:
    def __init__(self, wordSpace: WordsSpace):
        self._ws = wordSpace
        self._sim = {}

    def _n_largest(self, wId, n, length, predicate=None):
        if predicate is None:
            predicate = lambda x: True

        # get all sim values of the current word
        simValues = filter(predicate, ((i, self._sim[wId, i]) for i in range(length)))

        # filter only the n-largest
        largest = heapq.nlargest(n, simValues, key=lambda v: v[1])
        keys = [l[0] for l in largest]

        # if there is not enough - add zero similarities from remain column values
        if len(largest) < n:
            moreVals = islice(((i, 0.0) for i in range(length) if i not in keys),
                              n - len(largest))
            largest += list(moreVals)

        return largest


class CosSimilarity(Similarity):
    def __init__(self, wordSpace: WordsSpace):
        super().__init__(wordSpace)
        self._sim = {}
        self._normalized = modifiers.Normalize(wordSpace._matrix)

    def get_sim(self, a, b):
        aId, bId = self._ws.row2id(a), self._ws.row2id(b)

        self._precalculate(aId, bId)
        return self._sim[aId, bId]

    def get_neighbours(self, word, n):
        wId = self._ws.row2id(word)
        self._precalculate(wId)
        nRows, nCols = self._ws.shape()

        largest = self._n_largest(wId, n, nRows)

        return [self._ws.row2id(i) for i, v in largest]

    def _precalculate(self, aId, bId=None):
        nRows, nCols = self._ws.shape()

        aRow = self._normalized[aId, :]
        irange = range(nRows) if bId is None else [bId]
        for i in irange:
            if (aId, i) not in self._sim:
                bRow = self._normalized[i, :]
                sim = aRow.dot(bRow.getH())[0, 0]
                self._sim[aId, i] = sim
                self._sim[i, aId] = sim


class FirstOrderSimilarity(Similarity):
    def __init__(self, wordSpace: WordsSpace):
        super().__init__(wordSpace)
        self._sim = {}

    def get_neighbours(self, word, n):
        wId = self._ws.row2id(word)
        self._precalculate(wId)
        nRows, nCols = self._ws.shape()

        largest = self._n_largest(wId, n, nCols, lambda pair: pair[1] != 0)

        return [self._ws.col2id(i) for i, v in largest]

    def _precalculate(self, wId):
        nRows, nCols = self._ws.shape()

        aRow = self._ws._matrix[wId, :]
        aRowSum = aRow.sum()
        for i in range(nCols):
            if (wId, i) not in self._sim:
                if aRowSum == 0:
                    sim = 0.0
                else:
                    sim = aRow[0, i] / aRowSum
                self._sim[wId, i] = sim



__all__ = ["CosSimilarity", "FirstOrderSimilarity"]
