import abc
import heapq
from functools import lru_cache

from scipy.sparse import dok_matrix, csr_matrix
import numpy as np

from model.hashing import MagicHash
import typing


def _getAbsRow(id, matrix: csr_matrix):
    if id in _getAbsRow.cache:
        return _getAbsRow.cache[id]

    row = matrix.getrow(id)
    s = row.power(2, dtype=np.int64).sum()
    ret = _getAbsRow.cache[id] = np.sqrt(s)
    return ret
_getAbsRow.cache = {}

def _getDot(id1, id2, matrix):
    if (id1, id2) in _getDot.cahe:
        return _getDot.cahe[(id1,id2)]

    row1 = matrix.getrow(id1)
    row2 = matrix.getrow(id2)
    ret = _getDot.cahe[(id1, id2)] = row1.dot(row2.getH())[0,0]
    return ret
_getDot.cahe = {}

def _getSum(id, matrix):
    if id in _getDot.cahe:
        return _getDot.cahe[id]

    row = matrix.getrow(id)
    ret = _getDot.cahe[id] = row.sum()
    return ret
_getSum.cahe = {}

class Similarity:
    def __call__(self, aId, bId,
                 matrix: csr_matrix,
                 hashers: typing.Tuple[MagicHash, MagicHash]):
        raise NotImplementedError()

class Cossim(Similarity):
    def __call__(self, aId, bId,
                 matrix: csr_matrix,
                 hashers: typing.Tuple[MagicHash, MagicHash]):
        dotted = _getDot(aId, bId, matrix)
        row1abs = _getAbsRow(aId, matrix)
        row2abs = _getAbsRow(bId, matrix)

        return np.divide(dotted, (row1abs*row2abs))

class FirstOrder(Similarity):
    def __call__(self, aId, bId,
                 matrix: csr_matrix,
                 hashers: typing.Tuple[MagicHash, MagicHash]):
        # Maybe we just can return `np.divide(abVal, aRow.sum())`
        # but I don't know if this is consistent enough
        aRow = matrix.getrow(aId)
        bWord = hashers[0][bId]
        if bWord in hashers[1]:
            bColId = hashers[1][bWord]
            abVal = aRow[0, bColId]
            return np.divide(abVal, _getSum(aId, matrix))
        return 0.0


__all__ = ["Cossim", "FirstOrder"]