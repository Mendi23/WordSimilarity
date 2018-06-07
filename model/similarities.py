import abc

from scipy.sparse import dok_matrix, csr_matrix
import numpy as np

from model.hashing import MagicHash
import typing

class Similarity:
    def __call__(self, aId, bId,
                 matrix: csr_matrix,
                 hashers: typing.Tuple[MagicHash, MagicHash]):
        raise NotImplementedError()

class Cossim(Similarity):
    def __call__(self, aId, bId,
                 matrix: csr_matrix,
                 hashers: typing.Tuple[MagicHash, MagicHash]):
        aRow = matrix.getrow(aId)
        bRow = matrix.getrow(bId)
        dotted = aRow.dot(bRow.getH())[0, 0]
        row1abs = np.sqrt(aRow.dot(aRow.getH())[0, 0])
        row2abs = np.sqrt(bRow.dot(bRow.getH())[0, 0])

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
            return np.divide(abVal, aRow.sum())
        return 0.0

