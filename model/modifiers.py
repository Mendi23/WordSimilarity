from scipy.sparse import csr_matrix, dok_matrix
from scipy.sparse.linalg import norm
from sklearn.preprocessing.data import normalize
import numpy as np

from helpers.measuretime import measure

SMOOTH_POWER = 0.75


def _safe_divide(num, nparr):
    unsafe_indexes = nparr == 0
    nparr[unsafe_indexes] = 1
    nparr = num / nparr
    nparr[unsafe_indexes] = 0
    return nparr


class PMI:
    """
        PMI = log( p(c,w) / p(c)*p(w) )

            p(c,w) = #(c,w) / SUM
            p(c) = #(c,_) / SUM
            p(w) = #(_,w) / SUM

        Hence, PMI = log( #(c,w) * SUM ) / ( #(c,_) * #(_,w) )
    """

    def __init__(self, smooth=False):
        self.smooth = smooth

    def __call__(self, matrix: csr_matrix):
        SUM = matrix.sum()
        row_sum = matrix.sum(axis=1)
        col_sum = matrix.sum(axis=0)

        # do: 1 / each cell
        row_sum = _safe_divide(1, row_sum)
        col_sum = _safe_divide(1, col_sum)

        row_sum *= SUM

        if self.smooth:
            row_sum = np.power(row_sum, SMOOTH_POWER)

        res = matrix.multiply(row_sum).multiply(col_sum).tocsr()

        res.data = np.log(res.data)
        res.eliminate_zeros()

        return res

class PPMI(PMI):
    def __call__(self, matrix: csr_matrix):
        res = super().__call__(matrix)
        res[res < 0] = 0.0
        return res

def Normalize(matrix: csr_matrix):
    return normalize(matrix, axis=1)

__all__ = ["PMI", "PPMI"]

if __name__ == '__main__':
    dok = dok_matrix((5, 5))
    for i in range(5):
        dok[i, :] = np.arange(i * 5, (i + 1) * 5)

    dok2 = dok_matrix((5,5))
    dok2[1,2] = 4
    dok2[1,3] = 3