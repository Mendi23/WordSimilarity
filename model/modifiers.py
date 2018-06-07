from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
from sklearn.preprocessing.data import normalize
import numpy as np

def _safe_divide(num, nparr):
    zero_indexes = nparr == 0
    nparr[zero_indexes] = 1
    nparr = num / nparr
    nparr[zero_indexes] = 0

def _safe_log(sparse):
    sparse.data[sparse.data <= 1] = 1
    sparse.data = np.log(sparse.data)
    sparse.eliminate_zeros()


def PMI(matrix: csr_matrix):
    SUM = matrix.sum()
    row_sum = matrix.sum(axis=1)
    col_sum = matrix.sum(axis=0)

    _safe_divide(1, row_sum)
    _safe_divide(1, col_sum)

    res = matrix.multiply(SUM).multiply(row_sum).multiply(col_sum)

    _safe_log(res)

    return res


def Norm(matrix: csr_matrix):
    return normalize(matrix, axis=1)
