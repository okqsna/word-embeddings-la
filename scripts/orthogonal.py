import numpy as np
import alignment

def learn_orthogonal(x_ua: np.ndarray, y_eng: np.ndarray) -> np.ndarray:
    """
    Learn the transformation matrix W using the orthogonal Procrustes method.
    This method finds an orthogonal matrix W such that:
        W X ≈ Y
    with the constraint:
        W^T W = I
    The solution is obtained from the singular value decomposition (SVD):
        M = Y X^T = U Σ V^T
        W = U V^T

    Parameters
    :param x_ua: numpy.ndarray, source matrix Ukrainian embeddings, shape (d, n)
    :param y_eng: numpy.ndarraym, target matrix English embeddings, shape (d, n)
    :return: numpy.ndarray, orthogonal transformation matrix W of shape (d, d)
    """
    M = y_eng @ x_ua.T
    u, _, vt = np.linalg.svd(M, full_matrices=False)
    return u @ vt
