"""
Other approach to translation with embeddings for the research:
'Cross-Lingual Word Embeddings in the Context 
of Ukrainian-English Translation'
"""
import numpy as np

def learn_orthogonal(x_ua: np.ndarray, y_eng: np.ndarray) -> np.ndarray:
    """
    Function learns the transformation matrix W using the orthogonal Procrustes method.
    This method finds an orthogonal matrix W such that: W X ≈ Y
    with the constraint: W^T W = I
    The solution is obtained from the singular value decomposition (SVD):
        M = Y X^T = U Σ V^T
        W = U V^T

    :param x_ua: numpy.ndarray, source matrix Ukrainian embeddings
    :param y_eng: numpy.ndarraym, target matrix English embeddings
    :return: numpy.ndarray, orthogonal transformation matrix W
    """
    m_sim = y_eng @ x_ua.T
    u, _, vt = np.linalg.svd(m_sim, full_matrices=False)
    return u @ vt
