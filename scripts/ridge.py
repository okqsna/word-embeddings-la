import numpy as np

def learn_ridge(x_ua: np.ndarray, y_eng: np.ndarray, lam: float = 0.1) -> np.ndarray:
    """
    Learn the transformation matrix W using ridge regression.

    This method is a regularized version of least squares:
        W = Y X^T (X X^T + λI)^(-1)

    The additional λI term makes the matrix inversion more stable.

    :param x_ua: source matrix with Ukrainian embeddings, shape (d, n).
    :param y_eng: target matrix with English embeddings, shape (d, n).
    :param lam: regularization coefficient.
    :return: regularized transformation matrix W of shape (d, d).
    """
    d = x_ua.shape[0]
    identity = np.eye(d)
    return y_eng @ x_ua.T @ np.linalg.inv(x_ua @ x_ua.T + lam * identity)
