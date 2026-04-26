"""
Similarity metrics calculations for the research:
'Cross-Lingual Word Embeddings in the Context 
of Ukrainian-English Translation'
"""
import numpy as np

def normalized_cosine_similarity(x_new: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """
    Function computes cosine similarity for normalized vectors.

    :param: x_new, np.ndarray, possible word translation
    :param: candidates, np.ndarray, words to compare
    :return: cosine similarity value between -1 and 1
    """
    return candidates @ x_new

def euclidean_distance(x_new: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """
    Function computes Euclidean distance between a vector and candidate vectors.

    :param: x_new, np.ndarray, possible word translation
    :param: candidates, np.ndarray, words to compare
    :return: Euclidean distance value
    """
    return np.linalg.norm(candidates - x_new, axis=1)

def csls_similarity(x_new: np.ndarray, candidates: np.ndarray, k: int=10):
    """
    Function computes Cross-Domain Similarity Local Scaling(CSLS) using Cosine Similarity
    between a vector and candidate vectors.

    :param: x_new, np.ndarray, possible word translation
    :param: candidates, np.ndarray, words to compare
    :param: k, int, number of nearest neighbours to consider
    :return: CSLS value
    """
    cos_similarity = normalized_cosine_similarity(x_new, candidates)

    nearest_possible_neighbours = np.partition(cos_similarity, -k)[-k:]
    r_x = np.mean(nearest_possible_neighbours)

    similarity_matrix = candidates @ candidates.T
    np.fill_diagonal(similarity_matrix, -np.inf)

    candidates_neighbours = np.partition(similarity_matrix, -k, axis=1)[:, -k:]
    r_y = np.mean(candidates_neighbours, axis=1)

    return 2 * cos_similarity - r_x - r_y
