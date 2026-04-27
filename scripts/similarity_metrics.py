"""
Similarity metrics calculations for the research:
'Cross-Lingual Word Embeddings in the Context 
of Ukrainian-English Translation'
"""
import numpy as np

def normalized_cosine_similarity(x_new: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """
    Function computes cosine similarity for normalized vectors.

    :param x_new: np.ndarray, possible word translation
    :param candidates: np.ndarray, words to compare
    :return: cosine similarity value between -1 and 1
    """
    return candidates @ x_new

def csls_similarity(x_new: np.ndarray, candidates: np.ndarray, r_y: np.ndarray, k: int=5):
    """
    Function computes Cross-Domain Similarity Local Scaling(CSLS) using Cosine Similarity
    between a vector and candidate vectors.

    :param x_new: np.ndarray, possible word translation
    :param candidates: np.ndarray, words to compare
    :param r_y:np.ndarray, precomputed neighborhood densities for candidates
    :param k: int, number of nearest neighbours to consider
    :return: CSLS value
    """
    cos_similarity = normalized_cosine_similarity(x_new, candidates)

    nearest_possible_neighbours = np.partition(cos_similarity, -k)[-k:]
    r_x = np.mean(nearest_possible_neighbours)

    return 2 * cos_similarity - r_x - r_y


def candidates_neighbours_evaluation_csls(candidates: np.ndarray, k: int=5):
    """
    Function computes the average similarity of each candidate embedding
    to its k nearest neighbours, which is in CSLS similarity metric.

    :param candidates: np.ndarray, words to compare
    :param k: int, number of nearest neighbours to consider
    """
    similarity_matrix = candidates @ candidates.T
    np.fill_diagonal(similarity_matrix, -np.inf)

    candidates_neighbours = np.partition(similarity_matrix, -k, axis=1)[:, -k:]
    return np.mean(candidates_neighbours, axis=1)
