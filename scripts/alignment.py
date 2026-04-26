"""
Main calculations for the exploration data for the research
'Cross-Lingual word embeddings in the context 
of Ukrainian-English translation'
"""
import numpy as np
import similarity_metrics

def normalize(v):
    """
    Function normalizes the vector.

    :param v: word embedding to be normalized
    :return: normalized vector
    """
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v


def learn_least_squares(x_ua: np.ndarray, y_eng: np.ndarray) -> np.ndarray:
    """
    Function learns the transformation matrix W 
    using the least squares method - by minimizing the squared error
    and solving normal equation. 
    
    :param x_ua: numpy.ndarray, matrix with Ukrainian embeddings 
    :param y_eng: numpy.ndarray, target matrix with English embeddings
    :return: numpy.ndarray, transformation matrix W.
    """
    return y_eng @ x_ua.T @ np.linalg.inv(x_ua @ x_ua.T)


def build_candidate_matrices(english_words: list[str], model_en):
    """
    Function builds the matrix of English candidate word vectors by removing 
    duplicates, sorting, normalizing embedding vectors and creating candidate matrix
    used in cosine similarity.
    
    :param english_words: list[str], list of English words from the dataset.
    :param model_en: loaded English fastText embeddings model.

    :return: english_words - sorted unique English words;
    eng_candidates - matrix of normalized English candidate vectors.
    """
    english_words = sorted(set(english_words))
    eng_candidate = np.vstack([normalize(model_en.get_word_vector(w)) for w in english_words])
    return english_words, eng_candidate


def translate_word(ua_word: str, w_transformation: np.ndarray, model_ua, candidate_words: list[str],
    e_candidates: np.ndarray, top_k: int = 5,
    similarity_metric: str="cosine") -> list[tuple[str, float]]:
    """
    Function produces main prediction of the English translation of one Ukrainian word 
    by getting the Ukrainian word vector, mapping it into the English space
    using transformation matrix W, normalizing the mapped vector, computing cosine similarity scores
    with all English candidate vectors and returning the k best matches.

    :param ua_word: str, Ukrainian word to translate.
    :param w_transformation: numpy.ndarray, computed transformation matrix
    :param model_ua: Ukrainian fastText model.
    :param candidate_words: list[str], English candidate words corresponding to rows of e_candidates.
    :param e_candidates: numpy.ndarray, matrix of normalized English candidate vectors.
    :param top_k: int, default=5, number of best predictions to return.
    :return: list[tuple[str, float]], list of english_word and similarity_score pairs.
    """
    x = normalize(model_ua.get_word_vector(ua_word))
    x_new = normalize(w_transformation @ x)

    if similarity_metric == "cosine":
        scores = similarity_metrics.normalized_cosine_similarity(x_new, e_candidates)
    elif similarity_metric == "euclidean":
        scores = similarity_metrics.euclidean_distance(x_new, e_candidates)
        return [(candidate_words[i], float(scores[i])) for i in np.argsort(scores)[:top_k]]
    elif similarity_metric == "csls":
        scores = similarity_metrics.csls_similarity(x_new, e_candidates)

    return [(candidate_words[i], float(scores[i])) for i in np.argsort(scores)[-top_k:][::-1]]
