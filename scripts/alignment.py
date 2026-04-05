import numpy as np


def normalize(v):
    """
    Normalize a vector to unit length.

    Parameters
    v : numpy.ndarraynput vector.

    Returns
    -------
    numpy.ndarray, normalized vector with length 1, or the original vector if its norm is 0.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def learn_least_squares(X, Y):
    """
    Learn the transformation matrix W using the least squares method.
    The function solves the problem:
        minimize ||W X - Y||_F^2
    The closed-form solution is:
        W = Y X^T (X X^T)^-1

    Parameters
    X : numpy.ndarray, source matrix (Ukrainian embeddings), shape (d, n).
    Y : numpy.ndarray, target matrix (English embeddings), shape (d, n).

    Returns
    numpy.ndarray, transformation matrix W of shape (d, d).
    """
    return Y @ X.T @ np.linalg.inv(X @ X.T)

def learn_orthogonal(X, Y):
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
    X : numpy.ndarray, Source matrix (Ukrainian embeddings), shape (d, n).
    Y : numpy.ndarraym Target matrix (English embeddings), shape (d, n).

    Returns
    numpy.ndarray, orthogonal transformation matrix W of shape (d, d).
    """
    M = Y @ X.T
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def build_candidate_matrices(english_words, model_en):
    """
    Build the matrix of English candidate word vectors.
    1. removes duplicate words,
    2. sorts them,
    3. gets and normalizes their fastText vectors,
    4. stacks them into one matrix E.

    Parameters
    english_words : list[str], list of English candidate words.
    model_en : fasttext.FastText._FastText, loaded English fastText model.

    Returns
    english_words : list[str], sorted unique English words.
    E : numpy.ndarray, matrix of normalized English candidate vectors.

    """
    english_words = sorted(set(english_words))
    E = np.vstack([normalize(model_en.get_word_vector(w)) for w in english_words])
    return english_words, E


def translate_word(ua_word, W, model_uk, candidate_words, E, top_k=5):
    """
    Predict the English translation of one Ukrainian word.
    1. gets the Ukrainian word vector,
    2. maps it into the English space using W,
    3. normalizes the mapped vector,
    4. computes similarity scores with all English candidate vectors,
    5. returns the top-k best matches.

    Parameters
    ua_word : str, Ukrainian word to translate.
    W : numpy.ndarray, transformation matrix.
    model_uk : fasttext.FastText._FastText Ukrainian fastText model.
    candidate_words : list[str], English candidate words corresponding to rows of E.
    E : numpy.ndarray, matrix of normalized English candidate vectors.
    top_k : int, default=5, number of best predictions to return.

    Returns
    list[tuple[str, float]], list of (english_word, similarity_score) pairs.
    """
    x = normalize(model_uk.get_word_vector(ua_word))
    y_hat = normalize(W @ x)
    scores = E @ y_hat
    best = np.argsort(scores)[-top_k:][::-1]
    return [(candidate_words[i], float(scores[i])) for i in best]

def evaluate(correct_pairs, W, model_uk, candidate_words, E, top_k=1):
    """
    Evaluate translation quality on a test set.

    Parameters
    correct_pairs : list[tuple[str, str]], list of correct translated pairs.
    W : numpy.ndarray, transformation matrix.
    model_uk : fasttext.FastText._FastText, loaded Ukrainian fastText model.
    candidate_words : list[str], english candidate words.
    E : numpy.ndarray, matrix of normalized English candidate vectors.
    top_k : int, default=1, number of best predictions.

    Returns
    accuracy : float, correctness of the translated test words.
    results : list[dict] results for each test example.
    """
    correct = 0
    results = []
    for ua, true_en in correct_pairs:
        prediction = translate_word(ua, W, model_uk, candidate_words, E, top_k)
        predicted_words = [w for w, _ in prediction]
        hit = true_en in predicted_words
        if hit:
            correct += 1
        results.append({
            "ukrainian": ua,
            "true_english": true_en,
            "predictions": predicted_words,
            "correct": hit
        })

        accuracy = correct / len(correct_pairs) if correct_pairs else 0.0
        return accuracy, results
