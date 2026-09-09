"""
Engine agnostic face matching: compares a live embedding against a
set of known embeddings using cosine similarity and a configurable
threshold. Works identically regardless of which adapter produced
the embeddings.
"""

import numpy as np

# Single place to tune how strict recognition is.
# Cosine similarity ranges from -1 to 1; ArcFace embeddings typically
# need a fairly high threshold to avoid false matches.
SIMILARITY_THRESHOLD = 0.45


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(np.dot(a_norm, b_norm))


def match_embedding(live_embedding: np.ndarray, known_embeddings: dict):
    """
    known_embeddings: {student_id: embedding_vector}
    Returns (student_id, similarity) for the best match above threshold,
    or (None, best_similarity_seen) if nothing clears the threshold.
    """
    best_id = None
    best_score = -1.0

    for student_id, known_embedding in known_embeddings.items():
        score = cosine_similarity(live_embedding, known_embedding)
        if score > best_score:
            best_score = score
            best_id = student_id

    if best_score >= SIMILARITY_THRESHOLD:
        return best_id, best_score
    return None, best_score
