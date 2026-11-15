# Step 2: find passages, keep only the best few, best one FIRST.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def top_passages(question: str, passages: list[str], k: int = 3, min_score: float = 0.05):
    if not passages:
        return []
    vec = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform(passages + [question])
    scores = cosine_similarity(matrix[-1], matrix[:-1])[0]
    ranked = sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)
    # drop weak passages, keep best first
    return [p for s, p in ranked[:k] if s >= min_score]
