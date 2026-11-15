# Part 4: ask the same question n times. Same meaning = model is sure.
from itertools import combinations
from .llm import ask

def _prompt(question, passages):
    context = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    return (
        "Answer the question using ONLY the passages. Keep it short.\n\n"
        f"Passages:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

def _tfidf_similarity(answers):
    # Simple version: word overlap. Fast, no download.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    try:
        m = TfidfVectorizer().fit_transform(answers)
    except ValueError:  # empty words
        return 1.0 if len(set(answers)) == 1 else 0.0
    sims = cosine_similarity(m)
    pairs = list(combinations(range(len(answers)), 2))
    return sum(sims[i][j] for i, j in pairs) / len(pairs)

def _nli_similarity(answers):
    # Better version: two answers match if each one implies the other.
    # Uses the same DeBERTa model as verify.py (already downloaded).
    from .verify import _get_nli
    nli = _get_nli()
    def entails(a, b):
        out = nli({"text": a, "text_pair": b})
        out = out[0] if isinstance(out[0], list) else out
        probs = {o["label"].lower(): o["score"] for o in out}
        return probs.get("entailment", 0) >= 0.5
    pairs = list(combinations(range(len(answers)), 2))
    same = sum(1 for i, j in pairs
               if entails(answers[i], answers[j]) and entails(answers[j], answers[i]))
    return same / len(pairs)

def consistency_score(question: str, passages: list[str], n: int = 5,
                      temperature: float = 0.7, method: str = "tfidf") -> float:
    prompt = _prompt(question, passages)
    answers = [ask(prompt, temperature=temperature).strip() for _ in range(n)]
    answers = [a if a else "(empty)" for a in answers]
    if len(answers) < 2:
        return 1.0
    score = _nli_similarity(answers) if method == "nli" else _tfidf_similarity(answers)
    return max(0.0, min(1.0, float(score)))

if __name__ == "__main__":
    ask = lambda p, temperature=0.0: "Paris is the capital of France."
    print(consistency_score("Capital of France?", ["x"]))   # about 1.0
    import itertools
    outs = itertools.cycle(["Paris", "Berlin", "Madrid", "Rome", "Oslo"])
    ask = lambda p, temperature=0.0: next(outs)
    print(consistency_score("Capital of France?", ["x"]))   # about 0.0
