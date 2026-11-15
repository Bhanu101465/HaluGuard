import json
import os

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "qa_data.json")

def load_samples(n: int = 100) -> list[dict]:
    """
    Returns a list of samples:
    [{'question': ..., 'knowledge': ..., 'right_answer': ..., 'hallucinated_answer': ...}, ...]
    """
    samples = []
    if not os.path.exists(DATA_FILE):
        return samples

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            samples.append({
                "question": item.get("question", ""),
                "knowledge": item.get("knowledge", ""),
                "right_answer": item.get("right_answer", ""),
                "hallucinated_answer": item.get("hallucinated_answer", ""),
            })
            if n is not None and len(samples) >= n:
                break
    return samples

def get_passages() -> list[str]:
    """
    Returns all the knowledge texts from the dataset.
    """
    samples = load_samples(n=None)
    return [s["knowledge"] for s in samples if s["knowledge"]]

