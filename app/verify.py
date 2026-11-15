# Step 6a: check each fact against the passages with an NLI model.
# Labels: Supported / Unsupported / Not enough info
import os
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("HF_HOME", os.getenv("HF_HOME", r"D:\hf-cache"))

from transformers import pipeline

_nli = None
MODEL = "cross-encoder/nli-deberta-v3-base"

# Switch: join passages into one text vs check one-by-one
JOIN_PASSAGES = True

def _get_nli():
    global _nli
    if _nli is None:
        _nli = pipeline("text-classification", model=MODEL, top_k=None)
    return _nli

def check_fact(fact: str, passages: list[str], threshold: float = 0.7, join_passages: bool = None):
    nli = _get_nli()
    if not passages:
        return {"label": "Not enough info", "score": 0.0, "entailment_prob": 0.0, "evidence": None}

    use_join = JOIN_PASSAGES if join_passages is None else join_passages
    passages_to_check = ["\n".join(passages)] if use_join else passages

    best = {
        "label": "Not enough info",
        "score": 0.0,
        "entailment_prob": 0.0,
        "evidence": None,
    }
    max_entailment = 0.0

    for p in passages_to_check:
        out = nli({"text": p, "text_pair": fact})
        out = out[0] if isinstance(out[0], list) else out
        probs = {o["label"].lower(): o["score"] for o in out}

        entail = float(probs.get("entailment", 0.0))
        contradiction = float(probs.get("contradiction", 0.0))

        if entail > max_entailment:
            max_entailment = entail

        if entail >= threshold:
            return {
                "label": "Supported",
                "score": entail,
                "entailment_prob": entail,
                "evidence": p,
            }
        if contradiction >= threshold and contradiction > best["score"]:
            best = {
                "label": "Unsupported",
                "score": contradiction,
                "entailment_prob": entail,
                "evidence": p,
            }

    best["entailment_prob"] = max_entailment
    return best
