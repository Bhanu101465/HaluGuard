# Passage filter: checks each passage before writing the answer.
# Keeps only the passages that help answer the question (YES).
# If all passages are NO, keeps the single best one.

from .llm import ask

PROMPT_TEMPLATE = (
    "Does this passage help answer the question?\n"
    "Reply with only one word: YES or NO.\n\n"
    "Question: {question}\n\n"
    "Passage:\n{passage}"
)

def filter_passages(question: str, passages: list[str]) -> list[str]:
    """
    Look at each passage one by one and ask the model if it helps answer the question.
    Keep only the YES passages. If all are NO, keep the single best one (the first one).
    """
    if not passages:
        return []

    kept = []
    for p in passages:
        prompt = PROMPT_TEMPLATE.format(question=question, passage=p)
        try:
            reply = ask(prompt, temperature=0.0).strip().upper()
            if reply.startswith("YES"):
                kept.append(p)
        except Exception as e:
            print(f"Warning: passage filter call failed ({e}), keeping passage.")
            kept.append(p)

    # If all passages are NO, keep the single best one (first passage)
    if not kept:
        kept = [passages[0]]

    dropped = len(passages) - len(kept)
    print(f"Passage filter: kept {len(kept)}, dropped {dropped} (out of {len(passages)} passages)")
    return kept

if __name__ == "__main__":
    q = "What is the capital of France?"
    p_good = "Paris is the capital and most populous city of France."
    p_bad = "The kangaroo is a marsupial from the family Macropodidae."
    
    print("Testing filter_passages with one good and one bad passage:")
    results = filter_passages(q, [p_good, p_bad])
    print(f"Kept {len(results)} passage(s).")
