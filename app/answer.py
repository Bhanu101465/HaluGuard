# Step 3: write the draft answer using only the passages.
from .llm import ask

def write_answer(question: str, passages: list[str]) -> str:
    context = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    prompt = (
        "Answer the question using ONLY the passages below.\n"
        "If the passages do not have the answer, say: I don't know.\n\n"
        f"Passages:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    return ask(prompt)
