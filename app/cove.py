# Part 3: Chain of Verification.
# draft -> check questions -> answer them separately -> rewrite.
from .llm import ask

def _passages_text(passages):
    return "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))

def verify_and_rewrite(question: str, draft: str, passages: list[str]) -> str:
    context = _passages_text(passages)

    # a) make check questions from the draft
    q_prompt = (
        "Read the draft answer. Write 3 to 5 short check questions that "
        "test each fact in it. One question per line. No numbering.\n\n"
        f"Question: {question}\nDraft: {draft}"
    )
    lines = ask(q_prompt).splitlines()
    checks = [l.strip("-•0123456789. ").strip() for l in lines if l.strip()][:5]
    if not checks:
        return draft

    # b) answer each check question SEPARATELY, without the draft
    pairs = []
    for c in checks:
        a_prompt = (
            "Answer using ONLY the passages. If not there, say: not found.\n\n"
            f"Passages:\n{context}\n\nQuestion: {c}\nAnswer:"
        )
        pairs.append((c, ask(a_prompt).strip()))

    # c) + d) compare with the draft and rewrite
    checked = "\n".join(f"Q: {c}\nA: {a}" for c, a in pairs)
    r_prompt = (
        "Here is a draft answer and some checked facts.\n"
        "Rewrite the draft. Fix anything that does not match the checked "
        "facts. Remove anything marked 'not found'. Keep it short.\n"
        "Write only the final answer.\n\n"
        f"Question: {question}\nDraft: {draft}\n\nChecked facts:\n{checked}"
    )
    final = ask(r_prompt).strip()
    return final or draft

if __name__ == "__main__":
    def fake(p, temperature=0.0):
        if "check questions" in p:
            return "Is Paris the capital?\nIs France in Asia?"
        if "Rewrite the draft" in p:
            return "Paris is the capital of France."
        return "not found" if "Asia" in p else "Yes, Paris."
    ask = fake
    print(verify_and_rewrite("Capital of France?",
          "Paris is the capital of France, in Asia.", ["Paris is in France."]))
