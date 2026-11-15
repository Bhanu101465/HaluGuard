# LLM-as-judge: asks the LLM to decide if an answer is supported by a passage.
# Returns: verdict (SUPPORTED / NOT SUPPORTED), proof sentence, score (1.0 or 0.0).
# The LLM must copy the exact proof sentence word-for-word from the passage.
# If there is no exact sentence, proof is empty and verdict is NOT SUPPORTED.

from .llm import ask

PROMPT_TEMPLATE = """\
You are a strict fact-checker. Read the passage, the question, and the answer.

Decide: does the passage SUPPORT the answer?

Rules:
1. First line: write ONLY "SUPPORTED" or "NOT SUPPORTED".
2. Second line: copy the EXACT sentence from the passage that proves the answer, word for word.
   - You MUST copy the sentence directly from the passage text.
   - Do NOT write explanations or commentary.
   - Do NOT paraphrase or reword.
   - If there is no exact sentence in the passage that proves the answer, write: none

Passage:
{passage}

Question: {question}
Answer: {answer}
"""

def judge(question: str, answer: str, passages: list[str]) -> dict:
    """
    Ask the LLM to judge whether the answer is supported by the passages.

    Returns a dict with:
      verdict        : "SUPPORTED" or "NOT SUPPORTED"
      proof          : exact sentence copied from passage (or "" if none)
      score          : 1.0 if SUPPORTED, 0.0 if NOT SUPPORTED
      verified       : True if proof sentence really appears in passage text
      reject_reason  : reason for NOT SUPPORTED (or "" if SUPPORTED)
                       "llm_said_not_supported" | "no_proof_given" | "proof_not_in_passage"
    """
    passage_text = "\n\n".join(passages)

    prompt = PROMPT_TEMPLATE.format(
        passage=passage_text,
        question=question,
        answer=answer,
    )

    try:
        raw = ask(prompt).strip()
    except Exception as e:
        return {
            "verdict": "NOT SUPPORTED",
            "proof": "",
            "score": 0.0,
            "verified": False,
            "reject_reason": f"error: {e}",
        }

    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    # Parse raw verdict from first non-empty line
    first_line = lines[0].upper() if lines else ""
    if "NOT SUPPORTED" in first_line:
        raw_verdict = "NOT SUPPORTED"
    elif "SUPPORTED" in first_line:
        raw_verdict = "SUPPORTED"
    else:
        raw_verdict = "NOT SUPPORTED"

    # Parse proof from second non-empty line
    raw_proof = lines[1] if len(lines) > 1 else ""
    proof = raw_proof.strip().strip("\"'").strip()
    if proof.lower().startswith("proof:"):
        proof = proof[6:].strip().strip("\"'").strip()
    if proof.lower().startswith("sentence:"):
        proof = proof[9:].strip().strip("\"'").strip()
    if proof.lower() in ("none", "none.", "n/a", "no", "no proof", "not found"):
        proof = ""

    # Check whether proof sentence actually appears in the passage
    verified = False
    if proof:
        norm_proof = " ".join(proof.lower().split())
        norm_passage = " ".join(passage_text.lower().split())
        if norm_proof in norm_passage:
            verified = True
        else:
            p_strip = norm_proof.strip(".,;:!?\"'")
            if len(p_strip) >= 10 and p_strip in norm_passage:
                verified = True

    # Determine final verdict and reject reason
    if raw_verdict != "SUPPORTED":
        verdict = "NOT SUPPORTED"
        reject_reason = "llm_said_not_supported"
        # If LLM said NOT SUPPORTED, proof should be empty (no explanations)
        if not verified:
            proof = ""
    elif not proof:
        verdict = "NOT SUPPORTED"
        reject_reason = "no_proof_given"
    elif not verified:
        verdict = "NOT SUPPORTED"
        reject_reason = "proof_not_in_passage"
        proof = ""  # drop invalid proof
    else:
        verdict = "SUPPORTED"
        reject_reason = ""

    score = 1.0 if verdict == "SUPPORTED" else 0.0

    return {
        "verdict": verdict,
        "proof": proof,
        "score": score,
        "verified": verified,
        "reject_reason": reject_reason,
    }
