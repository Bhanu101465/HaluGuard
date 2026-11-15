# Step 5: break the answer into tiny facts (sentence splitter / FActScore idea).
import re
from .llm import ask

# Toggle if LLM should be used for splitting facts (default False to save calls/time)
USE_LLM_FACTS = False

def split_sentences(text: str) -> list[str]:
    """
    Splits text into sentences while protecting decimals (e.g. 6.213 km)
    and common abbreviations (e.g. Mr., U.S.).
    """
    # Protect dots between digits (e.g. 6.213 km)
    t = re.sub(r'(?<=\d)\.(?=\d)', '<DOT>', text)
    # Protect common abbreviations
    abbreviations = ['U.S.', 'U.K.', 'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'vs.', 'e.g.', 'i.e.', 'St.']
    for ab in abbreviations:
        pat = re.escape(ab)
        rep = ab.replace('.', '<DOT>')
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)
    # Split on periods, semicolons, and newlines
    parts = re.split(r'[.\n;]+', t)
    # Restore dots and strip whitespace
    res = [p.replace('<DOT>', '.').strip() for p in parts if p.strip()]
    return res or [text.strip()]

def split_into_facts(answer: str, use_llm: bool = USE_LLM_FACTS) -> list[str]:
    if use_llm:
        try:
            prompt = (
                "Break the text into tiny facts. One fact per line.\n"
                "Each fact must make sense alone (no 'he' or 'it').\n"
                "Write only the facts, no numbering.\n\n"
                f"Text: {answer}"
            )
            lines = ask(prompt).splitlines()
            facts = [l.strip("-• ").strip() for l in lines if l.strip()]
            if facts:
                return facts
        except Exception:
            pass

    return split_sentences(answer)

if __name__ == "__main__":
    test_text = "The track is 6.213 km long."
    facts = split_into_facts(test_text)
    print("Test sentence: ", test_text)
    print("Extracted facts:", facts)
    assert len(facts) == 1, f"Expected 1 fact, got {len(facts)}"
    print("Test passed: kept as 1 fact!")
