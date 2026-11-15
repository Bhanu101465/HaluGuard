# Part 2: look up a fact on the web (free, no key). Needs: pip install ddgs
def search_fact(fact: str, max_results: int = 3) -> list[str]:
    try:
        from ddgs import DDGS
        results = DDGS().text(fact, max_results=max_results)
        return [r.get("body", "") for r in results if r.get("body")]
    except Exception:
        return []  # never crash the pipeline

if __name__ == "__main__":
    for s in search_fact("Eiffel Tower height"):
        print("-", s[:120])
