import os
os.environ["HF_HOME"] = r"D:\hf-cache"

from app.data_loader import load_samples
from app.retrieve import top_passages
from app.answer import write_answer
from app.cove import verify_and_rewrite
from app.facts import split_into_facts
from app.verify import check_fact
from app.web_search import search_fact
from app.pipeline import make_search_query

samples = load_samples(5)

for idx in [0, 3]:  # Q1 and Q4 (0-indexed)
    s = samples[idx]
    q = s["question"]
    passages = top_passages(q, [s["knowledge"]]) or [s["knowledge"]]

    draft = write_answer(q, passages)
    answer = verify_and_rewrite(q, draft, passages)
    if len(answer.split()) < 5:
        from app.pipeline import rewrite_short_answer_to_claim
        answer = rewrite_short_answer_to_claim(q, answer)
    facts = split_into_facts(answer)

    print("=" * 75)
    print(f"QUESTION {idx+1}: {q}")
    print(f"Right Answer: {s['right_answer']}")
    print(f"Draft: {draft}")
    print(f"After CoVe: {answer}")
    print(f"Facts: {facts}")
    print()

    for f in facts:
        # Passage-only check first
        r_pass = check_fact(f, passages, join_passages=True)
        print(f"Fact: {f}")
        print(f"  Passage-only NLI: label={r_pass['label']}, entailment_prob={r_pass['entailment_prob']:.4f}")

        # Build enriched query
        query = make_search_query(f, q, passages)
        print(f"  Search Query: '{query}'")

        # Try web search
        try:
            snippets = search_fact(query, max_results=3)
            print(f"  Snippets returned: {len(snippets)}")
            if not snippets:
                print("  -> Web search returned 0 snippets (silent, no error)")
            for i, snip in enumerate(snippets, 1):
                r_snip = check_fact(f, [snip], join_passages=True)
                print(f"  Snippet {i} ({len(snip)} chars): {snip[:150]}")
                print(f"    NLI: label={r_snip['label']}, entailment_prob={r_snip['entailment_prob']:.4f}")
        except Exception as e:
            print(f"  -> Web search FAILED with error: {e}")
        print()

