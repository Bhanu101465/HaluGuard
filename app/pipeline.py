# The full HaluGuard pipeline connecting all modules.
import re
from .retrieve import top_passages
from .answer import write_answer
from .facts import split_into_facts
from .verify import check_fact
from .need_check import need_lookup
from .cove import verify_and_rewrite
from .web_search import search_fact
from .consistency import consistency_score
from .signals import signal_score
from .decision import decide, compute_final_score
from .judge import judge
from .passage_filter import filter_passages
from .llm import ask

# Feature switches (toggle parts on/off for testing)
USE_NEED_CHECK = True
USE_COVE = True
USE_WEB_SEARCH = True        # Option 1: turned ON
USE_CONSISTENCY = True
USE_SIGNALS = True
USE_JOIN_PASSAGES = True     # Check fact against all passages joined into one text
USE_SOFT_FACTSCORE = True    # Average raw entailment probabilities (soft score)
USE_CLAIM_REWRITE = True     # Turn short answers into a clear sentence using the LLM
USE_JUDGE = False            # LLM-as-judge: extra SUPPORTED/NOT SUPPORTED check (OFF by default)
USE_PASSAGE_FILTER = False   # Passage filter: drop unhelpful passages before drafting (OFF by default)

def make_search_query(fact: str, question: str, passages: list[str]) -> str:
    """
    Enriches the search query by extracting key entities, quotes, and names
    from the question and passage so search results are about the right entity.
    """
    entity_terms = []
    # 1. Quoted terms or proper nouns in question
    q_matches = re.findall(r'["\']([^"\']+)["\']|\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', question)
    for q_quote, q_cap in q_matches:
        term = (q_quote or q_cap).strip()
        if term and term.lower() not in ["what", "which", "who", "where", "when", "why", "how", "the", "are", "is"]:
            if term not in entity_terms:
                entity_terms.append(term)

    # 2. Key names from first passage
    if passages:
        p_matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', passages[0][:250])
        for p_cap in p_matches:
            term = p_cap.strip()
            if term.lower() not in ["the", "this", "that", "it", "he", "she", "in", "on", "at", "an", "and", "first"]:
                if term not in entity_terms:
                    entity_terms.append(term)

    extra_context = " ".join(entity_terms[:4])
    return f"{fact} {extra_context}".strip()

def rewrite_short_answer_to_claim(question: str, answer: str) -> str:
    """
    Turns short answers (<5 words) into one clear statement using the LLM.
    Falls back to simple template if LLM is unavailable or fails.
    """
    old_template = f"The answer to '{question}' is {answer}."
    if not USE_CLAIM_REWRITE:
        return old_template

    prompt = (
        "Write ONE clear sentence that states the question and this answer together. "
        "Do not add new facts. For yes/no answers, write the full statement (for 'no', write the negative statement). "
        "Write only the sentence.\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}"
    )
    try:
        rewritten = ask(prompt).strip()
        rewritten = rewritten.strip("\"' ")
        if rewritten:
            return rewritten
    except Exception as e:
        print(f"Warning: claim rewrite failed ({e}), using template.")

    return old_template

def run(question: str, passages: list[str], given_answer: str = None) -> dict:
    # 1. Need check: does question need fact lookup?
    needs = True
    if USE_NEED_CHECK:
        try:
            needs = need_lookup(question)
        except Exception as e:
            print(f"Warning: need_lookup failed ({e}), continuing with checks.")
            needs = True

    if not needs:
        try:
            answer = ask(question)
        except Exception:
            answer = "I don't know."
        return {
            "question": question,
            "passages_used": [],
            "answer": answer,
            "facts": [],
            "factscore": 1.0,
            "consistency": 1.0,
            "signal": 0.5,
            "final_score": 1.0,
            "decision": "answer",
            "skipped_checks": True,
        }

    # 2. Retrieve top passages
    passages_used = top_passages(question, passages)
    if not passages_used and passages:
        passages_used = passages[:3]

    # Optional passage filter (OFF by default)
    if USE_PASSAGE_FILTER:
        passages_used = filter_passages(question, passages_used)

    # 3. Draft answer & CoVe (or use given_answer)
    if given_answer is None:
        draft = write_answer(question, passages_used)

        # OPTION 2: If draft answer is "I don't know" (or equivalent),
        # skip CoVe, consistency, and signals, and return "I don't know".
        clean_draft = draft.strip().lower().rstrip(".!")
        unknown_phrases = ["i don't know", "i do not know", "unknown", "not found", "cannot answer", "not enough information"]
        if any(clean_draft == p or clean_draft.startswith(p) for p in unknown_phrases):
            return {
                "question": question,
                "passages_used": passages_used,
                "answer": "I don't know.",
                "facts": [],
                "factscore": 0.0,
                "consistency": 1.0,
                "signal": 0.5,
                "final_score": 0.0,
                "decision": "I don't know",
                "skipped_checks": True,
            }

        if USE_COVE:
            try:
                answer = verify_and_rewrite(question, draft, passages_used)
            except Exception as e:
                print(f"Warning: CoVe failed ({e}), using draft answer.")
                answer = draft
        else:
            answer = draft
    else:
        answer = given_answer

    # 4. Turn very short answers into full sentences for better NLI verification
    if len(answer.split()) < 5:
        answer = rewrite_short_answer_to_claim(question, answer)

    # 5. Split answer into facts
    facts = split_into_facts(answer)

    # 6. Check each fact with NLI (and web search if enabled)
    results = []
    web_helped = False
    for f in facts:
        r = check_fact(f, passages_used, join_passages=USE_JOIN_PASSAGES)
        label = r.get("label", "Not enough info")
        score = r.get("score", 0.0)
        entailment_prob = r.get("entailment_prob", 0.0)
        evidence = r.get("evidence", None)
        source = "passages"

        if label != "Supported" and USE_WEB_SEARCH:
            try:
                query = make_search_query(f, question, passages_used)
                snippets = search_fact(query)
                if snippets:
                    r_web = check_fact(f, snippets, join_passages=USE_JOIN_PASSAGES)
                    if r_web.get("label") == "Supported":
                        label = "Supported"
                        score = r_web.get("score", 0.0)
                        entailment_prob = r_web.get("entailment_prob", 0.0)
                        evidence = r_web.get("evidence", None)
                        source = "web"
                        web_helped = True
                    elif r_web.get("entailment_prob", 0.0) > entailment_prob:
                        entailment_prob = r_web.get("entailment_prob", 0.0)
            except Exception as e:
                print(f"Warning: search_fact failed ({e})")

        results.append({
            "fact": f,
            "label": label,
            "score": round(score, 2),
            "entailment_prob": round(entailment_prob, 4),
            "source": source,
            "evidence": evidence,
        })

    # 7. Factscore (soft average or hard ratio)
    if not results:
        factscore = 0.0
    elif USE_SOFT_FACTSCORE:
        factscore = round(sum(r["entailment_prob"] for r in results) / len(results), 2)
    else:
        supported = sum(1 for r in results if r["label"] == "Supported")
        factscore = round(supported / len(results), 2)

    # 8. Consistency (only if factscore < 0.9)
    if USE_CONSISTENCY and factscore < 0.9:
        try:
            consistency = round(consistency_score(question, passages_used), 2)
        except Exception as e:
            print(f"Warning: consistency_score failed ({e}), using 1.0.")
            consistency = 1.0
    else:
        consistency = 1.0

    # 9. Model signal
    if USE_SIGNALS:
        try:
            signal = round(signal_score(question, answer), 2)
        except Exception as e:
            print(f"Warning: signal_score failed ({e}), using 0.5.")
            signal = 0.5
    else:
        signal = 0.5

    # 10. Final score and decision
    final_score = round(compute_final_score(factscore, consistency, signal), 2)
    decision = decide(factscore, consistency, signal)

    # 11. LLM-as-judge (optional, OFF by default)
    judge_result = None
    if USE_JUDGE:
        try:
            judge_result = judge(question, answer, passages_used)
        except Exception as e:
            print(f"Warning: judge failed ({e}), skipping.")

    return {
        "question": question,
        "passages_used": passages_used,
        "answer": answer,
        "facts": results,
        "factscore": factscore,
        "consistency": consistency,
        "signal": signal,
        "final_score": final_score,
        "decision": decision,
        "web_helped": web_helped,
        "judge": judge_result,   # None when USE_JUDGE is False
    }

def run_sample(sample: dict, use_hallucinated: bool = False) -> dict:
    question = sample["question"]
    passages = [sample["knowledge"]]
    given_answer = sample["hallucinated_answer"] if use_hallucinated else sample["right_answer"]
    return run(question, passages, given_answer=given_answer)
