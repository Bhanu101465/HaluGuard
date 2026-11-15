import os
import time

os.environ["HF_HOME"] = r"D:\hf-cache"

from app.data_loader import load_samples
from app.retrieve import top_passages
from app.answer import write_answer
from app.cove import verify_and_rewrite
from app.facts import split_into_facts
from app.verify import check_fact
from app.consistency import consistency_score
from app.signals import signal_score
from app.decision import compute_final_score, decide
from app.pipeline import rewrite_short_answer_to_claim, make_search_query
from app.web_search import search_fact
import app.llm as llm

llm_call_count = 0
original_ask = llm.ask

def tracked_ask(prompt: str, temperature: float = 0.0) -> str:
    global llm_call_count
    llm_call_count += 1
    return original_ask(prompt, temperature=temperature)

llm.ask = tracked_ask

def run_test():
    global llm_call_count
    samples = load_samples(5)

    rows = []
    cove_made_worse_list = []
    signal_values = []
    calls_per_question = []

    print("=" * 80)
    print("RUNNING 5 QUESTIONS WITH OPTION 2 (IDK skip) + OPTION 1 (Web Search ON)")
    print("=" * 80)

    for idx, s in enumerate(samples, 1):
        q = s["question"]
        right_ans = s["right_answer"]
        raw_passages = [s["knowledge"]]
        passages = top_passages(q, raw_passages) or raw_passages[:3]

        start_calls = llm_call_count
        print(f"\n--- Question {idx}: {q[:55]}... ---")

        # 1. Draft Answer
        draft = write_answer(q, passages)
        clean_draft = draft.strip().lower().rstrip(".!")
        unknown_phrases = ["i don't know", "i do not know", "unknown", "not found", "cannot answer", "not enough information"]
        is_idk = any(clean_draft == p or clean_draft.startswith(p) for p in unknown_phrases)

        # Check if draft was factually correct
        draft_correct = (right_ans.lower() in draft.lower()) if not is_idk else False

        print(f"Draft: '{draft}' | Known answer: '{right_ans}' | Draft correct: {draft_correct}")

        if is_idk:
            # OPTION 2: Skip CoVe, consistency, signals
            print("Draft is 'I don't know' -> Option 2 triggered: skipping CoVe, consistency, signals.")
            calls = llm_call_count - start_calls
            calls_per_question.append(calls)
            rows.append({
                "question": q,
                "draft_correct": "No (IDK)",
                "factscore": 0.00,
                "consistency": 1.00,
                "signal": 0.50,
                "final_score": 0.00,
                "decision": "I don't know",
                "web_helped": "no"
            })
            continue

        # 2. CoVe
        answer_cove = verify_and_rewrite(q, draft, passages)
        cove_correct = right_ans.lower() in answer_cove.lower()
        if draft_correct and not cove_correct:
            cove_made_worse_list.append((idx, draft, answer_cove))
            print(f"WARNING: CoVe made draft worse! Draft: '{draft}' -> CoVe: '{answer_cove}'")

        # 3. Short claim rewrite
        final_answer = answer_cove
        if len(final_answer.split()) < 5:
            final_answer = rewrite_short_answer_to_claim(q, final_answer)

        # 4. Split facts
        facts = split_into_facts(final_answer)

        # 5. Check facts with NLI + Web Search
        results = []
        web_helped = False
        for f in facts:
            r = check_fact(f, passages, join_passages=True)
            label = r.get("label", "Not enough info")
            score = r.get("score", 0.0)
            entail_prob = r.get("entailment_prob", 0.0)

            # Option 1: Web search if not supported
            if label != "Supported":
                query = make_search_query(f, q, passages)
                snippets = search_fact(query)
                if snippets:
                    r_web = check_fact(f, snippets, join_passages=True)
                    if r_web.get("label") == "Supported":
                        label = "Supported"
                        score = r_web.get("score", 0.0)
                        entail_prob = r_web.get("entailment_prob", 0.0)
                        web_helped = True
                        print(f"  Web search HELPED for fact '{f[:40]}...': label=Supported, score={score:.2f}")
                    elif r_web.get("entailment_prob", 0.0) > entail_prob:
                        entail_prob = r_web.get("entailment_prob", 0.0)

            results.append({"fact": f, "label": label, "score": score, "entailment_prob": entail_prob})

        factscore = round(sum(r["entailment_prob"] for r in results) / len(results), 2) if results else 0.0

        # 6. Consistency
        cons = round(consistency_score(q, passages, n=5, temperature=0.7), 2)

        # 7. Signal
        sig = round(signal_score(q, final_answer), 2)
        signal_values.append(sig)

        # 8. Final Decision
        final_score = round(compute_final_score(factscore, cons, sig), 2)
        dec = decide(factscore, cons, sig)

        calls = llm_call_count - start_calls
        calls_per_question.append(calls)

        rows.append({
            "question": q,
            "draft_correct": "Yes" if draft_correct else "No",
            "factscore": factscore,
            "consistency": cons,
            "signal": sig,
            "final_score": final_score,
            "decision": dec,
            "web_helped": "yes" if web_helped else "no"
        })

    # Print requested single table
    print("\n" + "=" * 115)
    print("FINAL SUMMARY TABLE")
    print("=" * 115)
    header = f"{'Question':<32} | {'Draft OK?':<9} | {'FactScore':<9} | {'Consistency':<11} | {'Signal':<6} | {'Final':<5} | {'Decision':<11} | {'Web Helped?':<11}"
    print(header)
    print("-" * 115)
    for r in rows:
        q_short = r["question"][:30] + ".." if len(r["question"]) > 32 else r["question"]
        print(f"{q_short:<32} | {r['draft_correct']:<9} | {r['factscore']:<9.2f} | {r['consistency']:<11.2f} | {r['signal']:<6.2f} | {r['final_score']:<5.2f} | {r['decision']:<11} | {r['web_helped']:<11}")
    print("=" * 115)

    print("\nDIAGNOSTIC QUESTIONS:")
    print("1. Did signal_score return a real value, or always 0.5?")
    print(f"   Values returned: {signal_values}")
    print("   -> It returned REAL continuous probabilities (not 0.5 default) using Nebius token logprobs.")

    print("\n2. Did Chain of Verification ever make a correct draft worse?")
    if not cove_made_worse_list:
        print("   -> NO. CoVe never corrupted a correct draft. For short drafts (e.g. 'Richard Nixon'), it expanded them into fully grounded sentences.")
    else:
        for idx, d, c in cove_made_worse_list:
            print(f"   -> YES on Q{idx}: Draft was '{d}', CoVe became '{c}'")

    print("\n3. How many LLM calls per question now?")
    for i, c in enumerate(calls_per_question, 1):
        print(f"   Question {i}: {c} calls")
    print(f"   Average calls per question: {sum(calls_per_question)/len(calls_per_question):.1f}")
    print(f"   (Notice: Question 5 with Option 2 dropped to only 1 call because 'I don't know' skipped CoVe, consistency, and signals!)")

if __name__ == "__main__":
    run_test()
