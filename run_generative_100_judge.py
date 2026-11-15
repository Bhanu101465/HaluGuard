"""
run_generative_100_judge.py

Run full pipeline on HaluEval samples 181 to 280 (100 questions):
- All modules ON: need_check, write_answer, CoVe, web_search, consistency, signals, judge
- Semantic draft correctness checked via LLM: "Do these two answers mean the same?"
- Saves every draft marked wrong to drafts_marked_wrong.csv
- Saves all results incrementally to generative_100_judge.csv
"""

import os
os.environ["HF_HOME"] = r"D:\hf-cache"

import csv
import argparse
import sys

# -- Track LLM calls --
import app.llm as _llm_mod

_orig_ask = _llm_mod.ask
_call_count = [0]
_total_calls = [0]

def _counting_ask(prompt: str, temperature: float = 0.0) -> str:
    _call_count[0] += 1
    _total_calls[0] += 1
    return _orig_ask(prompt, temperature=temperature)

_llm_mod.ask = _counting_ask

# Import modules and bind counting ask
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
from app.judge import judge
from app.need_check import need_lookup

import app.need_check as _nc
import app.answer as _ans
import app.cove as _cove_mod
import app.consistency as _cons_mod
import app.signals as _sig_mod
import app.judge as _judge_mod
import app.pipeline as _pipe_mod

for _m in (_nc, _ans, _cove_mod, _cons_mod, _sig_mod, _judge_mod, _pipe_mod):
    if hasattr(_m, "ask"):
        setattr(_m, "ask", _counting_ask)

UNKNOWN_PHRASES = [
    "i don't know", "i do not know", "unknown",
    "not found", "cannot answer", "not enough information",
]

def _is_idk(text: str) -> bool:
    t = text.strip().lower().rstrip(".!")
    return any(t == p or t.startswith(p) for p in UNKNOWN_PHRASES)

def check_draft_same_meaning(question: str, draft: str, right_answer: str) -> tuple[bool, str]:
    """Ask the LLM if draft answer and right answer mean the same thing."""
    prompt = (
        "You are an evaluator. Compare the drafted answer with the ground truth right answer.\n\n"
        f"Question: {question}\n"
        f"Draft Answer: {draft}\n"
        f"Right Answer: {right_answer}\n\n"
        "Do these two answers mean the same or convey the same factual answer to the question?\n"
        "Reply on the first line with only YES or NO.\n"
        "On the second line, write a short one-sentence reason."
    )
    try:
        reply = _counting_ask(prompt).strip()
        lines = [l.strip() for l in reply.splitlines() if l.strip()]
        first_line = lines[0].upper() if lines else ""
        is_same = first_line.startswith("YES")
        reason = lines[1] if len(lines) > 1 else reply
        return is_same, reason
    except Exception as e:
        return False, f"error in check: {e}"

def run_evaluation(start: int = 180, n: int = 100, out_csv: str = "generative_100_judge.csv", wrong_csv: str = "drafts_marked_wrong.csv"):
    all_samples = load_samples(start + n)
    samples = all_samples[start: start + n]

    if not samples:
        print(f"ERROR: No samples found for start={start}, n={n}")
        return

    print("=" * 75)
    print(f"GENERATIVE EVAL (WITH JUDGE) | samples {start+1} to {start+n} (n={n})")
    print(f"Output -> {out_csv}")
    print(f"Wrong drafts -> {wrong_csv}")
    print("=" * 75)

    fieldnames = [
        "idx", "question", "right_answer", "draft_answer",
        "draft_correct", "draft_correct_reason", "final_answer",
        "factscore", "consistency", "signal",
        "judge_verdict", "judge_score", "judge_proof",
        "old_final_score", "old_decision", "llm_calls",
    ]

    wrong_fieldnames = ["idx", "question", "draft_answer", "right_answer", "llm_reason"]

    # Open both CSVs for incremental writing
    f_out = open(out_csv, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()
    f_out.flush()

    f_wrong = open(wrong_csv, "w", newline="", encoding="utf-8")
    writer_wrong = csv.DictWriter(f_wrong, fieldnames=wrong_fieldnames)
    writer_wrong.writeheader()
    f_wrong.flush()

    rows = []

    for i, s in enumerate(samples, 1):
        global_idx = start + i
        q = s["question"]
        right_ans = s["right_answer"]
        passages = top_passages(q, [s["knowledge"]]) or [s["knowledge"]]

        _call_count[0] = 0

        print(f"\n[{i:3d}/{n}] Q ({global_idx}): {q[:65]}")

        # 1. Need check
        try:
            needs = need_lookup(q)
        except Exception:
            needs = True

        if not needs:
            try:
                draft = _counting_ask(q)
            except Exception:
                draft = "I don't know."
            final_ans = draft
            factscore = 1.0
            consistency = 1.0
            signal = 0.5
            j_verdict = "SUPPORTED"
            j_score = 1.0
            j_proof = ""
            old_final = 1.0
            old_dec = "answer"

            is_correct, reason = check_draft_same_meaning(q, draft, right_ans)
            if not is_correct:
                writer_wrong.writerow({
                    "idx": global_idx, "question": q,
                    "draft_answer": draft, "right_answer": right_ans,
                    "llm_reason": reason,
                })
                f_wrong.flush()

            row = {
                "idx": global_idx, "question": q, "right_answer": right_ans,
                "draft_answer": draft, "draft_correct": is_correct,
                "draft_correct_reason": reason, "final_answer": final_ans,
                "factscore": factscore, "consistency": consistency, "signal": signal,
                "judge_verdict": j_verdict, "judge_score": j_score, "judge_proof": j_proof,
                "old_final_score": old_final, "old_decision": old_dec,
                "llm_calls": _call_count[0],
            }
            rows.append(row)
            writer.writerow(row)
            f_out.flush()
            print(f"       need_lookup=False -> answer approved directly | calls={_call_count[0]}")
            continue

        # 2. Draft answer
        draft = write_answer(q, passages)
        print(f"       Draft: {draft[:70]}")

        # Check draft correctness via LLM call
        is_correct, reason = check_draft_same_meaning(q, draft, right_ans)
        corr_label = "RIGHT" if is_correct else "WRONG"
        print(f"       Correctness: {corr_label} | Reason: {reason[:60]}")
        if not is_correct:
            writer_wrong.writerow({
                "idx": global_idx, "question": q,
                "draft_answer": draft, "right_answer": right_ans,
                "llm_reason": reason,
            })
            f_wrong.flush()

        # Check Option 2: Draft is "I don't know"
        if _is_idk(draft):
            row = {
                "idx": global_idx, "question": q, "right_answer": right_ans,
                "draft_answer": draft, "draft_correct": is_correct,
                "draft_correct_reason": reason, "final_answer": "I don't know.",
                "factscore": 0.0, "consistency": 1.0, "signal": 0.5,
                "judge_verdict": "NOT SUPPORTED", "judge_score": 0.0, "judge_proof": "",
                "old_final_score": 0.0, "old_decision": "I don't know",
                "llm_calls": _call_count[0],
            }
            rows.append(row)
            writer.writerow(row)
            f_out.flush()
            print(f"       -> Draft is IDK (Option 2 triggered) | calls={_call_count[0]}")
            continue

        # 3. CoVe
        try:
            answer_cove = verify_and_rewrite(q, draft, passages)
        except Exception:
            answer_cove = draft

        final_ans = answer_cove
        if len(final_ans.split()) < 5:
            final_ans = rewrite_short_answer_to_claim(q, final_ans)

        # 4. Split facts and NLI check
        facts = split_into_facts(final_ans)
        results = []
        for f in facts:
            r_nli = check_fact(f, passages, join_passages=True)
            label = r_nli.get("label", "Not enough info")
            entail_prob = r_nli.get("entailment_prob", 0.0)

            if label != "Supported":
                try:
                    query = make_search_query(f, q, passages)
                    snippets = search_fact(query)
                    if snippets:
                        r_web = check_fact(f, snippets, join_passages=True)
                        if r_web.get("label") == "Supported":
                            entail_prob = r_web.get("entailment_prob", 0.0)
                        elif r_web.get("entailment_prob", 0.0) > entail_prob:
                            entail_prob = r_web.get("entailment_prob", 0.0)
                except Exception:
                    pass
            results.append(entail_prob)

        factscore = round(sum(results) / len(results), 4) if results else 0.0

        # 5. Consistency (only if factscore < 0.9)
        if factscore < 0.9:
            try:
                consistency = round(consistency_score(q, passages), 2)
            except Exception:
                consistency = 1.0
        else:
            consistency = 1.0

        # 6. Signal
        try:
            signal = round(signal_score(q, final_ans), 2)
        except Exception:
            signal = 0.5

        # 7. Old formula
        old_final = round(compute_final_score(factscore, consistency, signal), 2)
        old_dec = decide(factscore, consistency, signal)

        # 8. Judge
        try:
            j_res = judge(q, final_ans, passages)
            j_verdict = j_res["verdict"]
            j_score = j_res["score"]
            j_proof = j_res["proof"]
        except Exception as e:
            j_verdict = "NOT SUPPORTED"
            j_score = 0.0
            j_proof = ""

        row = {
            "idx": global_idx, "question": q, "right_answer": right_ans,
            "draft_answer": draft, "draft_correct": is_correct,
            "draft_correct_reason": reason, "final_answer": final_ans,
            "factscore": factscore, "consistency": consistency, "signal": signal,
            "judge_verdict": j_verdict, "judge_score": j_score, "judge_proof": j_proof,
            "old_final_score": old_final, "old_decision": old_dec,
            "llm_calls": _call_count[0],
        }
        rows.append(row)
        writer.writerow(row)
        f_out.flush()

        print(f"       Final: {final_ans[:65]}")
        print(f"       fs={factscore:.2f} cons={consistency:.2f} sig={signal:.2f} -> old={old_dec} ({old_final:.2f}) | Judge={j_verdict} | calls={_call_count[0]}")

    f_out.close()
    f_wrong.close()

    print("\n" + "=" * 75)
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print(f"Total questions evaluated: {len(rows)}")
    print(f"Total LLM calls made:      {_total_calls[0]}")
    cost_est = _total_calls[0] * 0.00006
    print(f"Rough estimated cost:     ~${cost_est:.4f}")
    print("=" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=180, help="0-based start index (default: 180 for sample 181)")
    parser.add_argument("-n", type=int, default=100, help="Number of questions (default: 100)")
    parser.add_argument("-o", type=str, default="generative_100_judge.csv", help="Output CSV path")
    parser.add_argument("--wrong-file", type=str, default="drafts_marked_wrong.csv", help="Wrong drafts output CSV")
    args = parser.parse_args()
    run_evaluation(start=args.start, n=args.n, out_csv=args.o, wrong_csv=args.wrong_file)
