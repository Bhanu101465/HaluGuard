"""
generative_eval.py  —  Run the full HaluGuard pipeline on HaluEval samples
with no given_answer (true generative mode).

Extra metrics vs pipeline.run():
  - factscore_draft : NLI score of the DRAFT answer (before CoVe)
  - factscore_cove  : NLI score of the answer AFTER CoVe  (= normal factscore)
  - draft_correct   : True if right_answer text appears anywhere in draft (case-insensitive)
  - llm_calls       : LLM calls used for this question
  - web_helped      : yes/no

Usage:
  python generative_eval.py --start 150 -n 30 -o generative_30.csv
"""

import os
os.environ["HF_HOME"] = r"D:\hf-cache"

import argparse
import csv
import sys
import time

# -- LLM call counter (must patch before submodule imports) ------------------
import app.llm as _llm_mod

_orig_ask = _llm_mod.ask
_call_count = [0]

def _counting_ask(prompt: str, temperature: float = 0.0) -> str:
    _call_count[0] += 1
    return _orig_ask(prompt, temperature=temperature)

_llm_mod.ask = _counting_ask

# Now import all submodules — their local `ask` reference still points to
# the module attribute, so we also patch each module's own binding.
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
import app.need_check as _nc
import app.answer as _ans
import app.cove as _cove_mod
import app.consistency as _cons_mod
import app.signals as _sig_mod
import app.pipeline as _pipe_mod

# Patch every module's local `ask` reference
for _m in (_nc, _ans, _cove_mod, _cons_mod, _sig_mod, _pipe_mod):
    if hasattr(_m, "ask"):
        setattr(_m, "ask", _counting_ask)

# -- Helpers -----------------------------------------------------------------

UNKNOWN_PHRASES = [
    "i don't know", "i do not know", "unknown",
    "not found", "cannot answer", "not enough information",
]

def _is_idk(text: str) -> bool:
    t = text.strip().lower().rstrip(".!")
    return any(t == p or t.startswith(p) for p in UNKNOWN_PHRASES)

def _nli_factscore(answer: str, passages: list[str]) -> float:
    """Score an answer with NLI only — no LLM calls."""
    facts = split_into_facts(answer)
    if not facts:
        return 0.0
    total = sum(
        check_fact(f, passages, join_passages=True).get("entailment_prob", 0.0)
        for f in facts
    )
    return round(total / len(facts), 4)

# -- Main evaluation loop -----------------------------------------------------

def run_eval(start: int, n: int, out_path: str):
    all_samples = load_samples(start + n)
    samples = all_samples[start: start + n]

    if not samples:
        print(f"ERROR: no samples found for start={start}, n={n}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"GENERATIVE EVAL  |  samples {start+1} – {start+n}  |  n={n}")
    print(f"All pipeline modules ON (need_check, CoVe, web_search, consistency, signals)")
    print(f"Output -> {out_path}")
    print(f"{'='*70}\n")

    fieldnames = [
        "idx", "question", "right_answer",
        "draft", "draft_correct",
        "answer_cove",
        "factscore_draft", "factscore_cove",
        "consistency", "signal",
        "final_score", "decision",
        "llm_calls", "web_helped",
    ]

    rows = []
    total_calls = 0

    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for i, s in enumerate(samples, 1):
            q          = s["question"]
            right_ans  = s["right_answer"]
            passages   = top_passages(q, [s["knowledge"]]) or [s["knowledge"]]

            _call_count[0] = 0   # reset counter per question
            print(f"[{i:2d}/{n}] Q: {q[:70]}")

            # -- 1. Need-check ----------------------------------------------
            try:
                from app.need_check import need_lookup
                needs = need_lookup(q)
            except Exception as e:
                print(f"       need_lookup failed ({e}), assuming True")
                needs = True

            if not needs:
                # Factual lookup not needed -> pipeline trusts the LLM answer
                try:
                    draft = _counting_ask(q)
                except Exception:
                    draft = "I don't know."
                draft_correct  = right_ans.lower() in draft.lower()
                row = dict(
                    idx=start + i,
                    question=q, right_answer=right_ans,
                    draft=draft, draft_correct=draft_correct,
                    answer_cove=draft,
                    factscore_draft=1.0, factscore_cove=1.0,
                    consistency=1.0, signal=0.5,
                    final_score=1.0, decision="answer",
                    llm_calls=_call_count[0], web_helped="no",
                )
                rows.append(row)
                writer.writerow(row)
                fout.flush()
                total_calls += _call_count[0]
                print(f"       need_lookup=False -> skipped checks | calls={_call_count[0]}")
                continue

            # -- 2. Draft answer --------------------------------------------
            draft = write_answer(q, passages)
            print(f"       Draft: {draft[:80]}")

            draft_correct = right_ans.lower() in draft.lower()

            # Option 2: draft is "I don't know"
            if _is_idk(draft):
                row = dict(
                    idx=start + i,
                    question=q, right_answer=right_ans,
                    draft=draft, draft_correct=False,
                    answer_cove="I don't know.",
                    factscore_draft=0.0, factscore_cove=0.0,
                    consistency=1.0, signal=0.5,
                    final_score=0.0, decision="I don't know",
                    llm_calls=_call_count[0], web_helped="no",
                )
                rows.append(row)
                writer.writerow(row)
                fout.flush()
                total_calls += _call_count[0]
                print(f"       -> Option 2 (IDK draft) | calls={_call_count[0]}")
                continue

            # -- 3. Score DRAFT with NLI (no extra LLM calls) --------------
            # Rewrite short draft for NLI scoring if needed
            draft_for_nli = draft
            if len(draft.split()) < 5:
                draft_for_nli = rewrite_short_answer_to_claim(q, draft_for_nli)
            factscore_draft = _nli_factscore(draft_for_nli, passages)

            # -- 4. CoVe ----------------------------------------------------
            try:
                answer_cove = verify_and_rewrite(q, draft, passages)
            except Exception as e:
                print(f"       CoVe failed ({e}), using draft")
                answer_cove = draft

            # Short-answer rewrite for NLI
            answer_final = answer_cove
            if len(answer_final.split()) < 5:
                answer_final = rewrite_short_answer_to_claim(q, answer_final)

            # -- 5. Score CoVe answer with NLI (+ web search) --------------
            facts = split_into_facts(answer_final)
            results = []
            web_helped = False

            for f in facts:
                r = check_fact(f, passages, join_passages=True)
                label        = r.get("label", "Not enough info")
                entail_prob  = r.get("entailment_prob", 0.0)
                source       = "passages"

                if label != "Supported":
                    try:
                        query    = make_search_query(f, q, passages)
                        snippets = search_fact(query)
                        if snippets:
                            r_web = check_fact(f, snippets, join_passages=True)
                            if r_web.get("label") == "Supported":
                                label       = "Supported"
                                entail_prob = r_web.get("entailment_prob", 0.0)
                                source      = "web"
                                web_helped  = True
                            elif r_web.get("entailment_prob", 0.0) > entail_prob:
                                entail_prob = r_web.get("entailment_prob", 0.0)
                    except Exception as e:
                        print(f"       search_fact failed ({e})")

                results.append({"fact": f, "label": label, "entailment_prob": entail_prob, "source": source})

            factscore_cove = round(
                sum(r["entailment_prob"] for r in results) / len(results), 4
            ) if results else 0.0

            # -- 6. Consistency ---------------------------------------------
            if factscore_cove < 0.9:
                try:
                    consistency = round(consistency_score(q, passages), 2)
                except Exception as e:
                    print(f"       consistency_score failed ({e}), using 1.0")
                    consistency = 1.0
            else:
                consistency = 1.0

            # -- 7. Signal --------------------------------------------------
            try:
                signal = round(signal_score(q, answer_final), 2)
            except Exception as e:
                print(f"       signal_score failed ({e}), using 0.5")
                signal = 0.5

            # -- 8. Final score + decision ----------------------------------
            final_score = round(compute_final_score(factscore_cove, consistency, signal), 2)
            decision    = decide(factscore_cove, consistency, signal)

            calls_this_q = _call_count[0]
            total_calls += calls_this_q

            print(f"       CoVe: {answer_final[:70]}")
            print(f"       fs_draft={factscore_draft:.3f} fs_cove={factscore_cove:.3f} "
                  f"cons={consistency:.2f} sig={signal:.2f} "
                  f"final={final_score:.2f} -> {decision} | calls={calls_this_q}")

            row = dict(
                idx=start + i,
                question=q, right_answer=right_ans,
                draft=draft, draft_correct=draft_correct,
                answer_cove=answer_final,
                factscore_draft=factscore_draft,
                factscore_cove=factscore_cove,
                consistency=consistency,
                signal=signal,
                final_score=final_score,
                decision=decision,
                llm_calls=calls_this_q,
                web_helped="yes" if web_helped else "no",
            )
            rows.append(row)
            writer.writerow(row)
            fout.flush()   # write after every question so progress is saved

    # -- Summary --------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"DONE  |  {len(rows)} questions evaluated  |  saved to {out_path}")
    print(f"{'='*70}\n")

    _print_tables(rows, total_calls, n)


def _print_tables(rows: list[dict], total_calls: int, n: int):
    # Categorise
    cat = {"correct_answer": [], "correct_idk": [], "wrong_answer": [], "wrong_idk": []}
    for r in rows:
        dc  = bool(r["draft_correct"])
        dec = str(r["decision"])
        is_answer = (dec == "answer")
        if dc and is_answer:
            cat["correct_answer"].append(r)
        elif dc and not is_answer:
            cat["correct_idk"].append(r)
        elif not dc and is_answer:
            cat["wrong_answer"].append(r)
        else:
            cat["wrong_idk"].append(r)

    def avg(lst, key):
        vals = [float(r[key]) for r in lst if r[key] != ""]
        return sum(vals) / len(vals) if vals else 0.0

    print("=" * 65)
    print("TABLE A -- 4-Category Confusion Matrix")
    print("=" * 65)
    print(f"{'Category':<35} {'Count':>5}  {'% of total':>10}")
    print("-" * 55)

    labels = {
        "correct_answer": "[OK]  Correct draft   -> 'answer'",
        "correct_idk":    "[!!]  Correct draft   -> 'I don't know' (missed)",
        "wrong_answer":   "[XX]  Wrong draft     -> 'answer' (worst case!)",
        "wrong_idk":      "[>>]  Wrong draft     -> 'I don't know' (good catch)",
    }
    for key, label in labels.items():
        c   = len(cat[key])
        pct = 100 * c / n if n else 0
        print(f"  {label:<33} {c:>5}   {pct:>8.1f}%")

    print()
    print(f"  {'Metric':<20} {'Correct drafts':>14}  {'Wrong drafts':>14}")
    print("  " + "-" * 52)
    correct_rows = cat["correct_answer"] + cat["correct_idk"]
    wrong_rows   = cat["wrong_answer"]   + cat["wrong_idk"]
    for metric, key in [("Avg factscore_cove", "factscore_cove"),
                         ("Avg consistency",   "consistency"),
                         ("Avg signal",        "signal")]:
        cv = avg(correct_rows, key)
        wv = avg(wrong_rows,   key)
        print(f"  {metric:<20} {cv:>14.3f}  {wv:>14.3f}")

    print()
    # CoVe comparison table
    higher = equal = lower = 0
    for r in rows:
        fd = float(r["factscore_draft"])
        fc = float(r["factscore_cove"])
        if fc > fd + 1e-4:
            higher += 1
        elif fc < fd - 1e-4:
            lower += 1
        else:
            equal += 1

    print("+==============================================================+")
    print("|      TABLE B — Chain of Verification Effect on FactScore     |")
    print("+==============================================================+")
    print(f"  {'Outcome':<40} {'Count':>5}  {'%':>6}")
    print("  " + "-" * 55)
    for label, count in [("factscore_cove HIGHER than draft (CoVe helped)", higher),
                          ("factscore_cove EQUAL  to draft (no change)",     equal),
                          ("factscore_cove LOWER  than draft (CoVe hurt)",   lower)]:
        pct = 100 * count / len(rows) if rows else 0
        print(f"  {label:<40} {count:>5}  {pct:>5.1f}%")

    print()
    avg_calls = total_calls / n if n else 0
    cost_per_q = avg_calls * 0.00006   # ~$0.00006 per call (Nebius Qwen3-30B estimate)
    total_cost = cost_per_q * n
    print(f"  Avg LLM calls/question : {avg_calls:.1f}")
    print(f"  Rough cost/question    : ~${cost_per_q:.5f}")
    print(f"  Total cost for {n} Qs   : ~${total_cost:.4f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=150, help="0-based index of first sample")
    parser.add_argument("-n",      type=int, default=30,  help="Number of samples to run")
    parser.add_argument("-o",      type=str, default="generative_30.csv", help="Output CSV path")
    args = parser.parse_args()
    run_eval(args.start, args.n, args.o)

