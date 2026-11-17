import os
os.environ["HF_HOME"] = r"D:\hf-cache"

import time
import csv
from sklearn.metrics import roc_auc_score

from app.data_loader import load_samples
from app.consistency import sample_consistency_answers, score_answers_consistency
from app.semantic_entropy import (
    compute_semantic_entropy_score,
    apply_semantic_entropy_decision,
    is_doubtful,
    is_unsure
)
from app.decision import compute_final_score, decide
from app.need_check import need_lookup
from app.retrieve import top_passages
from app.pipeline import rewrite_short_answer_to_claim, make_search_query
from app.facts import split_into_facts
from app.verify import check_fact
from app.signals import signal_score
from app.web_search import search_fact

OUT_CSV = "semantic_entropy_100.csv"
START_IDX = 80   # 0-indexed -> sample 81
COUNT = 50       # 50 samples -> 81 to 130

def run_single_answer(q: str, passages: list, answer: str):
    """Runs pipeline on an answer and extracts consistency and semantic entropy."""
    passages_used = top_passages(q, passages)
    if not passages_used and passages:
        passages_used = passages[:3]

    if len(answer.split()) < 5:
        claim_answer = rewrite_short_answer_to_claim(q, answer)
    else:
        claim_answer = answer

    # Split into facts & check
    facts = split_into_facts(claim_answer)
    results = []
    for f in facts:
        r = check_fact(f, passages_used, join_passages=True)
        label = r.get("label", "Not enough info")
        score = r.get("score", 0.0)
        entail_prob = r.get("entailment_prob", 0.0)

        # Web search fallback if needed
        if label != "Supported":
            try:
                query = make_search_query(f, q, passages_used)
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

    factscore = round(sum(results) / len(results), 2) if results else 0.0

    # Consistency & Semantic Entropy
    sampled_answers = []
    if factscore < 0.9:
        try:
            sampled_answers = sample_consistency_answers(q, passages_used, n=5)
            consistency = round(score_answers_consistency(sampled_answers), 2)
        except Exception:
            consistency = 1.0
    else:
        consistency = 1.0

    # Semantic entropy
    if sampled_answers:
        entropy_score, _ = compute_semantic_entropy_score(sampled_answers)
    elif factscore >= 0.9:
        entropy_score = 1.0
    else:
        entropy_score = 0.0

    # Signal
    try:
        sig = round(signal_score(q, answer), 2)
    except Exception:
        sig = 0.5

    # Decision before
    final_score = round(compute_final_score(factscore, consistency, sig), 2)
    decision_before = decide(factscore, consistency, sig)

    # Decision after using the fixed rule:
    # If doubtful (0.3 <= factscore <= 0.7) and unsure (entropy_score < 0.5) -> "I don't know"
    decision_after = apply_semantic_entropy_decision(decision_before, factscore, entropy_score)

    return {
        "fact_score": factscore,
        "old_consistency_score": consistency,
        "entropy_score": entropy_score,
        "final_score": final_score,
        "signal_score": sig,
        "decision_before": decision_before,
        "decision_after": decision_after
    }

def main():
    print("=" * 75)
    print("EVALUATING SEMANTIC ENTROPY ON 50 NEW SAMPLES (81 to 130)")
    print(f"Output: {OUT_CSV}")
    print("=" * 75)

    all_samples = load_samples(START_IDX + COUNT)
    samples = all_samples[START_IDX: START_IDX + COUNT]
    if len(samples) < COUNT:
        print(f"Error: expected {COUNT} samples, got {len(samples)}")
        return

    fieldnames = [
        "sample_id",
        "label",
        "old_consistency_score",
        "entropy_score",
        "fact_score",
        "final_score",
        "decision_before",
        "decision_after"
    ]

    # Check for existing completed samples if resuming
    completed_ids = set()
    if os.path.exists(OUT_CSV):
        try:
            with open(OUT_CSV, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                counts = {}
                for row in reader:
                    sid = int(row["sample_id"])
                    counts[sid] = counts.get(sid, 0) + 1
                for sid, c in counts.items():
                    if c >= 2:
                        completed_ids.add(sid)
        except Exception:
            pass

    mode = "a" if completed_ids else "w"
    f_out = open(OUT_CSV, mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    if not completed_ids:
        writer.writeheader()
        f_out.flush()
    else:
        print(f"Resuming: found {len(completed_ids)} completed samples.")

    t0 = time.time()

    for idx, s in enumerate(samples, START_IDX + 1):
        if idx in completed_ids:
            continue

        q = s["question"]
        k = s["knowledge"]
        passages = [k]

        # 1. Right Answer
        res_r = run_single_answer(q, passages, s["right_answer"])
        writer.writerow({
            "sample_id": idx,
            "label": "right",
            "old_consistency_score": res_r["old_consistency_score"],
            "entropy_score": res_r["entropy_score"],
            "fact_score": res_r["fact_score"],
            "final_score": res_r["final_score"],
            "decision_before": res_r["decision_before"],
            "decision_after": res_r["decision_after"]
        })

        # 2. Hallucinated Answer
        res_h = run_single_answer(q, passages, s["hallucinated_answer"])
        writer.writerow({
            "sample_id": idx,
            "label": "hallucinated",
            "old_consistency_score": res_h["old_consistency_score"],
            "entropy_score": res_h["entropy_score"],
            "fact_score": res_h["fact_score"],
            "final_score": res_h["final_score"],
            "decision_before": res_h["decision_before"],
            "decision_after": res_h["decision_after"]
        })
        f_out.flush()

        done_count = idx - START_IDX
        if done_count % 10 == 0 or done_count == COUNT:
            elapsed = time.time() - t0
            print(f"[{done_count:2d}/{COUNT}] Done through sample {idx}. Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}m)")

    f_out.close()
    total_time = time.time() - t0
    print(f"\nAll 50 samples completed in {total_time:.1f}s ({total_time/60:.1f} minutes).")

    # Run Analysis
    run_analysis()

def run_analysis():
    with open(OUT_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print("\n" + "=" * 75)
    print("ANALYSIS RESULTS (100 Answers / 50 Samples)")
    print("=" * 75)

    y_true = [1 if r["label"] == "right" else 0 for r in rows]
    cons_scores = [float(r["old_consistency_score"]) for r in rows]
    entropy_scores = [float(r["entropy_score"]) for r in rows]

    # Test A: AUROC
    auc_cons = roc_auc_score(y_true, cons_scores)
    auc_entropy = roc_auc_score(y_true, entropy_scores)

    print(f"TEST A: AUROC SEPARATION (0.5 = random, 1.0 = perfect)")
    print(f"  Old Consistency AUROC : {auc_cons:.4f}")
    print(f"  Semantic Entropy AUROC: {auc_entropy:.4f}")
    if auc_entropy > auc_cons:
        print(f"  -> Semantic Entropy separates answers better (+{auc_entropy - auc_cons:.4f})")
    else:
        print(f"  -> Old Consistency separates answers better (+{auc_cons - auc_entropy:.4f})")

    # Test B: BEFORE vs AFTER
    rights = [r for r in rows if r["label"] == "right"]
    hallus = [r for r in rows if r["label"] == "hallucinated"]

    # Before
    right_appr_before = sum(1 for r in rights if r["decision_before"] == "answer")
    hallu_appr_before = sum(1 for h in hallus if h["decision_before"] == "answer")
    right_idk_before = sum(1 for r in rights if r["decision_before"] != "answer")
    hallu_idk_before = sum(1 for h in hallus if h["decision_before"] != "answer")

    # After
    right_appr_after = sum(1 for r in rights if r["decision_after"] == "answer")
    hallu_appr_after = sum(1 for h in hallus if h["decision_after"] == "answer")
    right_idk_after = sum(1 for r in rights if r["decision_after"] != "answer")
    hallu_idk_after = sum(1 for h in hallus if h["decision_after"] != "answer")

    hallu_drop = hallu_appr_before - hallu_appr_after
    right_drop = right_appr_before - right_appr_after

    print("\nTEST B: BEFORE vs AFTER DECISION COMPARISON")
    print("-" * 75)
    print(f"{'Metric':<38} | {'BEFORE':<14} | {'AFTER':<14} | {'Change':<8}")
    print("-" * 75)
    print(f"{'Hallucinated answers approved':<38} | {hallu_appr_before:2d}/50 ({hallu_appr_before/50*100:5.1f}%) | {hallu_appr_after:2d}/50 ({hallu_appr_after/50*100:5.1f}%) | -{hallu_drop}")
    print(f"{'Correct answers approved':<38} | {right_appr_before:2d}/50 ({right_appr_before/50*100:5.1f}%) | {right_appr_after:2d}/50 ({right_appr_after/50*100:5.1f}%) | -{right_drop}")
    print(f"{'\"I don\'t know\" on right answers':<38} | {right_idk_before:2d}/50 ({right_idk_before/50*100:5.1f}%) | {right_idk_after:2d}/50 ({right_idk_after/50*100:5.1f}%) | +{right_drop}")
    print(f"{'\"I don\'t know\" on hallucinations':<38} | {hallu_idk_before:2d}/50 ({hallu_idk_before/50*100:5.1f}%) | {hallu_idk_after:2d}/50 ({hallu_idk_after/50*100:5.1f}%) | +{hallu_drop}")
    print("-" * 75)

    is_useful = (hallu_drop >= 5) and (right_drop < hallu_drop)
    print(f"Useful test: hallu drop >= 5 ({hallu_drop} >= 5? {hallu_drop >= 5}) AND right drop < hallu drop ({right_drop} < {hallu_drop}? {right_drop < hallu_drop})")
    print(f"VERDICT: {'USEFUL' if is_useful else 'NOT USEFUL'}")

if __name__ == "__main__":
    main()
