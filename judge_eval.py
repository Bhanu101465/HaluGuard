"""
judge_eval.py  —  Test the LLM-as-judge on HaluEval samples.

For each sample it runs the judge twice:
  - once with the right answer
  - once with the hallucinated answer

Tracks:
  - Right answers approved (SUPPORTED)
  - Wrong answers rejected (NOT SUPPORTED)
  - Both correct (right=SUPPORTED and hallu=NOT SUPPORTED)
  - Rejection breakdown for right answers:
      * llm_said_not_supported
      * no_proof_given
      * proof_not_in_passage

Usage:
  python judge_eval.py --start 30 -n 120 -o judge_150.csv
"""

import os
os.environ["HF_HOME"] = r"D:\hf-cache"

import csv
import argparse
from app.data_loader import load_samples
from app.judge import judge

def run_eval(n: int, start: int, out_path: str):
    all_samples = load_samples(start + n)
    samples = all_samples[start: start + n]

    if not samples:
        print(f"ERROR: no samples found for start={start}, n={n}")
        return

    print("=" * 70)
    print(f"JUDGE EVAL  |  samples {start+1} to {start+n}  |  n={n}")
    print(f"Output -> {out_path}")
    print("=" * 70)

    fieldnames = [
        "idx", "question", "which_answer", "answer",
        "verdict", "proof", "verified", "score", "reject_reason",
    ]

    rows = []

    # Counters
    right_supported   = 0
    hallu_unsupported = 0
    right_rejections  = {
        "llm_said_not_supported": 0,
        "no_proof_given": 0,
        "proof_not_in_passage": 0,
    }

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, s in enumerate(samples, 1):
            idx      = start + i
            q        = s["question"]
            passage  = s["knowledge"]
            r_ans    = s["right_answer"]
            h_ans    = s["hallucinated_answer"]

            print(f"[{i:3d}/{n}] Q ({idx}): {q[:60]}")

            # 1. Judge right answer
            r = judge(q, r_ans, [passage])
            v_r = r["verdict"]
            p_r = r["proof"][:55] if r["proof"] else ""
            rr_r = r["reject_reason"]
            print(f"       RIGHT | {v_r:<13s} | verified={r['verified']} | reason={rr_r:<22s} | proof={p_r}")

            if v_r == "SUPPORTED":
                right_supported += 1
            else:
                reason = rr_r if rr_r in right_rejections else "other"
                right_rejections[reason] = right_rejections.get(reason, 0) + 1

            row_r = {
                "idx": idx, "question": q, "which_answer": "right",
                "answer": r_ans,
                "verdict": v_r, "proof": r["proof"],
                "verified": r["verified"], "score": r["score"],
                "reject_reason": rr_r,
            }
            rows.append(row_r)
            writer.writerow(row_r)

            # 2. Judge hallucinated answer
            h = judge(q, h_ans, [passage])
            v_h = h["verdict"]
            p_h = h["proof"][:55] if h["proof"] else ""
            rr_h = h["reject_reason"]
            print(f"       HALLU | {v_h:<13s} | verified={h['verified']} | reason={rr_h:<22s} | proof={p_h}")

            if v_h == "NOT SUPPORTED":
                hallu_unsupported += 1

            row_h = {
                "idx": idx, "question": q, "which_answer": "hallucinated",
                "answer": h_ans,
                "verdict": v_h, "proof": h["proof"],
                "verified": h["verified"], "score": h["score"],
                "reject_reason": rr_h,
            }
            rows.append(row_h)
            writer.writerow(row_h)

            f.flush()

    # Accurate both-correct count
    both_correct = 0
    for j in range(0, len(rows), 2):
        if rows[j]["verdict"] == "SUPPORTED" and rows[j+1]["verdict"] == "NOT SUPPORTED":
            both_correct += 1

    print()
    print("=" * 70)
    print(f"JUDGE RESULTS ({n} questions, samples {start+1} to {start+n})")
    print("=" * 70)
    print(f"1. Right answers approved (SUPPORTED)     : {right_supported:3d} / {n} ({100*right_supported/n:.1f}%)")
    print(f"2. Wrong answers rejected (NOT SUPPORTED) : {hallu_unsupported:3d} / {n} ({100*hallu_unsupported/n:.1f}%)")
    print(f"3. Questions where both are correct       : {both_correct:3d} / {n} ({100*both_correct/n:.1f}%)")
    print()
    print("4. Why right answers were rejected:")
    right_rejected = n - right_supported
    print(f"   Total right answers rejected: {right_rejected} / {n}")
    for reason, count in right_rejections.items():
        pct = (100 * count / right_rejected) if right_rejected > 0 else 0
        print(f"   - {reason:<24}: {count:3d} ({pct:.1f}% of rejected)")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--samples", type=int, default=120, help="Number of samples (default: 120)")
    parser.add_argument("--start", type=int, default=30, help="Start sample index (default: 30)")
    parser.add_argument("-o", "--output", type=str, default="judge_150.csv", help="Output CSV file path")
    args = parser.parse_args()
    run_eval(args.samples, args.start, args.output)
