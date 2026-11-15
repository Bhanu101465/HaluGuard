import os
import csv
import argparse

# Keep Hugging Face cache on D drive
os.environ["HF_HOME"] = r"D:\hf-cache"

from app.data_loader import load_samples
import app.pipeline as pipeline

def main():
    parser = argparse.ArgumentParser(description="Evaluate HaluGuard on HaluEval QA samples.")
    parser.add_argument("--start", type=int, default=0, help="Starting sample index (default: 0)")
    parser.add_argument("-n", "--samples", type=int, default=50, help="Number of samples (default: 50)")
    parser.add_argument("-o", "--output", type=str, default="results.csv", help="Output CSV file path")
    parser.add_argument("--join-passages", action="store_true", default=True, help="Join passages into single text for NLI")
    parser.add_argument("--soft-score", action="store_true", default=True, help="Compute factscore as average of entailment probabilities")
    parser.add_argument("--claim-rewrite", action="store_true", default=True, help="Use LLM to rewrite short answers into claims")
    args = parser.parse_args()

    # Turn off non-claim LLM modules for this test
    pipeline.USE_NEED_CHECK = False
    pipeline.USE_COVE = False
    pipeline.USE_CONSISTENCY = False
    pipeline.USE_SIGNALS = False
    pipeline.USE_JOIN_PASSAGES = args.join_passages
    pipeline.USE_SOFT_FACTSCORE = args.soft_score
    pipeline.USE_CLAIM_REWRITE = args.claim_rewrite

    all_samples = load_samples(n=args.start + args.samples)
    samples = all_samples[args.start : args.start + args.samples]
    if not samples:
        print("Error: No samples loaded from data/qa_data.json")
        return

    print("=" * 70)
    print(f"Evaluating HaluGuard on samples {args.start + 1} to {args.start + len(samples)} ({len(samples)} samples)")
    print(f"Configuration: JoinPassages={pipeline.USE_JOIN_PASSAGES}, SoftFactscore={pipeline.USE_SOFT_FACTSCORE}, ClaimRewrite={pipeline.USE_CLAIM_REWRITE}")
    print(f"Tie Threshold: Gap < 0.05 is counted as a TIE")
    print(f"Toggles: NLI=ON, WebSearch={pipeline.USE_WEB_SEARCH}, LLM modules=OFF (except claim rewrite)")
    print("=" * 70)

    csv_rows = []
    right_scores = []
    hallu_scores = []
    right_fact_counts = []
    hallu_fact_counts = []

    for i, s in enumerate(samples, 1):
        global_idx = args.start + i
        q = s["question"]
        passages = [s["knowledge"]]
        r_ans = s["right_answer"]
        h_ans = s["hallucinated_answer"]

        # Run with right answer
        res_right = pipeline.run(q, passages, given_answer=r_ans)
        # Run with hallucinated answer
        res_hallu = pipeline.run(q, passages, given_answer=h_ans)

        right_score = res_right["factscore"]
        hallu_score = res_hallu["factscore"]
        r_facts = len(res_right["facts"])
        h_facts = len(res_hallu["facts"])

        right_scores.append(right_score)
        hallu_scores.append(hallu_score)
        right_fact_counts.append(r_facts)
        hallu_fact_counts.append(h_facts)

        gap = right_score - hallu_score
        if abs(gap) < 0.05:
            outcome = "TIE"
            diff_sym = "="
        elif gap >= 0.05:
            outcome = "RIGHT_WIN"
            diff_sym = ">"
        else:
            outcome = "WRONG_WIN"
            diff_sym = "<"

        csv_rows.append({
            "sample_num": global_idx,
            "question": q,
            "which_answer": "right",
            "factscore": right_score,
            "num_facts": r_facts,
            "decision": res_right["decision"],
        })
        csv_rows.append({
            "sample_num": global_idx,
            "question": q,
            "which_answer": "hallucinated",
            "factscore": hallu_score,
            "num_facts": h_facts,
            "decision": res_hallu["decision"],
        })

        print(f"[{global_idx:03d}] Right: {right_score:.2f} {diff_sym} Hallu: {hallu_score:.2f} ({outcome:<9}) | facts: ({r_facts} vs {h_facts}) | Q: {q[:30]}...")

    # Save to CSV
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_num", "question", "which_answer", "factscore", "num_facts", "decision"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nSaved results to {args.output}")

    # Compute detailed statistics with tie threshold of 0.05
    right_wins = sum(1 for r, h in zip(right_scores, hallu_scores) if (r - h) >= 0.05)
    ties = sum(1 for r, h in zip(right_scores, hallu_scores) if abs(r - h) < 0.05)
    wrong_wins = sum(1 for r, h in zip(right_scores, hallu_scores) if (h - r) >= 0.05)

    avg_right = sum(right_scores) / len(right_scores)
    avg_hallu = sum(hallu_scores) / len(hallu_scores)
    avg_gap = (sum(right_scores) - sum(hallu_scores)) / len(samples)

    avg_r_facts = sum(right_fact_counts) / len(right_fact_counts)
    avg_h_facts = sum(hallu_fact_counts) / len(hallu_fact_counts)
    total_avg_facts = (sum(right_fact_counts) + sum(hallu_fact_counts)) / (2 * len(samples))

    print("\n" + "=" * 70)
    print(f"DETAILED EVALUATION REPORT: Samples {args.start + 1} to {args.start + len(samples)}")
    print("=" * 70)
    print(f"Total samples evaluated:             {len(samples)}")
    print(f"Right Answer Wins (gap >= +0.05):    {right_wins}/{len(samples)} ({right_wins/len(samples)*100:.1f}%)")
    print(f"Ties (|gap| < 0.05):                 {ties}/{len(samples)} ({ties/len(samples)*100:.1f}%)")
    print(f"Wrong Answer Wins (gap <= -0.05):    {wrong_wins}/{len(samples)} ({wrong_wins/len(samples)*100:.1f}%)")
    print("-" * 70)
    print(f"Average FactScore (Right Answers):   {avg_right:.2f}")
    print(f"Average FactScore (Hallucinated):    {avg_hallu:.2f}")
    print(f"Average Gap (Right - Hallu):         {avg_gap:+.2f}")
    print("-" * 70)
    print(f"Average facts per Right Answer:      {avg_r_facts:.2f}")
    print(f"Average facts per Hallu Answer:      {avg_h_facts:.2f}")
    print(f"Overall average facts per Answer:    {total_avg_facts:.2f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
