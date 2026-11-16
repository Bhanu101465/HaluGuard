# compare_mixes.py
# Reads scores_100.csv only (no model calls) and compares 3 weight mixes.

import csv

SCORES_FILE = "scores_100.csv"

def evaluate_mix(rows, w_fact, w_cons, w_sig, cutoff):
    right_approved = 0
    hallu_caught = 0
    total_right = 0
    total_hallu = 0

    for r in rows:
        label = r["label"].strip().lower()
        fact = float(r["fact_check_score"])
        cons = float(r["consistency_score"])
        sig = float(r["signal_score"])

        score = w_fact * fact + w_cons * cons + w_sig * sig

        if label == "right":
            total_right += 1
            if score >= cutoff:
                right_approved += 1
        elif label == "hallucinated":
            total_hallu += 1
            if score < cutoff:
                hallu_caught += 1

    total_correct = right_approved + hallu_caught
    total_samples = total_right + total_hallu

    return {
        "right_approved": right_approved,
        "total_right": total_right,
        "right_pct": right_approved / total_right * 100,
        "hallu_caught": hallu_caught,
        "total_hallu": total_hallu,
        "hallu_pct": hallu_caught / total_hallu * 100,
        "total_correct": total_correct,
        "total_samples": total_samples,
        "accuracy_pct": total_correct / total_samples * 100
    }

def main():
    with open(SCORES_FILE, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print("=" * 80)
    print(f"COMPARING 3 WEIGHT MIXES ON {SCORES_FILE} (100 Answers / 50 Samples)")
    print("=" * 80)

    mixes = [
        ("Current", 0.50, 0.25, 0.25, 0.60),
        ("Mix A",   0.40, 0.30, 0.30, 0.45),
        ("Mix B",   0.70, 0.10, 0.20, 0.30)
    ]

    print(f"{'Mix':<10} | {'Weights (Fact / Cons / Sig)':<28} | {'Cutoff':<6} | {'Right Appr %':<14} | {'Hallu Caught %':<14} | {'Accuracy %':<12}")
    print("-" * 92)

    for name, wf, wc, ws, cutoff in mixes:
        res = evaluate_mix(rows, wf, wc, ws, cutoff)
        w_str = f"{wf:.2f} / {wc:.2f} / {ws:.2f}"
        r_str = f"{res['right_approved']}/{res['total_right']} ({res['right_pct']:.1f}%)"
        h_str = f"{res['hallu_caught']}/{res['total_hallu']} ({res['hallu_pct']:.1f}%)"
        acc_str = f"{res['total_correct']}/{res['total_samples']} ({res['accuracy_pct']:.1f}%)"
        print(f"{name:<10} | {w_str:<28} | {cutoff:<6.2f} | {r_str:<14} | {h_str:<14} | {acc_str:<12}")

    print("=" * 92)

if __name__ == "__main__":
    main()
