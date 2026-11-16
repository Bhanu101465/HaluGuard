# tune_weights.py
# Reads scores_30.csv and tests different score weight combinations.
# Does NOT make any LLM or model calls.

import csv

SCORES_FILE = "scores_30.csv"

def load_data():
    with open(SCORES_FILE, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows

def evaluate(rows, w_fact, w_cons, w_sig, cutoff):
    """
    Evaluates a weight mix at a given cutoff.
    Returns: (right_approved, hallu_caught)
    """
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

    return right_approved, hallu_caught, total_right, total_hallu

def main():
    rows = load_data()
    n_total = len(rows)
    print("=" * 72)
    print(f"TUNING SCORE WEIGHTS (using {SCORES_FILE}, {n_total} answers)")
    print("=" * 72)

    # 1. Evaluate Current Pipeline Weights
    cur_wf, cur_wc, cur_ws = 0.50, 0.25, 0.25
    cur_cutoff = 0.60
    c_r, c_h, t_r, t_h = evaluate(rows, cur_wf, cur_wc, cur_ws, cur_cutoff)
    c_total = c_r + c_h
    print("\nCURRENT PIPELINE WEIGHTS:")
    print(f"  Weights : Fact={cur_wf:.2f}, Consistency={cur_wc:.2f}, Signal={cur_ws:.2f}")
    print(f"  Cutoff  : {cur_cutoff:.2f}")
    print(f"  Correct answers approved : {c_r}/{t_r} ({c_r/t_r*100:.1f}%)")
    print(f"  Hallucinations caught    : {c_h}/{t_h} ({c_h/t_h*100:.1f}%)")
    print(f"  Total correct decisions  : {c_total}/{n_total} ({c_total/n_total*100:.1f}%)")
    print("-" * 72)

    # 2. Grid Search: Steps of 0.1 (summing to 1.0)
    # Cutoff candidate thresholds: 0.10 to 0.90 in steps of 0.05
    cutoffs = [round(c * 0.05, 2) for c in range(2, 19)]

    mixes = []
    for i in range(11):
        for j in range(11 - i):
            k = 10 - i - j
            wf = round(i * 0.1, 1)
            wc = round(j * 0.1, 1)
            ws = round(k * 0.1, 1)
            mixes.append((wf, wc, ws))

    results = []
    for wf, wc, ws in mixes:
        best_cutoff = None
        best_total = -1
        best_r = 0
        best_h = 0
        best_diff = 999

        for cutoff in cutoffs:
            r_appr, h_caught, t_r, t_h = evaluate(rows, wf, wc, ws, cutoff)
            total = r_appr + h_caught
            diff = abs(r_appr - h_caught)

            # Optimization criterion:
            # 1. Highest total correct decisions (overall accuracy)
            # 2. Most balanced between right approved and hallucinations caught
            if total > best_total or (total == best_total and diff < best_diff):
                best_total = total
                best_diff = diff
                best_cutoff = cutoff
                best_r = r_appr
                best_h = h_caught

        results.append({
            "w_fact": wf,
            "w_cons": wc,
            "w_sig": ws,
            "cutoff": best_cutoff,
            "r_appr": best_r,
            "r_pct": best_r / t_r * 100,
            "h_caught": best_h,
            "h_pct": best_h / t_h * 100,
            "total": best_total,
            "acc": best_total / n_total * 100,
            "diff": best_diff
        })

    # Sort mixes: highest total correct first, then closest balance
    results.sort(key=lambda x: (x["total"], -x["diff"], x["r_appr"]), reverse=True)

    # 3. Print Top 5 Mixes
    print("\nTOP 5 WEIGHT MIXES (Steps of 0.1, Best Cutoff for each):")
    print(f"{'Rank':<5} | {'Weights (Fact / Cons / Sig)':<28} | {'Cutoff':<6} | {'Right Appr %':<14} | {'Hallu Caught %':<14} | {'Total Score':<12}")
    print("-" * 90)

    for rank in range(1, 6):
        m = results[rank - 1]
        w_str = f"{m['w_fact']:.1f} / {m['w_cons']:.1f} / {m['w_sig']:.1f}"
        r_str = f"{m['r_appr']}/{t_r} ({m['r_pct']:.1f}%)"
        h_str = f"{m['h_caught']}/{t_h} ({m['h_pct']:.1f}%)"
        tot_str = f"{m['total']}/{n_total} ({m['acc']:.1f}%)"
        print(f"#{rank:<4} | {w_str:<28} | {m['cutoff']:<6.2f} | {r_str:<14} | {h_str:<14} | {tot_str:<12}")

    print("=" * 90)

if __name__ == "__main__":
    main()
