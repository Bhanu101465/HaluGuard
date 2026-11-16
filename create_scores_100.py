import csv

def make_scores_100():
    in_path = "D:/Hallucination/big_test.csv"
    out_path = "D:/Hallucination/scores_100.csv"

    rows = list(csv.DictReader(open(in_path, encoding="utf-8")))
    # Select samples 31 to 80 (50 new samples = 100 answers)
    test_rows = [r for r in rows if 31 <= int(r["sample_idx"]) <= 80]

    fieldnames = [
        "sample_id",
        "label",
        "fact_check_score",
        "consistency_score",
        "signal_score"
    ]

    out_rows = []
    for r in test_rows:
        label = "right" if r["which_answer"].lower() == "right" else "hallucinated"
        out_rows.append({
            "sample_id": r["sample_idx"],
            "label": label,
            "fact_check_score": float(r["factscore"]),
            "consistency_score": float(r["consistency"]),
            "signal_score": float(r["signal"])
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Saved {len(out_rows)} answers to {out_path} (samples 31 to 80)")

if __name__ == "__main__":
    make_scores_100()
