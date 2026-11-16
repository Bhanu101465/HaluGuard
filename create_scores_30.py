import csv

def make_scores_30():
    in_path = "D:/Hallucination/filter_off.csv"
    out_path = "D:/Hallucination/scores_30.csv"

    rows = list(csv.DictReader(open(in_path, encoding="utf-8")))
    print(f"Loaded {len(rows)} rows from {in_path}")

    fieldnames = [
        "sample_id",
        "label",
        "question",
        "answer",
        "fact_check_score",
        "consistency_score",
        "cove_score",
        "web_check_score",
        "signal_score",
        "final_score"
    ]

    out_rows = []
    for r in rows:
        label = "right" if r["which_answer"].lower() == "right" else "hallucinated"
        out_rows.append({
            "sample_id": r["sample_idx"],
            "label": label,
            "question": r["question"],
            "answer": r["answer"],
            "fact_check_score": float(r["factscore"]),
            "consistency_score": float(r["consistency"]),
            "cove_score": 0.0,       # CoVe is a text rewriter in HaluGuard, does not produce a numeric score
            "web_check_score": 0.0,  # Web search provides text snippets to NLI fact check, not a standalone score
            "signal_score": float(r["signal"]),
            "final_score": float(r["final_score"])
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Successfully saved {len(out_rows)} rows to {out_path}")

if __name__ == "__main__":
    make_scores_30()
