import os
os.environ["HF_HOME"] = r"D:\hf-cache"

import time
import csv
from app.data_loader import load_samples
import app.pipeline as pipeline

OUT_CSV = "big_test.csv"
N_SAMPLES = 150  # 150 samples -> 150 right answers, 150 hallucinated answers = 300 answers

def main():
    # Enforce switches
    pipeline.USE_PASSAGE_FILTER = False
    pipeline.USE_JUDGE = False

    print("=" * 70)
    print(f"BIG TEST: {N_SAMPLES} HaluEval samples (300 answers: 150 right, 150 hallucinated)")
    print(f"USE_PASSAGE_FILTER = {pipeline.USE_PASSAGE_FILTER}")
    print(f"USE_JUDGE          = {pipeline.USE_JUDGE}")
    print(f"Output file        = {OUT_CSV}")
    print("=" * 70)

    samples = load_samples(N_SAMPLES)
    if not samples:
        print("ERROR: Could not load samples from data/qa_data.json")
        return

    fieldnames = [
        "sample_idx", "question", "which_answer", "answer",
        "factscore", "consistency", "signal", "final_score", "decision"
    ]

    # Check for existing completed samples if resuming
    completed_samples = set()
    file_exists = os.path.exists(OUT_CSV)
    if file_exists:
        try:
            with open(OUT_CSV, "r", encoding="utf-8") as f_in:
                reader = csv.DictReader(f_in)
                counts_by_idx = {}
                for row in reader:
                    idx = int(row["sample_idx"])
                    counts_by_idx[idx] = counts_by_idx.get(idx, 0) + 1
                # A sample is fully completed if it has both 'right' and 'hallucinated' (2 rows)
                for idx, c in counts_by_idx.items():
                    if c >= 2:
                        completed_samples.add(idx)
        except Exception as e:
            print(f"Warning reading existing {OUT_CSV}: {e}")

    mode = "a" if completed_samples else "w"
    f_out = open(OUT_CSV, mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    if not completed_samples:
        writer.writeheader()
        f_out.flush()
    else:
        print(f"Resuming: found {len(completed_samples)} previously completed samples in {OUT_CSV}.")

    correct_answers = 0
    hallucinations_caught = 0
    start_time = time.time()

    # If resuming, load existing stats from completed samples
    if completed_samples:
        with open(OUT_CSV, "r", encoding="utf-8") as f_in:
            reader = csv.DictReader(f_in)
            for row in reader:
                idx = int(row["sample_idx"])
                if idx in completed_samples:
                    if row["which_answer"] == "right" and row["decision"] == "answer":
                        correct_answers += 1
                    elif row["which_answer"] == "hallucinated" and row["decision"] != "answer":
                        hallucinations_caught += 1

    for idx, s in enumerate(samples, 1):
        if idx in completed_samples:
            continue

        q = s["question"]
        r_ans = s["right_answer"]
        h_ans = s["hallucinated_answer"]

        # 1. Right Answer evaluation
        res_r = pipeline.run_sample(s, use_hallucinated=False)
        if res_r["decision"] == "answer":
            correct_answers += 1

        writer.writerow({
            "sample_idx": idx,
            "question": q,
            "which_answer": "right",
            "answer": r_ans,
            "factscore": res_r["factscore"],
            "consistency": res_r["consistency"],
            "signal": res_r["signal"],
            "final_score": res_r["final_score"],
            "decision": res_r["decision"]
        })

        # 2. Hallucinated Answer evaluation
        res_h = pipeline.run_sample(s, use_hallucinated=True)
        if res_h["decision"] != "answer":
            hallucinations_caught += 1

        writer.writerow({
            "sample_idx": idx,
            "question": q,
            "which_answer": "hallucinated",
            "answer": h_ans,
            "factscore": res_h["factscore"],
            "consistency": res_h["consistency"],
            "signal": res_h["signal"],
            "final_score": res_h["final_score"],
            "decision": res_h["decision"]
        })
        f_out.flush()

        # Print update & save checkpoint notice every 25 samples
        if idx % 25 == 0 or idx == N_SAMPLES:
            elapsed = time.time() - start_time
            print(
                f"[CHECKPOINT {idx:3d}/{N_SAMPLES}] "
                f"Correct approved: {correct_answers}/{idx} ({correct_answers/idx*100:.1f}%) | "
                f"Hallucinations caught: {hallucinations_caught}/{idx} ({hallucinations_caught/idx*100:.1f}%) | "
                f"Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)"
            )

    f_out.close()
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS (300 Answers / 150 Samples)")
    print("=" * 70)
    print(f"Correct answers approved : {correct_answers}/{N_SAMPLES} ({correct_answers/N_SAMPLES*100:.1f}%)")
    print(f"Hallucinations caught    : {hallucinations_caught}/{N_SAMPLES} ({hallucinations_caught/N_SAMPLES*100:.1f}%)")
    print(f"Total time               : {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Saved to                 : {OUT_CSV}")
    print("=" * 70)

if __name__ == "__main__":
    main()
