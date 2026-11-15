import os
os.environ["HF_HOME"] = r"D:\hf-cache"

import time
import csv
from app.data_loader import load_samples
import app.pipeline as pipeline

def run_evaluation(filter_enabled: bool, out_csv: str, n_samples: int = 30):
    pipeline.USE_PASSAGE_FILTER = filter_enabled
    mode_name = "ON" if filter_enabled else "OFF"
    print("=" * 70)
    print(f"RUNNING EVALUATION: USE_PASSAGE_FILTER = {mode_name} ({n_samples} samples)")
    print(f"Saving to {out_csv}")
    print("=" * 70)

    samples = load_samples(n_samples)
    if not samples:
        raise ValueError("No samples found in data/qa_data.json")

    fieldnames = [
        "sample_idx", "question", "which_answer", "answer",
        "factscore", "consistency", "signal", "final_score", "decision",
        "passages_input", "passages_kept", "passages_dropped"
    ]

    f_out = open(out_csv, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()
    f_out.flush()

    correct_answers = 0
    hallucinations_caught = 0
    total_passages_kept = 0
    total_passages_dropped = 0
    total_runs = 0

    start_time = time.time()

    for idx, s in enumerate(samples, 1):
        q = s["question"]
        k = s["knowledge"]
        r_ans = s["right_answer"]
        h_ans = s["hallucinated_answer"]

        passages = [k]
        n_input = len(passages)

        # 1. Right Answer evaluation
        res_r = pipeline.run(q, passages, given_answer=r_ans)
        n_kept_r = len(res_r["passages_used"])
        n_dropped_r = n_input - n_kept_r
        is_r_correct = (res_r["decision"] == "answer")
        if is_r_correct:
            correct_answers += 1

        row_r = {
            "sample_idx": idx,
            "question": q,
            "which_answer": "right",
            "answer": r_ans,
            "factscore": res_r["factscore"],
            "consistency": res_r["consistency"],
            "signal": res_r["signal"],
            "final_score": res_r["final_score"],
            "decision": res_r["decision"],
            "passages_input": n_input,
            "passages_kept": n_kept_r,
            "passages_dropped": n_dropped_r
        }
        writer.writerow(row_r)

        # 2. Hallucinated Answer evaluation
        res_h = pipeline.run(q, passages, given_answer=h_ans)
        n_kept_h = len(res_h["passages_used"])
        n_dropped_h = n_input - n_kept_h
        is_h_caught = (res_h["decision"] != "answer")
        if is_h_caught:
            hallucinations_caught += 1

        row_h = {
            "sample_idx": idx,
            "question": q,
            "which_answer": "hallucinated",
            "answer": h_ans,
            "factscore": res_h["factscore"],
            "consistency": res_h["consistency"],
            "signal": res_h["signal"],
            "final_score": res_h["final_score"],
            "decision": res_h["decision"],
            "passages_input": n_input,
            "passages_kept": n_kept_h,
            "passages_dropped": n_dropped_h
        }
        writer.writerow(row_h)
        f_out.flush()

        total_passages_kept += (n_kept_r + n_kept_h)
        total_passages_dropped += (n_dropped_r + n_dropped_h)
        total_runs += 2

        print(f"[{idx:2d}/{n_samples}] Right: {res_r['decision']} ({res_r['final_score']}) | Hallu: {res_h['decision']} ({res_h['final_score']})")

    f_out.close()
    elapsed = time.time() - start_time

    avg_kept = total_passages_kept / total_runs
    avg_dropped = total_passages_dropped / total_runs

    summary = {
        "mode": f"USE_PASSAGE_FILTER = {mode_name}",
        "correct_answers": f"{correct_answers}/{n_samples} ({correct_answers/n_samples*100:.1f}%)",
        "hallucinations_caught": f"{hallucinations_caught}/{n_samples} ({hallucinations_caught/n_samples*100:.1f}%)",
        "avg_passages_kept": f"{avg_kept:.2f}",
        "avg_passages_dropped": f"{avg_dropped:.2f}",
        "time_taken": f"{elapsed:.1f}s ({elapsed/60:.1f} min)",
        "raw": {
            "correct_answers": correct_answers,
            "hallucinations_caught": hallucinations_caught,
            "avg_kept": avg_kept,
            "avg_dropped": avg_dropped,
            "time_taken": elapsed
        }
    }
    return summary

if __name__ == "__main__":
    print("Starting Filter OFF run...")
    summary_off = run_evaluation(filter_enabled=False, out_csv="filter_off.csv", n_samples=30)
    print("\nStarting Filter ON run...")
    summary_on = run_evaluation(filter_enabled=True, out_csv="filter_on.csv", n_samples=30)

    print("\n" + "=" * 70)
    print("COMPARISON RESULTS (30 Samples)")
    print("=" * 70)
    print(f"{'Metric':<32} | {'Filter OFF':<18} | {'Filter ON':<18}")
    print("-" * 70)
    print(f"{'Correct answers approved':<32} | {summary_off['correct_answers']:<18} | {summary_on['correct_answers']:<18}")
    print(f"{'Hallucinations caught':<32} | {summary_off['hallucinations_caught']:<18} | {summary_on['hallucinations_caught']:<18}")
    print(f"{'Avg passages kept':<32} | {summary_off['avg_passages_kept']:<18} | {summary_on['avg_passages_kept']:<18}")
    print(f"{'Avg passages dropped':<32} | {summary_off['avg_passages_dropped']:<18} | {summary_on['avg_passages_dropped']:<18}")
    print(f"{'Time taken':<32} | {summary_off['time_taken']:<18} | {summary_on['time_taken']:<18}")
    print("=" * 70)
