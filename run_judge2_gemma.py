"""
run_judge2_gemma.py

Run the second judge using `google/gemma-3-27b-it` on the same 100 questions and answers
from `generative_100_judge.csv`.
No pipeline rerun — only 100 judge calls.
"""

import os
os.environ["HF_HOME"] = r"D:\hf-cache"

import csv
import sys
from app.data_loader import load_samples
from app.llm import _get_client
from app.judge import PROMPT_TEMPLATE

MODEL_NAME = "google/gemma-3-27b-it"

client, _ = _get_client()

all_samples = load_samples(300)
sample_by_idx = {i + 1: s for i, s in enumerate(all_samples)}

in_csv = "generative_100_judge.csv"
out_csv = "generative_100_judge2.csv"

rows = list(csv.DictReader(open(in_csv, encoding="utf-8")))
print(f"Loaded {len(rows)} rows from {in_csv}")
print(f"Running judge with model: {MODEL_NAME}")

fieldnames = list(rows[0].keys()) + [
    "judge2_verdict", "judge2_score", "judge2_proof", "judge2_verified", "judge2_reject_reason"
]

f_out = open(out_csv, "w", newline="", encoding="utf-8")
writer = csv.DictWriter(f_out, fieldnames=fieldnames)
writer.writeheader()
f_out.flush()

for i, r in enumerate(rows, 1):
    idx = int(r["idx"])
    q = r["question"]
    ans = r["final_answer"]
    sample = sample_by_idx[idx]
    passage_text = sample["knowledge"]

    prompt = PROMPT_TEMPLATE.format(
        passage=passage_text,
        question=q,
        answer=ans,
    )

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=250,
        )
        raw = resp.choices[0].message.content or ""
        raw = raw.strip()
    except Exception as e:
        raw = f"NOT SUPPORTED\nerror: {e}"

    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    # Parse verdict
    first_line = lines[0].upper() if lines else ""
    if "NOT SUPPORTED" in first_line:
        raw_verdict = "NOT SUPPORTED"
    elif "SUPPORTED" in first_line:
        raw_verdict = "SUPPORTED"
    else:
        raw_verdict = "NOT SUPPORTED"

    # Parse proof
    raw_proof = lines[1] if len(lines) > 1 else ""
    proof = raw_proof.strip().strip("\"'").strip()
    if proof.lower().startswith("proof:"):
        proof = proof[6:].strip().strip("\"'").strip()
    if proof.lower().startswith("sentence:"):
        proof = proof[9:].strip().strip("\"'").strip()
    if proof.lower() in ("none", "none.", "n/a", "no", "no proof", "not found"):
        proof = ""

    # Exact copy verification
    verified = False
    if proof:
        norm_proof = " ".join(proof.lower().split())
        norm_passage = " ".join(passage_text.lower().split())
        if norm_proof in norm_passage:
            verified = True
        else:
            p_strip = norm_proof.strip(".,;:!?\"'")
            if len(p_strip) >= 10 and p_strip in norm_passage:
                verified = True

    # Final verdict
    if raw_verdict != "SUPPORTED":
        verdict = "NOT SUPPORTED"
        reject_reason = "llm_said_not_supported"
        if not verified:
            proof = ""
    elif not proof:
        verdict = "NOT SUPPORTED"
        reject_reason = "no_proof_given"
    elif not verified:
        verdict = "NOT SUPPORTED"
        reject_reason = "proof_not_in_passage"
        proof = ""
    else:
        verdict = "SUPPORTED"
        reject_reason = ""

    score = 1.0 if verdict == "SUPPORTED" else 0.0

    r_out = dict(r)
    r_out["judge2_verdict"] = verdict
    r_out["judge2_score"] = score
    r_out["judge2_proof"] = proof
    r_out["judge2_verified"] = verified
    r_out["judge2_reject_reason"] = reject_reason

    writer.writerow(r_out)
    f_out.flush()

    p_disp = proof[:50] if proof else "none"
    print(f"[{i:3d}/100] Q ({idx}): {verdict:<13s} | verified={verified} | proof={p_disp}")

f_out.close()
print(f"\nDone! Saved all 100 to {out_csv}")
