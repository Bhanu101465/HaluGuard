import os
import csv
import re

os.environ["HF_HOME"] = r"D:\hf-cache"

from app.data_loader import load_samples
import app.pipeline as pipeline

pipeline.USE_NEED_CHECK = False
pipeline.USE_COVE = False
pipeline.USE_CONSISTENCY = False
pipeline.USE_SIGNALS = False
pipeline.USE_JOIN_PASSAGES = True
pipeline.USE_SOFT_FACTSCORE = True

samples = load_samples(50)
losses = []

STOPWORDS = {
    'a', 'an', 'the', 'is', 'was', 'were', 'and', 'or', 'in', 'on', 'of',
    'to', 'for', 'with', 'that', 'this', 'by', 'at', 'it', 'as', 'be', 'from'
}

for i, s in enumerate(samples, 1):
    q = s['question']
    p = s['knowledge']
    r_ans = s['right_answer']
    h_ans = s['hallucinated_answer']

    res_r = pipeline.run(q, [p], given_answer=r_ans)
    res_h = pipeline.run(q, [p], given_answer=h_ans)

    r_score = res_r['factscore']
    h_score = res_h['factscore']

    if r_score < h_score:
        p_words = set(re.findall(r'\w+', p.lower()))
        h_words = set(re.findall(r'\w+', h_ans.lower()))
        overlap = sorted(list((p_words & h_words) - STOPWORDS))

        losses.append({
            'sample_num': i,
            'question': q,
            'passage': p,
            'right_answer': r_ans,
            'right_score': r_score,
            'right_facts': "; ".join([f"{f['fact']} (p={f['entailment_prob']:.2f})" for f in res_r['facts']]),
            'hallu_answer': h_ans,
            'hallu_score': h_score,
            'hallu_facts': "; ".join([f"{f['fact']} (p={f['entailment_prob']:.2f})" for f in res_h['facts']]),
            'overlap_words': ", ".join(overlap),
        })

print(f"Total losses found: {len(losses)}")

# Save to CSV
with open("losses.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "sample_num", "question", "passage", "right_answer", "right_score",
        "right_facts", "hallu_answer", "hallu_score", "hallu_facts", "overlap_words"
    ])
    writer.writeheader()
    writer.writerows(losses)

for l in losses:
    print("=" * 65)
    print(f"LOSS #{l['sample_num']}: Right={l['right_score']:.2f} < Hallu={l['hallu_score']:.2f}")
    print(f"Question:        {l['question']}")
    print(f"Passage:         {l['passage']}")
    print(f"Right Facts:     {l['right_facts']}")
    print(f"Hallu Facts:     {l['hallu_facts']}")
    print(f"Overlapping:     {l['overlap_words']}")
