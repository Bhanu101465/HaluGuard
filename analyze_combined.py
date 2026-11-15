import csv
import json

# 1. Load NLI scores from results.csv (1-50) and results_51_150.csv (51-150)
nli_data = {}  # sample_idx -> {'right': float, 'hallucinated': float, 'q': str}

rows50 = list(csv.DictReader(open('results.csv', encoding='utf-8')))
for i in range(0, len(rows50), 2):
    idx = (i // 2) + 1
    nli_data[idx] = {
        'q': rows50[i]['question'],
        'right': float(rows50[i]['factscore']),
        'hallucinated': float(rows50[i+1]['factscore']),
    }

rows100 = list(csv.DictReader(open('results_51_150.csv', encoding='utf-8')))
for i in range(0, len(rows100), 2):
    idx = int(rows100[i]['sample_num'])
    nli_data[idx] = {
        'q': rows100[i]['question'],
        'right': float(rows100[i]['factscore']),
        'hallucinated': float(rows100[i+1]['factscore']),
    }

# 2. Load Judge results from judge_150.csv
judge_data = {}  # sample_idx -> {'right': {...}, 'hallucinated': {...}}
judge_rows = list(csv.DictReader(open('judge_150.csv', encoding='utf-8')))
for i in range(0, len(judge_rows), 2):
    idx = int(judge_rows[i]['idx'])
    judge_data[idx] = {
        'right': {
            'verdict': judge_rows[i]['verdict'],
            'score': float(judge_rows[i]['score']),
            'proof': judge_rows[i]['proof'],
            'ans': judge_rows[i]['answer'],
        },
        'hallucinated': {
            'verdict': judge_rows[i+1]['verdict'],
            'score': float(judge_rows[i+1]['score']),
            'proof': judge_rows[i+1]['proof'],
            'ans': judge_rows[i+1]['answer'],
        }
    }

print(f"Total samples matched: {len(nli_data)} NLI, {len(judge_data)} Judge")

# 3. Create combined dataset for all 150 samples
combined = []
for idx in range(1, 151):
    combined.append({
        'idx': idx,
        'question': nli_data[idx]['q'],
        'nli_right': nli_data[idx]['right'],
        'nli_hallu': nli_data[idx]['hallucinated'],
        'judge_right_verdict': judge_data[idx]['right']['verdict'],
        'judge_right_score': judge_data[idx]['right']['score'],
        'judge_right_ans': judge_data[idx]['right']['ans'],
        'judge_right_proof': judge_data[idx]['right']['proof'],
        'judge_hallu_verdict': judge_data[idx]['hallucinated']['verdict'],
        'judge_hallu_score': judge_data[idx]['hallucinated']['score'],
        'judge_hallu_ans': judge_data[idx]['hallucinated']['ans'],
        'judge_hallu_proof': judge_data[idx]['hallucinated']['proof'],
    })

# Save side-by-side combined table
with open('combined_nli_judge_150.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(combined[0].keys()))
    writer.writeheader()
    writer.writerows(combined)

print("Saved combined_nli_judge_150.csv")

# 4. Evaluation function for a given decision rule
def evaluate_rule(subset, rule_fn):
    """
    rule_fn(nli_score, judge_score, judge_verdict) -> bool (True = approved, False = rejected)
    Returns:
      right_approved, wrong_rejected, both_correct, right_rejected, wrong_approved
    """
    n = len(subset)
    right_app = 0
    wrong_rej = 0
    both_corr = 0
    right_rej = 0
    wrong_app = 0

    for s in subset:
        app_r = rule_fn(s['nli_right'], s['judge_right_score'], s['judge_right_verdict'])
        app_h = rule_fn(s['nli_hallu'], s['judge_hallu_score'], s['judge_hallu_verdict'])

        if app_r:
            right_app += 1
        else:
            right_rej += 1

        if not app_h:
            wrong_rej += 1
        else:
            wrong_app += 1

        if app_r and (not app_h):
            both_corr += 1

    return {
        'n': n,
        'right_approved': right_app,
        'wrong_rejected': wrong_rej,
        'both_correct': both_corr,
        'right_rejected': right_rej,
        'wrong_approved': wrong_app,
    }

# Split into dev (1-75) and test (76-150)
dev_set = combined[:75]
test_set = combined[75:]
all_set = combined

rules = {
    # a. NLI only (various thresholds)
    'a1_nli_only_0.5': lambda nli, j_score, j_verdict: nli >= 0.5,
    'a2_nli_only_0.6': lambda nli, j_score, j_verdict: nli >= 0.6,
    'a3_nli_only_0.7': lambda nli, j_score, j_verdict: nli >= 0.7,
    # b. Judge only
    'b_judge_only': lambda nli, j_score, j_verdict: j_verdict == 'SUPPORTED',
    # c. Judge SUPPORTED AND NLI >= threshold
    'c1_judge_and_nli_0.2': lambda nli, j_score, j_verdict: (j_verdict == 'SUPPORTED') and (nli >= 0.2),
    'c2_judge_and_nli_0.3': lambda nli, j_score, j_verdict: (j_verdict == 'SUPPORTED') and (nli >= 0.3),
    'c3_judge_and_nli_0.4': lambda nli, j_score, j_verdict: (j_verdict == 'SUPPORTED') and (nli >= 0.4),
    # d. Average of NLI and Judge >= threshold
    'd1_avg_0.5': lambda nli, j_score, j_verdict: ((nli + j_score) / 2.0) >= 0.5,
    'd2_avg_0.6': lambda nli, j_score, j_verdict: ((nli + j_score) / 2.0) >= 0.6,
    'd3_avg_0.7': lambda nli, j_score, j_verdict: ((nli + j_score) / 2.0) >= 0.7,
}

print("\n" + "="*85)
print("EVALUATION ON DEV SET (Samples 1 to 75)")
print("="*85)
print(f"{'Rule':<25} | {'Right App':<10} | {'Wrong Rej':<10} | {'Both Corr':<10} | {'Right Rej':<10} | {'Wrong App':<10}")
print("-" * 85)
for name, fn in rules.items():
    res = evaluate_rule(dev_set, fn)
    print(f"{name:<25} | {res['right_approved']:>3d} ({res['right_approved']/75*100:4.1f}%) | {res['wrong_rejected']:>3d} ({res['wrong_rejected']/75*100:4.1f}%) | {res['both_correct']:>3d} ({res['both_correct']/75*100:4.1f}%) | {res['right_rejected']:>3d} ({res['right_rejected']/75*100:4.1f}%) | {res['wrong_approved']:>3d} ({res['wrong_approved']/75*100:4.1f}%)")

print("\n" + "="*85)
print("EVALUATION ON TEST SET (Samples 76 to 150)")
print("="*85)
print(f"{'Rule':<25} | {'Right App':<10} | {'Wrong Rej':<10} | {'Both Corr':<10} | {'Right Rej':<10} | {'Wrong App':<10}")
print("-" * 85)
for name, fn in rules.items():
    res = evaluate_rule(test_set, fn)
    print(f"{name:<25} | {res['right_approved']:>3d} ({res['right_approved']/75*100:4.1f}%) | {res['wrong_rejected']:>3d} ({res['wrong_rejected']/75*100:4.1f}%) | {res['both_correct']:>3d} ({res['both_correct']/75*100:4.1f}%) | {res['right_rejected']:>3d} ({res['right_rejected']/75*100:4.1f}%) | {res['wrong_approved']:>3d} ({res['wrong_approved']/75*100:4.1f}%)")

print("\n" + "="*85)
print("EVALUATION ON ALL 150 SAMPLES")
print("="*85)
print(f"{'Rule':<25} | {'Right App':<10} | {'Wrong Rej':<10} | {'Both Corr':<10} | {'Right Rej':<10} | {'Wrong App':<10}")
print("-" * 85)
for name, fn in rules.items():
    res = evaluate_rule(all_set, fn)
    print(f"{name:<25} | {res['right_approved']:>3d} ({res['right_approved']/150*100:4.1f}%) | {res['wrong_rejected']:>3d} ({res['wrong_rejected']/150*100:4.1f}%) | {res['both_correct']:>3d} ({res['both_correct']/150*100:4.1f}%) | {res['right_rejected']:>3d} ({res['right_rejected']/150*100:4.1f}%) | {res['wrong_approved']:>3d} ({res['wrong_approved']/150*100:4.1f}%)")
