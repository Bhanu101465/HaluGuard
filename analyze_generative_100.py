import csv

rows = list(csv.DictReader(open('generative_100_judge.csv', encoding='utf-8')))
print(f"Loaded {len(rows)} rows from generative_100_judge.csv")

def eval_subset(subset, name):
    print("\n" + "="*85)
    print(f"EVALUATION: {name} (n={len(subset)})")
    print("="*85)

    n_right = sum(1 for r in subset if r['draft_correct'] == 'True')
    n_wrong = sum(1 for r in subset if r['draft_correct'] == 'False')
    print(f"Ground truth drafts: {n_right} Right ({100*n_right/len(subset):.1f}%), {n_wrong} Wrong ({100*n_wrong/len(subset):.1f}%)")
    print("-" * 85)
    print(f"{'Rule':<30} | {'Right App':<11} | {'Right Rej':<11} | {'Wrong App':<11} | {'Wrong Rej':<11} | {'Accuracy':<10}")
    print("-" * 85)

    rules = {
        'a. Old formula only (>=0.6)': lambda r: float(r['old_final_score']) >= 0.6,
        'b. Judge only (SUPPORTED)': lambda r: r['judge_verdict'] == 'SUPPORTED',
        'c. Old formula AND Judge': lambda r: (float(r['old_final_score']) >= 0.6) and (r['judge_verdict'] == 'SUPPORTED'),
        'd. New formula (0.5*J+0.5*Old >= 0.6)': lambda r: (0.5 * float(r['judge_score']) + 0.5 * float(r['old_final_score'])) >= 0.6,
    }

    results = {}
    for r_name, r_fn in rules.items():
        ra = rr = wa = wr = 0
        for r in subset:
            is_right = (r['draft_correct'] == 'True')
            approved = r_fn(r)
            if is_right:
                if approved:
                    ra += 1
                else:
                    rr += 1
            else:
                if approved:
                    wa += 1
                else:
                    wr += 1
        acc = (ra + wr) / len(subset) * 100
        results[r_name] = {'ra': ra, 'rr': rr, 'wa': wa, 'wr': wr, 'acc': acc}
        print(f"{r_name:<30} | {ra:>4d} ({ra/n_right*100:4.1f}%) | {rr:>4d} ({rr/n_right*100:4.1f}%) | {wa:>4d} ({wa/n_wrong*100:4.1f}%) | {wr:>4d} ({wr/n_wrong*100:4.1f}%) | {acc:5.1f}%")

    return results

first_50 = rows[:50]
last_50 = rows[50:]

res_first = eval_subset(first_50, "FIRST 50 QUESTIONS (Samples 181 to 230 - Dev)")
res_last = eval_subset(last_50, "LAST 50 QUESTIONS (Samples 231 to 280 - Test)")
res_all = eval_subset(rows, "ALL 100 QUESTIONS (Samples 181 to 280)")

wrong_rows = list(csv.DictReader(open('drafts_marked_wrong.csv', encoding='utf-8')))
print("\n" + "="*85)
print(f"DRAFTS MARKED WRONG: {len(wrong_rows)} saved in drafts_marked_wrong.csv")
print("="*85)
