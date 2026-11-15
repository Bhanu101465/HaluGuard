import csv

rows = list(csv.DictReader(open('D:/Hallucination/generative_100_judge.csv', encoding='utf-8')))

weights_list = [
    ('0.50 / 0.25 / 0.25 (now)', 0.50, 0.25, 0.25),
    ('0.60 / 0.20 / 0.20', 0.60, 0.20, 0.20),
    ('0.40 / 0.30 / 0.30', 0.40, 0.30, 0.30),
    ('0.70 / 0.15 / 0.15', 0.70, 0.15, 0.15),
    ('0.50 / 0.50 / 0.00', 0.50, 0.50, 0.00),
]

cutoffs = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]

def eval_subset(subset, w, cutoff):
    w_f, w_c, w_s = w[1], w[2], w[3]
    approved = 0
    right_appr = 0
    wrong_appr = 0
    right_rej = 0
    wrong_rej = 0
    for r in subset:
        is_right = (r['draft_correct'].strip().lower() == 'true')
        f = float(r['factscore'])
        c = float(r['consistency'])
        s = float(r['signal'])
        score = w_f * f + w_c * c + w_s * s
        if score >= cutoff:
            approved += 1
            if is_right:
                right_appr += 1
            else:
                wrong_appr += 1
        else:
            if is_right:
                right_rej += 1
            else:
                wrong_rej += 1
    pct_right = (right_appr / approved * 100.0) if approved > 0 else 0.0
    return {
        'approved': approved,
        'pct_right': pct_right,
        'right_appr': right_appr,
        'right_rej': right_rej,
        'wrong_appr': wrong_appr,
        'wrong_rej': wrong_rej
    }

first_50 = rows[:50]
last_50 = rows[50:]
all_100 = rows

print("--- 1. Varying Cutoff (Weights: 0.5/0.25/0.25) ---")
print("First 50:")
for c in cutoffs:
    res = eval_subset(first_50, weights_list[0], c)
    print(f"Cutoff {c:.2f} | Appr: {res['approved']:2d} | Right%: {res['pct_right']:5.1f}% | R_Appr: {res['right_appr']:2d} | R_Rej: {res['right_rej']:2d} | W_Appr: {res['wrong_appr']:2d} | W_Rej: {res['wrong_rej']:2d}")

print("\nLast 50:")
for c in cutoffs:
    res = eval_subset(last_50, weights_list[0], c)
    print(f"Cutoff {c:.2f} | Appr: {res['approved']:2d} | Right%: {res['pct_right']:5.1f}% | R_Appr: {res['right_appr']:2d} | R_Rej: {res['right_rej']:2d} | W_Appr: {res['wrong_appr']:2d} | W_Rej: {res['wrong_rej']:2d}")

print("\nAll 100:")
for c in cutoffs:
    res = eval_subset(all_100, weights_list[0], c)
    print(f"Cutoff {c:.2f} | Appr: {res['approved']:2d} | Right%: {res['pct_right']:5.1f}% | R_Appr: {res['right_appr']:2d} | R_Rej: {res['right_rej']:2d} | W_Appr: {res['wrong_appr']:2d} | W_Rej: {res['wrong_rej']:2d}")

print("\n--- 2. Varying Weights (Cutoff: 0.6) ---")
print("First 50:")
for w in weights_list:
    res = eval_subset(first_50, w, 0.6)
    print(f"{w[0]:24s} | Appr: {res['approved']:2d} | Right%: {res['pct_right']:5.1f}% | R_Appr: {res['right_appr']:2d} | R_Rej: {res['right_rej']:2d} | W_Appr: {res['wrong_appr']:2d} | W_Rej: {res['wrong_rej']:2d}")

print("\nLast 50:")
for w in weights_list:
    res = eval_subset(last_50, w, 0.6)
    print(f"{w[0]:24s} | Appr: {res['approved']:2d} | Right%: {res['pct_right']:5.1f}% | R_Appr: {res['right_appr']:2d} | R_Rej: {res['right_rej']:2d} | W_Appr: {res['wrong_appr']:2d} | W_Rej: {res['wrong_rej']:2d}")

print("\nAll 100:")
for w in weights_list:
    res = eval_subset(all_100, w, 0.6)
    print(f"{w[0]:24s} | Appr: {res['approved']:2d} | Right%: {res['pct_right']:5.1f}% | R_Appr: {res['right_appr']:2d} | R_Rej: {res['right_rej']:2d} | W_Appr: {res['wrong_appr']:2d} | W_Rej: {res['wrong_rej']:2d}")

print("\n--- 3. Full Grid: All 30 combinations on First 50 ---")
grid = []
for w in weights_list:
    for c in cutoffs:
        rf = eval_subset(first_50, w, c)
        rl = eval_subset(last_50, w, c)
        ra = eval_subset(all_100, w, c)
        acc_f = (rf['right_appr'] + rf['wrong_rej']) / 50.0
        acc_l = (rl['right_appr'] + rl['wrong_rej']) / 50.0
        acc_a = (ra['right_appr'] + ra['wrong_rej']) / 100.0
        grid.append((w[0], c, rf, rl, ra, acc_f, acc_l, acc_a))

# Sort by accuracy on first 50, then right% on first 50
grid.sort(key=lambda x: (x[5], x[2]['pct_right'], x[2]['wrong_rej']), reverse=True)

print(f"{'Weights':24s} | Cutoff | F50 Acc | F50 Right% | F50 R_Rej | F50 W_Appr || L50 Acc | L50 Right% | L50 R_Rej | L50 W_Appr")
for g in grid[:15]:
    print(f"{g[0]:24s} | {g[1]:.2f}   | {g[5]*100:5.1f}%  | {g[2]['pct_right']:6.1f}%    | {g[2]['right_rej']:2d}        | {g[2]['wrong_appr']:2d}        || {g[6]*100:5.1f}%  | {g[3]['pct_right']:6.1f}%    | {g[3]['right_rej']:2d}        | {g[3]['wrong_appr']:2d}")
