import csv

rows = list(csv.DictReader(open('generative_30.csv', encoding='utf-8')))
print(f'Total rows: {len(rows)}')
print()

# Wrong -> answer (worst case)
bad = [r for r in rows if r['draft_correct'] == 'False' and r['decision'] == 'answer']
print(f'=== Wrong draft -> answer (worst case): {len(bad)} ===')
for r in bad:
    print(f"  Q:     {r['question'][:70]}")
    print(f"  Right: {r['right_answer']} | Draft: {r['draft'][:60]}")
    print(f"  fs_draft={r['factscore_draft']} fs_cove={r['factscore_cove']} sig={r['signal']} final={r['final_score']}")
    print()

# Correct -> IDK (missed)
missed = [r for r in rows if r['draft_correct'] == 'True' and r['decision'] != 'answer']
print(f'=== Correct draft -> IDK (missed): {len(missed)} ===')
for r in missed:
    print(f"  Q:     {r['question'][:70]}")
    print(f"  Right: {r['right_answer']} | Draft: {r['draft'][:60]}")
    print(f"  fs_draft={r['factscore_draft']} fs_cove={r['factscore_cove']} sig={r['signal']} final={r['final_score']}")
    print()

# CoVe hurt cases
hurt = [r for r in rows if float(r['factscore_cove']) < float(r['factscore_draft']) - 0.0001]
print(f'=== CoVe HURT (factscore_cove < factscore_draft): {len(hurt)} ===')
for r in hurt:
    print(f"  Q:     {r['question'][:70]}")
    print(f"  draft: {r['draft'][:60]}")
    print(f"  cove:  {r['answer_cove'][:60]}")
    print(f"  fs_draft={r['factscore_draft']} -> fs_cove={r['factscore_cove']}")
    print()

