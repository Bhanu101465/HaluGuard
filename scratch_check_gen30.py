import csv

rows = list(csv.DictReader(open(r'D:\Hallucination\generative_30.csv', encoding='utf-8')))

print('=== All 30 rows ===')
for i, r in enumerate(rows, 1):
    dc  = r['draft_correct']
    dec = r['decision']
    ra  = r['right_answer'][:35]
    dr  = r['draft'][:45]
    print(f'  [{i:2d}] dc={dc:5s}  dec={dec:<14s}  right={ra:<35s}  draft={dr}')

print()
print('=== Cases where draft_correct=False but answer might still be right ===')
wrong = [r for r in rows if r['draft_correct'] == 'False']
for r in wrong:
    print(f"  right_answer : {r['right_answer']}")
    print(f"  draft        : {r['draft'][:80]}")
    print(f"  decision     : {r['decision']}")
    print()
