import csv

rows = list(csv.DictReader(open(r'D:\Hallucination\generative_30.csv', encoding='utf-8')))

# Check each draft_correct=False case manually
wrong = [r for r in rows if r['draft_correct'] == 'False']
print('Checking each False case:')
for r in wrong:
    ra = r['right_answer']
    dr = r['draft']
    # Is the right answer actually IN the draft (just not matching exactly)?
    # Check case-insensitive and also check fragments
    ra_lower = ra.lower().strip()
    dr_lower = dr.lower()
    # check for key tokens of the right answer
    tokens = [t for t in ra_lower.split() if len(t) > 2]
    matches = [t for t in tokens if t in dr_lower]
    print(f"right_answer : {ra}")
    print(f"draft        : {dr[:100]}")
    print(f"decision     : {r['decision']}")
    print(f"key tokens   : {tokens}")
    print(f"matched      : {matches}")
    print(f"fraction     : {len(matches)}/{len(tokens)}")
    print()
