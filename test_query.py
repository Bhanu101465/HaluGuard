import os
os.environ["HF_HOME"] = r"D:\hf-cache"
from app.web_search import search_fact
from app.verify import check_fact

# Question 1 test
q1 = "Which magazine was started first Arthur's Magazine or First for Women?"
f1 = "Arthur's Magazine was started first, launching in 1844"
print("--- Q1 Tests ---")
query1 = "First for Women magazine launch year 1989 Arthur's Magazine"
snippets1 = search_fact(query1)
print(f"Query: '{query1}' -> {len(snippets1)} snippets")
for s in snippets1:
    print("  -", s[:130])
if snippets1:
    r1 = check_fact(f1, snippets1)
    print("  NLI result:", r1)

# Question 4 test
q4 = "What nationality was James Henry Miller's wife?"
f4 = "James Henry Miller's wife was American"
print("\n--- Q4 Tests ---")
query4 = "James Henry Miller Ewan MacColl wife Peggy Seeger American nationality"
snippets4 = search_fact(query4)
print(f"Query: '{query4}' -> {len(snippets4)} snippets")
for s in snippets4:
    print("  -", s[:130])
if snippets4:
    r4 = check_fact(f4, snippets4)
    print("  NLI result:", r4)

