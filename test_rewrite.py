import app.llm as llm

def rewrite(question: str, answer: str) -> str:
    prompt = (
        "Write ONE clear sentence that states the question and this answer together. "
        "Do not add new facts. For yes/no answers, write the full statement (for 'no', write the negative statement). "
        "Write only the sentence.\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}"
    )
    return llm.ask(prompt).strip("\"' ")

examples = [
    ("Are Gin and tonic and Paloma both cocktails based on tequila?", "no"),
    ("What is the length of the track where the 2013 Liqui Moly Bathurst 12 Hour was staged?", "6.213 km long"),
    ("Which actress played Kimberly Ann Hart?", "Amy Jo Johnson")
]

for q, a in examples:
    res = rewrite(q, a)
    print("Q:    ", q)
    print("A:    ", a)
    print("Claim:", res)
    print("-" * 65)

