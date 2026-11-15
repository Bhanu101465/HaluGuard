# Part 1: does this question need a fact lookup?
from .llm import ask

def need_lookup(question: str) -> bool:
    prompt = (
        "Does answering this question need looking up facts "
        "(people, places, dates, numbers, events)?\n"
        "Reply with only one word: YES or NO.\n"
        "Chit-chat, greetings, opinions or simple math = NO.\n\n"
        f"Question: {question}"
    )
    reply = ask(prompt).strip().upper()
    if reply.startswith("NO"):
        return False
    return True  # YES, or unclear -> look up (safe choice)

if __name__ == "__main__":
    ask = lambda p, temperature=0.0: "NO" if "hello" in p.lower() else "YES"
    print(need_lookup("hello there"))              # False
    print(need_lookup("Who wrote Hamlet?"))        # True
    ask = lambda p, temperature=0.0: "maybe??"
    print(need_lookup("anything"))                 # True (unclear)
