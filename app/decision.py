# Part 6: final decision. Mix the scores, then answer or say "I don't know".
# Change these numbers later to tune.
W_FACT = 0.5
W_CONSISTENCY = 0.25
W_SIGNAL = 0.25
THRESHOLD = 0.6

def compute_final_score(factscore: float, consistency: float, signal: float) -> float:
    return W_FACT * factscore + W_CONSISTENCY * consistency + W_SIGNAL * signal

def decide(factscore: float, consistency: float, signal: float,
           threshold: float = THRESHOLD) -> str:
    score = compute_final_score(factscore, consistency, signal)
    return "answer" if score >= threshold else "I don't know"

if __name__ == "__main__":
    print(decide(0.9, 0.9, 0.8))   # answer
    print(decide(0.2, 0.3, 0.5))   # I don't know
