# app/semantic_entropy.py
# Semantic entropy calculation for HaluGuard.
# Groups sampled answers by meaning using bidirectional NLI entailment.
# 1 group = sure (score 1.0). Many groups = unsure (score near 0.0).
#
# ==============================================================================
# FIXED RULES (do NOT change after seeing results):
# 1. Doubtful zone: fact score 0.3 to 0.7 (0.3 <= fact_score <= 0.7)
# 2. Unsure: entropy score below 0.5 (entropy_score < 0.5)
# 3. Rule: if the claim is doubtful AND unsure -> answer becomes "I don't know"
# ==============================================================================

import math
from typing import Tuple, List

DOUBTFUL_MIN = 0.3
DOUBTFUL_MAX = 0.7
UNSURE_THRESHOLD = 0.5

def cluster_answers(answers: List[str]) -> List[List[str]]:
    """
    Puts answers into groups by meaning.
    Same group = answer A supports B AND B supports A (bidirectional entailment).
    """
    if not answers:
        return []
    if len(answers) == 1:
        return [[answers[0]]]

    from .verify import _get_nli
    nli = _get_nli()

    cleaned = [a.strip() for a in answers]
    n = len(cleaned)

    # Prepare all pairs (i, j) where i != j
    pairs = []
    pair_indices = []
    for i in range(n):
        for j in range(n):
            if i != j:
                pairs.append({"text": cleaned[i], "text_pair": cleaned[j]})
                pair_indices.append((i, j))

    # Single batched forward pass for high efficiency
    entail_map = {}
    if pairs:
        results = nli(pairs)
        for (i, j), out in zip(pair_indices, results):
            out_item = out[0] if isinstance(out[0], list) else out
            probs = {o["label"].lower(): o["score"] for o in out_item}
            entail_map[(i, j)] = probs.get("entailment", 0.0) >= 0.5

    # Group answers: A and B belong to the same group iff A entails B and B entails A
    groups = []
    assigned = set()

    for i in range(n):
        if i in assigned:
            continue
        group = [cleaned[i]]
        assigned.add(i)
        for j in range(i + 1, n):
            if j not in assigned:
                if cleaned[i].lower() == cleaned[j].lower():
                    group.append(cleaned[j])
                    assigned.add(j)
                elif entail_map.get((i, j), False) and entail_map.get((j, i), False):
                    group.append(cleaned[j])
                    assigned.add(j)
        groups.append(group)

    return groups

def compute_semantic_entropy_score(answers: List[str]) -> Tuple[float, List[List[str]]]:
    """
    Computes semantic entropy score from 0.0 to 1.0 (1.0 = sure, 0.0 = unsure).
    1 group = sure. Many groups = unsure.
    """
    if not answers:
        return 1.0, []
    if len(answers) == 1:
        return 1.0, [[answers[0]]]

    groups = cluster_answers(answers)
    num_groups = len(groups)
    total_answers = len(answers)

    # 1 group = completely sure (H = 0 -> score = 1.0)
    if num_groups <= 1:
        return 1.0, groups

    # Shannon entropy over cluster probabilities
    # p_i = |group_i| / total_answers
    # H = - sum(p_i * ln(p_i))
    probs = [len(g) / total_answers for g in groups]
    h = -sum(p * math.log(p) for p in probs)
    h_max = math.log(total_answers)

    # Normalized entropy (0.0 = sure, 1.0 = max confusion)
    normalized_h = h / h_max

    # Score from 0 to 1 where 1.0 = sure, 0.0 = unsure
    entropy_score = round(max(0.0, min(1.0, 1.0 - normalized_h)), 2)
    return entropy_score, groups

def is_doubtful(fact_score: float) -> bool:
    """Doubtful zone: fact score between 0.3 and 0.7."""
    return DOUBTFUL_MIN <= fact_score <= DOUBTFUL_MAX

def is_unsure(entropy_score: float) -> bool:
    """Unsure: entropy score below 0.5."""
    return entropy_score < UNSURE_THRESHOLD

def apply_semantic_entropy_decision(current_decision: str, fact_score: float, entropy_score: float) -> str:
    """
    Rule: if the claim is doubtful AND unsure -> answer becomes "I don't know".
    Otherwise, keeps current decision.
    """
    if is_doubtful(fact_score) and is_unsure(entropy_score):
        return "I don't know"
    return current_decision

if __name__ == "__main__":
    test_sure = ["Paris is the capital of France.", "Paris is France's capital.", "The capital of France is Paris."]
    test_unsure = ["Paris", "Berlin", "Rome", "Madrid", "London"]
    s_sure, g_sure = compute_semantic_entropy_score(test_sure)
    s_unsure, g_unsure = compute_semantic_entropy_score(test_unsure)
    print(f"Sure score: {s_sure} (groups: {len(g_sure)})")
    print(f"Unsure score: {s_unsure} (groups: {len(g_unsure)})")
