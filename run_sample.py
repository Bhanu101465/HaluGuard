import os
import sys

# Ensure HF cache is strictly on D: drive
os.environ["HF_HOME"] = r"D:\hf-cache"

from app.data_loader import load_samples
from app.pipeline import run_sample

def main():
    print("=" * 60)
    print("HaluGuard: Testing Fact-Checker on HaluEval Sample")
    print("Cache Directory:", os.environ["HF_HOME"])
    print("=" * 60)

    # 1. Load 1 sample
    samples = load_samples(n=1)
    if not samples:
        print("Error: No samples found in data/qa_data.json")
        return

    sample = samples[0]
    print(f"\nQuestion: {sample['question']}")
    print(f"Knowledge:\n{sample['knowledge']}")

    # 2. Check Right Answer
    print("\n--- [1] Checking Right Answer ---")
    print(f"Answer: {sample['right_answer']}")
    right_result = run_sample(sample, use_hallucinated=False)
    for i, f in enumerate(right_result["facts"], 1):
        print(f"  Fact {i}: {f['fact']}")
        print(f"    Label: {f['label']} (score: {f['score']:.2f})")
    print(f"FactScore: {right_result['factscore']}")

    # 3. Check Hallucinated Answer
    print("\n--- [2] Checking Hallucinated Answer ---")
    print(f"Answer: {sample['hallucinated_answer']}")
    hallu_result = run_sample(sample, use_hallucinated=True)
    for i, f in enumerate(hallu_result["facts"], 1):
        print(f"  Fact {i}: {f['fact']}")
        print(f"    Label: {f['label']} (score: {f['score']:.2f})")
    print(f"FactScore: {hallu_result['factscore']}")

    # 4. Side-by-side comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"{'Answer Type':<25} | {'FactScore':<10}")
    print("-" * 40)
    print(f"{'Right Answer':<25} | {right_result['factscore']:<10.2f}")
    print(f"{'Hallucinated Answer':<25} | {hallu_result['factscore']:<10.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()

