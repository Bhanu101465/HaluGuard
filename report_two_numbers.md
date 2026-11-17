# HaluGuard: Two Numbers Report

This report evaluates HaluGuard using the real decisions of the default pipeline (weights: 0.50 fact, 0.25 consistency, 0.25 signal; cutoff: 0.60; switches at default).

---

## Key Question: Does "Hallucinations Caught" Equal "I don't know"?

**No, they are not equal in the total count.**
- In the pipeline, decisions are binary: `"answer"` (approved and returned) or `"I don't know"` (rejected). There is no third label like "flagged".
- Every **caught hallucination** receives the decision `"I don't know"`.
- However, the total count of `"I don't know"` answers is higher because it also includes **correct answers that were mistakenly rejected** (false alarms).
- Total `"I don't know"` = (Hallucinations caught) + (Correct answers rejected).

---

## 1. Big Test (300 Answers from HaluEval)

Data source: `big_test.csv` (150 right answers, 150 hallucinated answers).

| Metric | Count | Percentage |
| :--- | :---: | :---: |
| **BEFORE: Total hallucinations (all answers)** | 150 / 300 | **50.0%** |
| **AFTER: Hallucinations among approved answers** | 44 / 149 | **29.5%** |
| **Approved answers (shown to user)** | 149 / 300 | 49.7% |
| &nbsp;&nbsp;• Correct answers approved | 105 / 150 | 70.0% of right |
| &nbsp;&nbsp;• Hallucinated answers approved (leaked) | 44 / 150 | 29.3% of hallu |
| **"I don't know" answers (total)** | 151 / 300 | **50.3%** |
| &nbsp;&nbsp;• Right answers marked "I don't know" (lost) | 45 / 150 | 30.0% of right |
| &nbsp;&nbsp;• Hallucinations marked "I don't know" (caught) | 106 / 150 | 70.7% of hallu |
| **Hallucination rate drop (points)** | — | **-20.5 points** (50.0% → 29.5%) |
| **Hallucination rate drop (relative %)** | — | **-40.9%** relative drop |

---

## 2. 100-Answer Test (50 Samples, Samples 81 to 130)

Data source: `semantic_entropy_100.csv` (50 right answers, 50 hallucinated answers evaluated under default pipeline settings).

| Metric | Count | Percentage |
| :--- | :---: | :---: |
| **BEFORE: Total hallucinations (all answers)** | 50 / 100 | **50.0%** |
| **AFTER: Hallucinations among approved answers** | 15 / 47 | **31.9%** |
| **Approved answers (shown to user)** | 47 / 100 | 47.0% |
| &nbsp;&nbsp;• Correct answers approved | 32 / 50 | 64.0% of right |
| &nbsp;&nbsp;• Hallucinated answers approved (leaked) | 15 / 50 | 30.0% of hallu |
| **"I don't know" answers (total)** | 53 / 100 | **53.0%** |
| &nbsp;&nbsp;• Right answers marked "I don't know" (lost) | 18 / 50 | 36.0% of right |
| &nbsp;&nbsp;• Hallucinations marked "I don't know" (caught) | 35 / 50 | 70.0% of hallu |
| **Hallucination rate drop (points)** | — | **-18.1 points** (50.0% → 31.9%) |
| **Hallucination rate drop (relative %)** | — | **-36.2%** relative drop |

*(Note: On the alternative 100-answer set in `scores_100.csv` covering samples 31 to 80, the after rate is 15 of 53 = 28.3%, with 47 "I don't know".)*

---

## Summary in 3 Simple Lines

1. **What improved:** The chance that an approved answer is a hallucination dropped from 50.0% down to 29.5% (a 40.9% relative drop).
2. **What it cost:** To catch those hallucinations, the pipeline discarded 30.0% of good, correct answers by saying "I don't know".
3. **One limit:** About 29% to 32% of answers shown to the user are still hallucinations, so human verification is still needed.
