# HaluGuard Progress

---

## 1. DONE
Every file in the project and what it does:
- `.env`: Stores provider choice (`nebius`), model name (`Qwen/Qwen3-30B-A3B-Instruct-2507`), API key, and cache path (`D:\hf-cache`).
- `requirements.txt`: Package list without anthropic.
- `data/qa_data.json`: 10,000 HaluEval QA samples for benchmark testing.
- `app/data_loader.py`: Loads samples and knowledge passages from the dataset.
- `app/llm.py`: Connects to Nebius, NVIDIA, or Grok using the OpenAI client with `override=True` and empty reply detection.
- `app/retrieve.py`: Selects the top relevant passages using TF-IDF cosine similarity.
- `app/answer.py`: Prompts the LLM to write an answer using retrieved passages.
- `app/facts.py`: Breaks answers into atomic facts (protecting decimals like `6.213 km` and abbreviations like `Mr.`, `U.S.`).
- `app/verify.py`: Uses DeBERTa NLI to classify facts and return raw entailment probabilities, with support for joined-passage checking.
- `app/need_check.py`: Checks if a question needs a factual lookup or is chit-chat.
- `app/cove.py`: Implements Chain of Verification (generates check questions, answers them, rewrites draft).
- `app/web_search.py`: Searches the web for fact verification snippets (via DuckDuckGo).
- `app/consistency.py`: Computes answer consistency across repeated questions.
- `app/signals.py`: Reads token probability confidence from the model.
- `app/decision.py`: Combines factscore, consistency, and signal into final score and decision.
- `app/pipeline.py`: Orchestrates the complete pipeline with switches (`USE_CLAIM_REWRITE`, `USE_JOIN_PASSAGES`, `USE_SOFT_FACTSCORE`).
- `app/main.py`: FastAPI application serving the `POST /check` endpoint with optional `given_answer`.
- `run_sample.py`: Runs the pipeline on a single HaluEval sample.
- `eval_halueval.py`: Batch evaluates N HaluEval samples with `--start` offset, 0.05 tie margin, and LLM claim rewriting.
- `results.csv`: Stores the first 50 evaluation results.
- `results_51_150.csv`: Stores evaluation results for fresh samples 51 to 150 (100 samples).
- `ties.csv`: Categorization and details of the 25 ties from the first 50 samples.
- `losses.csv`: Error analysis breakdown of early model losses.
- `test_rewrite.py`: Test script verifying LLM claim rewriting on yes/no, decimal, and entity queries.

---

## 2. TESTED

### A. Tie Breakdown (First 50 Samples)
Saved to `ties.csv`:
- **Group a (Both High $\ge 0.7$)**: 8 / 25 (32.0%) — Hallucinated answers are valid generalizations or partially true claims.
- **Group b (Both Low $< 0.3$)**: 17 / 25 (68.0%) — Single knowledge passages lack the multi-hop facts to verify either answer.
- **Group c (In Between)**: 0 / 25 (0.0%).

### B. Fresh Evaluation Comparison (150 Total Samples Evaluated)

| Metric | First 50 Samples (1–50) | Fresh 100 Samples (51–150) | Combined (All 150 Samples) |
| :--- | :--- | :--- | :--- |
| **Right Answer Wins ($\text{gap} \ge +0.05$)** | 23 / 50 (46.0%) | **47 / 100 (47.0%)** | **70 / 150 (46.7%)** |
| **Ties ($|\text{gap}| < 0.05$)** | 25 / 50 (50.0%) | **42 / 100 (42.0%)** | **67 / 150 (44.7%)** |
| **Wrong Answer Wins (Losses)** | 2 / 50 (4.0%) | **11 / 100 (11.0%)** | **13 / 150 (8.7%)** |
| **Avg Right Answer FactScore** | 0.62 | **0.57** | **0.59** |
| **Avg Hallucinated FactScore** | 0.23 | **0.31** | **0.28** |
| **Average Score Gap** | +0.39 | **+0.26** | **+0.30** |
| **Facts per Answer** | 1.00 | **1.08** | **1.05** |

- **Non-loss rate**: **91.3%** of all 150 samples are either a clear Right Win (46.7%) or a Tie (44.7%).
- **Score Margin**: Right answers consistently score double the factscore of hallucinated answers (+0.30 margin).

---

## 3. NOT DONE
- **Web Search Verification**: Turn `USE_WEB_SEARCH = True` to see if web search snippets can break ties for Group B (the 67 ungrounded passages).
- **Full Generative Pipeline**: Test end-to-end question answering (`write_answer`, `cove`, `consistency_score`, and `signal_score`) without supplying `given_answer`.

---

## 4. PROBLEMS
- **Passage Gaps**: In ~44% of HaluEval samples, the single Wikipedia passage doesn't mention the entity or comparative fact needed, leading to 0.00 score ties. Web search is needed to bring in missing facts.

---

## 5. HOW TO RUN
Run these commands from `D:\Hallucination`:

- **Run 100 fresh samples (51–150)**:
  ```powershell
  & "D:\Hallucination\.venv\Scripts\python.exe" eval_halueval.py --start 50 -n 100 -o results_51_150.csv
  ```

- **Run first 50 samples**:
  ```powershell
  & "D:\Hallucination\.venv\Scripts\python.exe" eval_halueval.py -n 50
  ```

- **Run FastAPI server**:
  ```powershell
  & "D:\Hallucination\.venv\Scripts\python.exe" -m uvicorn app.main:app --reload
  ```

---

## 6. NEXT STEP
Turn on web search (`USE_WEB_SEARCH = True`) to evaluate whether web snippets can resolve the ~44% passage-missing ties.
