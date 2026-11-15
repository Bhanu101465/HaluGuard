# HaluGuard

HaluGuard checks whether an LLM answer is trustworthy or a hallucination.
It breaks the answer into small facts and checks each one against a passage
and the web. If the answer is not supported, it says "I don't know."

---

## How It Works

```
Question + passage
       |
  1. Need check  — Is this a factual question? (skip chit-chat)
       |
  2. Draft answer — LLM answers using only the passage
       |
  3. Chain of Verification (CoVe) — LLM checks its own facts and rewrites
       |
  4. Split into facts — break the answer into small sentences
       |
  5. NLI check — DeBERTa scores each fact vs the passage (0.0 to 1.0)
       |
  6. Web search — if NLI fails, search DuckDuckGo for more evidence
       |
  7. Consistency — ask the same question 5 times, measure agreement
       |
  8. Signal — read the model's own token confidence (logprobs)
       |
  9. Final score — weighted mix -> "answer" or "I don't know"
```

**Final score formula:**

```
score = 0.5 x factscore + 0.25 x consistency + 0.25 x signal
```

If score >= 0.6 -> `answer`. Otherwise -> `I don't know`.

---

## Results

### Big Test (300 Answers)
- **Test:** 300 answers from HaluEval (150 right, 150 hallucinated)
- **Right answers approved:** 105 / 150 (70.0%)
- **Hallucinations caught:** 106 / 150 (70.7%)
- **Time:** 40.4 minutes

### Optional Modules
- **Passage filter:** Tried it, no clear gain (each sample has only 1 passage), so it is OFF by default.
- **LLM judge:** Built, OFF by default.

---

## Earlier Tests

### Test 1 — Given-answer mode (150 questions)

The pipeline scored the right answer vs a hallucinated answer side by side.
No LLM calls for this test — NLI only.

| | First 50 | Next 100 | All 150 |
|--|--:|--:|--:|
| Right answer wins (gap >= 0.05) | 23 / 50 (46%) | 47 / 100 (47%) | **70 / 150 (47%)** |
| Ties (gap < 0.05) | 25 / 50 (50%) | 42 / 100 (42%) | **67 / 150 (45%)** |
| Wrong answer wins | 2 / 50 (4%) | 11 / 100 (11%) | **13 / 150 (9%)** |
| Avg right answer score | 0.62 | 0.57 | **0.59** |
| Avg hallucinated score | 0.23 | 0.31 | **0.28** |

Right answers score more than double hallucinated answers on average (0.59 vs 0.28).
Non-loss rate: **91%** (the pipeline does not make the wrong call 91% of the time).
This includes 45% ties, where the pipeline could not decide.

### Test 2 — Full generative mode (30 questions, samples 151-180)

The LLM writes its own draft. All modules ON. No given answer.
Avg 10.4 LLM calls per question. Total cost ~$0.019 for 30 questions.

| Outcome | Count | % |
|---------|------:|--:|
| Right answer -> approved ("answer") | 24 | 80% |
| Right answer -> rejected ("I don't know") | 5 | 17% |
| Wrong answer -> approved (worst case) | 0 | 0% |
| Wrong answer -> rejected (good catch) | 1 | 3% |

---

## Model and Tools

| Part | Tool |
|------|------|
| LLM | Qwen/Qwen3-30B-A3B-Instruct-2507 via Nebius AI |
| NLI checker | cross-encoder/nli-deberta-v3-base (HuggingFace, ~738 MB) |
| Web search | DuckDuckGo (free, no key needed) |
| API | FastAPI |

---

## Extra Features

- **IDK skip:** If the LLM says "I don't know" as its draft, skip CoVe,
  consistency and signals entirely. Saves ~6 LLM calls per question.
- **Short answer rewrite:** If the answer is under 5 words (e.g. "Richard Nixon"),
  the LLM rewrites it into a full sentence so NLI can check it properly.
- **Soft FactScore:** Instead of counting "supported / not supported", average the
  raw entailment probabilities (0.0-1.0) for a smoother score.
- **Joined passages:** Combine all retrieved passages into one NLI input so short
  passages do not cause false negatives.
- **Enriched web query:** Add entity names from the question and passage to the
  search query so results are about the right person or topic.
- **Real logprobs:** Nebius returns real token probabilities for "True/False". The
  signal score uses these instead of the 0.5 fallback.

---

## Limits

- **Ties from missing passages.** Questions tie when the Wikipedia passage does not
  contain the fact needed to decide.
- **Small passage sets.** HaluEval samples provide only 1 passage per sample, so passage
  filtering has minimal effect.
- **Weights not tuned.** The score formula weights (0.5, 0.25, 0.25) and threshold (0.6)
  can be tuned further for specific domains.
- **LLM fact splitting is off.** `app/facts.py` has a switch `USE_LLM_FACTS` (turned off
  by default to save LLM calls).

---

## Folder Structure

```
HaluGuard\
|
|-- app\
|   |-- llm.py            LLM client (Nebius / NVIDIA / Grok)
|   |-- answer.py         Draft answer from passages
|   |-- facts.py          Split answer into atomic facts
|   |-- verify.py         NLI fact checker (DeBERTa)
|   |-- need_check.py     Is this a factual question?
|   |-- cove.py           Chain of Verification
|   |-- web_search.py     DuckDuckGo fallback search
|   |-- consistency.py    5-sample consistency score
|   |-- signals.py        Token logprob confidence
|   |-- decision.py       Final score + threshold decision
|   |-- judge.py          Optional LLM-as-judge checker
|   |-- passage_filter.py Optional passage filter
|   |-- pipeline.py       Connects all steps; feature switches
|   |-- retrieve.py       TF-IDF passage retrieval (best first)
|   |-- data_loader.py    Load HaluEval samples
|   |-- main.py           FastAPI POST /check endpoint
|
|-- data\
|   |-- qa_data.json      10,000 HaluEval QA samples
|
|-- run_big_test.py       Run 300-answer evaluation (150 samples)
|-- run_filter_test.py    Evaluate passage filter ON vs OFF
|-- judge_eval.py         Test LLM-as-judge on HaluEval samples
|-- eval_halueval.py      Batch eval (given-answer mode)
|-- generative_eval.py    Full generative eval (no given answer)
|-- requirements.txt      Python packages
|-- .env.example          Template for your API key (safe to share)
```

---

## Install and Run

```powershell
# 1. Install packages (only needed once)
pip install -r requirements.txt

# 2. Copy .env.example to .env and fill in your key
copy .env.example .env

# 3. Start the API server
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
# Open http://127.0.0.1:8000/docs  ->  POST /check

# 4. Run the 300-answer test
.\.venv\Scripts\python.exe run_big_test.py

# 5. Run the passage filter comparison test
.\.venv\Scripts\python.exe run_filter_test.py
```

---

## API Usage

Send a POST request to `/check`:

```json
{
  "question": "Who wrote Hamlet?",
  "passages": ["Hamlet is a play written by William Shakespeare."],
  "given_answer": "William Shakespeare"
}
```

Response:

```json
{
  "decision": "answer",
  "factscore": 0.97,
  "consistency": 1.0,
  "signal": 0.95,
  "final_score": 0.97
}
```

Leave out `given_answer` to let HaluGuard write and check its own answer.

---

## Keep Your API Key Private

**Never upload `.env` to GitHub.** It contains your Nebius API key.

The `.gitignore` file in this repo already excludes it:

```
.env
```

Use `.env.example` as a safe template with empty values.

---

## References

These papers directly shaped the code. Each one is listed only if its idea
is actually in the code.

| Paper | Where it is used |
|-------|-----------------|
| Dhuliawala et al. (2023). *Chain-of-Verification Reduces Hallucination in Large Language Models.* | `app/cove.py` — generates check questions, answers them separately, rewrites the draft |
| Min et al. (2023). *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* | `app/facts.py` + `app/verify.py` — splits answer into atomic facts, checks each with NLI |
| Asai et al. (2023). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* | `app/need_check.py` — the LLM decides whether retrieval is needed (YES/NO), matching the Self-RAG retrieval decision token |
| Liu et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts.* | `app/retrieve.py` — ranks passages by TF-IDF and keeps the best passage first so the most relevant text is at the start of the context |
| Manakul et al. (2023). *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models.* | `app/consistency.py` — asks the same question 5 times and measures agreement between answers |
| Li et al. (2023). *HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models.* | `data/qa_data.json` — the 10,000-question dataset used for all tests |
