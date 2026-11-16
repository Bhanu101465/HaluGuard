# HaluGuard

HaluGuard is a hallucination detection pipeline for large language models.
It checks whether an LLM answer is factually supported by retrieved text and the web, falling back to "I don't know" when facts cannot be verified.

---

## How It Works

HaluGuard verifies answers through an 8-step pipeline:

```
1. Passage Retrieval (RAG)
   Finds relevant passages and places the best passage first.
       |
2. Draft Answer
   The LLM generates an answer using only the retrieved passage.
       |
3. Small Facts
   Breaks the drafted answer into individual atomic factual statements.
       |
4. Fact Check (NLI + Web)
   DeBERTa scores each fact against the passage (0.0 to 1.0).
   If evidence is missing, DuckDuckGo searches the web for snippets.
       |
5. Chain-of-Verification (CoVe)
   Drafts verification questions, answers them independently, and rewrites.
       |
6. Consistency (Ask 5 Times)
   Queries the question 5 times and measures agreement between answers.
       |
7. Confidence Signal
   Extracts token-level log probabilities from the LLM.
       |
8. Final Score & Decision
   Combines scores into a final weighted score:
   Score = 0.50 x Fact + 0.25 x Consistency + 0.25 x Signal
   If Score >= 0.60 -> "answer" (approved)
   Otherwise        -> "I don't know" (rejected)
```

**Optional Parts (OFF by default):**
- **LLM-as-judge (`app/judge.py`):** An independent LLM fact-checker that demands word-for-word proof from the passage.
- **Passage filter (`app/passage_filter.py`):** Drops irrelevant passages before answer drafting.

---

## Setup & Running

- **LLM:** `Qwen/Qwen3-30B-A3B-Instruct-2507` hosted on Nebius Token Factory
- **NLI Model:** `cross-encoder/nli-deberta-v3-base` (HuggingFace)
- **Dataset:** HaluEval QA benchmark (10,000 question-knowledge pairs)
- **Framework:** Python 3.12 with FastAPI

### Installation

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy the template and add your API key
copy .env.example .env

# 3. Start the FastAPI server
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

The API is accessible at `http://127.0.0.1:8000/docs` via `POST /check`.

---

## API Key Safety

- All API keys are kept strictly inside `.env`.
- `.env` is listed in `.gitignore` and is never committed to GitHub.
- A public template `.env.example` is provided with fake placeholder values.

---

## Results

### 1. Main Pipeline Test (300 Answers)
Tested on 150 HaluEval QA samples (150 right answers and 150 hallucinated answers) with default settings:
- **Right answers approved:** 105 / 150 (70.0%)
- **Hallucinations caught:** 106 / 150 (70.7%)
- **Total run time:** 40.4 minutes

### 2. Passage Filter Test (30 Samples / 60 Answers)
Tested with the passage filter turned OFF vs ON:
- **Filter OFF:** 17 / 30 correct approved (56.7%), 24 / 30 hallucinations caught (80.0%)
- **Filter ON:** 20 / 30 correct approved (66.7%), 21 / 30 hallucinations caught (70.0%)
- **Decision:** Kept **OFF** by default (each HaluEval sample has only 1 knowledge passage, so filtering gave no clear gain).

### 3. LLM-as-a-Judge Test (30 Samples in `judge_30.csv`)
Evaluated standalone on 30 sample pairs with exact word-for-word proof extraction:
- **Right answers approved (SUPPORTED):** 24 / 30 (80.0%)
- **Hallucinations caught (NOT SUPPORTED):** 26 / 30 (86.7%)
- **Both answers correct for sample:** 21 / 30 (70.0%)
- **Status:** Built in `app/judge.py`, kept **OFF** by default in the main pipeline.

### 4. Score Weight Tuning
- **On 30 samples (`scores_30.csv`):**
  - Current weights (`0.50 / 0.25 / 0.25`, cutoff `0.60`): 41 / 60 correct decisions (68.3%)
  - Best tuned mix (`0.40 / 0.30 / 0.30`, cutoff `0.45`): 46 / 60 correct decisions (76.7%)
- **Validation on 50 new samples (100 answers in `scores_100.csv`):**
  - Current mix (`0.50 / 0.25 / 0.25`, cutoff `0.60`): 73 / 100 (73.0% accuracy, 70.0% hallucinations caught)
  - Mix A (`0.40 / 0.30 / 0.30`, cutoff `0.45`): 75 / 100 (75.0% accuracy, 58.0% hallucinations caught)
  - Mix B (`0.70 / 0.10 / 0.20`, cutoff `0.30`): 74 / 100 (74.0% accuracy, 60.0% hallucinations caught)
- **Decision:** The accuracy gain was small (+2%) and Mix A caught significantly fewer hallucinations (58% vs 70%). Current weights were kept to maintain the strongest protection against hallucinations.

---

## Limits

- **~70% Accuracy:** The system is not perfect and still allows some hallucinations through or rejects difficult true facts.
- **Single Benchmark:** Evaluated exclusively on HaluEval QA; performance may differ on open-ended generation.
- **Latency:** Verifying an answer takes ~8 seconds due to multiple model calls and local NLI inferences.
- **Model Dependent:** Quality of verification questions, self-checks, and drafts depends on the underlying model.
- **Basic Web Search:** Web fallback relies on plain DuckDuckGo text snippets without complex multi-hop reasoning.
- **Small Test Sets:** Tuning and feature checks were evaluated on subsets (30 to 150 questions).
- **CoVe Skip Incomplete:** Skipping CoVe automatically when an answer drafts as "I don't know" is not fully finalized across every branch.

---

## Project Files

| File | Purpose |
| :--- | :--- |
| `app/main.py` | FastAPI application and `POST /check` endpoint |
| `app/pipeline.py` | Full 8-step verification pipeline and configuration toggles |
| `app/llm.py` | Unified LLM client for Nebius Token Factory |
| `app/retrieve.py` | TF-IDF passage retrieval and ranking (best passage first) |
| `app/answer.py` | Initial answer generation from retrieved passages |
| `app/facts.py` | Splits generated text into atomic factual statements |
| `app/verify.py` | Local DeBERTa cross-encoder NLI verification |
| `app/web_search.py` | DuckDuckGo search fallback for missing evidence |
| `app/cove.py` | Chain-of-Verification implementation |
| `app/consistency.py` | 5-sample self-consistency score |
| `app/signals.py` | Token confidence logprob extractor |
| `app/decision.py` | Scoring formula and threshold decision logic |
| `app/judge.py` | Optional LLM-as-judge with exact proof extraction |
| `app/passage_filter.py` | Optional passage relevance filter |
| `app/data_loader.py` | Parser for HaluEval QA samples |
| `run_big_test.py` | Runs 300-answer evaluation benchmark |
| `run_filter_test.py` | Evaluates passage filter ON vs OFF |
| `tune_weights.py` | Grid-search optimizer for scoring weights |
| `compare_mixes.py` | Compares weight configurations without model calls |
| `requirements.txt` | Python package dependencies |
| `.env.example` | Template for environment variables (safe to share) |

---

## References

The architecture and techniques in HaluGuard are based on these publications:

1. **Lewis, P. et al. (2020).** *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020. (Foundational RAG framework)
2. **Li, J. et al. (2023).** *HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models.* EMNLP 2023. (Evaluation dataset and methodology)
3. **Dhuliawala, S. et al. (2023).** *Chain-of-Verification Reduces Hallucination in Large Language Models.* arXiv:2309.11495. (`app/cove.py`)
4. **Min, S. et al. (2023).** *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP 2023. (`app/facts.py` and `app/verify.py`)
5. **Liu, N. F. et al. (2024).** *Lost in the Middle: How Language Models Use Long Contexts.* TACL 2024. (`app/retrieve.py` best-first ranking)
6. **Asai, A. et al. (2023).** *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* ICLR 2024. (`app/need_check.py` retrieval filtering)
7. **Manakul, P. et al. (2023).** *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models.* EMNLP 2023. (`app/consistency.py`)
8. **Orgad, H. et al. (2024).** *LLMs Know More Than They Show: On the Intrinsic Representation of Truth.* arXiv:2410.02707. (`app/signals.py` token confidence)
9. **Zheng, L. et al. (2023).** *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023. (`app/judge.py`)
