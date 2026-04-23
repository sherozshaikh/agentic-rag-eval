# agentic-rag-eval

**Multi-Hop Agentic RAG — Benchmarked on HotpotQA at Scale**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade agentic RAG system that decomposes multi-hop questions into sub-retrieval steps, adapts its retrieval strategy per query, re-ranks with a cross-encoder, and synthesizes answers via a ReAct loop — rigorously evaluated against a naive single-shot RAG baseline on 5,000 HotpotQA questions.

**Dual-backend:** runs fully local on a single GPU (Ollama) or via any OpenAI-compatible LLM API.

---

## Results

Evaluated on a stratified 5,000-question subset of HotpotQA using `qwen2.5:7b-instruct` on a single NVIDIA RTX A6000 (49GB VRAM).

| Pipeline | Questions | Exact Match | F1 | Avg Latency |
|---|---|---|---|---|
| **Agentic RAG** | 5,000 | **53.2%** | **61.6%** | 5,642 ms |
| Baseline RAG | 5,000 | 43.1% | 54.0% | 546 ms |
| **Delta** | | **+10.1%** | **+7.6%** | +5,096 ms |

**Agentic pipeline improves EM by +10.1 points and F1 by +7.6 points over single-shot dense RAG.**

### Retrieval Strategy Breakdown (Agentic)

| Strategy | Questions | EM | F1 |
|---|---|---|---|
| Sparse (BM25) | 3,961 (79%) | 53.9% | 62.0% |
| Hybrid (Dense + Sparse + RRF) | 814 (16%) | 50.9% | 59.8% |
| Dense | 225 (5%) | 50.7% | 59.8% |

The adaptive retriever selects BM25 for most multi-hop queries — factoid questions with named entities benefit from exact-term matching over semantic similarity.

### Latency Percentiles

| Percentile | Agentic | Baseline |
|---|---|---|
| p50 | 5,565 ms | 475 ms |
| p90 | 6,983 ms | 819 ms |
| p99 | 8,748 ms | 1,296 ms |

The latency cost (~5s/question) reflects the multi-step ReAct loop (3–7 LLM calls per question). Baseline is single-shot.

### Context: How This Compares to Published Results

> **Note:** Direct comparison across systems is not straightforward — evaluation splits, retrieval corpora, and passage access conditions differ across papers. The table below is for orientation only. Our setup: 5K stratified HotpotQA training subset, fullwiki-style (no gold passages provided), fully local inference.

| System | Model | EM | F1 | Setting | Source |
|---|---|---|---|---|---|
| **Ours (Agentic RAG)** | **qwen2.5:7b-instruct (local, free)** | **53.2%** | **61.6%** | **Fullwiki-style, no gold passages** | **This repo** |
| PRISM Agentic RAG | GPT-based | 54.2% | 67.0% | Fullwiki-style | [arXiv:2510.14278](https://arxiv.org/abs/2510.14278) |
| GPT-4o-mini Soft RAG | GPT-4o-mini | — | 60.9% | Hard questions subset | [arXiv:2604.09174](https://arxiv.org/abs/2604.09174) |
| Gemini-2.0-Flash Soft RAG | Gemini 2.0 Flash | — | 57.9% | Hard questions subset | [arXiv:2604.09174](https://arxiv.org/abs/2604.09174) |
| LLaMA-3-8B Soft RAG | LLaMA-3-8B | — | 29.6% | Hard questions subset | [arXiv:2604.09174](https://arxiv.org/abs/2604.09174) |
| StepChain GraphRAG (SOTA) | GPT-4o | 66.7% | 79.5% | Fullwiki-style | [arXiv:2510.02827](https://arxiv.org/abs/2510.02827) |

**Key takeaway:** A fully local 7B model with agentic RAG (zero API cost) achieves F1=61.6% — matching GPT-4o-mini Soft RAG (F1=60.9%) and beating Gemini-2.0-Flash Soft RAG (F1=57.9%) on hard multi-hop questions, while running entirely on-device.

---

## Architecture

```
Question
   │
   ▼
┌──────────────────────────────────────────┐
│            Decomposer                    │
│  Classifies query type (bridge/compare)  │
│  Generates sub-questions (JSON)          │
└───────────────────┬──────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────┐
│           ReAct Agent Loop               │
│  (up to 5 steps)                         │
│                                          │
│  Thought → Action → Observation          │
│                │                         │
│    ┌───────────▼────────────┐            │
│    │  Adaptive Retriever    │            │
│    │  ┌──────┬───────────┐  │            │
│    │  │Dense │ Sparse    │  │            │
│    │  │(BGE) │ (BM25)    │  │            │
│    │  └──┬───┴───────────┘  │            │
│    │     │ RRF Hybrid       │            │
│    │     ▼                  │            │
│    │  Cross-Encoder Rerank  │            │
│    │  (ms-marco-MiniLM)     │            │
│    └────────────────────────┘            │
│                                          │
└───────────────────┬──────────────────────┘
                    │
                    ▼
             Final Answer
                    │
                    ▼
┌──────────────────────────────────────────┐
│         Evaluation Pipeline              │
│  HotpotQA EM + F1  │  Retrieval metrics  │
│  RAGAS             │  DeepEval           │
│  LLM-as-Judge      │  Failure classifier │
│  DuckDB checkpoint (per-question)        │
└──────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | `qwen2.5:7b-instruct` via Ollama (local) or any OpenAI-compatible API |
| Embeddings | `BAAI/bge-small-en-v1.5` via FastEmbed |
| Sparse | `Qdrant/bm25` via FastEmbed |
| Vector DB | Qdrant (native dense + sparse + RRF hybrid) |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Agent | Custom ReAct loop (LangGraph-style, JSON actions) |
| Memory | Mem0 (long-term cross-session memory) |
| Evaluation | HotpotQA EM/F1 + RAGAS + DeepEval + LLM-as-Judge |
| Storage | DuckDB (per-question checkpointing + resume) |
| Serving | FastAPI + Jinja2 + Chart.js |
| Package manager | uv |

---

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/sherozshaikh/agentic-rag-eval.git
cd agentic-rag-eval
uv sync --extra dev
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set LLM_API_KEY / EVAL_LLM_API_KEY if using API backend
```

Key variables:

| Variable | Description | Default |
|---|---|---|
| `LLM_BACKEND` | `local` (Ollama) or `api` (OpenRouter) | `local` |
| `LLM_MODEL` | Reasoning model | `qwen2.5:7b-instruct` |
| `EVAL_LLM_MODEL` | Judge model | `google/gemini-2.5-flash` |
| `QDRANT_HOST` | Qdrant host | `localhost` |
| `MAX_AGENT_STEPS` | Max ReAct iterations | `5` |
| `CONTEXT_BUDGET_TOKENS` | Agent context window budget | `16000` |

### 3. Start Qdrant

```bash
docker compose -f docker/docker-compose.yml up -d qdrant
```

### 4. Seed HotpotQA corpus (one-time)

Downloads HotpotQA, builds a stratified 5K subset, indexes ~3M passages into Qdrant with dense + sparse vectors.

```bash
make seed
```

### 5. Run evaluations

```bash
make eval-debug     # 50 questions — smoke test
make eval           # 5,000 questions — full subset
make baseline       # naive RAG baseline for comparison
```

### 6. Analyze results

```bash
python scripts/analyze_results.py           # print metrics to terminal
python scripts/analyze_results.py --export  # also writes results/metrics_comparison.csv
```

### 7. Start the API + dashboard

```bash
make serve
# Dashboard:  http://localhost:8000/dashboard
# API docs:   http://localhost:8000/docs
```

---

## Evaluation Design

The evaluation compares two pipelines on the same 5,000-question HotpotQA subset:

**Agentic RAG** — question decomposition → ReAct loop → adaptive retrieval → cross-encoder rerank → synthesis

**Baseline RAG** — single dense retrieval (top-10) → one-shot LLM synthesis (no decomposition, no iteration)

The baseline is intentionally minimal — any improvement must come from architecture, not prompt engineering.

### Metrics

| Metric | Description |
|---|---|
| **Exact Match (EM)** | 1.0 if normalized prediction equals normalized gold answer |
| **F1** | Token-level F1 between prediction and gold (official HotpotQA scoring) |
| SF Precision / Recall / F1 | Supporting fact title overlap |
| Recall@5/10/20 | Passage retrieval recall |
| MRR, NDCG@10 | Ranking quality |

EM and F1 use the official HotpotQA normalization (lowercase, strip articles/punctuation).

### Checkpointing

Every question result is written to DuckDB immediately after processing. If a run crashes, it resumes from the last checkpoint automatically — no re-computation of completed questions.

---

## Project Structure

```
agentic-rag-eval/
├── src/agentic_rag_eval/
│   ├── agent/
│   │   ├── react_agent.py          # ReAct loop (Thought → Action → Observation)
│   │   ├── decomposer.py           # Query decomposition + type classification
│   │   └── memory.py               # Mem0 long-term memory store
│   ├── retrieval/
│   │   ├── retriever.py            # Qdrant dense + sparse + hybrid retrieval
│   │   ├── qdrant_client.py        # Qdrant connection + collection management
│   │   ├── adaptive.py             # Strategy selector (dense / sparse / hybrid)
│   │   ├── reranker.py             # Cross-encoder re-ranking
│   │   ├── embeddings.py           # FastEmbed dense + sparse encoders
│   │   └── strategy_selector.py    # Query-type → strategy routing
│   ├── baseline/
│   │   ├── naive_rag.py            # Single-shot dense RAG pipeline
│   │   └── run_baseline.py         # CLI runner for baseline eval
│   ├── evaluation/
│   │   ├── runner.py               # EvalRunner with per-question DuckDB checkpointing
│   │   ├── run_eval.py             # CLI entry point (--pipeline, --subset/--full, --limit)
│   │   ├── hotpotqa_metrics.py     # Official HotpotQA EM + F1 implementation
│   │   ├── retrieval_metrics.py    # Recall@K, MRR, NDCG
│   │   ├── judge.py                # LLM-as-Judge (coherence + completeness)
│   │   ├── failure_classifier.py   # Failure mode classification
│   │   ├── ragas_evaluator.py      # RAGAS integration
│   │   ├── deepeval_evaluator.py   # DeepEval integration
│   │   └── comparison.py           # Cross-run comparison matrix
│   ├── llm/
│   │   └── client.py               # OpenAI-compatible client (Ollama + API, tenacity retries)
│   ├── tracing/
│   │   ├── logger.py               # DuckDB trace logger (traces, spans, llm_calls, eval_records)
│   │   └── cleanup.py              # Trace retention / pruning
│   ├── serving/
│   │   ├── app.py                  # FastAPI app
│   │   ├── auth.py                 # API key authentication
│   │   ├── deps.py                 # FastAPI dependency injection
│   │   ├── validators.py           # Request validation helpers
│   │   └── routes/                 # dashboard, query, evaluate, health, metrics
│   ├── data/
│   │   ├── seed.py                 # Download + stratify + index HotpotQA
│   │   ├── loader.py               # HotpotQA dataset loader
│   │   ├── index_passages.py       # Qdrant indexing pipeline
│   │   ├── passages.py             # Passage schema + utilities
│   │   ├── subset.py               # Stratified subset builder
│   │   └── validate.py             # Data integrity checks
│   ├── dashboard/                  # Dashboard helpers
│   ├── prompts/
│   │   └── loader.py               # YAML prompt loader
│   ├── config.py                   # Pydantic settings (env / .env)
│   ├── schemas.py                  # Shared dataclasses (QueryResponse, EvalRecord, etc.)
│   ├── eval_run_id.py              # Deterministic run-ID generation
│   └── logging_setup.py            # Structured JSON / text logging
├── prompts/
│   ├── decompose.yaml              # Question decomposition prompt
│   ├── reason.yaml                 # ReAct reasoning prompt
│   ├── judge_coherence.yaml        # LLM judge — coherence scoring
│   ├── judge_completeness.yaml     # LLM judge — completeness scoring
│   └── failure_classify.yaml       # Failure mode classification prompt
├── static/
│   ├── css/dashboard.css           # Dashboard stylesheet
│   └── js/dashboard.js             # Dashboard Chart.js charts
├── templates/
│   ├── base.html                   # Jinja2 base layout
│   ├── dashboard.html              # Metrics dashboard
│   ├── run_detail.html             # Per-run drill-down
│   └── traces.html                 # Trace explorer
├── scripts/
│   └── analyze_results.py          # Metrics report from DuckDB → terminal + CSV
├── tests/
│   ├── unit/                       # Unit tests (config, metrics, retrieval, serving)
│   ├── integration/                # Integration tests (end-to-end agent, data)
│   └── smoke/                      # Smoke tests (Docker health, query endpoint)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml          # Qdrant + app + Ollama (GPU profile)
├── results/
│   ├── metrics_comparison.csv      # Exported eval results (agentic vs baseline)
│   ├── traces_agentic_5k.duckdb    # Agentic pipeline results — 5,000 questions
│   └── traces_baseline_5k.duckdb  # Baseline pipeline results — 5,000 questions
├── .env.example
├── Makefile
└── pyproject.toml
```

---

## Makefile Reference

```
Setup:
  make install          Install production dependencies
  make install-dev      Install dev dependencies

Data:
  make seed             Download HotpotQA, build subset, index into Qdrant
  make index            Re-index passages (assumes data already downloaded)

Evaluation:
  make eval-debug       50 questions — smoke test
  make eval             5,000 questions — full subset (agentic pipeline)
  make eval-full        Full HotpotQA validation set
  make baseline         Naive RAG baseline on subset

Development:
  make test             Run all tests
  make test-unit        Unit tests only
  make test-integration Integration tests only
  make lint             Ruff linter
  make format           isort + black + ruff
  make typecheck        mypy

Serving:
  make serve            FastAPI server (localhost:8000)

Docker:
  make docker-build     Build image
  make docker-up        Start app + Qdrant
  make docker-gpu-up    Start app + Qdrant + Ollama (GPU)
  make docker-down      Stop services
```

---

## Hardware Used

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX A6000 (49GB VRAM) |
| LLM | `qwen2.5:7b-instruct` via Ollama |
| Agentic eval (5K) | ~9 hours |
| Baseline eval (5K) | ~1 hour |

The system also runs on CPU-only (API backend) — set `LLM_BACKEND=api` and provide an OpenRouter key.

---

## Development

```bash
make test         # all tests
make lint         # ruff
make format       # isort + black + ruff
make typecheck    # mypy
```

---

## License

MIT

---

## Author

**Sheroz Shaikh** — [Portfolio](https://sherozshaikh.github.io/) | [GitHub](https://github.com/sherozshaikh) | [LinkedIn](https://www.linkedin.com/in/shaikh-sheroz-07s/)
