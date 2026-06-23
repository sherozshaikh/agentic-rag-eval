# agentic-rag-eval

**A component ablation of agentic RAG for multi-hop QA, on a local 7B model.**

This repository accompanies the paper *Dissecting Agentic RAG: A Component Ablation for Multi-Hop QA with a Local 7B Model*. It implements an agentic retrieval-augmented generation pipeline and a controlled ablation study that isolates the contribution of each component, run entirely on a local Qwen2.5-7B-Instruct model with no proprietary APIs.

📄 **Paper:** [arXiv:2606.21553](https://arxiv.org/abs/2606.21553)

The pipeline is **plan-and-execute** (not a per-step ReAct loop): a decomposer splits each question into sub-questions up front, an iterative retrieval loop gathers evidence for each (rule-based routing over Qdrant dense / sparse / RRF hybrid, then cross-encoder reranking), and a single synthesis call produces the answer. The model is not queried between retrieval steps.

---

## Key findings (5,000 HotpotQA distractor questions, Qwen2.5-7B-Instruct, local)

- **Fixed hybrid retrieval beats rule-based adaptive routing** by +1.8 EM / +1.9 F1 (*p* < 0.001). The routing heuristic fires on named entities and over-routes to BM25, forgoing the complementary dense signal.
- **Two retrieval iterations capture 95% of the gains of five** (*p* < 0.001 vs. one step) — deeper loops add nothing on two-hop questions.
- **Query decomposition (*p* = 0.004) and cross-encoder reranking (*p* < 0.001)** each contribute statistically significant but smaller gains.

| System | EM | F1 | Latency (ms) |
|---|---|---|---|
| Baseline (single-pass dense) | 43.1 | 54.0 | 546 |
| Agentic (full) | 53.2 | 61.6 | 5,642 |
| **Hybrid-only** | **55.0** | **63.5** | 5,688 |

Significance uses paired tests on the shared 5,000-question sample (McNemar's exact test for EM, Wilcoxon signed-rank for F1).

---

## Quick Start

### 1. Install dependencies
```bash
uv sync --extra dev
```

### 2. Configure environment
```bash
cp .env.example .env
# Defaults run fully local via Ollama (qwen2.5:7b-instruct). Edit only if using the API backend.
```

### 3. Start Qdrant
```bash
docker compose -f docker/docker-compose.yml up -d qdrant
```

### 4. Seed the HotpotQA corpus (one-time)
```bash
make seed
```

### 5. Run the evaluations
```bash
make baseline   # single-pass dense baseline
make eval       # full agentic pipeline
```

### 6. (Optional) API + dashboard
```bash
make serve
# Dashboard: http://localhost:8000/dashboard
# API docs:  http://localhost:8000/docs
```

---

## Reproducing the paper

The ablation runs write per-question traces to DuckDB under `results/`. The two main runs (`traces_agentic_5k.duckdb`, `traces_baseline_5k.duckdb`) are included; the ablation-variant databases are produced by the `ablation-*` targets in the `Makefile`.

Analysis scripts (read the DuckDB traces, write `results/*.csv` and figures):

```bash
python scripts/bootstrap_ci.py          # 95% bootstrap CIs per system
python scripts/paired_significance.py    # McNemar (EM) + Wilcoxon (F1) paired tests
python scripts/routing_breakdown.py      # retrieval-strategy distribution
python scripts/qualitative_analysis.py   # example failure/recovery cases
python scripts/generate_paper_figures.py # figures in results/figures/
```

The 5,000-question sample is drawn with a fixed seed (`HOTPOTQA_RANDOM_SEED=42`) for reproducibility.

---

## Stack

- **LLM:** Qwen2.5-7B-Instruct via Ollama (local), or an OpenAI-compatible API backend
- **Embeddings:** BGE-small-en-v1.5 via FastEmbed
- **Sparse:** Qdrant/bm25 via FastEmbed
- **Vector DB:** Qdrant (native dense + sparse + RRF fusion)
- **Reranker:** cross-encoder/ms-marco-MiniLM-L-6-v2
- **Agent:** plan-and-execute pipeline (decompose → iterative retrieval → synthesize)
- **Storage:** DuckDB (per-question traces)
- **Serving:** FastAPI + Jinja2 + Chart.js dashboard

The codebase also includes optional, off-by-default integrations (Mem0 long-term memory; RAGAS / DeepEval / LLM-as-judge evaluators). These are **not** used in the paper's results, which report Exact Match and token-level F1 from the official HotpotQA script.

---

## Environment Variables

See `.env.example` for the full list. Key variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_BACKEND` | `local` (Ollama) or `api` (OpenAI-compatible) | `local` |
| `LLM_MODEL` | Reasoning / generation model | `qwen2.5:7b-instruct` |
| `QDRANT_HOST` | Qdrant host | `localhost` |
| `MAX_AGENT_STEPS` | Max retrieval iterations | `5` |
| `CONTEXT_BUDGET_TOKENS` | Agent context budget | `16000` |
| `HOTPOTQA_SUBSET_SIZE` | Evaluation sample size | `5000` |
| `HOTPOTQA_RANDOM_SEED` | Sample seed | `42` |

The judge/eval model variables (`EVAL_LLM_*`) configure the optional evaluators and are not exercised in the paper.

---

## Development

```bash
make test          # run all tests
make lint          # ruff linter
make format        # ruff formatter
make typecheck     # mypy
```

---

## Citation

```bibtex
@misc{shaikh2026agenticrag,
  title         = {Dissecting Agentic RAG: A Component Ablation for Multi-Hop QA with a Local 7B Model},
  author        = {Shaikh, Sheroz},
  year          = {2026},
  eprint        = {2606.21553},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2606.21553}
}
```

---

## License

MIT
