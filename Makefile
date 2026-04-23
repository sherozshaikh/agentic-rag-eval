# ----- Setup -----
install:
	uv sync

install-dev:
	uv sync --extra dev

# ----- Data -----
seed:
	uv run python -m agentic_rag_eval.data.seed

index:
	uv run python -m agentic_rag_eval.data.index_passages

# ----- Evaluation -----
baseline:
	uv run python -m agentic_rag_eval.baseline.run_baseline

eval-debug:
	uv run python -m agentic_rag_eval.evaluation.run_eval --subset --pipeline agentic --limit 50 --no-judge --no-failure-classifier

eval:
	uv run python -m agentic_rag_eval.evaluation.run_eval --subset --pipeline agentic --no-judge --no-failure-classifier

eval-full:
	uv run python -m agentic_rag_eval.evaluation.run_eval --full --pipeline agentic --no-judge --no-failure-classifier

# ----- Testing -----
test:
	uv run pytest

test-unit:
	uv run pytest -m unit

test-integration:
	uv run pytest -m integration

lint:
	uv run ruff check src tests

format:
	isort . && black . && ruff check --fix . && ruff format .

clean-build:
	@echo "working on cleanup"

clean: clean-build
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -type f -delete
	find . -name "*.pyo" -type f -delete
	find . -name ".DS_Store" -type f -delete
	find . -type d -name ".vscode" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".idea" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".tox" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .verify_venv

typecheck:
	uv run mypy src

# ----- Serving -----
serve:
	uv run uvicorn agentic_rag_eval.serving.app:app --host 0.0.0.0 --port 8000 --reload

clean-traces:
	uv run python -m agentic_rag_eval.tracing.cleanup

# ----- Docker -----
docker-build:
	docker build -f docker/Dockerfile -t agentic-rag-eval:latest .

docker-up:
	docker compose -f docker/docker-compose.yml up -d

docker-gpu-up:
	docker compose -f docker/docker-compose.yml --profile gpu up -d

docker-down:
	docker compose -f docker/docker-compose.yml down

.PHONY: help install install-dev seed index baseline eval eval-full test test-unit test-integration lint format typecheck clean clean-traces serve docker-build docker-up docker-down docker-gpu-up

help:
	@echo "agentic-rag-eval — Makefile targets"
	@echo ""
	@echo "Setup:"
	@echo "  install           Install production dependencies (uv sync)"
	@echo "  install-dev       Install dev dependencies"
	@echo ""
	@echo "Data + indexing:"
	@echo "  seed              Download HotpotQA, stratified subset, index into Qdrant"
	@echo "  index             Re-index passages into Qdrant (assumes data is downloaded)"
	@echo ""
	@echo "Evaluation:"
	@echo "  baseline          Run naive RAG baseline on HotpotQA subset"
	@echo "  eval              Run full agentic pipeline eval on subset"
	@echo "  eval-full         Run full eval on 7.4K HotpotQA validation set"
	@echo ""
	@echo "Testing:"
	@echo "  test              Run all tests"
	@echo "  test-unit         Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  lint              Run ruff linter"
	@echo "  format            Run ruff formatter"
	@echo "  typecheck         Run mypy"
	@echo ""
	@echo "Serving:"
	@echo "  serve             Run FastAPI server locally"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean             Remove caches + build artifacts"
	@echo "  clean-traces      Remove trace data older than TRACE_RETENTION_DAYS"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build      Build Docker image"
	@echo "  docker-up         Start services (API mode: app + qdrant)"
	@echo "  docker-gpu-up     Start services with local LLM (app + qdrant + ollama)"
	@echo "  docker-down       Stop services"
