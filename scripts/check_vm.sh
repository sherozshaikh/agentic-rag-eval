#!/usr/bin/env bash
# ==============================================================================
# agentic-rag-eval — VM health check
# ------------------------------------------------------------------------------
# Verifies that the VM is ready to run evaluations after `vm_setup.sh`:
#
#   - Python 3.11+ available
#   - uv installed
#   - Ollama running and has qwen3.5:9b pulled
#   - Qdrant responding on :6333
#   - NVIDIA GPU visible (warning only)
#   - Disk space in cwd (warning below 20 GB)
#
# Exit status:
#   0 — all required checks passed
#   1 — one or more required checks failed
#
# Usage: ./scripts/check_vm.sh
# ==============================================================================

set -uo pipefail

# ----- Colors -----------------------------------------------------------------
if [[ -t 1 ]]; then
    C_RESET=$'\033[0m'; C_GREEN=$'\033[0;32m'; C_YELLOW=$'\033[0;33m'
    C_RED=$'\033[0;31m'; C_BLUE=$'\033[0;34m'; C_BOLD=$'\033[1m'
else
    C_RESET=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_BLUE=""; C_BOLD=""
fi

PASS="${C_GREEN}PASS${C_RESET}"
FAIL="${C_RED}FAIL${C_RESET}"
WARN="${C_YELLOW}WARN${C_RESET}"

FAILURES=0
WARNINGS=0

check()  { printf '  [%s] %s\n' "$1" "$2"; }
mark_fail() { FAILURES=$((FAILURES + 1)); check "${FAIL}" "$1"; [[ -n "${2:-}" ]] && printf '         %s\n' "$2"; }
mark_pass() { check "${PASS}" "$1"; [[ -n "${2:-}" ]] && printf '         %s\n' "$2"; }
mark_warn() { WARNINGS=$((WARNINGS + 1)); check "${WARN}" "$1"; [[ -n "${2:-}" ]] && printf '         %s\n' "$2"; }

section() { printf '\n%s%s%s\n' "${C_BOLD}" "$1" "${C_RESET}"; }

OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3.5:9b}"
QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"

# ==============================================================================
section "Python"
# ==============================================================================
if command -v python3 >/dev/null 2>&1; then
    PY_VERSION="$(python3 -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || echo unknown)"
    PY_MAJOR="$(python3 -c 'import sys; print(sys.version_info.major)' 2>/dev/null || echo 0)"
    PY_MINOR="$(python3 -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || echo 0)"
    if [[ "${PY_MAJOR}" -ge 3 && "${PY_MINOR}" -ge 11 ]]; then
        mark_pass "python3 ${PY_VERSION} (>= 3.11)"
    else
        mark_fail "python3 ${PY_VERSION} is too old; need >= 3.11"
    fi
else
    mark_fail "python3 not found in PATH"
fi

# ==============================================================================
section "uv"
# ==============================================================================
if command -v uv >/dev/null 2>&1; then
    mark_pass "$(uv --version)"
else
    mark_fail "uv not installed" "Run ./scripts/vm_setup.sh or: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

# ==============================================================================
section "Ollama"
# ==============================================================================
if command -v ollama >/dev/null 2>&1; then
    mark_pass "ollama CLI present: $(ollama --version 2>/dev/null || echo 'unknown')"

    if curl -fsS "${OLLAMA_URL}/api/version" >/dev/null 2>&1; then
        mark_pass "Ollama responding on ${OLLAMA_URL}"

        MODELS_JSON="$(curl -fsS "${OLLAMA_URL}/api/tags" 2>/dev/null || echo '')"
        if [[ -n "${MODELS_JSON}" ]] && printf '%s' "${MODELS_JSON}" | grep -q "\"${OLLAMA_MODEL}\""; then
            mark_pass "Model '${OLLAMA_MODEL}' is pulled"
        elif ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fxq "${OLLAMA_MODEL}"; then
            mark_pass "Model '${OLLAMA_MODEL}' is pulled (via ollama list)"
        else
            mark_fail "Model '${OLLAMA_MODEL}' not pulled" "Run: ollama pull ${OLLAMA_MODEL}"
        fi
    else
        mark_fail "Ollama not responding on ${OLLAMA_URL}" "Start with: sudo systemctl start ollama   (or: ollama serve)"
    fi
else
    mark_fail "ollama CLI not installed" "Install with: curl -fsSL https://ollama.com/install.sh | sh"
fi

# ==============================================================================
section "Qdrant"
# ==============================================================================
if curl -fsS "${QDRANT_URL}/readyz" >/dev/null 2>&1; then
    mark_pass "Qdrant /readyz on ${QDRANT_URL}"
elif curl -fsS "${QDRANT_URL}/" >/dev/null 2>&1; then
    mark_pass "Qdrant responding on ${QDRANT_URL} (no /readyz — older version)"
else
    mark_fail "Qdrant not responding on ${QDRANT_URL}" "Start with: docker compose -f docker/docker-compose.yml up -d qdrant"
fi

# ==============================================================================
section "GPU"
# ==============================================================================
if command -v nvidia-smi >/dev/null 2>&1; then
    if GPU_INFO="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)" && [[ -n "${GPU_INFO}" ]]; then
        while IFS= read -r line; do
            [[ -n "${line}" ]] && mark_pass "GPU: ${line}"
        done <<< "${GPU_INFO}"
    else
        mark_warn "nvidia-smi present but returned no GPUs"
    fi
else
    mark_warn "nvidia-smi not found — GPU mode unavailable (OK for API-only deployments)"
fi

# ==============================================================================
section "Disk space"
# ==============================================================================
if command -v df >/dev/null 2>&1; then
    CWD_FREE_KB="$(df -Pk . | awk 'NR==2 {print $4}' 2>/dev/null || echo 0)"
    CWD_FREE_GB=$(( CWD_FREE_KB / 1024 / 1024 ))
    CWD_MOUNT="$(df -P . | awk 'NR==2 {print $6}' 2>/dev/null || echo .)"
    if [[ "${CWD_FREE_GB}" -ge 20 ]]; then
        mark_pass "${CWD_FREE_GB} GB free on ${CWD_MOUNT}"
    elif [[ "${CWD_FREE_GB}" -ge 5 ]]; then
        mark_warn "${CWD_FREE_GB} GB free on ${CWD_MOUNT} (< 20 GB; evaluations may fill disk)"
    else
        mark_fail "${CWD_FREE_GB} GB free on ${CWD_MOUNT} (< 5 GB; too low)"
    fi
else
    mark_warn "df not available; cannot check disk space"
fi

# ==============================================================================
section "Summary"
# ==============================================================================
if [[ "${FAILURES}" -eq 0 ]]; then
    printf '  %sAll required checks passed%s' "${C_GREEN}${C_BOLD}" "${C_RESET}"
    if [[ "${WARNINGS}" -gt 0 ]]; then
        printf ' (%d warnings)\n' "${WARNINGS}"
    else
        printf '\n'
    fi
    exit 0
else
    printf '  %s%d check(s) failed%s' "${C_RED}${C_BOLD}" "${FAILURES}" "${C_RESET}"
    if [[ "${WARNINGS}" -gt 0 ]]; then
        printf ', %d warning(s)\n' "${WARNINGS}"
    else
        printf '\n'
    fi
    exit 1
fi
