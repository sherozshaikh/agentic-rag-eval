#!/usr/bin/env bash
# ==============================================================================
# agentic-rag-eval — VM setup script
# ------------------------------------------------------------------------------
# Provisions a fresh Ubuntu/Debian VM for running the hybrid deployment:
#   - uv + Python deps (native)
#   - Ollama (native, for GPU access)
#   - Qdrant (Docker container, named volume)
#
# Usage:
#   ./scripts/vm_setup.sh [--skip-ollama] [--skip-model] [--skip-qdrant]
#
# Flags:
#   --skip-ollama   Don't install Ollama or start its service.
#   --skip-model    Don't pull the default LLM (qwen2.5:7b-instruct).
#   --skip-qdrant   Don't start the Qdrant container.
#   -h, --help      Show this help.
# ==============================================================================

set -euo pipefail

# ----- Colors -----------------------------------------------------------------
if [[ -t 1 ]] && command -v tput >/dev/null 2>&1 && [[ $(tput colors 2>/dev/null || echo 0) -ge 8 ]]; then
    C_RESET=$'\033[0m'
    C_GREEN=$'\033[0;32m'
    C_YELLOW=$'\033[0;33m'
    C_RED=$'\033[0;31m'
    C_BLUE=$'\033[0;34m'
    C_BOLD=$'\033[1m'
else
    C_RESET=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_BLUE=""; C_BOLD=""
fi

log_info()  { printf '%s[INFO]%s  %s\n'  "${C_BLUE}"   "${C_RESET}" "$*"; }
log_ok()    { printf '%s[ OK ]%s  %s\n'  "${C_GREEN}"  "${C_RESET}" "$*"; }
log_warn()  { printf '%s[WARN]%s  %s\n'  "${C_YELLOW}" "${C_RESET}" "$*" >&2; }
log_error() { printf '%s[FAIL]%s  %s\n'  "${C_RED}"    "${C_RESET}" "$*" >&2; }
log_step()  { printf '\n%s==> %s%s\n'    "${C_BOLD}"   "$*" "${C_RESET}"; }

die() { log_error "$*"; exit 1; }

# ----- Flags ------------------------------------------------------------------
SKIP_OLLAMA=0
SKIP_MODEL=0
SKIP_QDRANT=0

usage() {
    sed -n '2,20p' "$0"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-ollama) SKIP_OLLAMA=1 ;;
        --skip-model)  SKIP_MODEL=1 ;;
        --skip-qdrant) SKIP_QDRANT=1 ;;
        -h|--help)     usage; exit 0 ;;
        *) die "Unknown flag: $1 (try --help)" ;;
    esac
    shift
done

# ----- Paths ------------------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/docker/docker-compose.yml"
ENV_EXAMPLE="${PROJECT_ROOT}/.env.example"
ENV_FILE="${PROJECT_ROOT}/.env"

cd "${PROJECT_ROOT}"

OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:7b-instruct}"

# ==============================================================================
# Step 1 — Verify OS
# ==============================================================================
log_step "Step 1/9: Verifying operating system"

if [[ ! -f /etc/os-release ]]; then
    die "/etc/os-release not found; this script targets Ubuntu/Debian Linux."
fi
# shellcheck source=/dev/null
. /etc/os-release
OS_ID="${ID:-unknown}"
OS_LIKE="${ID_LIKE:-}"
OS_VERSION="${VERSION_ID:-unknown}"

case "${OS_ID} ${OS_LIKE}" in
    *ubuntu*|*debian*)
        log_ok "Detected: ${PRETTY_NAME:-${OS_ID} ${OS_VERSION}}"
        ;;
    *)
        log_warn "This script is tested on Ubuntu/Debian. Detected: ${OS_ID} ${OS_VERSION}"
        log_warn "Continuing anyway, but some steps may fail."
        ;;
esac

if [[ "$(uname -s)" != "Linux" ]]; then
    die "This script requires Linux. Detected: $(uname -s)"
fi

# ==============================================================================
# Step 2 — Install base system dependencies
# ==============================================================================
log_step "Step 2/9: Checking base dependencies (curl, git, docker)"

missing_bins=()
for bin in curl git; do
    if ! command -v "${bin}" >/dev/null 2>&1; then
        missing_bins+=("${bin}")
    fi
done

if (( ${#missing_bins[@]} > 0 )); then
    log_warn "Missing: ${missing_bins[*]} — attempting to install with apt."
    if command -v sudo >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y --no-install-recommends "${missing_bins[@]}"
    else
        apt-get update
        apt-get install -y --no-install-recommends "${missing_bins[@]}"
    fi
fi
log_ok "curl and git available."

if ! command -v docker >/dev/null 2>&1; then
    log_warn "docker is not installed."
    log_warn "Install Docker Engine + compose plugin following https://docs.docker.com/engine/install/"
    if [[ ${SKIP_QDRANT} -eq 0 ]]; then
        die "Docker is required to start Qdrant. Re-run with --skip-qdrant to bypass."
    fi
else
    if ! docker compose version >/dev/null 2>&1; then
        die "Docker is installed but the 'compose' plugin is missing. Install docker-compose-plugin."
    fi
    log_ok "docker + compose plugin available."
fi

# ==============================================================================
# Step 3 — Install uv
# ==============================================================================
log_step "Step 3/9: Installing uv"

if command -v uv >/dev/null 2>&1; then
    log_ok "uv already installed: $(uv --version)"
else
    log_info "Downloading uv installer from https://astral.sh/uv/install.sh"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Make uv visible in THIS shell even before the user re-sources their profile.
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
    if ! command -v uv >/dev/null 2>&1; then
        die "uv installation failed — uv not on PATH after install."
    fi
    log_ok "uv installed: $(uv --version)"
fi

# ==============================================================================
# Step 4 — uv sync
# ==============================================================================
log_step "Step 4/9: Resolving Python dependencies (uv sync --extra dev)"

if [[ ! -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
    die "pyproject.toml not found at ${PROJECT_ROOT}"
fi

uv sync --extra dev
log_ok "Python environment ready at ${PROJECT_ROOT}/.venv"

# ==============================================================================
# Step 5 — Install Ollama
# ==============================================================================
log_step "Step 5/9: Installing Ollama"

if (( SKIP_OLLAMA )); then
    log_warn "Skipping Ollama install (--skip-ollama)."
else
    if command -v ollama >/dev/null 2>&1; then
        log_ok "Ollama already installed: $(ollama --version 2>/dev/null || echo 'version unknown')"
    else
        log_info "Downloading Ollama installer from https://ollama.com/install.sh"
        curl -fsSL https://ollama.com/install.sh | sh
        if ! command -v ollama >/dev/null 2>&1; then
            die "Ollama installation failed."
        fi
        log_ok "Ollama installed: $(ollama --version 2>/dev/null || echo 'version unknown')"
    fi
fi

# ==============================================================================
# Step 6 — Start Ollama service
# ==============================================================================
log_step "Step 6/9: Starting Ollama systemd service"

if (( SKIP_OLLAMA )); then
    log_warn "Skipping (--skip-ollama)."
elif ! command -v systemctl >/dev/null 2>&1; then
    log_warn "systemctl not found — Ollama service must be started manually."
else
    SUDO=""
    if [[ ${EUID} -ne 0 ]]; then
        SUDO="sudo"
    fi
    if ${SUDO} systemctl enable --now ollama 2>/dev/null; then
        log_ok "ollama.service enabled and started."
    else
        log_warn "Could not enable ollama.service (is it installed as a systemd unit?)."
        log_warn "You may need to run 'ollama serve' manually."
    fi

    # Wait briefly for it to accept connections.
    for i in 1 2 3 4 5 6 7 8 9 10; do
        if curl -fsS "http://127.0.0.1:11434/api/version" >/dev/null 2>&1; then
            log_ok "Ollama is responding on :11434"
            break
        fi
        sleep 1
        if [[ ${i} -eq 10 ]]; then
            log_warn "Ollama did not respond on :11434 after 10s."
        fi
    done
fi

# ==============================================================================
# Step 7 — Pull default model
# ==============================================================================
log_step "Step 7/9: Pulling LLM model (${OLLAMA_MODEL})"

if (( SKIP_MODEL )) || (( SKIP_OLLAMA )); then
    log_warn "Skipping model pull."
elif ! command -v ollama >/dev/null 2>&1; then
    log_warn "ollama binary not available; skipping model pull."
else
    log_info "This may take a while (multi-GB download)."
    if ollama pull "${OLLAMA_MODEL}"; then
        log_ok "Model pulled: ${OLLAMA_MODEL}"
    else
        log_warn "Failed to pull ${OLLAMA_MODEL}. You can retry later with: ollama pull ${OLLAMA_MODEL}"
    fi
fi

# ==============================================================================
# Step 8 — Start Qdrant container
# ==============================================================================
log_step "Step 8/9: Starting Qdrant container"

if (( SKIP_QDRANT )); then
    log_warn "Skipping Qdrant start (--skip-qdrant)."
elif [[ ! -f "${COMPOSE_FILE}" ]]; then
    die "docker-compose file not found: ${COMPOSE_FILE}"
else
    docker compose -f "${COMPOSE_FILE}" up -d qdrant
    log_ok "Qdrant container started."

    log_info "Waiting for Qdrant to become healthy..."
    for i in $(seq 1 20); do
        if curl -fsS "http://127.0.0.1:6333/readyz" >/dev/null 2>&1 \
           || curl -fsS "http://127.0.0.1:6333/" >/dev/null 2>&1; then
            log_ok "Qdrant is responding on :6333"
            break
        fi
        sleep 1
        if [[ ${i} -eq 20 ]]; then
            log_warn "Qdrant did not respond on :6333 after 20s."
        fi
    done
fi

# ==============================================================================
# Step 9 — .env file and final verification
# ==============================================================================
log_step "Step 9/9: .env file and final verification"

if [[ -f "${ENV_FILE}" ]]; then
    log_ok ".env already exists — leaving untouched."
elif [[ -f "${ENV_EXAMPLE}" ]]; then
    cp "${ENV_EXAMPLE}" "${ENV_FILE}"
    log_ok "Copied .env.example → .env (edit this file to add API keys)."
else
    log_warn ".env.example not found; .env not created."
fi

# ----- Final summary ----------------------------------------------------------
printf '\n%s========================================%s\n' "${C_GREEN}" "${C_RESET}"
printf '%sVM setup complete%s\n' "${C_GREEN}${C_BOLD}" "${C_RESET}"
printf '%s========================================%s\n\n' "${C_GREEN}" "${C_RESET}"

cat <<EOF
${C_BOLD}Next steps:${C_RESET}

  1. Edit your .env file and fill in API keys if using hosted LLMs:
       ${C_BLUE}\$EDITOR ${ENV_FILE}${C_RESET}

  2. Seed the vector store and dataset (one-time, ~10-20 min):
       ${C_BLUE}make seed${C_RESET}

  3. Run a quick evaluation to verify everything works:
       ${C_BLUE}make eval${C_RESET}

  4. For a long evaluation that survives SSH disconnects:
       ${C_BLUE}./scripts/run_eval_tmux.sh --pipeline full${C_RESET}

  5. Quick health check at any time:
       ${C_BLUE}./scripts/check_vm.sh${C_RESET}

${C_BOLD}Services:${C_RESET}
  - Qdrant   : http://localhost:6333  (Docker container)
  - Ollama   : http://localhost:11434 (native systemd service)
  - App      : run with ${C_BLUE}make serve${C_RESET} → http://localhost:8000
EOF
