#!/usr/bin/env bash
# ==============================================================================
# agentic-rag-eval — run evaluation inside a detached tmux session
# ------------------------------------------------------------------------------
# Starts `make eval-full` (or another pipeline) inside a tmux session and
# detaches, so the evaluation survives SSH disconnects.
#
# Usage:
#   ./scripts/run_eval_tmux.sh [--pipeline baseline|agentic|full] [--session NAME]
#
# Flags:
#   --pipeline {baseline|agentic|full}   Which Make target to run. Default: full.
#   --session NAME                       tmux session name. Default: agentic-rag-eval.
#   -h, --help                           Show this help.
#
# After the session starts, reattach with:
#   tmux attach -t <session-name>
#
# Logs are written to logs/eval_<timestamp>.log alongside the tmux output.
# ==============================================================================

set -euo pipefail

# ----- Colors -----------------------------------------------------------------
if [[ -t 1 ]]; then
    C_RESET=$'\033[0m'; C_GREEN=$'\033[0;32m'; C_YELLOW=$'\033[0;33m'
    C_RED=$'\033[0;31m'; C_BLUE=$'\033[0;34m'; C_BOLD=$'\033[1m'
else
    C_RESET=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_BLUE=""; C_BOLD=""
fi

log_info()  { printf '%s[INFO]%s  %s\n' "${C_BLUE}"  "${C_RESET}" "$*"; }
log_ok()    { printf '%s[ OK ]%s  %s\n' "${C_GREEN}" "${C_RESET}" "$*"; }
log_warn()  { printf '%s[WARN]%s  %s\n' "${C_YELLOW}" "${C_RESET}" "$*" >&2; }
log_error() { printf '%s[FAIL]%s  %s\n' "${C_RED}"   "${C_RESET}" "$*" >&2; }

die() { log_error "$*"; exit 1; }

# ----- Defaults ---------------------------------------------------------------
PIPELINE="full"
SESSION_NAME="agentic-rag-eval"

usage() { sed -n '2,22p' "$0"; }

# ----- Parse flags ------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pipeline)
            [[ $# -ge 2 ]] || die "--pipeline requires an argument"
            PIPELINE="$2"
            shift 2
            ;;
        --session)
            [[ $# -ge 2 ]] || die "--session requires an argument"
            SESSION_NAME="$2"
            shift 2
            ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            die "Unknown flag: $1 (try --help)" ;;
    esac
done

case "${PIPELINE}" in
    baseline) MAKE_TARGET="eval-baseline" ;;
    agentic)  MAKE_TARGET="eval-agentic" ;;
    full)     MAKE_TARGET="eval-full" ;;
    *) die "Invalid --pipeline '${PIPELINE}' (expected: baseline|agentic|full)" ;;
esac

# ----- Preconditions ----------------------------------------------------------
command -v tmux >/dev/null 2>&1 || die "tmux is not installed. Install with: sudo apt-get install -y tmux"
command -v make >/dev/null 2>&1 || die "make is not installed."

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
cd "${PROJECT_ROOT}"

[[ -f Makefile ]] || die "Makefile not found at ${PROJECT_ROOT}/Makefile"

# ----- Prep log file ----------------------------------------------------------
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/eval_${TIMESTAMP}.log"

# ----- Refuse to clobber an existing session ---------------------------------
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    die "tmux session '${SESSION_NAME}' already exists. Attach with: tmux attach -t ${SESSION_NAME}"
fi

# ----- Build the command to run inside tmux ----------------------------------
# `script` would be nicer but is not always installed; use `tee` + PIPESTATUS.
INNER_CMD=$(cat <<EOF
set -o pipefail
cd "${PROJECT_ROOT}"
echo "=== agentic-rag-eval :: ${MAKE_TARGET} ==="
echo "=== started: \$(date -Iseconds) ==="
echo "=== host:    \$(hostname) ==="
echo "=== log:     ${LOG_FILE} ==="
echo
make ${MAKE_TARGET} 2>&1 | tee -a "${LOG_FILE}"
rc=\${PIPESTATUS[0]}
echo
echo "=== finished: \$(date -Iseconds) (exit=\${rc}) ==="
echo "Press any key to close this tmux window..."
read -n 1 -s -r || true
exit \${rc}
EOF
)

log_info "Starting tmux session '${SESSION_NAME}' running 'make ${MAKE_TARGET}'"
log_info "Log file: ${LOG_FILE}"

tmux new-session -d -s "${SESSION_NAME}" "bash -lc $(printf '%q' "${INNER_CMD}")"

# Detach is implicit because we used -d; verify the session is actually running.
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    log_ok "Detached tmux session '${SESSION_NAME}' is running."
    cat <<EOF

${C_BOLD}Attach:${C_RESET}   ${C_BLUE}tmux attach -t ${SESSION_NAME}${C_RESET}
${C_BOLD}Tail log:${C_RESET} ${C_BLUE}tail -f ${LOG_FILE}${C_RESET}
${C_BOLD}Kill:${C_RESET}     ${C_BLUE}tmux kill-session -t ${SESSION_NAME}${C_RESET}

You can safely disconnect from SSH; the evaluation will continue running.
EOF
    exit 0
else
    die "Failed to start tmux session '${SESSION_NAME}'."
fi
