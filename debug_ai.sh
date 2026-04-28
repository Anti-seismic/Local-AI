#!/usr/bin/env bash
# =============================================================================
# debug_ai.sh
# Local AI kill, clean, and status tool
#
# Usage:
#   ./debug_ai.sh           full kill + clean + status report
#   ./debug_ai.sh --kill    kill + clean only
#   ./debug_ai.sh --status  status report only
#
# Called by run_ai.sh at startup (--kill) to ensure a clean slate.
# =============================================================================

set -uo pipefail   # -e intentionally omitted: we want best-effort cleanup

SCRIPT_PATH="$(readlink -f "$0")"
BASE_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"

LOG_DIR="${BASE_DIR}/logs"
# vLLM venv python — has torch, used for GPU memory purge
PYTHON_VLLM="${BASE_DIR}/virtual_Env/Qwen3.5-2B-AWQ-4bit/bin/python3"
# UI venv python — used for any UI/tool related checks
PYTHON_UI="${BASE_DIR}/virtual_Env/ProjectUI/bin/python3"
STATIC_PORT=8080
VLLM_PORT=8770
WS_PORT=8765
UPLOAD_PORT=8771
VLLM_API_PORT=8000
TMUX_SESSION="localai"
LOG_MAX=$((50 * 1024 * 1024))

MODE="full"
case "${1:-}" in
  --kill)   MODE="kill"   ;;
  --status) MODE="status" ;;
esac

# Colour helpers
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; RESET='\033[0m'
ok()   { echo -e "${GREEN}✅ $*${RESET}"; }
warn() { echo -e "${YELLOW}⚠️  $*${RESET}"; }
err()  { echo -e "${RED}❌ $*${RESET}"; }
info() { echo -e "${CYAN}ℹ️  $*${RESET}"; }

# =============================================================================
# SECTION 1: KILL + CLEAN
# =============================================================================
do_kill() {
  echo ""
  echo "════════════════════════════════════════════"
  echo " KILL — stopping all LocalAI services"
  echo "════════════════════════════════════════════"

  # --- Graceful SIGTERM to known processes ---
  echo "--- Graceful kill (SIGTERM) ---"
  for pat in \
    "${BASE_DIR}/vllm_launcher.py" \
    "${BASE_DIR}/upload_server.py" \
    "${BASE_DIR}/tray.py" \
    "vllm.entrypoints.openai.api_server" \
    "http.server ${STATIC_PORT}"
  do
    if pgrep -f "${pat}" >/dev/null 2>&1; then
      warn "Killing: ${pat}"
      pkill -TERM -f "${pat}" 2>/dev/null || true
    fi
  done

  sleep 2

  # --- Force SIGKILL anything still alive ---
  echo "--- Force kill (SIGKILL) remaining ---"
  for pat in \
    "${BASE_DIR}/vllm_launcher.py" \
    "${BASE_DIR}/upload_server.py" \
    "${BASE_DIR}/tray.py" \
    "vllm.entrypoints.openai.api_server" \
    "http.server ${STATIC_PORT}"
  do
    if pgrep -f "${pat}" >/dev/null 2>&1; then
      warn "Force killing: ${pat}"
      pkill -KILL -f "${pat}" 2>/dev/null || true
    fi
  done

  # --- Kill by port (catches anything not matched by pattern) ---
  echo "--- Kill by port ---"
  _kill_port() {
    local port="$1"
    local pids=""
    if command -v lsof >/dev/null 2>&1; then
      pids="$(lsof -t -i :"${port}" 2>/dev/null || true)"
    elif command -v fuser >/dev/null 2>&1; then
      pids="$(fuser "${port}/tcp" 2>/dev/null | tr ' ' '\n' || true)"
    fi
    if [ -n "${pids}" ]; then
      warn "Port ${port} occupied by PIDs: ${pids} — killing."
      for pid in ${pids}; do
        kill -TERM "${pid}" 2>/dev/null || true
      done
      sleep 1
      for pid in ${pids}; do
        kill -KILL "${pid}" 2>/dev/null || true
      done
    else
      ok "Port ${port} is free."
    fi
  }

  for port in ${VLLM_API_PORT} ${VLLM_PORT} ${WS_PORT} ${UPLOAD_PORT} ${STATIC_PORT}; do
    _kill_port "${port}"
  done

  # --- Kill tmux session ---
  if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    warn "Killing tmux session '${TMUX_SESSION}'."
    tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
  else
    ok "No tmux session '${TMUX_SESSION}' found."
  fi

  sleep 1

  # --- GPU/VRAM memory release (uses venv python3 which has torch) ---
  echo "--- Releasing GPU/VRAM memory ---"
  if [ -f "${PYTHON_VLLM}" ]; then
    "${PYTHON_VLLM}" - <<'PYEOF' 2>/dev/null \
      && ok "GPU memory released." \
      || warn "GPU memory purge skipped (torch unavailable or no GPU)."
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print("torch.cuda.empty_cache() done.")
else:
    print("No CUDA device found.")
PYEOF
  else
    warn "vLLM venv python3 not found at ${PYTHON_VLLM}. Skipping GPU purge."
  fi

  # --- GPU memory report ---
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "--- GPU state after cleanup ---"
    nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total \
               --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '{printf "  GPU: %s | Used: %s MB | Free: %s MB | Total: %s MB\n",$1,$2,$3,$4}' \
    || true
  fi

  # --- Drop Linux page cache (optional, requires sudo) ---
  echo "--- Dropping page cache (requires sudo) ---"
  if sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then
    ok "Page cache dropped."
  else
    warn "Page cache drop skipped (no sudo or not supported)."
  fi

  # --- Remove Python __pycache__ in project dir ---
  echo "--- Removing __pycache__ ---"
  find "${BASE_DIR}" -maxdepth 3 -type d -name "__pycache__" \
       -not -path "*/virtual_Env/*" \
       -exec rm -rf {} + 2>/dev/null || true
  ok "__pycache__ cleaned."

  # --- Truncate logs that have exceeded 50 MB ---
  echo "--- Checking log sizes ---"
  if [ -d "${LOG_DIR}" ]; then
    for logfile in "${LOG_DIR}"/*.log; do
      [ -f "${logfile}" ] || continue
      size=$(stat -c%s "${logfile}" 2>/dev/null || echo 0)
      if [ "${size}" -gt "${LOG_MAX}" ]; then
        warn "Log ${logfile} is $(( size / 1024 / 1024 )) MB — truncating."
        > "${logfile}"
      fi
    done
  fi
  ok "Log check done."

  ok "System clean. Ready for a fresh launch."
}

# =============================================================================
# SECTION 2: STATUS REPORT
# =============================================================================
do_status() {
  echo ""
  echo "════════════════════════════════════════════"
  echo " STATUS — LocalAI service report"
  echo "════════════════════════════════════════════"
  echo "Base dir : ${BASE_DIR}"
  echo "Logs     : ${LOG_DIR}"
  echo ""

  # Process checks
  _check_proc() {
    local label="$1"; local pattern="$2"
    if pgrep -f "${pattern}" >/dev/null 2>&1; then
      ok "${label}: RUNNING (PIDs: $(pgrep -f "${pattern}" | tr '\n' ' '))"
    else
      err "${label}: not running"
    fi
  }

  _check_proc "vllm_launcher.py"                   "${BASE_DIR}/vllm_launcher.py"
  _check_proc "vllm API server"                    "vllm.entrypoints.openai.api_server"
  _check_proc "upload_server.py"                   "${BASE_DIR}/upload_server.py"
  _check_proc "Static HTTP :${STATIC_PORT}"        "http.server ${STATIC_PORT}"
  _check_proc "tray.py"                            "${BASE_DIR}/tray.py"

  # tmux session
  echo ""
  if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    ok "tmux session '${TMUX_SESSION}': ACTIVE"
    tmux list-panes -t "${TMUX_SESSION}" 2>/dev/null || true
  else
    err "tmux session '${TMUX_SESSION}': not found"
  fi

  # Port checks
  echo ""
  echo "--- Port status ---"
  _check_port() {
    local port="$1"; local label="$2"
    if command -v lsof >/dev/null 2>&1; then
      if lsof -i :"${port}" -P -n >/dev/null 2>&1; then
        ok "Port ${port} (${label}): IN USE"
      else
        err "Port ${port} (${label}): free"
      fi
    elif command -v ss >/dev/null 2>&1; then
      if ss -ltnp 2>/dev/null | grep -q ":${port}"; then
        ok "Port ${port} (${label}): IN USE"
      else
        err "Port ${port} (${label}): free"
      fi
    else
      warn "Port ${port} (${label}): cannot check (lsof/ss not found)"
    fi
  }

  _check_port "${VLLM_API_PORT}" "vLLM API"
  _check_port "${VLLM_PORT}"     "Launcher microservice"
  _check_port "${WS_PORT}"       "WebSocket"
  _check_port "${UPLOAD_PORT}"   "Upload server"
  _check_port "${STATIC_PORT}"   "Static HTTP (UI)"

  # HTTP health checks
  echo ""
  echo "--- HTTP health ---"
  _http() {
    local url="$1"; local label="$2"
    if curl -sf --max-time 3 "${url}" >/dev/null 2>&1; then
      ok "${label}: responding"
    else
      err "${label}: not responding (${url})"
    fi
  }

  _http "http://127.0.0.1:8000/v1/models" "vLLM API"
  _http "http://127.0.0.1:8770/status"    "Launcher microservice"
  _http "http://127.0.0.1:8771/status"    "Upload server"
  _http "http://127.0.0.1:${STATIC_PORT}/" "Static HTTP (UI)"

  # GPU status
  echo ""
  echo "--- GPU/VRAM ---"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total,utilization.gpu \
               --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '{printf "  GPU: %s | Used: %s MB | Free: %s MB | Total: %s MB | Util: %s%%\n",$1,$2,$3,$4,$5}' \
    || warn "nvidia-smi query failed."
  else
    warn "nvidia-smi not found."
  fi

  # Log tails
  echo ""
  echo "--- Recent log entries (last 5 lines each) ---"
  if [ -d "${LOG_DIR}" ]; then
    for logfile in "${LOG_DIR}"/*.log; do
      [ -f "${logfile}" ] || continue
      echo "  ▶ $(basename "${logfile}"):"
      tail -5 "${logfile}" 2>/dev/null | sed 's/^/    /' || true
      echo ""
    done
  else
    warn "Log directory not found."
  fi

  info "Status report complete."
}

# =============================================================================
# Entry point
# =============================================================================
case "${MODE}" in
  kill)   do_kill ;;
  status) do_status ;;
  full)   do_kill; do_status ;;
esac
