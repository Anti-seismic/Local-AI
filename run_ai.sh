#!/usr/bin/env bash
# =============================================================================
# run_ai.sh
# Local AI orchestrator
#
# Role: Start all services, wait for readiness, open browser once, attach tmux.
#
# Primary display: tmux session "localai" with 4 panes:
#   Pane 0 (top-left):    vllm_launcher.py  — proxy, API, WS, storage
#   Pane 1 (top-right):   launcher structured log — tail -f logs/vllm_launcher.log
#   Pane 2 (bottom-left): upload_server.py
#   Pane 3 (bottom-right):static HTTP server (serves the UI on :8080)
#
# Fallback display (if tmux fails): gnome-terminal tabs, then xterm windows.
#
# tray.py: detached background process, only when a valid X display exists.
# Browser: opened exactly once, after the model is confirmed ready.
# Monitoring: services_failure_order.log updated in background.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 1. Resolve the true script location regardless of how it was called.
# -----------------------------------------------------------------------------
SCRIPT_PATH="$(readlink -f "$0")"
BASE_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
cd "${BASE_DIR}"

# -----------------------------------------------------------------------------
# 2. Configuration
# -----------------------------------------------------------------------------
LOG_DIR="${BASE_DIR}/logs"

# vLLM venv: torch, vllm, websockets — runs vllm_launcher.py
VENV_VLLM="${BASE_DIR}/virtual_Env/Qwen3.5-2B-AWQ-4bit/bin/activate"
PYTHON_VLLM="${BASE_DIR}/virtual_Env/Qwen3.5-2B-AWQ-4bit/bin/python3"

# UI venv: flask, pystray, pillow, requests — runs upload_server.py, tray.py
VENV_UI="${BASE_DIR}/virtual_Env/ProjectUI/bin/activate"
PYTHON_UI="${BASE_DIR}/virtual_Env/ProjectUI/bin/python3"

STATIC_PORT=8080
SESSION="localai"
UI_URL="http://127.0.0.1:${STATIC_PORT}/chat.html"
FAILURE_LOG="${LOG_DIR}/services_failure_order.log"

mkdir -p "${LOG_DIR}"

log() { echo "[run_ai] $(date '+%H:%M:%S') $*"; }

# -----------------------------------------------------------------------------
# 3. Sanity checks
# -----------------------------------------------------------------------------
for cmd in curl xdg-open; do
  command -v "${cmd}" >/dev/null 2>&1 \
    || { echo "ERROR: '${cmd}' is required but not installed." >&2; exit 1; }
done
[ -f "${VENV_VLLM}" ]          || { echo "ERROR: vLLM venv not found at ${VENV_VLLM}" >&2; exit 1; }
[ -f "${PYTHON_VLLM}" ]        || { echo "ERROR: vLLM python not found at ${PYTHON_VLLM}" >&2; exit 1; }
[ -f "${VENV_UI}" ]            || { echo "ERROR: UI venv not found at ${VENV_UI}" >&2; exit 1; }
[ -f "${PYTHON_UI}" ]          || { echo "ERROR: UI python not found at ${PYTHON_UI}" >&2; exit 1; }
[ -f "${BASE_DIR}/chat.html" ] || { echo "ERROR: chat.html not found in ${BASE_DIR}" >&2; exit 1; }

# -----------------------------------------------------------------------------
# 4. Clean up any previous instance via the single authoritative path.
# -----------------------------------------------------------------------------
log "Cleaning up any previous LocalAI instance..."
bash "${BASE_DIR}/debug_ai.sh" --kill
log "Cleanup done."

# -----------------------------------------------------------------------------
# 5. Launch services.
#
#    upload_server and the static HTTP server are started here as background
#    processes — BEFORE and INDEPENDENT of the terminal display strategy.
#    This guarantees they run regardless of tmux pane behaviour, desktop-icon
#    launch context, or which terminal strategy is chosen.
#
#    vllm_launcher is the "core" process and is started by the chosen display
#    strategy so it remains visible in a terminal pane/tab/window.
#
#    Terminal display strategies (tried in order):
#      A. tmux session with 4 panes       — best experience
#      B. gnome-terminal with 4 tabs      — Ubuntu desktop fallback
#      C. xterm with 4 separate windows   — universal X11 fallback
#      D. background process              — last resort, logs only
#
#    In all cases panes/tabs show log tails so the terminal view is read-only
#    and never becomes a single point of failure for the underlying services.
# -----------------------------------------------------------------------------

DISPLAY_MODE="none"   # will be set to: tmux | gnome | xterm | background

# ── Pre-display: start upload_server and static HTTP unconditionally ─────────
# Using the venv's python3 executable directly (no 'source activate' needed —
# Python resolves site-packages relative to its own executable path).
# stdout/stderr are discarded here: upload_server uses a RotatingFileHandler
# that writes directly to upload_server.log; http.server output is captured
# separately below via shell redirection.

log "Starting upload server in background (logs → upload_server.log)..."
setsid "${PYTHON_UI}" "${BASE_DIR}/upload_server.py" \
    >/dev/null 2>&1 &
disown $!

log "Starting static HTTP server on :${STATIC_PORT} in background (logs → static_http.log)..."
setsid "${PYTHON_UI}" -m http.server "${STATIC_PORT}" \
    --directory "${BASE_DIR}" \
    >> "${LOG_DIR}/static_http.log" 2>&1 &
disown $!

# ── Strategy A: tmux ────────────────────────────────────────────────────────
if command -v tmux >/dev/null 2>&1; then
  log "Attempting tmux session..."

  # Kill any leftover session from a previous run.
  tmux kill-session -t "${SESSION}" 2>/dev/null || true

  # Create a detached session WITHOUT -x/-y.
  # tmux 3.x assigns a default size (80x24) and auto-resizes on attach.
  # Passing -x/-y causes "size missing" in non-interactive / no-TTY contexts
  # on certain tmux builds and is never necessary for detached sessions.
  if tmux new-session -d -s "${SESSION}" -n "LocalAI" \
       "source '${VENV_VLLM}' && python3 '${BASE_DIR}/vllm_launcher.py' ; \
        echo '[vllm_launcher exited — press Enter]' ; read" 2>/dev/null; then

    # Pane 1 (top-right): launcher structured log (timestamps, calibration, WS events).
    # Pane 0 already shows all vLLM output live with [vLLM] prefix, making a second
    # tail of vllm_api.log redundant. This pane shows clean launcher decisions only.
    # vllm_api.log remains on disk for manual inspection: tail -f logs/vllm_api.log
    tmux split-window -t "${SESSION}:0.0" -h \
      "echo '=== Launcher log (tail -f logs/vllm_launcher.log) ===' && \
       sleep 1 && \
       tail -f '${LOG_DIR}/vllm_launcher.log' ; \
       echo '[stream ended — press Enter]' ; read" 2>/dev/null || true

    # Pane 2 (bottom-left): tail upload_server.log
    # upload_server runs as a pre-display background process (started above).
    # This pane shows its live log output without running the service itself,
    # eliminating the single-point-of-failure that caused the desktop-icon bug.
    tmux split-window -t "${SESSION}:0.0" -v \
      "echo '=== Upload server (tail -f logs/upload_server.log) ===' && \
       sleep 2 && \
       tail -f '${LOG_DIR}/upload_server.log' ; \
       echo '[stream ended — press Enter]' ; read" 2>/dev/null || true

    # Pane 3 (bottom-right): tail static_http.log
    # http.server runs as a pre-display background process (started above).
    tmux split-window -t "${SESSION}:0.2" -h \
      "echo '=== Static HTTP :${STATIC_PORT} (tail -f logs/static_http.log) ===' && \
       sleep 2 && \
       tail -f '${LOG_DIR}/static_http.log' ; \
       echo '[stream ended — press Enter]' ; read" 2>/dev/null || true

    DISPLAY_MODE="tmux"
    log "tmux session '${SESSION}' created with 4 panes."
  else
    log "tmux new-session failed — trying next fallback."
    tmux kill-session -t "${SESSION}" 2>/dev/null || true
  fi
fi

# ── Strategy B: gnome-terminal ───────────────────────────────────────────────
if [ "${DISPLAY_MODE}" = "none" ] && command -v gnome-terminal >/dev/null 2>&1; then
  log "Attempting gnome-terminal tabs..."

  gnome-terminal \
    --tab --title="vllm_launcher" \
      -- bash -c "source '${VENV_VLLM}' && python3 '${BASE_DIR}/vllm_launcher.py' ; \
                  echo '[exited]' ; read" \
    --tab --title="launcher_log" \
      -- bash -c "sleep 1 && tail -f '${LOG_DIR}/vllm_launcher.log' ; read" \
    --tab --title="upload_server" \
      -- bash -c "sleep 2 && tail -f '${LOG_DIR}/upload_server.log' ; read" \
    --tab --title="static_http" \
      -- bash -c "sleep 2 && tail -f '${LOG_DIR}/static_http.log' ; read" \
    2>/dev/null && DISPLAY_MODE="gnome" || true

  [ "${DISPLAY_MODE}" = "gnome" ] \
    && log "gnome-terminal launched with 4 tabs." \
    || log "gnome-terminal failed — trying next fallback."
fi

# ── Strategy C: xterm ────────────────────────────────────────────────────────
if [ "${DISPLAY_MODE}" = "none" ] && command -v xterm >/dev/null 2>&1; then
  log "Attempting xterm windows..."

  xterm -T "vllm_launcher" \
    -e bash -c "source '${VENV_VLLM}' && python3 '${BASE_DIR}/vllm_launcher.py' ; read" &
  disown $!

  xterm -T "launcher_log" \
    -e bash -c "sleep 1 && tail -f '${LOG_DIR}/vllm_launcher.log' ; read" &
  disown $!

  xterm -T "upload_server" \
    -e bash -c "sleep 2 && tail -f '${LOG_DIR}/upload_server.log' ; read" &
  disown $!

  xterm -T "static_http" \
    -e bash -c "sleep 2 && tail -f '${LOG_DIR}/static_http.log' ; read" &
  disown $!

  DISPLAY_MODE="xterm"
  log "xterm windows launched."
fi

# ── Strategy D: background processes (last resort) ───────────────────────────
if [ "${DISPLAY_MODE}" = "none" ]; then
  log "WARNING: No terminal emulator available. Launching vllm_launcher as background process."
  log "         upload_server and http.server are already running (started above)."
  log "         Monitor progress via: tail -f ${LOG_DIR}/*.log"

  # Use setsid + the venv's python3 directly — no 'source activate' needed.
  # setsid detaches from the controlling TTY so the process survives terminal close.
  # >/dev/null: vllm_launcher has a RotatingFileHandler; stdout duplication avoided.
  setsid "${PYTHON_VLLM}" "${BASE_DIR}/vllm_launcher.py" \
    >/dev/null 2>&1 &
  disown $!

  DISPLAY_MODE="background"
  log "vllm_launcher launched in background."
fi

# -----------------------------------------------------------------------------
# 6. tray.py — detached background process, only with a valid X display.
# -----------------------------------------------------------------------------
X_DISPLAY="${DISPLAY:-}"
X_SOCKET_NUM="${X_DISPLAY#*:}"
X_SOCKET_NUM="${X_SOCKET_NUM%%.*}"
X_SOCKET="/tmp/.X11-unix/X${X_SOCKET_NUM}"

if [ -n "${X_DISPLAY}" ] && [ -e "${X_SOCKET}" ]; then
  log "X display '${X_DISPLAY}' confirmed. Starting tray.py..."
  export DISPLAY="${X_DISPLAY}"
  export XAUTHORITY="${XAUTHORITY:-${HOME}/.Xauthority}"
  setsid "${PYTHON_UI}" "${BASE_DIR}/tray.py" >> "${LOG_DIR}/tray.log" 2>&1 &
  disown $!
  log "tray.py started."
else
  log "No valid X display (DISPLAY='${X_DISPLAY:-unset}'). Skipping tray icon."
fi

# -----------------------------------------------------------------------------
# 7. Service Failure Order Monitor — background, exits when services end.
# -----------------------------------------------------------------------------
_start_failure_monitor() {
  # Polls all services every 10 s. First failure of each service is written
  # to services_failure_order.log with a sequential number and timestamp.
  #
  # Implementation note: bash nested functions do NOT close over parent
  # local variables. We therefore avoid nested functions entirely and use
  # the log file line count as state — no local variable arithmetic needed.

  local fail_log="${FAILURE_LOG}"
  local base_dir="${BASE_DIR}"
  local static_port="${STATIC_PORT}"
  local session="${SESSION}"
  local mode="${DISPLAY_MODE}"

  {
    echo "# Service Failure Order Log"
    echo "# Session started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "# Display mode: ${mode}"
    echo "# Format: <order>. <service> — <timestamp>"
  } > "${fail_log}"

  # Track which services have already been recorded using a plain string.
  # We cannot use associative arrays reliably across subshells.
  local recorded=""

  _already_recorded() {
    # Returns 0 (true) if the service name appears in $recorded
    case "${recorded}" in *"|$1|"*) return 0 ;; esac
    return 1
  }

  while pgrep -f "${base_dir}/vllm_launcher.py" >/dev/null 2>&1      || { [ "${mode}" = "tmux" ] && tmux has-session -t "${session}" 2>/dev/null; }; do
    sleep 10

    # For each service: run check, record first failure.
    # We read the current line count from the log to get the order number.
    _check() {
      local name="$1"
      local is_up="$2"   # 0 = up, non-zero = down
      if [ "${is_up}" -ne 0 ] && ! _already_recorded "${name}"; then
        recorded="${recorded}|${name}|"
        local order
        # Count only data lines (not comment lines starting with #)
        order=$(grep -c "^[0-9]" "${fail_log}" 2>/dev/null || echo 0)
        order=$(( order + 1 ))
        printf "%d. %s — %s
" "${order}" "${name}"           "$(date '+%Y-%m-%d %H:%M:%S')" >> "${fail_log}"
      fi
    }

    pgrep -f "${base_dir}/vllm_launcher.py" >/dev/null 2>&1
    _check "vllm_launcher.py" $?

    curl -sf --max-time 3 "http://127.0.0.1:8000/v1/models" >/dev/null 2>&1
    _check "vLLM API (port 8000)" $?

    curl -sf --max-time 3 "http://127.0.0.1:8771/status" >/dev/null 2>&1
    _check "upload_server.py (port 8771)" $?

    curl -sf --max-time 3 "http://127.0.0.1:8770/status" >/dev/null 2>&1
    _check "launcher microservice (port 8770)" $?

    if command -v lsof >/dev/null 2>&1; then
      lsof -i :8765 -P -n >/dev/null 2>&1
      _check "WebSocket server (port 8765)" $?
    fi

    curl -sf --max-time 3 "http://127.0.0.1:${static_port}/" >/dev/null 2>&1
    _check "static HTTP server (port ${static_port})" $?

  done
}

_start_failure_monitor &
disown $!
log "Service failure monitor started. Log: ${FAILURE_LOG}"

# -----------------------------------------------------------------------------
# 8. Open tmux and manage readiness wait.
#
#    Strategy A — gnome-terminal available:
#      Open the tmux session in a NEW terminal window so all 4 panes are
#      visible from second 0. The original terminal runs the readiness wait
#      and calls xdg-open directly (full X11 context — browser opens reliably).
#      This is the confirmed working path: vision tests were conducted with it.
#
#    Strategy B — no gnome-terminal:
#      Attach to tmux immediately. A background job waits for readiness and
#      calls xdg-open. DISPLAY is explicitly exported to the subshell.
#
#    For non-tmux modes: wait in this terminal, then open browser.
# -----------------------------------------------------------------------------
if [ "${DISPLAY_MODE}" = "tmux" ]; then

  # Ensure DISPLAY is exported for any child process that needs X11
  export DISPLAY="${DISPLAY:-:0}"
  export XAUTHORITY="${XAUTHORITY:-${HOME}/.Xauthority}"

  # ── Strategy A: gnome-terminal new window (confirmed working) ────────────
  _OPENED_IN_NEW_WINDOW=false
  if command -v gnome-terminal >/dev/null 2>&1 && [ -n "${DISPLAY}" ]; then
    log "Opening tmux panes in a new terminal window (LocalAI Services)..."
    gnome-terminal       --window       --title="LocalAI Services"       -- bash -c "tmux attach-session -t '${SESSION}'; exec bash"       2>/dev/null     && _OPENED_IN_NEW_WINDOW=true     || log "gnome-terminal failed — falling back to in-terminal attach."
  fi

  if [ "${_OPENED_IN_NEW_WINDOW}" = "true" ]; then
    # Panes are open in the new window.
    # Run readiness wait HERE (original terminal, full X11 context).
    # xdg-open called directly — browser opens reliably.
    log "Panes open in 'LocalAI Services' window."
    log "Waiting for vLLM model (http://127.0.0.1:8000/v1/models)..."
    READY=0
    for i in $(seq 1 180); do
      if curl -sf "http://127.0.0.1:8000/v1/models" >/dev/null 2>&1; then
        READY=1; log "✅ vLLM model ready after ${i}s."; break
      fi
      (( i % 15 == 0 )) && log "  Still loading... (${i}s elapsed)"
      sleep 1
    done
    [ "${READY}" -eq 0 ] && log "⚠️  Model not ready after 180 s. Check vLLM output."
    for i in $(seq 1 15); do
      curl -sf "http://127.0.0.1:8770/status" >/dev/null 2>&1         && { log "✅ Launcher microservice ready."; break; }
      sleep 1
    done
    log "Opening UI: ${UI_URL}"
    xdg-open "${UI_URL}" 2>/dev/null       || log "⚠️  xdg-open failed — open ${UI_URL} manually."
    log "Done. Panes are in the 'LocalAI Services' window."
    log "To stop everything: ./debug_ai.sh --kill"

  else
    # ── Strategy B: attach immediately, background wait + browser ─────────
    log "Attaching to tmux immediately. Browser opens when model is ready."
    log "CTRL+B then D to detach. To stop everything: ./debug_ai.sh --kill"

    _UI_URL="${UI_URL}"
    _LOG_DIR="${LOG_DIR}"
    _DISPLAY="${DISPLAY}"
    _XAUTHORITY="${XAUTHORITY:-${HOME}/.Xauthority}"
    (
      export DISPLAY="${_DISPLAY}"
      export XAUTHORITY="${_XAUTHORITY}"
      READY=0
      for i in $(seq 1 180); do
        curl -sf "http://127.0.0.1:8000/v1/models" >/dev/null 2>&1           && { READY=1; break; }
        sleep 1
      done
      if [ "${READY}" -eq 1 ]; then
        for i in $(seq 1 15); do
          curl -sf "http://127.0.0.1:8770/status" >/dev/null 2>&1 && break
          sleep 1
        done
        echo "$(date '+%H:%M:%S') ✅ Model ready — opening ${_UI_URL}"           >> "${_LOG_DIR}/startup.log"
        xdg-open "${_UI_URL}" 2>>"${_LOG_DIR}/startup.log"           || echo "$(date '+%H:%M:%S') ⚠️  xdg-open failed"              >> "${_LOG_DIR}/startup.log"
      else
        echo "$(date '+%H:%M:%S') ⚠️  Model not ready after 180 s."           >> "${_LOG_DIR}/startup.log"
      fi
    ) &
    disown $!

    tmux select-pane -t "${SESSION}:0.1" 2>/dev/null || true
    tmux attach-session -t "${SESSION}"
  fi

else
  # Non-tmux display modes: wait in this terminal, then open browser.
  log "Waiting for vLLM model (http://127.0.0.1:8000/v1/models)..."
  READY=0
  for i in $(seq 1 180); do
    curl -sf "http://127.0.0.1:8000/v1/models" >/dev/null 2>&1       && { READY=1; log "✅ Model ready after ${i}s."; break; }
    (( i % 15 == 0 )) && log "  Still loading... (${i}s elapsed)"
    sleep 1
  done
  [ "${READY}" -eq 0 ] && log "⚠️  Model not ready after 180 s."
  log "Opening UI: ${UI_URL}"
  xdg-open "${UI_URL}" 2>/dev/null     || log "⚠️  xdg-open failed — open ${UI_URL} manually."
  log "Display mode: ${DISPLAY_MODE}. Services are running."
  log "To stop everything: ./debug_ai.sh --kill"
fi

# -----------------------------------------------------------------------------
# 9. Keep this terminal open.
#
#    When launched via the desktop icon (Terminal=true in .desktop), the
#    gnome-terminal would close as soon as the script exits. This would send
#    SIGHUP to all processes in the same session — potentially killing services
#    that setsid did not protect (e.g. processes inheriting from this shell).
#
#    exec bash replaces the script process with an interactive shell, keeping
#    the terminal open indefinitely. This matches the behaviour of a manual
#    ./run_ai.sh launch (where the terminal stays open at the shell prompt).
#    The user can type 'exit' or press Ctrl+D to close the terminal.
# -----------------------------------------------------------------------------
log "Services launched. Terminal kept open (type 'exit' or Ctrl+D to close)."
exec bash
