#!/usr/bin/env python3
# =============================================================================
# vllm_launcher.py
# Backend orchestrator for Local AI.
#
# Responsibilities:
#   - Launch and manage the vLLM subprocess, streaming its output to
#     vllm_api.log (and to pane 0 stdout with [vLLM] prefix)
#   - HTTP microservice on :8770 (all API endpoints)
#   - WebSocket server on :8765 (connection tracking)
#   - Background FIFO cleanup (conversations older than 7 days)
#   - Rotating log output (50 MB cap, overwrite)
#   - Full multimodal support: text, image (base64), video (URL)
#   - Tool selection: built-in tool definitions, per-request override
#
# Uses ThreadingHTTPServer — vLLM can fetch /files/ while /chat is active.
# Uses sys.executable — vLLM always runs in the vLLM virtualenv.
# This script must be launched with the vLLM venv:
#   virtual_Env/Qwen3.5-2B-AWQ-4bit/bin/python3 vllm_launcher.py
# Required in vLLM venv: websockets, vllm, torch
# =============================================================================

import asyncio
import base64
import json
import logging
import logging.handlers
import mimetypes
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import websockets

# ---------------------------------------------------------------------------
# Paths & identity
# ---------------------------------------------------------------------------
BASE_DIR    = Path("/home/ai-broker/LocalAI")
MODELS_ROOT = BASE_DIR / "models"
DATA_ROOT   = BASE_DIR / "data"
LOG_DIR     = BASE_DIR / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_ROOT.mkdir(parents=True, exist_ok=True)

MACHINE_NAME = socket.gethostname()
try:
    USER_NAME = (
        os.environ.get("USER")
        or os.environ.get("LOGNAME")
        or subprocess.check_output(["whoami"], text=True).strip()
    )
except Exception:
    USER_NAME = "ai-broker"

IDENTITY      = f"{MACHINE_NAME}_{USER_NAME}"
USER_DATA_DIR = DATA_ROOT / IDENTITY
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_MAX_BYTES = 50 * 1024 * 1024


def _make_rotating_handler(filename: str) -> logging.handlers.RotatingFileHandler:
    path = LOG_DIR / filename
    h = logging.handlers.RotatingFileHandler(
        path, maxBytes=LOG_MAX_BYTES, backupCount=1, encoding="utf-8"
    )
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    return h


log = logging.getLogger("launcher")
log.setLevel(logging.DEBUG)
log.propagate = False
log.addHandler(_make_rotating_handler("vllm_launcher.log"))
log.addHandler(logging.StreamHandler(sys.stdout))

# ---------------------------------------------------------------------------
# Built-in tool definitions (Qwen3.5 native tool format for vLLM)
# These are the tools shown in the GUI tool panel. The model info JSON can
# override or extend them via a "custom_tools" list.
# ---------------------------------------------------------------------------
BUILTIN_TOOLS: Dict[str, Dict] = {
    "code_interpreter": {
        "type": "function",
        "function": {
            "name": "code_interpreter",
            "description": (
                "Generate and execute Python code to solve computational problems, "
                "analyse data, process files, create visualisations, or perform "
                "any task that benefits from running code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type":        "string",
                        "description": "The Python code to execute",
                    }
                },
                "required": ["code"],
            },
        },
    },
    "web_search": {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information, news, facts, or any topic "
                "that may require up-to-date data not present in training knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type":        "string",
                        "description": "The search query",
                    },
                    "num_results": {
                        "type":        "integer",
                        "description": "Number of results to return (default 5)",
                        "default":     5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    "image_analysis": {
        "type": "function",
        "function": {
            "name": "image_analysis",
            "description": (
                "Perform detailed analysis of an image: object detection, text "
                "recognition, scene description, visual question answering, or "
                "comparison between multiple images."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type":        "string",
                        "description": "The analysis task: 'describe', 'detect', 'read_text', 'compare', 'answer'",
                    },
                    "question": {
                        "type":        "string",
                        "description": "Specific question about the image (optional)",
                    },
                },
                "required": ["task"],
            },
        },
    },
    "calculator": {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate a mathematical expression and return the result. "
                "Supports arithmetic, algebra, and basic statistics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type":        "string",
                        "description": "The mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        },
    },
}

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
vllm_proc: Optional[subprocess.Popen] = None
connected_clients: set = set()
lock = threading.Lock()

current_model_id   = "Qwen3.5-2B-AWQ-4bit"
current_model_path = str((MODELS_ROOT / current_model_id).resolve())  # no trailing slash
thinking_enabled   = False
last_restart_ts    = None

CONVERSATION_TTL_DAYS = 7

# ---------------------------------------------------------------------------
# Model info helpers
# ---------------------------------------------------------------------------

def infer_quantization(name: str) -> str:
    lo = name.lower()
    if "4bit" in lo or ("awq" in lo and "4" in lo): return "4bit"
    if "8bit" in lo: return "8bit"
    if "fp16" in lo or "16bit" in lo: return "fp16"
    return "unknown"


def dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


def load_model_info(model_id: str) -> Dict[str, Any]:
    info_path = MODELS_ROOT / "info" / f"{model_id}.json"
    if info_path.is_file():
        try:
            with info_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning("Failed to load model info for %s: %s", model_id, e)
    return {}


def model_has_vision(model_info: Dict[str, Any]) -> bool:
    """Return True when the model declares any visual input capability."""
    caps = [c.lower() for c in model_info.get("capabilities", [])]
    return any(c in caps for c in ("vision", "image", "video", "multimodal"))


def list_models() -> List[Dict[str, Any]]:
    models = []
    if not MODELS_ROOT.is_dir():
        return models
    for entry in MODELS_ROOT.iterdir():
        if not entry.is_dir() or entry.name.startswith(".") or entry.name == "info":
            continue
        model_id = entry.name
        path     = str(entry.resolve())  # no trailing slash — transformers 4.57+ rejects it
        info     = load_model_info(model_id)
        model    = {
            "id":                      model_id,
            "path":                    info.get("path", path),
            "quantization":            info.get("quantization", infer_quantization(model_id)),
            "size_bytes":              info.get("size_bytes", dir_size_bytes(entry)),
            "vram_gb":                 info.get("vram_gb"),
            "params_b":                info.get("params_b"),
            "type":                    info.get("type", "chat"),
            "name":                    info.get("name", model_id),
            "description":             info.get("description", ""),
            "license":                 info.get("license", ""),
            "capabilities":            info.get("capabilities", []),
            "max_model_len":           info.get("max_model_len", 9216),
            "gpu_memory_utilization":  info.get("gpu_memory_utilization", 0.98),
            "max_num_batched_tokens":  info.get("max_num_batched_tokens", 4096),
            "tool_call_parser":        info.get("tool_call_parser", "qwen3_coder"),
            "reasoning_parser":        info.get("reasoning_parser", "qwen3"),
            "chat_template":           info.get("chat_template"),
            "vision_service":          info.get("vision_service"),
            "video_service":           info.get("video_service"),
            "image_input_type":        info.get("image_input_type"),
            "host":                    info.get("host", "127.0.0.1"),
            "port":                    info.get("port", 8000),
            "enable_auto_tool_choice": info.get("enable_auto_tool_choice", True),
            # Available tools for this model (IDs from BUILTIN_TOOLS + custom_tools)
            "available_tools":         list(BUILTIN_TOOLS.keys()) +
                                       [t["id"] for t in info.get("custom_tools", [])],
        }
        # Ensure no trailing slash on stored path
        model["path"] = model["path"].rstrip("/")
        models.append(model)
    return models

# ---------------------------------------------------------------------------
# vLLM process management
# ---------------------------------------------------------------------------

def build_vllm_cmd(model_path: str, model_info: Dict[str, Any]) -> List[str]:
    """
    Build the vLLM command entirely from model_info JSON — no hardcoded values.
    Uses sys.executable so vLLM always runs in the correct virtualenv.
    Optional parameters are only added when present in the JSON.
    """
    # Strip any trailing slash from the model path.
    # transformers 4.57+ rejects local paths ending with "/" in cached_file(),
    # treating them as malformed HuggingFace repo IDs.
    model_path = model_path.rstrip("/")

    host        = str(model_info.get("host", "127.0.0.1"))
    port        = str(model_info.get("port", 8000))
    gpu_util    = str(model_info.get("gpu_memory_utilization", 0.98))
    max_len     = str(model_info.get("max_model_len", 9216))
    max_batched = str(model_info.get("max_num_batched_tokens", 4096))
    tool_parser      = model_info.get("tool_call_parser", "qwen3_coder")
    reasoning_parser = model_info.get("reasoning_parser", "qwen3")
    auto_tool        = model_info.get("enable_auto_tool_choice", True)

    allowed_origins = json.dumps(model_info.get(
        "allowed_origins",
        [f"http://{host}:8080", "http://localhost:8080", "*"]
    ))

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model",                  model_path,
        "--host",                   host,
        "--port",                   port,
        "--gpu-memory-utilization", gpu_util,
        "--max-model-len",          max_len,
        "--max-num-batched-tokens", max_batched,
        "--tool-call-parser",       tool_parser,
        "--reasoning-parser",       reasoning_parser,
        "--allowed-origins",        allowed_origins,
    ]

    # Optional parameters — only included when explicitly set in the JSON.
    # --chat-template is intentionally omitted: Qwen3.5 bundles its chat
    # template inside tokenizer_config.json and vLLM reads it automatically.
    # Passing a name like "qwen2" causes vLLM to look for a file by that name
    # and fail with ValueError when it does not exist.

    # Multimodal per-prompt limits — controls how many images/videos per request.
    # Routing is automatic: vLLM reads the model config and detects the VL
    # architecture. No --vision-service or --video-service flags exist in vLLM.
    mm_limits = model_info.get("limit_mm_per_prompt", {})
    if mm_limits:
        # vLLM expects JSON format: '{"image":1,"video":1}'
        import json as _json
        cmd += ["--limit-mm-per-prompt", _json.dumps(mm_limits, separators=(",", ":"))]

    # --quantization: explicit quantization backend for this model.
    # "vllm_quantization" is used to distinguish from the descriptive
    # "quantization" display field (e.g. "4bit"). vLLM auto-detects from
    # the model config.json, but being explicit is safer and clearer.
    vllm_quant = model_info.get("vllm_quantization", "")
    if vllm_quant:
        cmd += ["--quantization", vllm_quant]

    if auto_tool:
        cmd.append("--enable-auto-tool-choice")

    # Any extra arbitrary flags: {"extra_vllm_args": ["--flag", "value"]}
    extra = model_info.get("extra_vllm_args", [])
    if isinstance(extra, list):
        cmd.extend(extra)

    return cmd


def _notify(msg: str):
    try:
        subprocess.Popen(
            ["notify-send", "LocalAI", msg],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception:
        pass


def _vllm_api_base(model_info: Optional[Dict[str, Any]] = None) -> str:
    if model_info:
        return f"http://{model_info.get('host', '127.0.0.1')}:{model_info.get('port', 8000)}"
    return "http://127.0.0.1:8000"


def _stream_vllm_output(pipe):
    """
    Read vLLM subprocess output line by line.
    Writes to vllm_api.log AND to our own stdout (pane 0) with [vLLM] prefix
    so tracebacks appear both in the dedicated log and in the launcher pane.
    Pane 3 in tmux does: tail -f logs/vllm_api.log
    """
    log_path = LOG_DIR / "vllm_api.log"
    LOG_MAX  = 50 * 1024 * 1024
    try:
        f = log_path.open("ab")
        for raw_line in pipe:
            # Forward to pane 0 with prefix for visual distinction
            sys.stdout.buffer.write(b"[vLLM] " + raw_line)
            sys.stdout.buffer.flush()
            # Write to dedicated vllm_api.log
            f.write(raw_line)
            f.flush()
            # Simple rotation: truncate when exceeding 50 MB
            try:
                if log_path.stat().st_size > LOG_MAX:
                    f.close()
                    log_path.write_bytes(b"")
                    f = log_path.open("ab")
            except OSError:
                pass
    except Exception as e:
        log.error("vLLM output streaming thread error: %s", e)
    finally:
        try:
            f.close()
        except Exception:
            pass


def start_vllm():
    global vllm_proc, last_restart_ts
    with lock:
        if vllm_proc is not None and vllm_proc.poll() is None:
            log.info("vLLM already running (pid %s), skipping start.", vllm_proc.pid)
            return
        info = load_model_info(current_model_id)
        cmd  = build_vllm_cmd(current_model_path, info)
        log.info("Starting vLLM: %s", " ".join(cmd))
        try:
            # Set offline mode so transformers 5.x / huggingface_hub 1.11+
            # never call hf_hub_download with a local filesystem path.
            # maybe_override_with_speculators() passes the local model path to
            # PretrainedConfig.get_config_dict() which in transformers 5.x
            # reaches hf_hub_download and fails repo-ID validation.
            # TRANSFORMERS_OFFLINE=1 forces local_files_only=True internally,
            # bypassing the hub entirely. The model is fully local — no hub
            # access is ever needed.
            vllm_env = os.environ.copy()
            vllm_env["TRANSFORMERS_OFFLINE"] = "1"
            vllm_env["HF_HUB_OFFLINE"]       = "1"
            # NOTE: Do NOT set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1.
            # On RTX 5060 8GB, it reduces effective gpu_memory_utilization from
            # 0.98 to ~0.9395, shrinking KV cache from 68,000 to 61,472 tokens.
            # The 0.98 value is the empirical OOM ceiling for this GPU and must
            # not be raised to 1.0 either (causes immediate OOM on startup).

            # Capture stdout+stderr so we can stream to vllm_api.log
            vllm_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=vllm_env,
            )
            last_restart_ts = time.time()
            log.info("vLLM process started (pid %s)", vllm_proc.pid)
            threading.Thread(
                target=_stream_vllm_output,
                args=(vllm_proc.stdout,),
                daemon=True,
                name="vllm-output-stream",
            ).start()
            _notify(f"Model starting: {current_model_id}")
            # Schedule a deferred config validation.
            # Polls vllm_api.log every 5 s until both the KV cache and encoder
            # cache calibration lines appear, then runs _validate_model_config.
            # This is event-driven rather than a fixed sleep, so it works correctly
            # regardless of how long torch.compile + warmup takes on first boot
            # (empirically 90–150 s; a fixed 90 s sleep misses the KV cache line).
            def _deferred_validate():
                POLL_INTERVAL = 5    # seconds between log checks
                TIMEOUT       = 300  # hard ceiling (torch.compile + warmup ≤ ~150 s)
                start = time.monotonic()
                while time.monotonic() - start < TIMEOUT:
                    time.sleep(POLL_INTERVAL)
                    actual = _read_vllm_actual_config()
                    if "kv_cache_tokens" in actual and "encoder_cache_tokens" in actual:
                        log.debug(
                            "config-validator: both calibration lines detected "
                            "after %.0f s", time.monotonic() - start
                        )
                        break
                else:
                    log.warning(
                        "config-validator: timed out after %d s waiting for vLLM "
                        "calibration lines. Proceeding with partial data.", TIMEOUT
                    )
                info = load_model_info(current_model_id)
                _validate_model_config(info)
            threading.Thread(target=_deferred_validate, daemon=True,
                             name="config-validator").start()
        except FileNotFoundError as e:
            log.error(
                "FATAL: vLLM executable not found: %s — Command: %s",
                e, " ".join(cmd)
            )
            vllm_proc = None
        except Exception as e:
            log.error("Failed to start vLLM: %s\n%s", e, traceback.format_exc())
            vllm_proc = None


def stop_vllm():
    global vllm_proc
    with lock:
        if vllm_proc is not None and vllm_proc.poll() is None:
            log.info("Stopping vLLM (pid %s)", vllm_proc.pid)
            vllm_proc.terminate()
            try:
                vllm_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                vllm_proc.kill()
        vllm_proc = None


def restart_vllm(new_model_path: Optional[str] = None,
                 new_model_id: Optional[str] = None):
    global current_model_path, current_model_id
    if new_model_path:
        # Strip trailing slash — transformers 4.57+ rejects paths ending with /
        current_model_path = new_model_path.rstrip("/")
    if new_model_id:
        current_model_id = new_model_id
    stop_vllm()
    start_vllm()


_vllm_alive_cache: tuple = (0.0, False)   # (timestamp, result)
_VLLM_ALIVE_CACHE_TTL = 3.0              # seconds

def _read_vllm_actual_config() -> dict:
    """
    Parse vllm_api.log after startup to extract the actual values vLLM allocated.
    Returns a dict with any of: kv_cache_tokens, encoder_cache_tokens, block_size.

    Target log lines:
      "GPU KV cache size: 68,000 tokens"
      "Encoder cache will be initialized with a budget of 16384 tokens"
      "Setting attention block size to 544 tokens to ensure that attention page size..."
    Falls back to empty dict if the log isn't readable or values not yet written.
    """
    log_path = LOG_DIR / "vllm_api.log"
    result = {}
    if not log_path.is_file():
        return result
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"GPU KV cache size: ([\d,]+) tokens", text)
        if m:
            result["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
        m = re.search(r"Encoder cache will be initialized with a budget of (\d+) tokens", text)
        if m:
            result["encoder_cache_tokens"] = int(m.group(1))
        m = re.search(r"Setting attention block size to (\d+) tokens", text)
        if m:
            result["block_size"] = int(m.group(1))
    except Exception as e:
        log.debug("_read_vllm_actual_config: %s", e)
    return result


def _validate_model_config(model_info: dict) -> None:
    """
    After vLLM starts, compare JSON-specified values with what vLLM actually
    allocated. Log warnings when the JSON values are inconsistent with reality.

    max_model_len auto-calibration:
      Reads the actual KV cache capacity and block_size reported by vLLM.

      Formula: optimal = (kv_cache_tokens // block_size) * block_size
        Since vLLM always allocates a whole number of blocks, kv_cache_tokens
        is already block-aligned, so this equals kv_cache_tokens exactly.
        Using kv_cache_tokens directly creates a STABLE FIXED POINT:
          max_model_len = kv_cache_tokens → same VRAM profiling overhead →
          same kv_cache_tokens on next boot → no further update.

      Why NOT kv_cache_tokens − SAFETY_MARGIN:
        The block_size on this hardware is 544 tokens (set by mamba constraints).
        A 128-token margin is smaller than 1 block, so it doesn't guarantee a
        full-block boundary. Worse, it creates an oscillation: a smaller
        max_model_len recovers 1 block → KV rises by 544 → optimal rises by ~544
        → max_model_len rises → KV drops by 544 → repeat indefinitely.

      Floor (65536): never calibrate below this minimum for this hardware.
      Falls back to 65536 if detection fails.
    """
    actual = _read_vllm_actual_config()
    if not actual:
        return

    max_model_len         = model_info.get("max_model_len", 65536)
    vision_encoder_budget = model_info.get("vision_encoder_budget",
                                           max(4096, max_model_len // 2))
    json_kv               = model_info.get("kv_cache_tokens_expected", 0)

    if "kv_cache_tokens" in actual:
        kv         = actual["kv_cache_tokens"]
        block_size = actual.get("block_size", 0)
        log.info("Runtime calibration: vLLM KV cache = %s tokens "
                 "(block_size=%s)", f"{kv:,}",
                 f"{block_size}" if block_size else "unknown")

        if json_kv and abs(kv - json_kv) > 1000:
            log.warning(
                "KV cache mismatch: JSON expected %s but vLLM allocated %s. "
                "Check gpu_memory_utilization.",
                f"{json_kv:,}", f"{kv:,}"
            )

        # Stable block-aligned calibration (see docstring above).
        FALLBACK = 65536
        if block_size and block_size > 0:
            optimal_max = max(FALLBACK, (kv // block_size) * block_size)
        else:
            optimal_max = max(FALLBACK, kv)   # kv already block-aligned

        if optimal_max != max_model_len:
            log.info(
                "Auto-calibrating max_model_len: %d → %d "
                "(KV cache = %s tokens, block_size = %s)",
                max_model_len, optimal_max, f"{kv:,}",
                f"{block_size}" if block_size else "unknown"
            )
            info_path = MODELS_ROOT / "info" / f"{current_model_id}.json"
            if info_path.is_file():
                try:
                    with info_path.open("r", encoding="utf-8") as fj:
                        raw = json.load(fj)
                    raw["max_model_len"] = optimal_max
                    with info_path.open("w", encoding="utf-8") as fj:
                        json.dump(raw, fj, indent=4, ensure_ascii=False)
                    model_info["max_model_len"] = optimal_max
                    log.info(
                        "Calibrated max_model_len=%d written to %s. "
                        "Takes effect on next restart.",
                        optimal_max, info_path
                    )
                except Exception as e:
                    log.warning(
                        "Failed to write calibrated max_model_len to JSON: %s. "
                        "Using fallback=%d on next restart.",
                        e, FALLBACK
                    )
            else:
                log.warning(
                    "Model info JSON not found at %s — "
                    "calibrated max_model_len=%d was NOT persisted.",
                    info_path, optimal_max
                )
        else:
            log.info(
                "max_model_len=%d matches calibrated optimum — no JSON update needed.",
                max_model_len
            )

    if "encoder_cache_tokens" in actual:
        log.info("Runtime calibration: vLLM encoder cache = %s tokens",
                 f"{actual['encoder_cache_tokens']:,}")
        if vision_encoder_budget != actual["encoder_cache_tokens"]:
            log.info(
                "Updating vision_encoder_budget from JSON %s → actual %s",
                vision_encoder_budget, actual["encoder_cache_tokens"]
            )
            model_info["vision_encoder_budget"] = actual["encoder_cache_tokens"]
            model_info["vision_token_budget"]   = max(
                1024, actual["encoder_cache_tokens"] // 4
            )


def vllm_is_alive() -> bool:
    """Check vLLM API liveness with a 3-second result cache.
    Prevents flooding /v1/models when the browser polls every 4 s
    and the failure monitor polls every 10 s simultaneously.
    """
    global _vllm_alive_cache
    now = time.monotonic()
    if now - _vllm_alive_cache[0] < _VLLM_ALIVE_CACHE_TTL:
        return _vllm_alive_cache[1]
    info = load_model_info(current_model_id)
    base = _vllm_api_base(info)
    try:
        req = urllib.request.Request(f"{base}/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=3) as r:
            result = r.status == 200
    except Exception:
        result = False
    _vllm_alive_cache = (now, result)
    return result

# ---------------------------------------------------------------------------
# Periodic heartbeat — gap in log signals a crash
# ---------------------------------------------------------------------------

def _heartbeat():
    while True:
        time.sleep(300)
        with lock:
            running = vllm_proc is not None and vllm_proc.poll() is None
        log.info(
            "HEARTBEAT — launcher alive | vLLM process: %s | vLLM API: %s",
            "running" if running else "stopped",
            "alive" if vllm_is_alive() else "not responding",
        )

# ---------------------------------------------------------------------------
# Conversation storage helpers
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text[:60]


def conv_folder(conv_id: str) -> Path:
    folder = USER_DATA_DIR / conv_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "files").mkdir(exist_ok=True)
    return folder


def conv_meta_path(conv_id: str) -> Path:
    return conv_folder(conv_id) / "messages.jsonl"


def list_conversations() -> List[Dict[str, Any]]:
    convs = []
    if not USER_DATA_DIR.is_dir():
        return convs
    for entry in sorted(USER_DATA_DIR.iterdir(), key=lambda p: p.stat().st_mtime):
        if not entry.is_dir():
            continue
        meta = entry / "messages.jsonl"
        display_name = entry.name
        created_at   = None
        if meta.is_file():
            try:
                with meta.open("r", encoding="utf-8") as f:
                    first = f.readline()
                if first:
                    obj          = json.loads(first)
                    display_name = obj.get("conv_display_name", entry.name)
                    created_at   = obj.get("conv_created_at")
            except Exception:
                pass
        convs.append({
            "id": entry.name, "display_name": display_name, "created_at": created_at
        })
    return convs


def create_conversation(display_name: str) -> Dict[str, Any]:
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug    = _slugify(display_name) or "chat"
    conv_id = f"{slug}_{ts}"
    folder  = conv_folder(conv_id)
    meta    = {
        "type": "meta", "conv_id": conv_id,
        "conv_display_name": display_name, "conv_created_at": ts,
    }
    with (folder / "messages.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    log.info("Created conversation %s ('%s')", conv_id, display_name)
    return {"id": conv_id, "display_name": display_name, "created_at": ts}


def rename_conversation(conv_id: str, new_name: str) -> bool:
    meta_path = conv_meta_path(conv_id)
    if not meta_path.is_file():
        log.warning("rename_conversation: %s not found", conv_id)
        return False
    lines = meta_path.read_text(encoding="utf-8").splitlines(keepends=True)
    if not lines:
        return False
    try:
        obj = json.loads(lines[0])
        obj["conv_display_name"] = new_name
        lines[0] = json.dumps(obj, ensure_ascii=False) + "\n"
        meta_path.write_text("".join(lines), encoding="utf-8")
        log.info("Renamed conversation %s to '%s'", conv_id, new_name)
        return True
    except Exception as e:
        log.error("rename_conversation error: %s", e)
        return False


def delete_conversation(conv_id: str) -> bool:
    import shutil
    folder = USER_DATA_DIR / conv_id
    if not folder.is_dir():
        return False
    shutil.rmtree(folder)
    log.info("Deleted conversation %s", conv_id)
    return True


def load_history(conv_id: str) -> List[Dict[str, Any]]:
    meta_path = conv_meta_path(conv_id)
    if not meta_path.is_file():
        return []
    messages = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") == "meta" or "role" not in obj:
                continue
            if obj.get("role") in ("user", "assistant"):
                messages.append({
                    "role":              obj["role"],
                    "content":           obj.get("content", ""),
                    "timestamp":         obj.get("timestamp", ""),
                    "reasoning_content": obj.get("reasoning_content"),
                    "attachments":       obj.get("attachments"),
                })
    return messages


def append_message(conv_id: str, role: str, content: str,
                   reasoning: Optional[str] = None,
                   attachments: Optional[List] = None) -> str:
    meta_path = conv_meta_path(conv_id)
    ts        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record: Dict[str, Any] = {
        "type": "message", "role": role, "content": content, "timestamp": ts
    }
    if reasoning:
        record["reasoning_content"] = reasoning
    if attachments:
        record["attachments"] = attachments
    with meta_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return ts

# ---------------------------------------------------------------------------
# FIFO cleanup (background thread, hourly)
# ---------------------------------------------------------------------------

def _cleanup_old_conversations():
    while True:
        try:
            import shutil
            cutoff = datetime.now() - timedelta(days=CONVERSATION_TTL_DAYS)
            for entry in USER_DATA_DIR.iterdir():
                if not entry.is_dir():
                    continue
                meta = entry / "messages.jsonl"
                if not meta.is_file():
                    continue
                try:
                    with meta.open("r", encoding="utf-8") as f:
                        first = f.readline()
                    obj = json.loads(first)
                    created_dt = datetime.strptime(
                        obj.get("conv_created_at", ""), "%Y%m%d_%H%M%S"
                    )
                    if created_dt < cutoff:
                        shutil.rmtree(entry)
                        log.info("FIFO cleanup: removed conversation %s", entry.name)
                except Exception:
                    pass
        except Exception as e:
            log.error("Cleanup thread error: %s", e)
        time.sleep(3600)

# ---------------------------------------------------------------------------
# Multimodal content builder
# ---------------------------------------------------------------------------

def _file_to_base64_data_url(filepath: Path, mime: str) -> str:
    """Read a file from disk and return a base64 data URL."""
    data = filepath.read_bytes()
    b64  = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _build_user_content(
    user_text: str,
    attachments: List[Dict],
    conv_id: str,
    is_vision: bool,
) -> Any:
    """
    Build the vLLM-compatible content for a user message.

    For vision-capable models:
      - Images → image_url content part with base64 data URL (read from disk)
      - Videos → video_url content part with serve URL (file may be large)
      - Documents → text note with filename

    For text-only models:
      - All attachments become text parts (filename note).
      - No image_url or video_url parts are ever sent.

    This safety check prevents vLLM 500 errors from unsupported content types.
    """
    if not attachments:
        return user_text

    parts       = [{"type": "text", "text": user_text}] if user_text else []
    vision_used = False

    for att in attachments:
        att_type = att.get("type", "document")
        name     = att.get("name", "")
        mime     = att.get("mime", "application/octet-stream")
        url      = att.get("url", "")

        if att_type == "image":
            if is_vision:
                # Read the image from disk and send as base64 data URL.
                # This avoids any potential aiohttp fetch issues.
                safe_name = url.rsplit("/", 1)[-1]
                filepath  = USER_DATA_DIR / conv_id / "files" / safe_name
                if filepath.is_file():
                    try:
                        data_url = _file_to_base64_data_url(filepath, mime)
                        parts.append({
                            "type":      "image_url",
                            "image_url": {"url": data_url},
                        })
                        vision_used = True
                    except Exception as e:
                        log.warning(
                            "Failed to encode image %s as base64: %s. "
                            "Sending as text context only.", name, e
                        )
                        parts.append({
                            "type": "text",
                            "text": f"[Image attached: {name} — could not encode]",
                        })
                else:
                    log.warning("Image file not found on disk: %s", filepath)
                    parts.append({
                        "type": "text",
                        "text": f"[Image attached: {name} — file not found on server]",
                    })
            else:
                # Text-only model: describe the image without sending pixel data.
                parts.append({
                    "type": "text",
                    "text": (
                        f"[Image attached: {name}. "
                        "This model does not support vision input.]"
                    ),
                })

        elif att_type == "video":
            if is_vision:
                # Send videos as a URL (they can be large — base64 is impractical).
                parts.append({
                    "type":      "video_url",
                    "video_url": {"url": url},
                })
            else:
                parts.append({
                    "type": "text",
                    "text": f"[Video attached: {name} — this model does not support video input]",
                })

        else:
            # Document/other: send as a filename note; native vision handles content.
            parts.append({
                "type": "text",
                "text": f"[File attached: {name}]",
            })

    # Log the exact combination of information sources being sent to vLLM.
    if attachments:
        mode = ["NATIVE VISION (image_url base64)"] if vision_used else ["FILE NAME/TYPE ONLY (no vision)"]
        log.info("Content build for %d attachment(s): %s", len(attachments), " + ".join(mode))
    return parts if parts else user_text


# ---------------------------------------------------------------------------
# Tool resolution
# ---------------------------------------------------------------------------

def _resolve_tools(selected_tools: List[str],
                   model_info: Dict[str, Any]) -> List[Dict]:
    """
    Return vLLM-compatible tool definitions for the selected tool IDs.
    Returns empty list when selected_tools is empty or ["auto"]
    (auto mode: vLLM uses --enable-auto-tool-choice without an explicit list).
    """
    if not selected_tools or selected_tools == ["auto"]:
        return []

    # Merge builtin tools with any custom tools from the model JSON
    all_tools = dict(BUILTIN_TOOLS)
    for custom in model_info.get("custom_tools", []):
        if "id" in custom and "definition" in custom:
            all_tools[custom["id"]] = custom["definition"]

    result = []
    for tool_id in selected_tools:
        if tool_id in all_tools:
            result.append(all_tools[tool_id])
        else:
            log.warning("Unknown tool requested: %s — skipping.", tool_id)
    return result

# ---------------------------------------------------------------------------
# LLM proxy
# ---------------------------------------------------------------------------

def _call_vllm_streaming(
    conv_id: str,
    user_content: Any,
    model_path: str,
    model_info: Dict[str, Any],
    think: bool,
    tools: List[Dict],
) -> Tuple[str, Optional[str], Any]:
    """
    Assemble conversation history and call vLLM with streaming.
    Returns (final_content, reasoning_content, tool_calls).
    """
    history       = load_history(conv_id)
    vllm_messages = [{"role": m["role"], "content": m["content"]} for m in history]

    if isinstance(user_content, list):
        vllm_messages.append({"role": "user", "content": user_content})
    else:
        vllm_messages.append({"role": "user", "content": str(user_content)})

    # ── All generation budget values from model config — NO hardcoded constants.
    # Override per model in the JSON. Defaults derived from max_model_len so
    # the calculation scales correctly on any hardware (8GB laptop → H200 cluster).
    max_model_len         = model_info.get("max_model_len",          9216)
    max_output_tokens     = model_info.get("max_output_tokens",      max(512,  max_model_len // 4))
    min_output_tokens     = model_info.get("min_output_tokens",      max(64,   max_output_tokens // 8))
    vision_encoder_budget = model_info.get("vision_encoder_budget",  max(4096, max_model_len // 2))
    vision_token_budget   = model_info.get("vision_token_budget",    max(1024, vision_encoder_budget // 4))

    def _text_chars(msg):
        """Count characters of text-only parts. Excludes base64 image data."""
        c = msg.get("content", "")
        if isinstance(c, str):
            return len(c)
        return sum(
            len(part.get("text", ""))
            for part in c
            if isinstance(part, dict) and part.get("type") == "text"
        )

    has_image = any(
        isinstance(m.get("content"), list) and
        any(p.get("type") == "image_url" for p in m["content"] if isinstance(p, dict))
        for m in vllm_messages
    )

    # ── Conversation history pruning ──────────────────────────────────────
    # When accumulated history exceeds the context budget, drop the oldest
    # turns so the current turn always fits. Allows unlimited session length.
    # Reserve max_output_tokens for generation + headroom for image tokens.
    history_token_limit = max_model_len - max_output_tokens - (
        vision_token_budget if has_image else 0
    )
    while len(vllm_messages) > 1:
        if (sum(_text_chars(m) for m in vllm_messages) // 4) <= history_token_limit:
            break
        vllm_messages.pop(0)

    # ── Generation budget ─────────────────────────────────────────────────
    input_tokens_est = sum(_text_chars(m) for m in vllm_messages) // 4

    if has_image:
        # Reserve vision_token_budget for the visual encoder output.
        # The remainder of the context window is available for generation.
        available = max_model_len - vision_token_budget - input_tokens_est - 64
    else:
        available = max_model_len - input_tokens_est - 64

    safe_max = max(min_output_tokens, min(max_output_tokens, available))

    log.debug(
        "Token budget: max_model_len=%d, input≈%d, vision=%d, safe_max=%d",
        max_model_len, input_tokens_est,
        vision_token_budget if has_image else 0, safe_max,
    )

    base    = _vllm_api_base(model_info)
    api_url = f"{base}/v1/chat/completions"

    payload: Dict[str, Any] = {
        "model":            model_path,
        "messages":         vllm_messages,
        "max_tokens":       safe_max,
        "temperature":      1.0 if think else 0.7,
        "top_p":            0.95 if think else 0.8,
        "presence_penalty": 1.5,
        "stream":           True,
        "extra_body": {"top_k": 20, "enable_thinking": think},
    }

    # Include tool definitions when specific tools are selected.
    # When empty (auto mode), vLLM uses --enable-auto-tool-choice freely.
    if tools:
        payload["tools"]       = tools
        payload["tool_choice"] = "auto"

    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    content_parts   = []
    reasoning_parts = []
    tool_calls_raw  = None

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except Exception:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                c = delta.get("content") or ""
                if c:
                    content_parts.append(c)
                r = delta.get("reasoning_content") or ""
                if r:
                    reasoning_parts.append(r)
                tc = delta.get("tool_calls")
                if tc:
                    tool_calls_raw = tc

    except urllib.error.HTTPError as e:
        # Read and log the full vLLM error body for immediate diagnosis.
        try:
            error_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            error_body = "(could not read error body)"
        log.error(
            "vLLM HTTP %s error.\nURL: %s\nBody: %s",
            e.code, api_url, error_body[:4000]
        )
        raise RuntimeError(
            f"vLLM returned HTTP {e.code}. "
            f"Full error logged to vllm_launcher.log. "
            f"Summary: {error_body[:300]}"
        ) from e

    except Exception as e:
        log.error("vLLM streaming error: %s\n%s", e, traceback.format_exc())
        raise

    raw_content   = "".join(content_parts)
    raw_reasoning = "".join(reasoning_parts) if reasoning_parts else None

    # Bulletproof fallback: use reasoning as content when content is empty.
    final_content   = raw_content if raw_content else (raw_reasoning or "")
    final_reasoning = raw_reasoning if raw_content else None

    log.debug(
        "vLLM response: content=%d chars, reasoning=%s chars, tools=%s",
        len(final_content),
        len(final_reasoning) if final_reasoning else 0,
        bool(tool_calls_raw),
    )
    return final_content, final_reasoning, tool_calls_raw

# ---------------------------------------------------------------------------
# Threading HTTP server — prevents /files/ deadlock
# ---------------------------------------------------------------------------

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        err = traceback.format_exc()
        if "BrokenPipe" not in err and "ConnectionReset" not in err:
            log.error("HTTP handler error for %s:\n%s", client_address, err)

# ---------------------------------------------------------------------------
# HTTP microservice
# ---------------------------------------------------------------------------

class MicroserviceHandler(BaseHTTPRequestHandler):

    def log_message(self, *args):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Conv-Id")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")

    def _json(self, code: int, payload: Any):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _read_body(self) -> Dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return {}

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        p = self.path.split("?")[0].rstrip("/")

        if p == "/status":
            with lock:
                running = vllm_proc is not None and vllm_proc.poll() is None
                pid     = vllm_proc.pid if running else None
            self._json(200, {
                "running":            running,
                "vllm_alive":         vllm_is_alive(),
                "clients":            len(connected_clients),
                "current_model_id":   current_model_id,
                "current_model_path": current_model_path,
                "pid":                pid,
                "last_restart_ts":    last_restart_ts,
                "thinking_enabled":   thinking_enabled,
                "identity":           IDENTITY,
            })

        elif p == "/models":
            self._json(200, {"models": list_models()})

        elif p == "/tools":
            # Return the full builtin tool catalogue for the frontend.
            self._json(200, {
                "builtin_tools": [
                    {
                        "id":          tid,
                        "name":        tdef["function"]["name"],
                        "description": tdef["function"]["description"],
                    }
                    for tid, tdef in BUILTIN_TOOLS.items()
                ]
            })

        elif p == "/health":
            with lock:
                proc_ok = vllm_proc is not None and vllm_proc.poll() is None
            self._json(200, {"launcher_running": proc_ok, "vllm_alive": vllm_is_alive()})

        elif p == "/conversations":
            self._json(200, {"conversations": list_conversations(), "identity": IDENTITY})

        elif p.startswith("/conversations/") and p.endswith("/messages"):
            conv_id = p[len("/conversations/"):-len("/messages")]
            self._json(200, {"messages": load_history(conv_id)})

        elif p.startswith("/conversations/") and "/files/" in p:
            parts    = p[len("/conversations/"):].split("/files/", 1)
            conv_id  = parts[0]
            filename = parts[1] if len(parts) > 1 else ""
            filepath = USER_DATA_DIR / conv_id / "files" / filename
            if not filepath.is_file():
                self._json(404, {"error": "file not found"})
                return
            mime, _ = mimetypes.guess_type(str(filepath))
            mime     = mime or "application/octet-stream"
            data     = filepath.read_bytes()
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type",   mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        p = self.path.split("?")[0].rstrip("/")

        if p == "/restart":
            data       = self._read_body()
            model_id   = data.get("model_id")
            model_path = data.get("model_path")
            if model_id and not model_path:
                model_path = str((MODELS_ROOT / model_id).resolve()) + "/"
            if model_path and not Path(model_path).is_dir():
                self._json(400, {"error": "invalid model_path"})
                return
            threading.Thread(
                target=restart_vllm,
                kwargs={"new_model_path": model_path, "new_model_id": model_id},
                daemon=True,
            ).start()
            self._json(200, {
                "status":     "restarting",
                "model_id":   current_model_id,
                "model_path": current_model_path,
            })

        elif p == "/shutdown":
            stop_vllm()
            self._json(200, {"status": "shutting down"})
            threading.Timer(1.0, lambda: sys.exit(0)).start()

        elif p == "/thinking":
            global thinking_enabled
            data             = self._read_body()
            thinking_enabled = bool(data.get("enabled", False))
            alive            = vllm_is_alive()
            log.info("Thinking mode set to %s (vllm_alive=%s)", thinking_enabled, alive)
            self._json(200, {
                "thinking_enabled": thinking_enabled,
                "vllm_alive":       alive,
                "ok":               alive,
            })

        elif p == "/conversations":
            data = self._read_body()
            name = (data.get("display_name") or "").strip()
            if not name:
                self._json(400, {"error": "display_name required"})
                return
            self._json(200, create_conversation(name))

        elif p.startswith("/conversations/") and p.endswith("/rename"):
            conv_id  = p[len("/conversations/"):-len("/rename")]
            data     = self._read_body()
            new_name = (data.get("display_name") or "").strip()
            if not new_name:
                self._json(400, {"error": "display_name required"})
                return
            ok = rename_conversation(conv_id, new_name)
            self._json(200 if ok else 404, {"ok": ok})

        elif p.startswith("/conversations/") and p.endswith("/chat"):
            conv_id = p[len("/conversations/"):-len("/chat")]
            data    = self._read_body()

            user_text      = data.get("message", "").strip()
            attachments    = data.get("attachments", [])
            selected_tools = data.get("selected_tools", [])   # [] = auto

            if not user_text and not attachments:
                self._json(400, {"error": "message required"})
                return

            m_info    = load_model_info(current_model_id)
            is_vision = model_has_vision(m_info)

            # Build multimodal content with safety checks.
            user_content = _build_user_content(
                user_text, attachments, conv_id, is_vision
            )

            # Resolve the tool definitions for the selected tools.
            tools = _resolve_tools(selected_tools, m_info)

            # Persist user message.
            user_ts = append_message(
                conv_id, "user", user_text,
                attachments=[a.get("name") for a in attachments] if attachments else None,
            )

            try:
                final_content, reasoning, tool_calls = _call_vllm_streaming(
                    conv_id, user_content, current_model_path,
                    m_info, thinking_enabled, tools
                )
            except Exception as e:
                log.error("Chat error: %s", e)
                self._json(500, {"error": str(e)})
                return

            ai_ts = append_message(conv_id, "assistant", final_content, reasoning=reasoning)

            self._json(200, {
                "content":           final_content,
                "reasoning_content": reasoning,
                "tool_calls":        tool_calls,
                "thinking_enabled":  thinking_enabled,
                "user_timestamp":    user_ts,
                "ai_timestamp":      ai_ts,
            })

        else:
            self._json(404, {"error": "not found"})

    def do_DELETE(self):
        p = self.path.split("?")[0].rstrip("/")
        if p.startswith("/conversations/"):
            conv_id = p[len("/conversations/"):]
            ok = delete_conversation(conv_id)
            self._json(200 if ok else 404, {"ok": ok})
        else:
            self._json(404, {"error": "not found"})


def run_http_server():
    server = ThreadingHTTPServer(("127.0.0.1", 8770), MicroserviceHandler)
    log.info("HTTP microservice listening on :8770 (threaded)")
    server.serve_forever()

# ---------------------------------------------------------------------------
# WebSocket server
# ---------------------------------------------------------------------------

async def ws_handler(websocket):
    with lock:
        connected_clients.add(websocket)
    log.info("WS client connected (total %d)", len(connected_clients))
    try:
        async for _ in websocket:
            pass
    finally:
        with lock:
            connected_clients.discard(websocket)
        log.info("WS client disconnected (total %d)", len(connected_clients))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _shutdown_handler():
    log.info("Shutdown signal received.")
    stop_vllm()
    sys.exit(0)


async def main():
    log.info("LocalAI launcher starting (identity=%s)", IDENTITY)
    log.info("Using Python executable: %s", sys.executable)

    threading.Thread(target=_cleanup_old_conversations, daemon=True,
                     name="fifo-cleanup").start()
    threading.Thread(target=_heartbeat, daemon=True,
                     name="heartbeat").start()

    start_vllm()

    threading.Thread(target=run_http_server, daemon=True,
                     name="http-server").start()

    await websockets.serve(ws_handler, "127.0.0.1", 8765)
    log.info("WebSocket server listening on :8765")

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig, lambda: asyncio.create_task(_shutdown_handler())
        )

    await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — shutting down.")
        stop_vllm()
    except SystemExit:
        pass
    except Exception as e:
        log.critical(
            "FATAL: unhandled exception in main loop: %s\n%s",
            e, traceback.format_exc()
        )
        stop_vllm()
        sys.exit(1)
