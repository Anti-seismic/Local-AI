#!/usr/bin/env python3
# =============================================================================
# vllm_launcher.py — Backend orchestrator for LocalAI.
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

# input_guard: server-side prompt injection blocking (defence in depth).
# Must be imported before any WebSocket message reaches the model.
try:
    from input_guard import check_message as _guard_check
    _INPUT_GUARD = True
except ImportError:
    _INPUT_GUARD = False

# notes_catalog: contextual operator notes injected into model JSON on first boot
# and on every environment re-detection.  Gracefully absent — notes simply won't
# be enriched if the file is missing (no crash, no degraded functionality).
try:
    from notes_catalog import get_notes_for_context, infer_model_family, NoteContext
    _NOTES_CATALOG = True
except ImportError:
    _NOTES_CATALOG = False

import websockets
import constants

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

# CORS allowed origins — module-level so HTTP handler methods can read it
# without needing access to model_info (a local variable in start_vllm).
# Updated from the model JSON once start_vllm() loads it.
# Wildcard intentionally absent — see upload_server.py for full rationale.
_CORS_ALLOWED_ORIGINS: frozenset = frozenset([
    constants.URL_STATIC,
    f"http://localhost:{constants.PORT_STATIC}",
])
thinking_enabled   = False
last_restart_ts    = None

# ---------------------------------------------------------------------------
# System-ready semaphore
# ---------------------------------------------------------------------------
# This is the authoritative server-side gate that prevents any /chat request
# from reaching the model before the full stack is ready.
# The UI send button is greyed as a UX hint only — this lock is the real gate.
# Set to True only when vLLM has passed its liveness check after startup.
# Reset to False immediately on restart/stop.  Never read from client input.
_system_ready = threading.Event()   # cleared = not ready, set = ready

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


def _model_info_path(model_id: str) -> Path:
    """Return the canonical JSON path for a model — always derived from model_id."""
    return MODELS_ROOT / "info" / f"{model_id}.json"


def _scaffold_model_info(model_id: str) -> Dict[str, Any]:
    """
    Build a minimal, correct JSON template for a model that has never been
    configured before.  Called on first boot of any new model.

    All values are either auto-detectable at runtime (environment block) or
    set to safe conservative defaults that the calibration loop will refine.
    The operator should fill in model-specific fields (params_b, description,
    license, capabilities) after first boot.

    Notes are populated by notes_catalog if available, falling back to a small
    set of universal notes if the catalog module is absent.
    """
    model_path = str((MODELS_ROOT / model_id).resolve())
    quantization = infer_quantization(model_id)

    # Best-effort notes from catalog.  At scaffold time we have no detected
    # hardware yet (torch not yet imported in the vLLM subprocess context here),
    # so we use zeroed / unknown values for hardware fields.  The catalog
    # predicates that require real hardware data (e.g. Hopper-specific notes)
    # will be evaluated again in detect_and_update_environment() once vLLM has
    # started and torch is available.
    if _NOTES_CATALOG:
        ctx = NoteContext(
            model_id       = model_id,
            model_family   = infer_model_family(model_id),
            quantization   = quantization,
            has_vision     = False,       # unknown at scaffold time
            has_reasoning  = False,       # unknown at scaffold time
            has_tool_calls = False,       # unknown at scaffold time
            vllm_version   = "",
            compute_cap    = "",
            gpu_vram_mb    = 0,
            gpu_name       = "",
            existing_notes = [],
        )
        notes = get_notes_for_context(ctx)
    else:
        # Fallback: minimal universal notes when catalog is not available
        notes = [
            "Auto-generated template on first boot. "
            "Update params_b, description, license, and capabilities.",
            "gpu_memory_utilization starts at 0.90 (safe default). "
            "Raise toward 0.98 only after verifying the model loads without OOM.",
            "max_model_len starts at 65536 and is auto-calibrated upward "
            "by vllm_launcher.py after the first successful start.",
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is set in the "
            "vLLM subprocess environment to prevent CUDA allocator fragmentation "
            "OOM on large vision batches. Safe on all hardware.",
        ]

    return {
        "id":                       model_id,
        "name":                     model_id,           # operator should update
        "path":                     model_path,
        "quantization":             quantization,
        "type":                     "chat",
        "params_b":                 None,               # operator should fill
        "max_model_len":            65536,              # conservative; calibrated up
        "max_num_batched_tokens":   None,               # auto-derived at launch
        "gpu_memory_utilization":   0.90,               # safe default; operator may raise
        "description":              "",
        "license":                  "",
        "capabilities":             ["chat"],
        "tool_call_parser":         "hermes",           # most common default
        "reasoning_parser":         None,
        "enable_auto_tool_choice":  True,
        "host":                     "127.0.0.1",
        "port":                     8000,
        "allowed_origins":          [
            constants.URL_STATIC,
            f"http://localhost:{constants.PORT_STATIC}",
        ],
        "environment":              {"notes": notes},
        "limit_mm_per_prompt":      {"image": None, "video": 1},
        "vllm_quantization":        "",
        "model_weights_vram_gb":    None,
    }


def load_model_info(model_id: str) -> Dict[str, Any]:
    info_path = _model_info_path(model_id)
    if info_path.is_file():
        try:
            with info_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning("Failed to load model info for %s: %s", model_id, e)
            return {}

    # First boot for this model — create a scaffold JSON so all subsequent
    # code paths (detect_and_update_environment, _validate_model_config, etc.)
    # can write to the file normally without special-casing a missing file.
    log.info(
        "No config JSON found for model '%s' — creating first-boot scaffold at %s",
        model_id, info_path,
    )
    scaffold = _scaffold_model_info(model_id)
    info_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with info_path.open("w", encoding="utf-8") as f:
            json.dump(scaffold, f, indent=4, ensure_ascii=False)
        log.info("Scaffold written. Edit %s to add model-specific metadata.", info_path)
    except Exception as e:
        log.warning("Could not write scaffold JSON for %s: %s", model_id, e)
    return scaffold


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
            # Weight-only VRAM footprint (excludes KV cache and activation memory).
            # Set by the admin in the model JSON; displayed in the UI model info panel.
            "model_weights_vram_gb":   info.get("model_weights_vram_gb"),
            # Available tools for this model (IDs from BUILTIN_TOOLS + custom_tools)
            "available_tools":         list(BUILTIN_TOOLS.keys()) +
                                       [t["id"] for t in info.get("custom_tools", [])],
        }
        # Ensure no trailing slash on stored path
        model["path"] = model["path"].rstrip("/")
        models.append(model)
    return models

# ---------------------------------------------------------------------------
# Hardware / software environment auto-detection
# ---------------------------------------------------------------------------

def detect_and_update_environment(model_info: dict, info_path: Path) -> None:
    """
    Detect current hardware and software environment at launch time.
    Writes detected values to the JSON's 'environment' block.
    Only the 'notes' key is manually managed; everything else is auto-detected.
    Falls back gracefully: if a value cannot be detected, it is simply absent.
    """
    env: Dict[str, Any] = {}

    # vLLM version
    try:
        import vllm as _vllm_mod
        env["vllm_version"] = _vllm_mod.__version__
    except Exception:
        pass

    # PyTorch version and CUDA
    try:
        import torch as _torch
        env["torch_version"] = _torch.__version__
        env["torch_cuda"]    = _torch.version.cuda or "n/a"
        if _torch.cuda.is_available():
            cap = _torch.cuda.get_device_capability()
            env["compute_capability"] = f"sm_{cap[0]}{cap[1]}"
    except Exception:
        pass

    # GPU via nvidia-smi (primary — covers NVIDIA)
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            parts = [p.strip() for p in r.stdout.strip().split(",")]
            if len(parts) >= 3:
                env["gpu"]            = parts[0]
                env["gpu_vram_mb"]    = int(parts[1])
                env["driver_version"] = parts[2]
    except Exception:
        pass

    # AMD GPU fallback via rocm-smi
    if "gpu" not in env:
        try:
            r = subprocess.run(
                ["rocm-smi", "--showproductname", "--showmeminfo", "vram",
                 "--csv"],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0:
                env["gpu"] = "AMD GPU (rocm-smi detected)"
        except Exception:
            pass

    # Read the environment block from disk rather than from the in-memory
    # model_info dict.  model_info was loaded at start_vllm() time, before
    # the previous run's detect_and_update_environment had written its notes.
    # Reading from disk here gives us the notes that were persisted by the
    # previous run, which is what the idempotency tag check needs.
    try:
        with info_path.open("r", encoding="utf-8") as _f:
            _disk_raw = json.load(_f)
        old_env = _disk_raw.get("environment", {})
    except Exception:
        old_env = model_info.get("environment", {})
    notes     = list(old_env.get("notes", []))   # mutable copy
    vllm_venv = old_env.get("vllm_venv", "")
    ui_venv   = old_env.get("ui_venv",   "")

    # Enrich notes via catalog now that we have real hardware data.
    # get_notes_for_context() is idempotent: it checks each note's tag against
    # the existing notes list and skips anything already present.
    if _NOTES_CATALOG:
        caps = [c.lower() for c in model_info.get("capabilities", [])]
        ctx  = NoteContext(
            model_id       = current_model_id,
            model_family   = infer_model_family(current_model_id),
            quantization   = model_info.get("quantization",
                                            infer_quantization(current_model_id)),
            has_vision     = any(c in caps for c in
                                 ("vision", "image", "video", "multimodal")),
            has_reasoning  = bool(model_info.get("reasoning_parser")),
            has_tool_calls = any(c in caps for c in ("tool_calls", "tools")),
            vllm_version   = env.get("vllm_version", ""),
            compute_cap    = env.get("compute_capability", ""),
            gpu_vram_mb    = env.get("gpu_vram_mb", 0),
            gpu_name       = env.get("gpu", ""),
            existing_notes = notes,
        )
        new_catalog_notes = get_notes_for_context(ctx)
        if new_catalog_notes:
            notes.extend(new_catalog_notes)
            log.info(
                "notes_catalog: added %d note(s) to %s JSON",
                len(new_catalog_notes), current_model_id,
            )

    new_env: Dict[str, Any] = {**env}
    # Persist the PYTORCH_CUDA_ALLOC_CONF value so operators know it is
    # intentionally set. This prevents fragmentation OOM in the unquantised
    # vision encoder when processing large image batches (e.g. 20-page PPTX).
    new_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if notes:
        new_env["notes"]    = notes
    if vllm_venv:
        new_env["vllm_venv"] = vllm_venv
    if ui_venv:
        new_env["ui_venv"]   = ui_venv

    model_info["environment"] = new_env

    if info_path.is_file():
        try:
            with info_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            raw["environment"] = new_env
            with info_path.open("w", encoding="utf-8") as f:
                json.dump(raw, f, indent=4, ensure_ascii=False)
            log.info("Environment auto-detection written to %s", info_path)
        except Exception as e:
            log.warning("Failed to persist detected environment to JSON: %s", e)


# ---------------------------------------------------------------------------
# vLLM process management
# ---------------------------------------------------------------------------

def build_vllm_cmd(model_path: str, model_info: Dict[str, Any]) -> List[str]:
    """
    Build the vLLM command entirely from model_info JSON — no hardcoded values.
    Uses sys.executable so vLLM always runs in the correct virtualenv.
    Optional parameters are only added when present in the JSON.

    Two-tier max_num_batched_tokens:
      Tier 1 (auto): max_num_batched_tokens_auto — derived from max_model_len.
      Tier 2 (override): max_num_batched_tokens — set by admin. Prevails if not null.
    The auto value is written to the JSON at build time.
    """
    # Strip any trailing slash from the model path.
    model_path = model_path.rstrip("/")

    host        = str(model_info.get("host", "127.0.0.1"))
    port        = str(model_info.get("port", 8000))
    gpu_util    = str(model_info.get("gpu_memory_utilization", 0.98))
    max_len     = str(model_info.get("max_model_len", 9216))

    # Two-tier max_num_batched_tokens
    auto_batched   = min(int(model_info.get("max_model_len", 9216)), 8192)
    admin_override = model_info.get("max_num_batched_tokens")   # None if absent or null
    effective_batched = admin_override if admin_override is not None else auto_batched
    model_info["max_num_batched_tokens_auto"] = auto_batched   # keep in memory for display
    if effective_batched != auto_batched:
        log.info(
            "max_num_batched_tokens: admin override %d prevails over auto %d",
            effective_batched, auto_batched
        )
    else:
        log.info(
            "max_num_batched_tokens: using auto-derived value %d "
            "(set 'max_num_batched_tokens' in JSON to override)",
            auto_batched
        )
    max_batched = str(effective_batched)
    # None (from a null in the JSON) means "this model doesn't support this
    # feature" — the corresponding flag is omitted from the cmd entirely (see below).
    tool_parser      = model_info.get("tool_call_parser")   # None → omit --tool-call-parser
    reasoning_parser = model_info.get("reasoning_parser")   # None → omit --reasoning-parser
    auto_tool        = model_info.get("enable_auto_tool_choice", True)

    allowed_origins = json.dumps(model_info.get(
        "allowed_origins",
        [constants.URL_STATIC, f"http://localhost:{constants.PORT_STATIC}"]
    ))

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model",                  model_path,
        "--host",                   host,
        "--port",                   port,
        "--gpu-memory-utilization", gpu_util,
        "--max-model-len",          max_len,
        "--max-num-batched-tokens", max_batched,
        "--allowed-origins",        allowed_origins,
    ]

    # --tool-call-parser and --reasoning-parser are omitted when null/absent:
    # null in JSON means "this model doesn't support this feature" (scaffold
    # default for a new unknown model). Passing the flag with an empty or
    # incorrect value causes vLLM to crash or silently misbehave.
    if tool_parser:
        cmd += ["--tool-call-parser", tool_parser]
    if reasoning_parser:
        cmd += ["--reasoning-parser", reasoning_parser]

    # Optional parameters — only included when explicitly set in the JSON.
    # --chat-template is intentionally omitted: Qwen3.5 bundles its chat
    # template inside tokenizer_config.json and vLLM reads it automatically.
    # Passing a name like "qwen2" causes vLLM to look for a file by that name
    # and fail with ValueError when it does not exist.

    # ── Multimodal per-prompt limits — two-tier: auto-derived + admin override ─
    # Tier 1 (auto): image = min(30, vision_encoder_budget // 256).
    #   256 tokens is the minimum useful vision budget per image; the formula
    #   ensures at least 30 images can be served even on small encoder budgets.
    # Tier 2 (admin): set limit_mm_per_prompt.image to a non-null integer to
    #   override the auto value.  null = use auto.
    # The auto value and a note are written to the JSON at launch.
    _enc_budget       = model_info.get("vision_encoder_budget", 16384)
    _auto_img_limit   = min(30, _enc_budget // 256)
    _mm_raw           = model_info.get("limit_mm_per_prompt", {})
    _admin_img        = _mm_raw.get("image")           # None if absent or null
    _eff_img          = _admin_img if _admin_img is not None else _auto_img_limit
    _video_limit      = _mm_raw.get("video", 1)

    # Persist auto value + note to JSON for admin visibility
    _mm_raw_update = {
        "image":      _mm_raw.get("image"),    # preserve admin override (may be null)
        "image_auto": _auto_img_limit,
        "image_note": (
            f"auto = min(30, vision_encoder_budget={_enc_budget} // 256) = {_auto_img_limit}. "
            "Set 'image' to a non-null integer to override."
        ),
        "video": _video_limit,
    }
    model_info["limit_mm_per_prompt"] = _mm_raw_update
    _info_path = _model_info_path(current_model_id)
    if _info_path.is_file():
        try:
            with _info_path.open("r", encoding="utf-8") as _f:
                _raw_json = json.load(_f)
            _raw_json["limit_mm_per_prompt"] = _mm_raw_update
            with _info_path.open("w", encoding="utf-8") as _f:
                json.dump(_raw_json, _f, indent=4, ensure_ascii=False)
        except Exception as _e:
            log.warning("Could not persist limit_mm_per_prompt to JSON: %s", _e)

    if _admin_img is not None:
        log.info("limit_mm_per_prompt.image: admin override=%d (auto=%d)",
                 _eff_img, _auto_img_limit)
    else:
        log.info("limit_mm_per_prompt.image: auto-derived=%d", _auto_img_limit)

    cmd += ["--limit-mm-per-prompt",
            json.dumps({"image": _eff_img, "video": _video_limit},
                       separators=(",", ":"))]

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
    return constants.URL_VLLM_API


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
    global vllm_proc, last_restart_ts, _CORS_ALLOWED_ORIGINS
    with lock:
        if vllm_proc is not None and vllm_proc.poll() is None:
            log.info("vLLM already running (pid %s), skipping start.", vllm_proc.pid)
            return
        info      = load_model_info(current_model_id)
        info_path = _model_info_path(current_model_id)
        # Auto-detect hardware/software environment and persist to JSON.
        detect_and_update_environment(info, info_path)
        # Update CORS allowed origins from model JSON.
        # Done here so _cors() never needs to reference model_info directly
        # (model_info is local to this function, not a module-level global).
        _CORS_ALLOWED_ORIGINS = frozenset(info.get(
            "allowed_origins",
            [constants.URL_STATIC, f"http://localhost:{constants.PORT_STATIC}"]
        ))
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
            # Mark system as NOT ready. Cleared here so any in-flight /chat
            # requests block immediately — before the old process is even dead.
            _system_ready.clear()
            log.info("System semaphore: CLOSED (vLLM restarting)")

            vllm_env = os.environ.copy()
            vllm_env["TRANSFORMERS_OFFLINE"] = "1"
            vllm_env["HF_HUB_OFFLINE"]       = "1"
            # Prevent CUDA allocator fragmentation OOM on large vision batches.
            # When 20 images are processed simultaneously through the unquantised
            # vision encoder, the allocator may have enough reserved-but-fragmented
            # VRAM to satisfy the request but fail to find a contiguous block.
            # expandable_segments:True allows growing into fragmented reserved
            # space, eliminating the OOM without any VRAM overhead.
            # Safe on all hardware — no effect when fragmentation is absent.
            vllm_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
                POLL_INTERVAL   = 5    # seconds between log checks
                TIMEOUT         = 300  # hard ceiling (torch.compile + warmup ≤ ~150 s)
                # Extra wait after calibration lines appear before HTTP server opens.
                # The KV cache / encoder lines come from EngineCore log stream;
                # the HTTP server only opens after the APIServer finishes multimodal
                # warmup (~16 s on this hardware, longer on first boot).
                # We therefore keep polling vllm_is_alive() after the calibration
                # lines are detected rather than checking exactly once and giving up.
                ALIVE_POLL      = 2    # seconds between liveness checks post-calibration
                ALIVE_TIMEOUT   = 150  # seconds to wait for HTTP after calibration lines.
                                       # Breakdown on cached run: torch.compile load (~17 s)
                                       # + CUDA graph capture (~9 s) + multimodal warmup
                                       # (~17 s) = ~43 s minimum.  On a fast-calibration boot
                                       # (5 s to detect lines), this gives >100 s headroom.
                                       # Observed worst case: 91 s (03:33:01 → 03:34:32).
                                       # 150 s is safe on all hardware.

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

                # Poll vllm_is_alive() until the HTTP server responds.
                # This bridges the gap between EngineCore calibration lines and
                # the APIServer multimodal warmup completing (up to ~30 s).
                alive_start = time.monotonic()
                while time.monotonic() - alive_start < ALIVE_TIMEOUT:
                    # Bypass the liveness cache for this check — we need a
                    # fresh HTTP probe, not a cached result from a previous poll.
                    global _vllm_alive_cache
                    _vllm_alive_cache = (0.0, False)   # invalidate cache
                    if vllm_is_alive():
                        _system_ready.set()
                        log.info(
                            "System semaphore: OPEN (vLLM ready after %.0f s)",
                            time.monotonic() - start,
                        )
                        return
                    time.sleep(ALIVE_POLL)

                log.warning(
                    "System semaphore: CLOSED — vLLM HTTP server did not respond "
                    "within %d s after calibration lines appeared. "
                    "Stack not ready. Check vllm_api.log for errors.",
                    ALIVE_TIMEOUT,
                )
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
        _system_ready.clear()
        log.info("System semaphore: CLOSED (stop_vllm)")
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
_VLLM_ALIVE_CACHE_TTL = 10.0   # seconds — reduces /v1/models polling noise by ~70%.
                                # 3 s was too short: overlapping browser poll (4 s)
                                # and /status handler calls expired the cache on
                                # every cycle, generating ~400 redundant requests
                                # per session.  10 s is safe — vLLM failure
                                # detection does not need sub-10 s resolution
                                # (heartbeat runs every 300 s).

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
            info_path = _model_info_path(current_model_id)
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
        enc_actual = actual["encoder_cache_tokens"]
        log.info("Runtime calibration: vLLM encoder cache = %s tokens",
                 f"{enc_actual:,}")

        # Persist vision_encoder_budget to JSON so upload_server.py can read
        # the calibrated value for its DPI safety cap (_get_encoder_budget()).
        # Also add vision capabilities to the JSON when the encoder is detected
        # on a scaffold that only had ["chat"] — evidence of a vision model.
        info_path = _model_info_path(current_model_id)
        if info_path.is_file():
            try:
                with info_path.open("r", encoding="utf-8") as fj:
                    raw = json.load(fj)

                raw["vision_encoder_budget"] = enc_actual
                raw["vision_token_budget"]   = max(1024, enc_actual // 4)

                # Auto-promote capabilities: when the encoder cache is found,
                # the model is a vision model. Add vision/image caps if absent.
                # This repairs scaffold JSONs that default to ["chat"] only.
                current_caps = raw.get("capabilities", [])
                added_caps   = []
                for cap in ("vision", "image"):
                    if cap not in current_caps:
                        current_caps.append(cap)
                        added_caps.append(cap)
                if added_caps:
                    raw["capabilities"] = current_caps
                    log.info(
                        "Auto-detected vision encoder — added capabilities %s "
                        "to JSON (was scaffold default ['chat'] only).",
                        added_caps,
                    )

                with info_path.open("w", encoding="utf-8") as fj:
                    json.dump(raw, fj, indent=4, ensure_ascii=False)
                log.info(
                    "vision_encoder_budget=%d written to %s.",
                    enc_actual, info_path,
                )
            except Exception as e:
                log.warning(
                    "Failed to persist vision_encoder_budget to JSON: %s", e
                )

        if vision_encoder_budget != enc_actual:
            log.info(
                "Updating vision_encoder_budget from JSON %s → actual %s",
                vision_encoder_budget, enc_actual,
            )
        model_info["vision_encoder_budget"] = enc_actual
        model_info["vision_token_budget"]   = max(1024, enc_actual // 4)


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
# vLLM process watchdog — exits launcher when vLLM dies unexpectedly
# ---------------------------------------------------------------------------

def _vllm_watchdog():
    """
    Monitor the vLLM subprocess. When it exits without stop_vllm() being
    called (unexpected crash), log the event and exit the launcher cleanly.

    This ensures that when the admin kills vLLM (or it crashes), the entire
    service tree collapses: the launcher exits, run_ai.sh's EXIT trap fires,
    and debug_ai.sh --kill cleans up upload_server, http.server, and tray.py.

    Does NOT fire when stop_vllm() is called deliberately — stop_vllm() sets
    vllm_proc = None before this watchdog can see the dead process.
    """
    while True:
        time.sleep(5)
        with lock:
            proc = vllm_proc
        if proc is None:
            # Deliberate stop — watchdog sleeps and re-checks.
            continue
        if proc.poll() is not None:
            # Process exited without stop_vllm() being called.
            rc = proc.returncode
            log.critical(
                "vLLM process exited unexpectedly (rc=%s). "
                "Launcher shutting down to allow clean service teardown.",
                rc,
            )
            _system_ready.clear()
            # Give the log handlers a moment to flush.
            time.sleep(1)
            sys.exit(1)


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


PDF_CONTEXT_DPI_UNKNOWN = "__unknown__"   # sentinel for _build_user_content PDF note


def _enumerate_pages(first, last, pdf_name: str) -> str:
    """
    Return 'Page A of Name, Page A+1 of Name, ..., Page B of Name'.

    Used in the multi-batch protocol to explicitly enumerate every page a
    reply is responsible for, in plain English with no symbols ("page of
    the document", never "image") and no arrow/bracket notation that could
    be misread by the model.
    """
    try:
        first_i, last_i = int(first), int(last)
    except (TypeError, ValueError):
        return f"Page {first} of {pdf_name} to Page {last} of {pdf_name}"
    return ", ".join(f"Page {p} of {pdf_name}" for p in range(first_i, last_i + 1))


def _build_multibatch_preamble(
    reply_num:        int,
    batch_first_page,
    batch_last_page,
    pdf_name:         str,
    doc_total_pages,
) -> str:
    """
    Build the per-reply preamble for the multi-batch document analysis
    protocol. Identical structure for reply 1 (batch 0) through the last
    image-batch reply — only reply_num and the batch's page range differ.

    batch_first_page / batch_last_page: the first and last page of THIS
    BATCH (not the document's overall first/last page — named explicitly
    to avoid that ambiguity).

    The preamble ends with a period (never a colon) and never repeats the
    user's original request text — the user's request is sent as its own
    message (reply 1) or is already established in conversation history
    (subsequent replies), referenced here only as "the tasks requested by
    the user".

    The executive summary (final reply) has its own dedicated builder —
    see _build_executive_summary_note.
    """
    page_list = _enumerate_pages(batch_first_page, batch_last_page, pdf_name)
    return (
        f"The work to be done is compartmentalised to be delivered in "
        f"multiple independent replies, each having its own output token budget.\n"
        f"This is reply {reply_num}. In this reply, the analysis is restrained to {page_list}.\n"
        f"Reply {reply_num} is strictly limited to pages "
        f"[{batch_first_page}–{batch_last_page}] of the document "
        f"'{pdf_name}' ({doc_total_pages} pages total). "
        f"Any page of the document outside this bracket must be disregarded.\n"
        f"Address every page in this range in order, individually for pages containing critical information "
        f"and collectively for the ones of less importance as everything must fit in this reply. "
        f"All selected pages must appear in the response. Omit blank pages and Table of Contents.\n"
        f"Source of each piece of information in your reply, reference the page of the document it comes from.\n"
        f"Title this reply 'Detailed Analysis of pages {batch_first_page} to {batch_last_page}'.\n"
        f"For the moment, perform the tasks requested by the user from "
        f"page {batch_first_page} to page {batch_last_page} inclusive of the document only.\n"
        f"Take all the time you need to think, organise your thoughts and structure your answer before to reply."
    )


def _build_executive_summary_note(
    total_replies: int,
    pdf_name:      str,
    total_pages,
    prior_analyses: List[Dict[str, Any]],
) -> str:
    """
    Build the final reply's prompt — an executive summary synthesising the
    analyses produced by all prior page-batch replies. The prior analyses
    are quoted verbatim (Solution 1) so the model has direct access to the
    full-document coverage without relying on an abstract "your history
    covers everything" claim.
    """
    quoted_analyses = ""
    for i, m in enumerate(prior_analyses, start=1):
        quoted_analyses += (
            f"\n\n--- ANALYSIS {i} OF {len(prior_analyses)} "
            f"(Reply {i}) ---\n"
            f"{m.get('content', '')}"
        )

    return (
        f"This is reply {total_replies}, the final reply — an "
        f"executive summary.\n"
        f"Document '{pdf_name}' — {total_pages} pages total.\n"
        f"Write an executive summary of the entire document: global "
        f"context, key findings across all {total_pages} pages, and "
        f"a conclusion. Below are the {len(prior_analyses)} analyses "
        f"produced in the previous replies, covering all {total_pages} "
        f"pages. Write the executive summary based on these analyses. "
        f"Do NOT analyse any new pages of the document — base your "
        f"summary entirely on the analyses provided below."
        f"{quoted_analyses}"
    )


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
            # Per-page label, inserted immediately before the image itself.
            # This gives the model an explicit, unambiguous mapping from
            # "the next image you see" to "Page N of <document>" — no
            # counting, no reliance on an aggregate "pages are in order"
            # claim, and no "image sequence" guessing.
            pdf_src  = att.get("pdf_source")
            pdf_page = att.get("pdf_page")
            if pdf_src and pdf_page:
                parts.append({
                    "type": "text",
                    "text": f"Page {pdf_page} of {pdf_src}",
                })

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

        elif att_type == "security_blocked":
            # File was blocked by security scan — inform the model.
            threats = att.get("security_report", {}).get("threats", [])
            parts.append({
                "type": "text",
                "text": (
                    f"[SECURITY ALERT: '{name}' was blocked before being sent to you. "
                    f"Reason: {'; '.join(threats)}. "
                    "The file content was NOT shown to you. The user has been warned.]"
                ),
            })

        else:
            # Document/other: send as a filename note; native vision handles content.
            parts.append({
                "type": "text",
                "text": f"[File attached: {name}]",
            })

    # ── PDF source context note ───────────────────────────────────────────
    # When multiple consecutive images originate from the same PDF, prepend a
    # single text part so the model understands it is reading a document
    # rather than a collection of unrelated images.
    pdf_sources: Dict[str, Dict] = {}
    for att in attachments:
        src = att.get("pdf_source")
        if src:
            if src not in pdf_sources:
                pdf_sources[src] = {
                    "pages":       0,
                    "total":       att.get("pdf_total_pages", "?"),
                    "dpi":         att.get("pdf_dpi", PDF_CONTEXT_DPI_UNKNOWN),
                }
            pdf_sources[src]["pages"] += 1

    for pdf_name, info in pdf_sources.items():
        note = (
            f"The following are {info['pages']} of {info['total']} pages "
            f"of the document '{pdf_name}'"
            + (f", rendered at {info['dpi']} DPI" if info["dpi"] != PDF_CONTEXT_DPI_UNKNOWN else "")
            + ". Each page is preceded by its page label."
        )
        # Insert after the user's text part (index 0), before the first page label.
        insert_idx = 1 if parts and parts[0].get("type") == "text" else 0
        parts.insert(insert_idx, {"type": "text", "text": note})

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
    conv_id:      str,
    user_content: Any,
    model_path:   str,
    model_info:   Dict[str, Any],
    think:        bool,
    tools:        List[Dict],
) -> Tuple[str, Optional[str], Any]:
    """
    Assemble conversation history and call vLLM with streaming.
    Returns (final_content, reasoning_content, tool_calls).
    """
    history = load_history(conv_id)

    # Every call site does append_message(conv_id, "user", <plain text>, ...)
    # immediately before calling this function — for display/audit purposes.
    # `user_content` is the RICHER version of that SAME turn (preamble +
    # page labels + images for multi-batch document analysis). Without this
    # step, vllm_messages would end with TWO consecutive "user" role
    # messages: the bare placeholder just persisted, and user_content
    # appended below — confusing the model (this was the root cause of
    # reply 1's preamble appearing to be ignored: for a brand-new
    # conversation, vllm_messages became exactly
    # [{"user": bare_text}, {"user": [preamble+images]}] with nothing else,
    # the most extreme case of this duplication).
    # Drop the bare duplicate so exactly ONE user message represents this turn.
    if history and history[-1].get("role") == "user":
        history = history[:-1]

    vllm_messages = [{"role": m["role"], "content": m["content"]} for m in history]

    if isinstance(user_content, list):
        vllm_messages.append({"role": "user", "content": user_content})
    else:
        vllm_messages.append({"role": "user", "content": str(user_content)})

    # ── All generation budget values — fully dynamic, derived from calibrated
    # runtime values. No hardcoded constants. Scales automatically from an
    # 8GB laptop to an H200 cluster.
    # Override behaviour: JSON field present and not null → admin override wins.
    # JSON field absent or null → auto-derived value is used.
    max_model_len         = model_info.get("max_model_len", 9216)
    vision_encoder_budget = model_info.get("vision_encoder_budget", max(4096, max_model_len // 16))
    # Two-tier: admin "image" key may be null (use auto), auto may be null
    # (hardware detection not yet run), final fallback = 20 (proven PoC floor).
    _lm_cfg   = model_info.get("limit_mm_per_prompt", {})
    img_limit = _lm_cfg.get("image") or _lm_cfg.get("image_auto") or 20

    # Dynamic vision token budget per image:
    #   MAX = full encoder budget (one image can use all encoder capacity)
    #   MIN = encoder budget divided by (2 × image limit) → reasonable floor
    #         even when the context is packed with many pages
    max_vision_per_img = vision_encoder_budget
    min_vision_per_img = max(64, vision_encoder_budget // (img_limit * 2))

    # Count visual attachments in the current turn only (last user message).
    num_vision_atts = sum(
        1
        for part in (
            vllm_messages[-1].get("content", [])
            if isinstance(vllm_messages[-1].get("content"), list) else []
        )
        if isinstance(part, dict) and part.get("type") in ("image_url", "video_url")
    )
    has_image = num_vision_atts > 0

    # Per-image token budget scales inversely with image count.
    vision_per_img  = max(min_vision_per_img,
                          min(max_vision_per_img, vision_encoder_budget // max(1, num_vision_atts)))
    total_vision    = num_vision_atts * vision_per_img

    # Dynamic output cap:
    #   MAX = 25% of max_model_len, hard-capped at 16384 (≈ 12,000 words)
    #   MIN = scales with model size, floor at 256
    # Admin override: if "max_output_tokens" is set in JSON (not null), it prevails.
    auto_max_output = min(max_model_len // 4, 16384)
    auto_min_output = max(256, max_model_len // 1000)
    max_output_tokens = model_info.get("max_output_tokens") or auto_max_output
    min_output_tokens = model_info.get("min_output_tokens") or auto_min_output

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

    # ── Conversation history pruning ──────────────────────────────────────
    # When accumulated history exceeds the context budget, drop the oldest
    # turns so the current turn always fits.
    history_token_limit = max_model_len - max_output_tokens - total_vision
    while len(vllm_messages) > 1:
        if (sum(_text_chars(m) for m in vllm_messages) // 4) <= history_token_limit:
            break
        vllm_messages.pop(0)

    # ── Generation budget ─────────────────────────────────────────────────
    input_tokens_est = sum(_text_chars(m) for m in vllm_messages) // 4
    available        = max_model_len - total_vision - input_tokens_est - 64
    safe_max         = max(min_output_tokens, min(max_output_tokens, available))

    log.debug(
        "Token budget: max_model_len=%d, input≈%d, vision=%d (%d×%d), safe_max=%d",
        max_model_len, input_tokens_est,
        total_vision, num_vision_atts, vision_per_img, safe_max,
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

    # Detect silent empty response — indicates vLLM crashed or OOMed mid-stream.
    # The stream connected successfully (no HTTPError) but yielded no data.
    # Without this check the caller silently stores an empty assistant message
    # and the user sees a blank bubble with no indication of what happened.
    if not raw_content and not raw_reasoning and not tool_calls_raw:
        log.error(
            "vLLM returned an empty response for conv=%s. "
            "The engine may have crashed (OOM or internal error). "
            "Check vllm_api.log for details.",
            conv_id,
        )
        # Also clear the system semaphore — if the engine died, we are not ready.
        _system_ready.clear()
        log.info("System semaphore: CLOSED (empty vLLM response — engine may be down)")
        raise RuntimeError(
            "The AI returned an empty response. This usually means the model ran "
            "out of GPU memory (too many large images at once) or the engine "
            "crashed mid-generation. Check vllm_api.log for the full error. "
            "The system will require a restart."
        )

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
        # Use the module-level constant — safe at any point in startup.
        # model_info is local to start_vllm() and cannot be referenced here.
        origin = self.headers.get("Origin", "")
        if origin in _CORS_ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin",  origin)
            self.send_header("Vary",                         "Origin")
        elif not origin:
            # Direct/non-browser request (no Origin header) — allow.
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
            alive = vllm_is_alive()
            ready = _system_ready.is_set()
            self._json(200, {
                "running":            running,
                "vllm_alive":         alive,
                "system_ready":       ready,
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
            # Path.name strips any directory components — prevents path traversal
            # (e.g. "../../etc/passwd" collapses to "passwd", which won't exist
            # under the conversation files/ directory).
            filename = Path(parts[1]).name if len(parts) > 1 else ""
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

            # ── Server-side system-ready semaphore (the real gate) ────────
            # This check is performed BEFORE reading the request body further.
            # The UI send button is greyed as UX only — this is the enforced
            # lock. No console trick, crafted packet, or race condition can
            # bypass it: _system_ready is a threading.Event set exclusively by
            # the deferred calibration thread after vLLM passes its liveness
            # check, and cleared immediately on any stop/restart.
            if not _system_ready.is_set():
                self._json(503, {
                    "error": (
                        "System not ready. The AI model is still loading or "
                        "restarting. Please wait until the status shows ✅ ready."
                    ),
                    "system_ready": False,
                })
                return

            user_text      = data.get("message", "").strip()
            selected_tools = data.get("selected_tools", [])   # [] = auto

            # ── Server-side prompt injection guard (defence in depth) ──────
            # This is the real security gate; GUI greying is supplementary UX.
            # vLLM has zero input filtering. Blocked messages never reach it.
            if _INPUT_GUARD and user_text:
                guard = _guard_check(user_text)
                if guard["blocked"]:
                    self._json(400, {"error": guard["reason"], "blocked": True})
                    return

            m_info    = load_model_info(current_model_id)
            is_vision = model_has_vision(m_info)

            # ── Multi-batch document analysis ─────────────────────────────
            # Attachments from multi-page documents arrive pre-split into
            # batches by doc_converter (batch_index / total_batches fields).
            # Batch 0 is sent with the user's prompt. Batches 1..N are sent
            # automatically as continuation turns, each with an injected note
            # explaining context, so the full document is always analysed.
            # Single-batch and non-document uploads are unaffected (fast path).
            all_attachments = data.get("attachments", [])
            total_batches   = max(
                (a.get("total_batches", 1) for a in all_attachments),
                default=1,
            )

            if not user_text and not all_attachments:
                self._json(400, {"error": "message required"})
                return

            def _attachments_for_batch(batch_idx: int) -> List[Dict]:
                if total_batches == 1:
                    return all_attachments
                return [a for a in all_attachments
                        if a.get("batch_index", 0) == batch_idx]

            attachments = _attachments_for_batch(0)

            # ── Multi-batch structured analysis protocol ──────────────────
            # Only applied when hardware cannot one-shot all pages at once
            # (total_batches > 1). When total_batches == 1 the hardware fits
            # all images in a single call — no protocol, model decides freely.
            #
            # Protocol when total_batches > 1:
            #   Reply 0 (batch 0 call): global document context
            #   Reply 1 (continuation 1): detailed analysis of pages 1–N
            #   Reply 2 (continuation 2): detailed analysis of pages N+1–2N
            #   …
            #   Reply K (final): detailed analysis of last pages + conclusion
            #
            # Each reply is independent: the model reads only its assigned
            # pages and generates a fresh response with its own full output
            # token budget. No budget division — each reply gets safe_max
            # computed from its own input + vision tokens.
            # Initialised here so the reference in self._json() below is always
            # defined regardless of branch, even though it is only used when
            # total_batches > 1 (the ternary guards the actual value sent).
            pdf_name_pre = None
            if total_batches > 1:
                pdf_name_pre  = next(
                    (a.get("pdf_source") for a in all_attachments if a.get("pdf_source")),
                    "document",
                )
                doc_total_pages = all_attachments[-1].get("pdf_total_pages", "?")
                batch_0_atts    = _attachments_for_batch(0)
                batch_last_page = batch_0_atts[-1].get("pdf_page", "?") if batch_0_atts else "?"
                total_replies   = total_batches + 1   # context reply + one per batch

                log.info(
                    "Multi-batch protocol: %s pages → %d replies "
                    "(context + %d batch(es))",
                    doc_total_pages, total_replies, total_batches,
                )

                preamble = _build_multibatch_preamble(
                    reply_num=1,
                    batch_first_page=1,
                    batch_last_page=batch_last_page,
                    pdf_name=pdf_name_pre,
                    doc_total_pages=doc_total_pages,
                )
                # User's request FIRST, preamble LAST — the preamble's final
                # sentence ("...page 1 to page N of the document only.") is
                # the freshest instruction the model reads, immediately
                # before the page images. The user's request is established
                # earlier in the same message and referenced by the preamble
                # as "the tasks requested by the user" — never repeated.
                content_text = user_text + "\n\n" + preamble
            else:
                content_text = user_text

            # Build multimodal content for batch 0 — uses preamble if multi-batch.
            user_content = _build_user_content(
                content_text, attachments, conv_id, is_vision
            )

            tools = _resolve_tools(selected_tools, m_info)

            # Persist the FULL content_text (user request + preamble) so the
            # multi-batch protocol is auditable for reply 1 exactly as it is
            # for replies 2..N — consistent history reconstruction on restart.
            user_ts = append_message(
                conv_id, "user", content_text,
                attachments=[a.get("name") for a in all_attachments] if all_attachments else None,
            )

            # ── Batch 0: respond immediately ──────────────────────────────
            try:
                final_content, reasoning, tool_calls = _call_vllm_streaming(
                    conv_id, user_content, current_model_path,
                    m_info, thinking_enabled, tools,
                )
            except Exception as e:
                log.error("Chat error: %s", e)
                self._json(500, {"error": str(e)})
                return

            ai_ts = append_message(conv_id, "assistant", final_content, reasoning=reasoning)

            next_batch = 1 if total_batches > 1 else None

            self._json(200, {
                "content":           final_content,
                "reasoning_content": reasoning,
                "tool_calls":        tool_calls,
                "thinking_enabled":  thinking_enabled,
                "user_timestamp":    user_ts,
                "ai_timestamp":      ai_ts,
                "next_batch_index":  next_batch,
                "total_batches":     total_batches if total_batches > 1 else None,
                "pdf_name":          pdf_name_pre if total_batches > 1 else None,
            })

        elif p.startswith("/conversations/") and p.endswith("/chat_continue"):
            # ── Client-driven multi-batch continuation ────────────────────
            # Called automatically by the UI after each batch response when
            # next_batch_index is set. Each call processes one batch and
            # returns immediately, giving every batch its own full output
            # token budget and allowing progressive rendering in the UI.
            conv_id = p[len("/conversations/"):-len("/chat_continue")]
            data    = self._read_body()

            if not _system_ready.is_set():
                self._json(503, {"error": "System not ready.", "system_ready": False})
                return

            batch_idx      = int(data.get("batch_index", 1))
            total_batches  = int(data.get("total_batches", 1))
            all_atts       = data.get("attachments", [])
            pdf_name       = data.get("pdf_name", "document")
            selected_tools = data.get("selected_tools", [])
            is_summary     = bool(data.get("summary", False))
            total_replies  = total_batches + 1

            if is_summary:
                # Executive summary — no images, synthesise from history text.
                total_pages = all_atts[0].get("pdf_total_pages", "?") if all_atts else "?"
                first_page  = "N/A"
                last_page   = "N/A"

                # ── Solution 1: inject prior batch analyses verbatim ────────
                # The model previously complained it lacked "data" for pages
                # beyond the first batch when asked to synthesise from
                # "conversation history" in the abstract. Here we extract the
                # last `total_batches` assistant replies (the page-batch
                # analyses) and quote them directly in the prompt, removing
                # any ambiguity about what data is available.
                hist = load_history(conv_id)
                assistant_msgs = [m for m in hist if m.get("role") == "assistant"]
                prior_analyses = assistant_msgs[-total_batches:] if total_batches > 0 else []

                cont_note = _build_executive_summary_note(
                    total_replies, pdf_name, total_pages, prior_analyses,
                )

                m_info       = load_model_info(current_model_id)
                tools        = _resolve_tools(selected_tools, m_info)
                cont_content = _build_user_content(cont_note, [], conv_id, False)

                log.info(
                    "Multi-batch: executive summary for '%s' (%s pages), "
                    "%d prior analyses injected (%d chars)",
                    pdf_name, total_pages, len(prior_analyses), len(cont_note),
                )

                append_message(conv_id, "user", cont_note)

                try:
                    cont_str, cont_reasoning, tool_calls = _call_vllm_streaming(
                        conv_id, cont_content, current_model_path,
                        m_info, thinking_enabled, tools,
                    )
                except Exception as e:
                    log.error("Executive summary error: %s", e)
                    self._json(500, {"error": str(e)})
                    return

                ai_ts = append_message(conv_id, "assistant", cont_str,
                                       reasoning=cont_reasoning)

                self._json(200, {
                    "content":           cont_str,
                    "reasoning_content": cont_reasoning,
                    "tool_calls":        tool_calls,
                    "batch_index":       batch_idx,
                    "first_page":        first_page,
                    "last_page":         last_page,
                    "total_pages":       total_pages,
                    "next_batch_index":  None,
                    "summary":           True,
                    "ai_timestamp":      ai_ts,
                })
                return

            # ── Image batch continuation ───────────────────────────────────
            batch_atts = [a for a in all_atts
                          if a.get("batch_index", 0) == batch_idx]
            if not batch_atts:
                self._json(400, {"error": f"No attachments for batch {batch_idx}"})
                return

            batch_first_page = batch_atts[0].get("pdf_page", "?")
            batch_last_page  = batch_atts[-1].get("pdf_page", "?")
            total_pages      = batch_atts[0].get("pdf_total_pages", "?")
            reply_num        = batch_idx + 1   # batch 1 → reply 2, etc.
            is_last_img      = (batch_idx == total_batches - 1)

            cont_note = _build_multibatch_preamble(
                reply_num=reply_num,
                batch_first_page=batch_first_page,
                batch_last_page=batch_last_page,
                pdf_name=pdf_name,
                doc_total_pages=total_pages,
            )

            m_info    = load_model_info(current_model_id)
            is_vision = model_has_vision(m_info)
            tools     = _resolve_tools(selected_tools, m_info)

            cont_content = _build_user_content(
                cont_note, batch_atts, conv_id, is_vision
            )

            log.info(
                "Multi-batch: continuation batch %d/%d (pages %s–%s) for '%s'",
                batch_idx + 1, total_batches, batch_first_page, batch_last_page, pdf_name,
            )

            append_message(conv_id, "user", cont_note)

            try:
                cont_str, cont_reasoning, tool_calls = _call_vllm_streaming(
                    conv_id, cont_content, current_model_path,
                    m_info, thinking_enabled, tools,
                )
            except Exception as e:
                log.error("Continuation batch %d error: %s", batch_idx, e)
                self._json(500, {"error": str(e)})
                return

            ai_ts      = append_message(conv_id, "assistant", cont_str,
                                        reasoning=cont_reasoning)
            # Signal next image batch, or null if all image batches done
            # (client will then fire the summary call separately)
            next_batch = (batch_idx + 1) if not is_last_img else None

            self._json(200, {
                "content":           cont_str,
                "reasoning_content": cont_reasoning,
                "tool_calls":        tool_calls,
                "batch_index":       batch_idx,
                "first_page":        batch_first_page,
                "last_page":         batch_last_page,
                "total_pages":       total_pages,
                "next_batch_index":  next_batch,
                "summary_pending":   is_last_img,   # client fires summary after this
                "total_batches":     total_batches,
                "pdf_name":          pdf_name,
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
    server = ThreadingHTTPServer((constants.HOST_BIND, constants.PORT_LAUNCHER), MicroserviceHandler)
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
    log.info("Build marker: multi-batch-protocol-v3-numerals-only (2026-06-12)")
    log.info("Using Python executable: %s", sys.executable)

    threading.Thread(target=_cleanup_old_conversations, daemon=True,
                     name="fifo-cleanup").start()
    threading.Thread(target=_heartbeat, daemon=True,
                     name="heartbeat").start()
    threading.Thread(target=_vllm_watchdog, daemon=True,
                     name="vllm-watchdog").start()

    start_vllm()

    threading.Thread(target=run_http_server, daemon=True,
                     name="http-server").start()

    await websockets.serve(ws_handler, constants.HOST_BIND, constants.PORT_WEBSOCKET)
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
