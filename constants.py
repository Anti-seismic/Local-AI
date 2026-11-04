#!/usr/bin/env python3
"""
constants.py — Shared port and address constants for all LocalAI microservices.

All Python microservices import from here. A single edit here changes every
service simultaneously — no per-file hunt for hardcoded values.

Future datacenter note
──────────────────────
This file is the correct solution for a single-machine deployment where all
microservices are co-located. For multi-node / datacenter deployment (H100,
H200 clusters), these should become environment variables or a distributed
config source (e.g. etcd, Consul) so each service can be addressed
independently across nodes. At that point, this file should be replaced by
a thin loader:
    import os
    PORT_LAUNCHER = int(os.environ.get("LOCALAI_PORT_LAUNCHER", 8770))
    ...
until that refactor is done, keep all port constants here.
"""

# ── Bind host ────────────────────────────────────────────────────────────────
# All services bind to loopback only. Change to "0.0.0.0" for network access,
# but add firewall rules before doing so.
HOST_BIND = "127.0.0.1"

# ── Service ports ────────────────────────────────────────────────────────────
PORT_VLLM_API   = 8000   # vLLM OpenAI-compatible API (managed by vllm_launcher)
PORT_LAUNCHER   = 8770   # vllm_launcher.py HTTP microservice
PORT_WEBSOCKET  = 8765   # vllm_launcher.py WebSocket (browser live updates)
PORT_UPLOAD     = 8771   # upload_server.py
PORT_STATIC     = 8080   # python -m http.server (serves chat.html, app.js, etc.)

# ── Derived base URLs ─────────────────────────────────────────────────────────
URL_LAUNCHER    = f"http://{HOST_BIND}:{PORT_LAUNCHER}"
URL_UPLOAD      = f"http://{HOST_BIND}:{PORT_UPLOAD}"
URL_STATIC      = f"http://{HOST_BIND}:{PORT_STATIC}"
URL_VLLM_API    = f"http://{HOST_BIND}:{PORT_VLLM_API}"
URL_WEBSOCKET   = f"ws://{HOST_BIND}:{PORT_WEBSOCKET}"
