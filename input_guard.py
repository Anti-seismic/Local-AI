#!/usr/bin/env python3
"""
input_guard.py — Server-side user input validation and prompt injection blocking.

Responsibilities (one file, one purpose):
  - Prompt injection detection and blocking (server-side enforcement)
  - Called by vllm_launcher.py BEFORE any content reaches the model
  - Client-side UI greying is supplementary UX only; this is the real gate

Defence-in-depth rationale:
  vLLM is a pure inference engine with zero input filtering.
  The model's safety training is not a security control — it can be bypassed.
  All blocking is done here, server-side. The model is never called for
  blocked inputs.

Blocked inputs are logged (category label only, NOT full text, to prevent
log poisoning). The client receives a generic error message; the specific
pattern matched is never revealed (prevents targeted bypass engineering).
"""

import logging
import logging.handlers
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Logger  →  shared security audit log
# ---------------------------------------------------------------------------
# Dual-process note: file_security.py also opens a RotatingFileHandler on
# security_audit.log. This is intentional — input_guard runs inside the
# vllm_launcher process and file_security runs inside upload_server.
# They are separate OS processes; each must configure its own handler.
# Linux kernel serialises concurrent append-mode writes at the VFS layer,
# so both handlers write safely to the same file without interleaving.
# If this module is ever imported into the upload_server process, the handler
# would be duplicated within that process — audit log lines would appear twice.
# That is not the current architecture and is not expected to change.
# ---------------------------------------------------------------------------
_BASE    = Path(__file__).resolve().parent
_LOG_DIR = _BASE / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("input_guard")
log.setLevel(logging.DEBUG)
log.propagate = False
_fh = logging.handlers.RotatingFileHandler(
    _LOG_DIR / "security_audit.log",
    maxBytes=100 * 1024 * 1024, backupCount=10, encoding="utf-8"
)
_fh.setFormatter(logging.Formatter(
    "%(asctime)s.%(msecs)03d [%(levelname)-7s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S"
))
log.addHandler(_fh)

# ---------------------------------------------------------------------------
# Pattern definitions  (raw string, human label)
# Compiled once at import time for performance.
# ---------------------------------------------------------------------------
_RAW: list[tuple[str, str]] = [
    # Instruction override
    (r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|context|prompt|rules?)",
     "instruction-override:ignore-previous"),
    (r"disregard\s+(the\s+)?(above|previous|following|all)",
     "instruction-override:disregard"),
    (r"forget\s+(all\s+)?(previous\s+)?(instructions?|context|rules?|guidelines?)",
     "instruction-override:forget"),
    (r"override\s+(your\s+)?(instructions?|rules?|guidelines?|system\s*prompt)",
     "instruction-override:explicit"),
    (r"do\s+not\s+follow\s+(your\s+)?(instructions?|guidelines?|rules?)",
     "instruction-override:do-not-follow"),
    # Role / persona injection
    (r"you\s+are\s+now\s+\w+\s+(without|with\s+no)\s+",
     "role-injection:unrestricted-persona"),
    (r"pretend\s+(you\s+are|to\s+be)\s+(a|an)?\s*(unrestricted|uncensored|evil|dangerous)",
     "role-injection:malicious-persona"),
    (r"act\s+as\s+(if\s+)?(you\s+(have|are|were)\s+)?(no\s+)?(restrictions?|limits?|filters?|rules?|guidelines?)",
     "role-injection:no-restrictions"),
    (r"roleplay\s+as\s+(a|an)?\s*(unrestricted|uncensored|evil|hacker|attacker)",
     "role-injection:adversarial-roleplay"),
    # LLM token injection
    (r"<\|system\|>",       "token-injection:<|system|>"),
    (r"<\|assistant\|>",    "token-injection:<|assistant|>"),
    (r"<\|im_start\|>",     "token-injection:<|im_start|>"),
    (r"<\|im_end\|>",       "token-injection:<|im_end|>"),
    (r"\[INST\]",           "token-injection:[INST]"),
    (r"<<SYS>>",            "token-injection:<<SYS>>"),
    (r"</?s>",              "token-injection:</s>"),
    # System prompt extraction
    (r"(print|show|reveal|output|display|repeat|write\s+out)\s+(your\s+)?"
     r"(system\s+prompt|instructions?|rules?|guidelines?|initial\s+prompt)",
     "extraction:system-prompt"),
    (r"what\s+(are\s+)?(your\s+)?(exact\s+)?(instructions?|system\s+prompt|rules?)\s*\?",
     "extraction:instructions-query"),
    # Jailbreak keywords
    (r"\bDAN\b",                           "jailbreak:DAN"),
    (r"\bdeveloper\s+mode\b",              "jailbreak:developer-mode"),
    (r"\bgod\s+mode\b",                    "jailbreak:god-mode"),
    (r"\bjailbreak\b",                     "jailbreak:keyword"),
    (r"\b(uncensored|unrestricted)\s+mode\b", "jailbreak:mode-keyword"),
    (r"enable\s+(developer|god|unrestricted|uncensored|jailbreak)\s+mode",
     "jailbreak:mode-activation"),
    # Context manipulation
    (r"from\s+now\s+on\s+(you\s+(are|will|should|must))",
     "context-manipulation:from-now-on"),
    (r"for\s+(this|the\s+rest\s+of\s+(this|the))\s+conversation\s+(you\s+(are|pretend|act))",
     "context-manipulation:conversation-scope"),
    (r"in\s+this\s+hypothetical\s+.{0,50}(no\s+(rules?|laws?|restrictions?|guidelines?))",
     "context-manipulation:hypothetical-no-rules"),
]

_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(raw, re.IGNORECASE | re.DOTALL), label)
    for raw, label in _RAW
]

_USER_MSG = (
    "\u26a0\ufe0f Message blocked: it contains patterns associated with "
    "prompt injection attacks. Your message was not sent to the model."
)
_DOC_MSG  = (
    "\u26a0\ufe0f Document '{name}' contains embedded prompt injection patterns "
    "and was not forwarded to the model."
)


def check_message(text: str) -> dict:
    """
    Scan a user message for prompt injection.

    Returns {"blocked": bool, "reason": str, "label": str}.
    `reason` is safe to show to the user.
    `label`  is for security logging only — never sent to the client.
    """
    if not text or not text.strip():
        return {"blocked": False, "reason": "", "label": ""}
    for pattern, label in _PATTERNS:
        if pattern.search(text):
            log.warning(
                "[Guard: BLOCKED] category=%s  preview=%s",
                label, repr(text[:60])
            )
            return {"blocked": True, "reason": _USER_MSG, "label": label}
    return {"blocked": False, "reason": "", "label": ""}


def check_file_content(text: str, source: str) -> dict:
    """
    Scan text extracted from an uploaded document for embedded injection.
    `source` is a filename used only in logging.
    """
    if not text or not text.strip():
        return {"blocked": False, "reason": "", "label": ""}
    for pattern, label in _PATTERNS:
        if pattern.search(text):
            log.warning(
                "[Guard: DOC-INJECTION] source=%s  category=%s", source, label
            )
            return {
                "blocked": True,
                "reason":  _DOC_MSG.format(name=source),
                "label":   label,
            }
    return {"blocked": False, "reason": "", "label": ""}
