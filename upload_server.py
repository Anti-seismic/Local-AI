#!/usr/bin/env python3
"""
upload_server.py — Lean upload orchestrator.

Responsibilities (one file, one purpose):
  - Receive uploaded files, enforce size limits, persist to disk
  - Delegate security scanning/sanitization to file_security.py
  - Delegate format conversion to doc_converter.py
  - Serve the /upload, /unlock, /pdf_info, and /status endpoints

Password-protected PDFs:
  /upload detects encryption and returns {password_required, saved_name}.
  The file is already on disk. /unlock reads it back, decrypts with the
  user-supplied password (never logged), then runs the normal pipeline.
"""

import json
import logging
import logging.handlers
import mimetypes
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from flask import Flask, jsonify, request

import constants
import doc_converter
import file_security
from file_security import sec   # security audit logger

# ---------------------------------------------------------------------------
# Identity and paths
# ---------------------------------------------------------------------------
_MACHINE = socket.gethostname()
_USER    = os.getenv("USER", "unknown")
IDENTITY = f"{_MACHINE}_{_USER}"

BASE_DIR      = Path(__file__).resolve().parent
USER_DATA_DIR = BASE_DIR / "data" / IDENTITY
LOG_DIR       = BASE_DIR / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging (general operational log)
# ---------------------------------------------------------------------------
log = logging.getLogger("upload_server")
log.setLevel(logging.DEBUG)
log.propagate = False
_fh = logging.handlers.RotatingFileHandler(
    LOG_DIR / "upload_server.log",
    maxBytes=50 * 1024 * 1024, backupCount=1, encoding="utf-8"
)
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
log.addHandler(_fh)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024   # 500 MB hard ceiling

# ---------------------------------------------------------------------------
# CORS
# Wildcard ("*") is intentionally excluded. Only the local static HTTP server
# that serves chat.html is a legitimate origin. Any other origin — including
# a malicious site open in another browser tab — is blocked from reading
# responses. The echo-back pattern (returning the exact requesting origin
# rather than "*") is required for future credentials-bearing requests and
# is more precise than a wildcard.
# Add entries for bastion/LAN hostnames if the UI is accessed remotely.
# ---------------------------------------------------------------------------
_ALLOWED_ORIGINS = frozenset([
    constants.URL_STATIC,
    f"http://localhost:{constants.PORT_STATIC}",
])


@app.after_request
def _cors(resp):
    origin = request.headers.get("Origin", "")
    if origin in _ALLOWED_ORIGINS:
        resp.headers["Access-Control-Allow-Origin"]  = origin   # exact, not *
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Conv-Id"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Vary"]                         = "Origin"
    return resp


@app.before_request
def _handle_preflight():
    """Handle CORS preflight OPTIONS requests before Flask routing."""
    if request.method == "OPTIONS":
        resp   = app.make_response("")
        origin = request.headers.get("Origin", "")
        if origin in _ALLOWED_ORIGINS:
            resp.headers["Access-Control-Allow-Origin"]  = origin
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Conv-Id"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            resp.headers["Vary"]                         = "Origin"
        return resp

# ---------------------------------------------------------------------------
# Upload limits (bytes) per extension
# ---------------------------------------------------------------------------
DEFAULT_SIZE_LIMIT = 50  * 1024 * 1024   # 50 MB
FILE_SIZE_LIMITS   = {
    ".pdf":  200 * 1024 * 1024,
    ".docx":  50 * 1024 * 1024,
    ".pptx": 100 * 1024 * 1024,
    ".xlsx":  50 * 1024 * 1024,
    ".doc":   50 * 1024 * 1024,
    ".ppt":  100 * 1024 * 1024,
    ".xls":   50 * 1024 * 1024,
    ".png":   30 * 1024 * 1024,
    ".jpg":   30 * 1024 * 1024,
    ".jpeg":  30 * 1024 * 1024,
}

IMAGE_EXTS  = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}
OFFICE_EXTS = {".docx", ".pptx", ".xlsx", ".dotx", ".potx", ".xltx",
               ".doc",  ".ppt",  ".xls"}

# ---------------------------------------------------------------------------
# Vision encoder DPI safety cap
#
# The Qwen3-VL vision encoder has a fixed token budget (default 16,384).
# Each page at DPI d renders at approximately:
#   tokens_per_page ≈ (d × W_in / 28) × (d × H_in / 28) / merge_factor
# where merge_factor=4 (2×2 patch merge), W_in≈8.27", H_in≈11.69" (A4/letter).
# Simplified: tokens_per_page ≈ d² × 0.1103
#
# Allocating the full encoder budget equally across all pages:
#   max_tokens_per_page = ENCODER_BUDGET / num_pages
#   d_max = sqrt(max_tokens_per_page / 0.1103)
#
# This cap is a hard server-side limit regardless of what DPI the user chose
# in the UI — the vision encoder will OOM if we exceed it, killing the engine.
# ---------------------------------------------------------------------------
# Default fallback used only when the model JSON cannot be read (first boot
# before calibration, or misconfigured path).  The launcher writes the real
# calibrated value to vision_encoder_budget in the model JSON after start.
_VISION_ENCODER_BUDGET_FALLBACK = 16_384
_TOKENS_PER_DPI_SQUARED = 0.1103  # empirical constant for A4/letter pages

# Path to the models/info/ directory — mirrors vllm_launcher.py's layout.
# Upload server has no concept of "current model"; it reads the active model
# JSON via the same path convention used by the launcher.
_MODELS_INFO_DIR = BASE_DIR / "models" / "info"


def _get_encoder_budget(model_id: str | None = None) -> int:
    """
    Read the calibrated vision_encoder_budget from the active model JSON.
    Falls back to the compile-time default when the JSON is unavailable or
    when calibration has not yet run (first boot of a new model/hardware).
    Logs a warning to upload_server.log when the fallback is used — this
    signals that auto-probing/calibration has not completed successfully.
    """
    if model_id is None:
        # Discover the most recently modified model JSON as the active model.
        # The launcher always writes to the JSON of the model it is running,
        # so the most recently modified file is the active one.
        try:
            jsons = sorted(_MODELS_INFO_DIR.glob("*.json"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
            if not jsons:
                raise FileNotFoundError("No model JSON files found")
            info_path = jsons[0]
        except Exception as e:
            log.warning(
                "VISION_ENCODER_BUDGET: could not locate model JSON (%s). "
                "Using fallback %d tokens. "
                "This means auto-probing/calibration has not completed — "
                "run the system twice on new hardware or after a model change.",
                e, _VISION_ENCODER_BUDGET_FALLBACK,
            )
            return _VISION_ENCODER_BUDGET_FALLBACK
    else:
        info_path = _MODELS_INFO_DIR / f"{model_id}.json"

    try:
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
        budget = info.get("vision_encoder_budget")
        if budget is None:
            log.warning(
                "VISION_ENCODER_BUDGET: 'vision_encoder_budget' not set in %s. "
                "Using fallback %d tokens. "
                "This means auto-probing/calibration has not completed — "
                "run the system twice on new hardware or after a model change.",
                info_path.name, _VISION_ENCODER_BUDGET_FALLBACK,
            )
            return _VISION_ENCODER_BUDGET_FALLBACK
        return int(budget)
    except Exception as e:
        log.warning(
            "VISION_ENCODER_BUDGET: could not read %s (%s). "
            "Using fallback %d tokens.",
            info_path, e, _VISION_ENCODER_BUDGET_FALLBACK,
        )
        return _VISION_ENCODER_BUDGET_FALLBACK


def _safe_dpi(requested_dpi: int, num_pages: int,
              encoder_budget: int | None = None) -> int:
    """
    Return the highest DPI that keeps per-page vision tokens within budget
    for a single batch of pages.

    The denominator is min(num_pages, PDF_MAX_PAGES) — the actual number of
    pages sent in one vLLM call — NOT the total page count of the document.
    Multi-page documents are split into batches of PDF_MAX_PAGES; each batch
    is processed independently, so the DPI ceiling is determined by the batch
    size rather than the document size. Using total pages would underestimate
    the safe DPI for every batch after the first.

    Formula: d_max = sqrt(encoder_budget / (batch_size × _TOKENS_PER_DPI_SQUARED))
    Floor: 72 DPI (minimum legible for any document).
    """
    if encoder_budget is None:
        encoder_budget = _get_encoder_budget()
    if num_pages <= 0:
        return requested_dpi
    batch_size          = min(num_pages, doc_converter.PDF_MAX_PAGES)
    max_tokens_per_page = encoder_budget / batch_size
    d_max = int((max_tokens_per_page / _TOKENS_PER_DPI_SQUARED) ** 0.5)
    d_max = max(72, d_max)
    return min(requested_dpi, d_max)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _conv_files_dir(conv_id: str) -> Path:
    d = USER_DATA_DIR / conv_id / "files"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _serve_url(conv_id: str, filename: str) -> str:
    return f"{constants.URL_LAUNCHER}/conversations/{conv_id}/files/{filename}"


def _safe_filename(original: str) -> str:
    """Timestamp-prefixed safe filename; strips path components."""
    name = Path(original).name
    return f"{int(time.time() * 1000)}_{name}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/status", methods=["GET"])
def status():
    return jsonify({"status": "ok", "identity": IDENTITY})


@app.route("/upload", methods=["POST"])
def upload():
    """
    Main upload endpoint.

    Query params:
      conv_id  — conversation ID (required)
      pdf_dpi  — DPI for PDF page rendering (optional, default 150)

    Returns JSON:
      { "files": [ <attachment_dict>, ... ], "total_size": <int> }

    Special returns:
      { "password_required": true, "saved_name": "<name>", "filename": "<original>" }
        — encrypted PDF detected; call /unlock with the saved_name and password.
    """
    conv_id = (
        request.args.get("conv_id")
        or request.form.get("conv_id")
        or request.headers.get("X-Conv-Id", "")
    ).strip()
    if not conv_id:
        return jsonify({"error": "conv_id is required"}), 400

    try:
        pdf_dpi = int(request.args.get("pdf_dpi") or request.form.get("pdf_dpi") or 150)
        pdf_dpi = max(72, min(1200, pdf_dpi))
    except (ValueError, TypeError):
        pdf_dpi = 150

    uploaded   = request.files.getlist("files")
    if not uploaded:
        return jsonify({"error": "No files provided"}), 400

    files_dir  = _conv_files_dir(conv_id)
    processed  = []
    total_size = 0

    for fobj in uploaded:
        if not fobj or not fobj.filename:
            continue

        original_name = Path(fobj.filename).name
        ext           = Path(original_name).suffix.lower()
        data          = fobj.read()
        size          = len(data)

        # Size limit
        limit = FILE_SIZE_LIMITS.get(ext, DEFAULT_SIZE_LIMIT)
        if size > limit:
            return jsonify({
                "error": f"'{original_name}' exceeds the {limit // 1024 // 1024} MB limit"
            }), 413
        total_size += size

        # Magic-byte validation
        magic = file_security.validate_magic(data, ext)
        if not magic["valid"]:
            log.warning(
                "Magic mismatch for '%s': claimed %s, detected %s",
                original_name, magic["claimed"], magic["detected"]
            )
            sec.warning(
                "[Security: MAGIC-MISMATCH] '%s': claimed=%s detected=%s",
                original_name, magic["claimed"], magic["detected"]
            )

        # Always save original to disk first (audit trail + /unlock fallback)
        safe_name = _safe_filename(original_name)
        dest      = files_dir / safe_name
        try:
            dest.write_bytes(data)
        except OSError as e:
            log.error("Cannot save '%s': %s", safe_name, e)
            return jsonify({"error": f"Storage error: {e}"}), 500

        # ── PDF ──────────────────────────────────────────────────────────
        if ext == ".pdf":
            if file_security.detect_pdf_encryption(data):
                sec.info(
                    "[Security: ENCRYPTED] '%s' is password-protected — "
                    "awaiting /unlock call (saved as %s)", original_name, safe_name
                )
                return jsonify({
                    "password_required": True,
                    "saved_name":        safe_name,
                    "filename":          original_name,
                })

            sec_result = file_security.full_pipeline(data, ext, original_name)
            if sec_result["blocked"]:
                processed.append({
                    "id": safe_name, "name": original_name,
                    "type": "security_blocked",
                    "block_reason": sec_result["block_reason"],
                    "threats": sec_result["threats"],
                    "size": size,
                })
                continue

            work_data = sec_result["data"]

            # Pre-read page count to apply the DPI safety cap before conversion.
            # This avoids an OOM crash in the vision encoder when the requested
            # DPI would produce per-page tensors too large for Flash Attention.
            pdf_info_pre = doc_converter.get_pdf_info(work_data)
            num_pages    = pdf_info_pre.get("page_count", 1) or 1
            safe_dpi     = _safe_dpi(pdf_dpi, num_pages)
            if safe_dpi < pdf_dpi:
                log.warning(
                    "'%s': requested DPI %d capped to %d "
                    "(%d pages × encoder budget %d tokens).",
                    original_name, pdf_dpi, safe_dpi, num_pages, _get_encoder_budget()
                )

            page_atts = doc_converter.pdf_to_images(
                work_data, safe_dpi, original_name, conv_id, files_dir
            )
            if page_atts:
                if sec_result["sanitized"]:
                    for a in page_atts:
                        a["sanitized"]       = True
                        a["security_warning"] = sec_result.get("warning")
                processed.extend(page_atts)
                log.info("Uploaded PDF '%s' → %d page image(s)", original_name, len(page_atts))
            else:
                # Conversion failed — attach as raw document
                processed.append({
                    "id": safe_name, "name": original_name,
                    "url": _serve_url(conv_id, safe_name),
                    "type": "document", "mime": "application/pdf", "size": size,
                })
            continue

        # ── Office documents ──────────────────────────────────────────────
        if ext in OFFICE_EXTS:
            sec_result = file_security.full_pipeline(data, ext, original_name)
            if sec_result["blocked"]:
                processed.append({
                    "id": safe_name, "name": original_name,
                    "type": "security_blocked",
                    "block_reason": sec_result["block_reason"],
                    "threats": sec_result["threats"],
                    "size": size,
                })
                continue

            work_data = sec_result["data"]

            # Office docs go PDF→images via LibreOffice; apply the same DPI cap.
            # Page count is not known until after conversion, so we conservatively
            # assume doc_converter.PDF_MAX_PAGES (20) for the budget calculation —
            # this is the worst case and ensures we never OOM even on the longest
            # supported document.
            safe_dpi_office = _safe_dpi(pdf_dpi, doc_converter.PDF_MAX_PAGES)
            if safe_dpi_office < pdf_dpi:
                log.warning(
                    "'%s': office DPI %d capped to %d (max %d pages assumed).",
                    original_name, pdf_dpi, safe_dpi_office, doc_converter.PDF_MAX_PAGES
                )

            page_atts = doc_converter.office_to_images(
                work_data, ext, safe_dpi_office, original_name, conv_id, files_dir
            )
            if page_atts:
                if sec_result["sanitized"]:
                    for a in page_atts:
                        a["sanitized"]        = True
                        a["security_warning"] = sec_result.get("warning")
                processed.extend(page_atts)
                log.info("Uploaded '%s' → %d page image(s)", original_name, len(page_atts))
            else:
                mime = mimetypes.guess_type(original_name)[0] or "application/octet-stream"
                processed.append({
                    "id": safe_name, "name": original_name,
                    "url": _serve_url(conv_id, safe_name),
                    "type": "document", "mime": mime, "size": size,
                })
            continue

        # ── Images ───────────────────────────────────────────────────────
        if ext in IMAGE_EXTS:
            norm_data, mime, ok = doc_converter.normalize_image(data, ext)
            if ok and norm_data != data:
                # Save normalised copy
                norm_name = _safe_filename(Path(original_name).stem + ".jpg")
                norm_dest = files_dir / norm_name
                norm_dest.write_bytes(norm_data)
                processed.append({
                    "id": norm_name, "name": original_name,
                    "url": _serve_url(conv_id, norm_name),
                    "type": "image", "mime": "image/jpeg",
                    "size": len(norm_data),
                })
            else:
                mime = mimetypes.guess_type(original_name)[0] or "image/jpeg"
                processed.append({
                    "id": safe_name, "name": original_name,
                    "url": _serve_url(conv_id, safe_name),
                    "type": "image", "mime": mime, "size": size,
                })
            log.info("Uploaded image '%s' (%d bytes)", original_name, size)
            continue

        # ── Everything else (text, code, …) ──────────────────────────────
        mime = mimetypes.guess_type(original_name)[0] or "application/octet-stream"
        processed.append({
            "id": safe_name, "name": original_name,
            "url": _serve_url(conv_id, safe_name),
            "type": "document", "mime": mime, "size": size,
        })
        log.info("Uploaded '%s' (%d bytes, type=document)", original_name, size)

    return jsonify({"files": processed, "total_size": total_size})


@app.route("/unlock", methods=["POST"])
def unlock():
    """
    Decrypt a password-protected document that was already saved to disk.

    Form fields:
      conv_id    — conversation ID
      saved_name — filename returned by /upload in the password_required response
      password   — document password  ← NEVER logged, NEVER stored

    Returns the same {files, total_size} structure as /upload.
    """
    conv_id    = (request.form.get("conv_id") or "").strip()
    saved_name = Path(request.form.get("saved_name") or "").name   # strip any path
    password   = request.form.get("password", "")                  # opaque — never log

    if not conv_id or not saved_name:
        return jsonify({"error": "conv_id and saved_name are required"}), 400

    file_path = _conv_files_dir(conv_id) / saved_name
    if not file_path.is_file():
        return jsonify({"error": "File not found on server"}), 404

    data          = file_path.read_bytes()
    original_name = "_".join(saved_name.split("_")[1:])   # strip timestamp prefix
    ext           = Path(original_name).suffix.lower()

    # Read pdf_dpi from query or form (default 150)
    try:
        pdf_dpi = int(request.args.get("pdf_dpi") or request.form.get("pdf_dpi") or 150)
        pdf_dpi = max(72, min(1200, pdf_dpi))
    except (ValueError, TypeError):
        pdf_dpi = 150

    # Decrypt — password never appears in any log
    decrypted, ok, err = file_security.decrypt_pdf(data, password)
    if not ok:
        sec.error(
            "[Security: DECRYPT-FAIL] '%s': %s (password NOT logged)",
            original_name, err
        )
        return jsonify({"error": "Incorrect password or unsupported encryption",
                        "password_invalid": True}), 400

    sec.info(
        "[Security: DECRYPTED] '%s' unlocked successfully (password NOT logged)",
        original_name
    )

    # Security scan + conversion on decrypted data
    sec_result = file_security.full_pipeline(decrypted, ext, original_name)
    if sec_result["blocked"]:
        return jsonify({
            "files": [{
                "id": saved_name, "name": original_name,
                "type": "security_blocked",
                "block_reason": sec_result["block_reason"],
                "threats": sec_result["threats"],
            }],
            "total_size": len(data),
        })

    files_dir  = _conv_files_dir(conv_id)
    work_data  = sec_result["data"]

    # Apply DPI cap: get page count from the decrypted data before converting.
    pdf_info_pre = doc_converter.get_pdf_info(work_data)
    num_pages    = pdf_info_pre.get("page_count", 1) or 1
    safe_dpi     = _safe_dpi(pdf_dpi, num_pages)
    if safe_dpi < pdf_dpi:
        log.warning(
            "'%s' (unlocked): DPI %d capped to %d (%d pages).",
            original_name, pdf_dpi, safe_dpi, num_pages
        )

    page_atts  = doc_converter.pdf_to_images(
        work_data, safe_dpi, original_name, conv_id, files_dir
    )

    if sec_result["sanitized"]:
        for a in page_atts:
            a["sanitized"]        = True
            a["security_warning"] = sec_result.get("warning")

    return jsonify({"files": page_atts, "total_size": len(data)})


@app.route("/pdf_info", methods=["POST"])
def pdf_info():
    """
    Lightweight PDF analysis before upload.
    Returns: page_count, encrypted, recommended_dpi, content_hint, security.
    """
    fobj = request.files.get("file")
    if not fobj:
        return jsonify({"error": "No file provided"}), 400

    data   = fobj.read()
    info   = doc_converter.get_pdf_info(data)
    s_high, s_med = file_security.scan_pdf(data)
    info["security"] = {
        "severity": "HIGH" if s_high else ("MEDIUM" if s_med else "CLEAN"),
        "threats":  s_high + s_med,
    }
    # Inform the UI of the DPI cap using the batch size as denominator.
    # A 46-page document split into 20-page batches has a safe DPI of
    # sqrt(16384 / (20 × 0.1103)) ≈ 86 DPI, not sqrt(16384 / (46 × 0.1103)) ≈ 57.
    num_pages = info.get("page_count", 1) or 1
    rec_dpi   = info.get("recommended_dpi") or 150
    capped    = _safe_dpi(rec_dpi, num_pages)
    info["dpi_safe_max"] = capped
    if capped < rec_dpi:
        batch_size = min(num_pages, doc_converter.PDF_MAX_PAGES)
        info["dpi_cap_warning"] = (
            f"At {num_pages} pages (batch size {batch_size}) the maximum safe DPI "
            f"is {capped} (vision encoder budget {_get_encoder_budget()} tokens). "
            f"The recommended DPI of {rec_dpi} has been adjusted automatically."
        )
    return jsonify(info)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log.info("Upload server starting on :8771 (identity=%s)", IDENTITY)
    try:
        app.run(host=constants.HOST_BIND, port=constants.PORT_UPLOAD, threaded=True)
    except Exception as e:
        log.error("Flask startup failed: %s", e, exc_info=True)
        raise
