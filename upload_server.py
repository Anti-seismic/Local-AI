#!/usr/bin/env python3
# =============================================================================
# upload_server.py
# File upload microservice for Local AI
#
# Endpoints:
#   POST /upload?conv_id=<id>  upload files and store in conversation folder
#   GET  /status               health check
#
# Must be launched with the UI venv python:
#   virtual_Env/ProjectUI/bin/python3 upload_server.py
# Dependencies in UI venv: flask
#
# Note: OCR (PaddleOCR) has been removed. Image and document understanding
# is handled entirely by the model's native vision capability
# =============================================================================

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

# Silence Flask/Werkzeug's access log — our logger handles all output.
import logging as _stdlib_logging
_stdlib_logging.getLogger("werkzeug").setLevel(_stdlib_logging.WARNING)

# ---------------------------------------------------------------------------
# Paths & identity
# ---------------------------------------------------------------------------
BASE_DIR  = Path("/home/ai-broker/LocalAI")
LOG_DIR   = BASE_DIR / "logs"
DATA_ROOT = BASE_DIR / "data"

LOG_DIR.mkdir(parents=True, exist_ok=True)

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
# Logging — single rotating handler + stdout. log.propagate=False prevents
# duplicate entries from bubbling to the root logger.
# ---------------------------------------------------------------------------
LOG_MAX_BYTES = 50 * 1024 * 1024

log = logging.getLogger("upload_server")
log.setLevel(logging.DEBUG)
log.propagate = False

_fh = logging.handlers.RotatingFileHandler(
    LOG_DIR / "upload_server.log",
    maxBytes=LOG_MAX_BYTES, backupCount=1, encoding="utf-8"
)
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
log.addHandler(_fh)
log.addHandler(logging.StreamHandler(sys.stdout))

# ---------------------------------------------------------------------------
# File type / size policy
# ---------------------------------------------------------------------------
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

FILE_SIZE_LIMITS: dict = {
    ".png":  100 * 1024 * 1024,
    ".jpg":  100 * 1024 * 1024,
    ".jpeg": 100 * 1024 * 1024,
    ".gif":  100 * 1024 * 1024,
    ".webp": 100 * 1024 * 1024,
    ".bmp":  100 * 1024 * 1024,
    ".tiff": 100 * 1024 * 1024,
    ".pdf":  200 * 1024 * 1024,
    ".xlsx":  50 * 1024 * 1024,
    ".xls":   50 * 1024 * 1024,
    ".ppt":   50 * 1024 * 1024,
    ".pptx":  50 * 1024 * 1024,
    ".docx": 200 * 1024 * 1024,
    ".doc":  200 * 1024 * 1024,
    ".txt":   30 * 1024 * 1024,
    ".py":   100 * 1024 * 1024,
    ".js":   100 * 1024 * 1024,
    ".mp4":  5000 * 1024 * 1024,
    ".avi":  5000 * 1024 * 1024,
    ".mov":  5000 * 1024 * 1024,
}
DEFAULT_SIZE_LIMIT    = 200 * 1024 * 1024
MAX_FILES_PER_REQUEST = 20


def _check_disk_space():
    import shutil
    total, _, free = shutil.disk_usage(DATA_ROOT)
    free_pct = (free / total) * 100
    if free_pct < 11:
        return False, f"Disk space critical: only {free_pct:.1f}% free"
    return True, None

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5500 * 1024 * 1024


@app.after_request
def _cors(resp):
    resp.headers["Access-Control-Allow-Origin"]  = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Conv-Id"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "running":  True,
        "identity": IDENTITY,
    })


@app.route("/upload", methods=["POST", "OPTIONS"])
def upload_files():
    if request.method == "OPTIONS":
        return jsonify({"ok": True})

    ok, msg = _check_disk_space()
    if not ok:
        return jsonify({"error": msg}), 413

    conv_id = (
        request.args.get("conv_id")
        or request.form.get("conv_id")
        or request.headers.get("X-Conv-Id", "")
    ).strip()
    if not conv_id:
        return jsonify({"error": "conv_id is required"}), 400

    files_dir = USER_DATA_DIR / conv_id / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    uploaded = request.files.getlist("files")
    if not uploaded:
        return jsonify({"error": "No files provided"}), 400
    if len(uploaded) > MAX_FILES_PER_REQUEST:
        return jsonify({"error": f"Max {MAX_FILES_PER_REQUEST} files per request"}), 413

    processed  = []
    total_size = 0

    for fobj in uploaded:
        if not fobj or not fobj.filename:
            continue

        original_name = Path(fobj.filename).name
        ext           = Path(original_name).suffix.lower()
        data          = fobj.read()
        size          = len(data)

        limit = FILE_SIZE_LIMITS.get(ext, DEFAULT_SIZE_LIMIT)
        if size > limit:
            return jsonify({
                "error": f"'{original_name}' exceeds limit of {limit // 1024 // 1024} MB"
            }), 413

        total_size += size

        safe_name = f"{int(time.time() * 1000)}_{original_name}"
        dest      = files_dir / safe_name
        try:
            dest.write_bytes(data)
        except OSError as e:
            log.error("Failed to save %s: %s", safe_name, e)
            return jsonify({"error": f"Failed to save file: {e}"}), 500

        mime, _ = mimetypes.guess_type(original_name)
        mime     = mime or fobj.content_type or "application/octet-stream"

        if ext in VIDEO_EXTS:
            file_type = "video"
        elif ext in IMAGE_EXTS:
            file_type = "image"
        else:
            file_type = "document"

        # Serve URL: the launcher exposes files at this path via ThreadingHTTPServer.
        serve_url = (
            f"http://127.0.0.1:8770/conversations/{conv_id}/files/{safe_name}"
        )

        processed.append({
            "id":   safe_name,
            "name": original_name,
            "url":  serve_url,
            "type": file_type,
            "mime": mime,
            "size": size,
        })
        log.info(
            "Uploaded '%s' (%d bytes, type=%s)",
            original_name, size, file_type,
        )

    return jsonify({"files": processed, "total_size": total_size})


if __name__ == "__main__":
    log.info("Upload server starting on :8771 (identity=%s)", IDENTITY)
    app.run(host="0.0.0.0", port=8771, debug=False, threaded=True)
