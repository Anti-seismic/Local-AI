#!/usr/bin/env bash
# =============================================================================
# project_env.sh — Build the ProjectUI virtualenv
#
# Cleans and rebuilds ONLY the ProjectUI venv.
# When called standalone (partial reinstall), the vLLM venv is untouched.
# When called from setup_all.sh, the parent has already wiped everything.
#
# Usage (standalone partial reinstall):
#   bash project_env.sh
# =============================================================================

set -euo pipefail

ENV_DIR="$HOME/LocalAI/virtual_Env/ProjectUI"

echo "🚀 ProjectUI Environment: $ENV_DIR"

# === LOCAL HYGIENE ===
# Remove only this venv (redundancy for standalone partial reinstalls).
# setup_all.sh wipes the entire virtual_Env directory before calling this.
rm -rf "$ENV_DIR"
python3 -m venv "$ENV_DIR"
# shellcheck disable=SC1090
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools

# === FLASK STACK (upload_server.py) ===
python -m pip install \
  "flask>=3.0,<4.0" \
  "werkzeug==3.0.4"

# === PDF PROCESSING STACK (upload_server.py / doc_converter.py) ===
# pdf2image: PDF → JPEG page images (requires poppler-utils system package)
# pypdf:     PDF security scanning and sanitization
# oletools:  Legacy binary Office (.doc/.xls/.ppt) macro scanning
# System packages required (run once as root):
#   sudo apt install poppler-utils libreoffice libmagic1
python -m pip install \
  pdf2image \
  pypdf \
  oletools


# pystray requires Pillow for icon image rendering.
# requests is used for HTTP calls to the launcher microservice on :8770.
python -m pip install \
  pystray \
  Pillow \
  requests

# === VERIFY ===
# Use importlib.metadata.version() throughout:
#   - pystray 0.19.x has no __version__ attribute
#   - flask 3.0+ deprecated flask.__version__
python -c "
import importlib.metadata

flask_ver     = importlib.metadata.version('flask')
pystray_ver   = importlib.metadata.version('pystray')
pillow_ver    = importlib.metadata.version('Pillow')
requests_ver  = importlib.metadata.version('requests')
pdf2img_ver   = importlib.metadata.version('pdf2image')
pypdf_ver     = importlib.metadata.version('pypdf')

print('✅ Flask:      ', flask_ver)
print('✅ pystray:    ', pystray_ver)
print('✅ Pillow:     ', pillow_ver)
print('✅ requests:   ', requests_ver)
print('✅ pdf2image:  ', pdf2img_ver)
print('✅ pypdf:      ', pypdf_ver)
print('✅ ProjectUI ready')
"

echo "✅ ProjectUI: $ENV_DIR"
