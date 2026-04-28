#!/usr/bin/env bash
# ================================================================
# project_env.sh
# Build or clean then rebuild the "ProjectUI" virtual environment
# ================================================================

set -euo pipefail

ENV_DIR="$HOME/LocalAI/virtual_Env/ProjectUI"

echo "ProjectUI Environment: $ENV_DIR"

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

# === TRAY ICON (tray.py) ===
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

flask_ver    = importlib.metadata.version('flask')
pystray_ver  = importlib.metadata.version('pystray')
pillow_ver   = importlib.metadata.version('Pillow')
requests_ver = importlib.metadata.version('requests')

print('Flask:     ', flask_ver)
print('pystray:   ', pystray_ver)
print('Pillow:    ', pillow_ver)
print('requests:  ', requests_ver)
print('ProjectUI ready')
"

echo "ProjectUI: $ENV_DIR"
