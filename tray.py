#!/usr/bin/env python3
# =============================================================================
# tray.py
# System tray icon for Loca lAI
#
# Role: System tray icon ONLY
#   - Provides a taskbar icon with a small menu
#   - Does NOT start services (run_ai.sh does that)
#   - Does NOT open the browser on startup (run_ai.sh does that once)
#   - Exits cleanly and immediately if no X display is accessible
#
# Menu actions:
#   Open UI          → opens http://127.0.0.1:8080/chat.html in the browser
#   Restart Backend  → POST /restart to the launcher microservice
#   Shutdown Backend → POST /shutdown to the launcher microservice
#   Quit             → removes the tray icon and exits this process
# Must be launched with the UI venv python:
#   virtual_Env/ProjectUI/bin/python3 tray.py
# Required in UI venv: pystray, pillow, requests (optional)
# =============================================================================

import os
import sys
import warnings

# Suppress the requests/urllib3 version mismatch warning.
# The real fix is: pip install --upgrade requests
# This suppression ensures the tray log stays clean in the meantime.
warnings.filterwarnings("ignore", message="urllib3", category=Warning)
warnings.filterwarnings("ignore", message="chardet", category=Warning)
warnings.filterwarnings("ignore", message="charset_normalizer", category=Warning)

# -----------------------------------------------------------------------------
# 1. Check X display accessibility BEFORE importing pystray.
#    pystray's _xorg backend tries to open the display at IMPORT TIME,
#    raising Xlib.error.DisplayConnectionError if no display is available.
#    Detect this early and exit cleanly rather than crashing.
# -----------------------------------------------------------------------------
_display = os.environ.get("DISPLAY", "").strip()
if not _display:
    print("tray.py: No DISPLAY environment variable. Tray icon not available.", flush=True)
    sys.exit(0)

# Verify the X11 socket file exists before attempting any X connection.
_display_num = _display.lstrip(":").split(".")[0]
_x11_socket  = f"/tmp/.X11-unix/X{_display_num}"
if not os.path.exists(_x11_socket):
    print(f"tray.py: X11 socket {_x11_socket} not found. Tray icon not available.", flush=True)
    sys.exit(0)

# -----------------------------------------------------------------------------
# 2. Import pystray and PIL. Catch ALL exceptions — DisplayConnectionError
#    is not an ImportError, so we must catch Exception broadly.
# -----------------------------------------------------------------------------
try:
    import pystray
    from PIL import Image, ImageDraw
except Exception as e:
    print(f"tray.py: Cannot initialise tray ({type(e).__name__}: {e}). Exiting.", flush=True)
    sys.exit(0)

# -----------------------------------------------------------------------------
# 3. HTTP helper — calls the launcher microservice.
#    Uses urllib (stdlib only) so we don't depend on requests here.
# -----------------------------------------------------------------------------
import urllib.request
import urllib.error
import webbrowser

LAUNCHER = "http://127.0.0.1:8770"
UI_URL   = "http://127.0.0.1:8080/chat.html"


def _post(path: str) -> None:
    """Fire-and-forget POST to the launcher. Silently ignores failures."""
    try:
        req = urllib.request.Request(
            f"{LAUNCHER}{path}",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=3)
    except Exception:
        pass

# -----------------------------------------------------------------------------
# 4. Tray icon image
# -----------------------------------------------------------------------------

def _make_icon_image() -> Image.Image:
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d   = ImageDraw.Draw(img)
    d.ellipse((4, 4, 60, 60), fill="#10b981")
    try:
        d.text((16, 20), "AI", fill="white")
    except Exception:
        pass
    return img

# -----------------------------------------------------------------------------
# 5. Menu callbacks
# -----------------------------------------------------------------------------

def on_open_ui(icon, item):
    webbrowser.open(UI_URL)


def on_restart(icon, item):
    _post("/restart")


def on_shutdown(icon, item):
    _post("/shutdown")
    try:
        icon.stop()
    except Exception:
        pass


def on_quit(icon, item):
    try:
        icon.stop()
    except Exception:
        pass

# -----------------------------------------------------------------------------
# 6. Main — run the tray icon. Any X11 crash exits cleanly.
# -----------------------------------------------------------------------------

def main():
    try:
        image = _make_icon_image()
        menu  = pystray.Menu(
            pystray.MenuItem("🌐 Open UI",           on_open_ui),
            pystray.MenuItem("🔄 Restart Backend",   on_restart),
            pystray.MenuItem("⏹️  Shutdown Backend", on_shutdown),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("✖ Quit Tray",          on_quit),
        )
        icon = pystray.Icon("LocalAI", image, "LocalAI Agent", menu)
        print("tray.py: Tray icon active.", flush=True)
        icon.run()
    except Exception as e:
        print(f"tray.py: Tray icon stopped ({type(e).__name__}: {e}).", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
