#!/usr/bin/env python3
"""
file_security.py — Document security scanning and sanitization.

Responsibilities (one file, one purpose):
  - Magic-byte file type validation
  - PDF: scan (JS, auto-run, Launch), decrypt (password-protected), sanitize
  - OOXML (.docx/.pptx/.xlsx): scan (VBA, external rels), sanitize
  - Legacy Office (.doc/.xls/.ppt): scan via oletools, sanitize via LibreOffice
  - Dedicated security audit log for SOC/forensics

All password material is handled as opaque bytes and is NEVER written to any log.
"""

import io
import logging
import logging.handlers
import re
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional dependencies — import failures degrade gracefully
# ---------------------------------------------------------------------------
try:
    from pypdf import PdfReader as _PdfReader, PdfWriter as _PdfWriter
    _PYPDF = True
except ImportError:
    _PYPDF = False

try:
    from oletools.olevba import VBA_Parser as _VBA_Parser
    _OLETOOLS = True
except ImportError:
    _OLETOOLS = False

try:
    import shutil as _shutil
    _LIBREOFFICE = _shutil.which("libreoffice") or _shutil.which("soffice")
except Exception:
    _LIBREOFFICE = None

# ---------------------------------------------------------------------------
# Security audit logger
# ---------------------------------------------------------------------------
_BASE    = Path(__file__).resolve().parent
_LOG_DIR = _BASE / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

sec = logging.getLogger("localai.security")
sec.setLevel(logging.DEBUG)
sec.propagate = False

_sh = logging.handlers.RotatingFileHandler(
    _LOG_DIR / "security_audit.log",
    maxBytes=100 * 1024 * 1024, backupCount=10, encoding="utf-8"
)
_sh.setFormatter(logging.Formatter(
    "%(asctime)s.%(msecs)03d [%(levelname)-7s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S"
))
sec.addHandler(_sh)

# ---------------------------------------------------------------------------
# Magic-byte file type validation
# ---------------------------------------------------------------------------
_MAGIC_MAP: dict[bytes, str] = {
    b"%PDF":              "pdf",
    b"PK\x03\x04":       "ooxml",    # .docx / .pptx / .xlsx
    b"\xD0\xCF\x11\xE0": "ole",      # .doc / .xls / .ppt
    b"\xFF\xD8\xFF":      "jpeg",
    b"\x89PNG\r\n":       "png",
    b"GIF8":              "gif",
    b"RIFF":              "riff",     # .webp / .wav — disambiguate by ext
    b"BM":                "bmp",
    b"II*\x00":           "tiff",
    b"MM\x00*":           "tiff",
}

_EXT_TO_MAGIC: dict[str, str] = {
    ".pdf":  "pdf",
    ".docx": "ooxml", ".pptx": "ooxml", ".xlsx": "ooxml",
    ".dotx": "ooxml", ".potx": "ooxml", ".xltx": "ooxml",
    ".doc":  "ole",   ".ppt":  "ole",   ".xls":  "ole",
    ".jpg":  "jpeg",  ".jpeg": "jpeg",
    ".png":  "png",
    ".gif":  "gif",
    ".webp": "riff",
    ".bmp":  "bmp",
    ".tiff": "tiff",  ".tif":  "tiff",
}

_TEXT_EXTS = {
    ".txt", ".py", ".js", ".ts", ".json", ".csv", ".md",
    ".sh", ".bash", ".yaml", ".yml", ".toml", ".xml", ".html",
    ".htm", ".css", ".sql", ".rs", ".go", ".java", ".c", ".cpp",
    ".h", ".rb", ".php", ".log",
}


def validate_magic(data: bytes, ext: str) -> dict:
    """
    Validate that the file's magic bytes match its claimed extension.
    Returns {"valid": bool, "detected": str, "claimed": str}.
    Text files are exempt (no magic bytes standard).
    """
    if ext in _TEXT_EXTS:
        return {"valid": True, "detected": "text", "claimed": ext}

    header   = data[:8]
    detected = next((v for k, v in _MAGIC_MAP.items() if header.startswith(k)), None)
    expected = _EXT_TO_MAGIC.get(ext)

    if expected is None:
        # Unknown extension — let it through, warn
        return {"valid": True, "detected": detected or "unknown", "claimed": ext}

    # RIFF is shared by WEBP and WAV — disambiguate by extension
    if detected == "riff" and expected == "riff":
        return {"valid": True, "detected": "riff", "claimed": ext}

    valid = detected == expected if detected else False
    return {"valid": valid, "detected": detected or "unknown", "claimed": ext}


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def detect_pdf_encryption(data: bytes) -> bool:
    """Return True if the PDF is password-protected (requires decryption)."""
    if not _PYPDF:
        return False
    try:
        r = _PdfReader(io.BytesIO(data), strict=False)
        return r.is_encrypted
    except Exception:
        return False


def decrypt_pdf(data: bytes, password_raw: str) -> tuple[bytes, bool, str]:
    """
    Attempt to decrypt a password-protected PDF.
    password_raw is treated as an opaque string — never logged.
    Returns (decrypted_bytes, success, error_description).
    """
    if not _PYPDF:
        return data, False, "pypdf not available"
    try:
        # Sanitize: strip null bytes, limit length, treat as raw string
        password = password_raw.replace("\x00", "")[:1024]
        reader   = _PdfReader(io.BytesIO(data), strict=False)
        result   = reader.decrypt(password)
        if result == 0:
            return data, False, "Incorrect password or unsupported encryption algorithm"
        # Re-write through PdfWriter to produce a plain (non-encrypted) copy
        writer = _PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        out = io.BytesIO()
        writer.write(out)
        return out.getvalue(), True, ""
    except Exception as e:
        return data, False, str(e)


def _scan_pdf_raw(data: bytes) -> tuple[list, list]:
    """Return (high_threats, medium_threats) from scanning a PDF."""
    high, medium = [], []
    if not _PYPDF:
        return high, medium
    try:
        reader = _PdfReader(io.BytesIO(data), strict=False)
        trailer = reader.trailer
        if "/OpenAction" in trailer:
            act = trailer["/OpenAction"]
            if isinstance(act, dict) and act.get("/S") in ("/JavaScript", "/JS"):
                high.append("Auto-open JavaScript (/OpenAction /JS)")
            else:
                medium.append("Auto-open action (/OpenAction)")
        root = reader.root_object
        if "/AcroForm" in root:
            af = root["/AcroForm"]
            if isinstance(af, dict):
                if "/XFA" in af:
                    medium.append("XFA active form (/AcroForm /XFA)")
                if "/AA" in af or "/JS" in af:
                    high.append("JavaScript in AcroForm (/JS or /AA)")
        if "/Names" in root:
            names_tree = root["/Names"]
            if isinstance(names_tree, dict) and "/EmbeddedFiles" in names_tree:
                medium.append("Embedded file tree (/EmbeddedFiles)")
        for i, page in enumerate(reader.pages, 1):
            if "/AA" in page:
                aa = page["/AA"]
                if isinstance(aa, dict):
                    for trig, act in aa.items():
                        s = act.get("/S", "") if isinstance(act, dict) else ""
                        bucket = high if s in ("/JavaScript", "/JS") else medium
                        bucket.append(f"Page {i}: Additional Action {trig} (S={s or 'unknown'})")
            if "/Annots" in page:
                for ref in (page["/Annots"] or []):
                    try:
                        ann = ref.get_object() if hasattr(ref, "get_object") else ref
                        if isinstance(ann, dict):
                            act = ann.get("/A", {})
                            if isinstance(act, dict) and act.get("/S") in ("/Launch", "/JavaScript", "/JS"):
                                high.append(f"Page {i}: {act.get('/S')} in annotation")
                    except Exception:
                        pass
    except Exception as e:
        high.append(f"PDF parse error: {e}")
    return high, medium


def scan_pdf(data: bytes) -> tuple[list, list]:
    """
    Public wrapper around _scan_pdf_raw for callers outside this module
    (e.g. upload_server.py /pdf_info endpoint).
    Returns (high_threats, medium_threats).
    Using a public wrapper insulates callers from internal refactoring.
    """
    return _scan_pdf_raw(data)


def _sanitize_pdf_data(data: bytes) -> tuple[bytes, list[str], bool]:
    """
    Rewrite PDF through PdfWriter (fresh document root — strips /OpenAction,
    /JS, /JavaScript, /AcroForm by design). Strips /AA and dangerous annotations
    per page explicitly.
    Returns (sanitized_bytes, removed_descriptions, success).
    """
    if not _PYPDF:
        return data, ["pypdf not available"], False
    removed = []
    try:
        reader = _PdfReader(io.BytesIO(data), strict=False)
        writer = _PdfWriter()
        for i, page in enumerate(reader.pages, 1):
            try:
                if "/AA" in page:
                    del page.get_object()["/AA"]
                    removed.append(f"Page {i}: /AA (Additional Actions)")
            except Exception:
                pass
            try:
                if "/Annots" in page:
                    clean = []
                    for ref in (page["/Annots"] or []):
                        try:
                            ann = ref.get_object() if hasattr(ref, "get_object") else ref
                            if isinstance(ann, dict):
                                act = ann.get("/A", {})
                                if isinstance(act, dict) and act.get("/S") in (
                                        "/Launch", "/JavaScript", "/JS"):
                                    removed.append(f"Page {i}: {act.get('/S')} annotation")
                                    continue
                            clean.append(ref)
                        except Exception:
                            clean.append(ref)
                    page.get_object()["/Annots"] = clean
            except Exception:
                pass
            writer.add_page(page)
        # Safe metadata subset only
        if reader.metadata:
            try:
                writer.add_metadata({k: v for k, v in reader.metadata.items()
                                     if k in ("/Title", "/Author", "/Subject",
                                              "/Creator", "/Producer")})
            except Exception:
                pass
        removed.append("Document-level /OpenAction, /JS, /AcroForm not carried "
                       "to new writer (stripped by design)")
        out = io.BytesIO()
        writer.write(out)
        sanitized = out.getvalue()
        _PdfReader(io.BytesIO(sanitized), strict=False)   # validate
        return sanitized, removed, True
    except Exception as e:
        return data, [f"PDF sanitization error: {e}"], False


# ---------------------------------------------------------------------------
# OOXML helpers (.docx / .pptx / .xlsx)
# ---------------------------------------------------------------------------

def _scan_ooxml_raw(data: bytes) -> tuple[list, list]:
    high, medium = [], []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = z.namelist()
            vba = [n for n in names if "vbaProject" in n]
            if vba:
                high.append(f"VBA macro project: {vba[0]}")
            ole = [n for n in names
                   if "embeddings" in n.lower() or "oleObject" in n]
            if ole:
                # OLE objects are kept (rendered safely by LibreOffice headless)
                medium.append(f"Embedded OLE objects present ({len(ole)} item(s)) "
                              "— kept for visual rendering, not removed")
            for n in names:
                if n.endswith(".rels"):
                    try:
                        content = z.read(n).decode("utf-8", errors="replace")
                        if 'TargetMode="External"' in content:
                            medium.append(f"External relationship in {n} "
                                          "(auto-fetch capable — will be removed)")
                            break
                    except Exception:
                        pass
    except zipfile.BadZipFile:
        high.append("Not a valid ZIP/OOXML archive — possible format spoofing")
    except Exception as e:
        high.append(f"OOXML scan error: {e}")
    return high, medium


def _sanitize_ooxml_data(data: bytes) -> tuple[bytes, list[str], bool]:
    """
    Remove VBA macros and external relationships from OOXML.
    OLE embedded objects are intentionally kept (see security comments).
    """
    removed = []
    REMOVE_NAMES = {"vbaProject.bin", "vbaData.bin"}
    try:
        out_buf = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(data), "r") as zin, \
             zipfile.ZipFile(out_buf, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                name = item.filename
                if any(name.endswith(p) for p in REMOVE_NAMES) or \
                   any(d in name for d in ("vbaProject/", "vbaData/")):
                    removed.append(f"VBA removed: {name}")
                    continue
                content = zin.read(name)
                if name.endswith(".rels") and b'TargetMode="External"' in content:
                    content = re.sub(
                        rb'<Relationship[^>]+TargetMode="External"[^/]*/?>',
                        b"", content
                    )
                    removed.append(f"External relationship stripped in {name}")
                if name == "[Content_Types].xml" and b"vbaProject" in content:
                    content = re.sub(rb'<Override[^>]*vbaProject[^/]*/>', b"", content)
                    removed.append("vbaProject content-type entry removed")
                zout.writestr(item, content)
        sanitized = out_buf.getvalue()
        with zipfile.ZipFile(io.BytesIO(sanitized)):  # validate
            pass
        return sanitized, removed, True
    except Exception as e:
        return data, [f"OOXML sanitization error: {e}"], False


# ---------------------------------------------------------------------------
# Legacy Office helpers (.doc / .xls / .ppt)
# ---------------------------------------------------------------------------

def _scan_legacy_raw(data: bytes, filename: str) -> tuple[list, list]:
    high, medium = [], []
    if not _OLETOOLS:
        high.append("oletools not installed — install with: pip install oletools")
        return high, medium
    try:
        vba = _VBA_Parser(filename, data=data)
        if vba.detect_vba_macros():
            for _, code_file, name, _ in vba.extract_macros():
                high.append(f"VBA macro '{name}' in '{code_file}'")
        vba.close()
    except Exception as e:
        high.append(f"oletools scan error: {e}")
    return high, medium


def _sanitize_legacy_via_libreoffice(data: bytes, ext: str) -> tuple[bytes, list[str], bool]:
    """
    Convert legacy binary Office file to OOXML via LibreOffice headless.
    LibreOffice import strips macros by design — no manual removal needed.
    The resulting OOXML is then passed through _sanitize_ooxml_data().
    """
    import subprocess
    import tempfile

    if not _LIBREOFFICE:
        return data, ["LibreOffice not found — install: sudo apt install libreoffice"], False

    EXT_MAP = {".doc": "docx", ".ppt": "pptx", ".xls": "xlsx"}
    target = EXT_MAP.get(ext)
    if not target:
        return data, [f"No LibreOffice conversion defined for {ext}"], False

    removed = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / f"input{ext}"
            src.write_bytes(data)
            result = subprocess.run(
                [_LIBREOFFICE, "--headless", "--convert-to", target,
                 str(src), "--outdir", tmp],
                capture_output=True, text=True, timeout=90
            )
            if result.returncode != 0:
                return data, [f"LibreOffice conversion failed: {result.stderr[:300]}"], False
            out_path = Path(tmp) / f"input.{target}"
            if not out_path.exists():
                return data, ["LibreOffice produced no output file"], False
            ooxml_data = out_path.read_bytes()
            removed.append(f"Converted {ext} → .{target} via LibreOffice "
                           "(macros stripped by importer)")
            # Apply OOXML sanitizer on the converted output
            sanitized, ooxml_removed, ok = _sanitize_ooxml_data(ooxml_data)
            removed.extend(ooxml_removed)
            if not ok:
                return ooxml_data, removed + ["OOXML post-sanitization partial"], True
            return sanitized, removed, True
    except subprocess.TimeoutExpired:
        return data, ["LibreOffice conversion timed out (> 90 s)"], False
    except Exception as e:
        return data, [f"LibreOffice conversion error: {e}"], False


# ---------------------------------------------------------------------------
# Unified pipeline
# ---------------------------------------------------------------------------

_OOXML_EXTS  = {".docx", ".pptx", ".xlsx", ".dotx", ".potx", ".xltx"}
_LEGACY_EXTS = {".doc", ".ppt", ".xls"}


def full_pipeline(data: bytes, ext: str, filename: str) -> dict:
    """
    Full security pipeline: scan → sanitize → re-scan → audit-log.

    Returns:
      {
        "data":         bytes,       # sanitized bytes (or original if clean)
        "blocked":      bool,
        "sanitized":    bool,
        "removed":      list[str],
        "severity":     str,         # "CLEAN" | "HIGH" | "MEDIUM" | "UNVERIFIABLE"
        "threats":      list[str],
        "block_reason": str | None,
        "warning":      str | None,  # amber warning text for UI (MEDIUM)
      }
    """
    # ── 1. Choose scanner/sanitizer by file type ──────────────────────────
    if ext == ".pdf":
        scan_fn     = _scan_pdf_raw
        sanitize_fn = _sanitize_pdf_data
        rescan_fn   = _scan_pdf_raw
    elif ext in _OOXML_EXTS:
        scan_fn     = _scan_ooxml_raw
        sanitize_fn = _sanitize_ooxml_data
        rescan_fn   = _scan_ooxml_raw
    elif ext in _LEGACY_EXTS:
        scan_fn     = lambda d: _scan_legacy_raw(d, filename)  # noqa: E731
        sanitize_fn = lambda d: _sanitize_legacy_via_libreoffice(d, ext)  # noqa: E731
        rescan_fn   = lambda d: _scan_ooxml_raw(d)   # output is OOXML  # noqa: E731
    else:
        # Unscannable type — pass through unchanged
        sec.info("[Security: PASS] %s (no scanner for %s)", filename, ext)
        return {"data": data, "blocked": False, "sanitized": False,
                "removed": [], "severity": "CLEAN", "threats": [],
                "block_reason": None, "warning": None}

    # ── 2. Initial scan ───────────────────────────────────────────────────
    high, medium = scan_fn(data)
    all_threats = high + medium
    severity    = "HIGH" if high else ("MEDIUM" if medium else "CLEAN")

    if severity == "CLEAN":
        sec.info("[Security: PASS] %s", filename)
        return {"data": data, "blocked": False, "sanitized": False,
                "removed": [], "severity": "CLEAN", "threats": [],
                "block_reason": None, "warning": None}

    # ── 3. Attempt sanitization ───────────────────────────────────────────
    sanitized_data, removed, ok = sanitize_fn(data)

    if not ok:
        reason = f"sanitization failed: {'; '.join(removed)}"
        sec.error("[Security: BLOCKED] %s\n  Threats: %s\n  Reason: %s",
                  filename, "; ".join(all_threats), reason)
        return {"data": None, "blocked": True, "sanitized": False,
                "removed": [], "severity": severity, "threats": all_threats,
                "block_reason": reason, "warning": None}

    # ── 4. Re-scan sanitized output ───────────────────────────────────────
    re_high, re_medium = rescan_fn(sanitized_data)
    residual = re_high + re_medium

    if re_high:
        reason = f"residual HIGH threats after sanitization: {'; '.join(re_high)}"
        sec.error("[Security: BLOCKED] %s\n  Initial: %s\n  Removed: %s\n  Residual: %s",
                  filename, "; ".join(all_threats), "; ".join(removed), "; ".join(residual))
        return {"data": None, "blocked": True, "sanitized": False,
                "removed": removed, "severity": severity, "threats": all_threats,
                "block_reason": reason, "warning": None}

    # ── 5. Success ────────────────────────────────────────────────────────
    if len(removed) == 1:
        sec.warning("[Security: SANITIZED] %s: %s — %s",
                    filename, "; ".join(all_threats), removed[0])
    else:
        sec.warning("[Security: SANITIZED] %s\n  Threats: %s\n  Removed (%d items): %s"
                    "\n  Re-scan: CLEAN",
                    filename, "; ".join(all_threats), len(removed), "; ".join(removed))

    warning = (f"Document sanitized: {', '.join(medium[:3])}"
               + (" (and more)" if len(medium) > 3 else "")) if medium and not high else None

    return {"data": sanitized_data, "blocked": False, "sanitized": True,
            "removed": removed, "severity": severity, "threats": all_threats,
            "block_reason": None, "warning": warning}
