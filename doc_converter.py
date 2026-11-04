#!/usr/bin/env python3
"""
doc_converter.py — Document and image format conversion.

Responsibilities (one file, one purpose):
  - PDF → JPEG page images (via pdf2image / poppler)
  - Legacy/OOXML → OOXML → images (via LibreOffice headless)
  - Raw image format normalisation (BMP, TIFF → JPEG)
  - PDF metadata pre-analysis (page count, content type, encryption)

Called by upload_server.py after security clearance.
Never touches security logic — that lives in file_security.py.
"""

import io
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional
import constants

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    from pdf2image import convert_from_bytes as _pdf_convert
    _PDF2IMAGE = True
except ImportError:
    _PDF2IMAGE = False

try:
    from pypdf import PdfReader as _PdfReader
    _PYPDF = True
except ImportError:
    _PYPDF = False

try:
    from PIL import Image as _PILImage
    _PIL = True
except ImportError:
    _PIL = False

_LIBREOFFICE: Optional[str] = shutil.which("libreoffice") or shutil.which("soffice")

# ---------------------------------------------------------------------------
# Module logger — propagates to the caller's (upload_server) log handler.
# No separate file handler here; avoid two RotatingFileHandlers on the same
# upload_server.log file in the same process.
# ---------------------------------------------------------------------------
log = logging.getLogger("doc_converter")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PDF_MAX_PAGES  = 20
PDF_DPI_DEFAULT = 150
PDF_QUALITY    = 85     # JPEG quality for page images

_SERVE_BASE = constants.URL_LAUNCHER

# ---------------------------------------------------------------------------
# PDF page image conversion
# ---------------------------------------------------------------------------

def pdf_to_images(
    data: bytes,
    dpi: int,
    original_name: str,
    conv_id: str,
    files_dir: Path,
    max_pages: int = PDF_MAX_PAGES,
) -> list[dict]:
    """
    Convert ALL pages of a PDF to JPEG images and return them split into
    batches of max_pages each.

    Every attachment dict carries:
      pdf_source, pdf_page, pdf_total_pages, pdf_dpi  — existing fields
      batch_index   — 0-based batch number
      total_batches — total number of batches for this document

    When total_batches == 1 (document fits in one batch), behaviour is
    identical to before. When total_batches > 1, the launcher will
    automatically fire a continuation request for each subsequent batch
    so the full document is analysed regardless of hardware limits.
    """
    if not _PDF2IMAGE:
        log.error("pdf2image not installed — cannot convert PDF to images")
        return []

    dpi = max(72, min(1200, dpi))

    try:
        pages = _pdf_convert(data, dpi=dpi, fmt="jpeg")
    except Exception as e:
        log.error("pdf2image conversion failed for '%s': %s", original_name, e)
        return []

    total_pages   = len(pages)
    total_batches = max(1, (total_pages + max_pages - 1) // max_pages)

    if total_batches > 1:
        log.info(
            "'%s': %d pages total — split into %d batch(es) of up to %d pages.",
            original_name, total_pages, total_batches, max_pages
        )

    attachments = []
    ts_base     = int(time.time() * 1000)

    for page_num, page_img in enumerate(pages, start=1):
        img_buf = io.BytesIO()
        page_img.save(img_buf, format="JPEG", quality=PDF_QUALITY)
        img_bytes = img_buf.getvalue()

        page_safe = f"{ts_base}_{page_num:03d}_{original_name}.jpg"
        page_dest = files_dir / page_safe
        try:
            page_dest.write_bytes(img_bytes)
        except OSError as e:
            log.error("Cannot save page image %s: %s", page_safe, e)
            continue

        batch_index = (page_num - 1) // max_pages

        attachments.append({
            "id":              page_safe,
            "name":            f"{original_name} — page {page_num}/{total_pages}",
            "url":             f"{_SERVE_BASE}/conversations/{conv_id}/files/{page_safe}",
            "type":            "image",
            "mime":            "image/jpeg",
            "size":            len(img_bytes),
            "pdf_source":      original_name,
            "pdf_page":        page_num,
            "pdf_total_pages": total_pages,
            "pdf_dpi":         dpi,
            "batch_index":     batch_index,
            "total_batches":   total_batches,
        })

    log.info(
        "PDF '%s': %d/%d pages → JPEG (DPI=%d, %d batch(es))",
        original_name, len(attachments), total_pages, dpi, total_batches
    )
    return attachments


# ---------------------------------------------------------------------------
# LibreOffice-based conversion (OOXML / legacy Office → JPEG pages)
# ---------------------------------------------------------------------------

def libreoffice_to_pdf(data: bytes, src_ext: str) -> tuple[bytes, bool]:
    """
    Convert any Office document to PDF via LibreOffice headless.
    Returns (pdf_bytes, success).
    """
    if not _LIBREOFFICE:
        log.error("LibreOffice not found — install: sudo apt install libreoffice")
        return b"", False
    try:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / f"input{src_ext}"
            src.write_bytes(data)
            result = subprocess.run(
                [_LIBREOFFICE, "--headless", "--convert-to", "pdf",
                 str(src), "--outdir", tmp],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                log.error("LibreOffice PDF conversion failed: %s", result.stderr[:300])
                return b"", False
            out = Path(tmp) / "input.pdf"
            if not out.exists():
                log.error("LibreOffice produced no PDF for %s", src_ext)
                return b"", False
            return out.read_bytes(), True
    except subprocess.TimeoutExpired:
        log.error("LibreOffice conversion timed out")
        return b"", False
    except Exception as e:
        log.error("LibreOffice error: %s", e)
        return b"", False


def office_to_images(
    data: bytes,
    src_ext: str,
    dpi: int,
    original_name: str,
    conv_id: str,
    files_dir: Path,
    max_pages: int = PDF_MAX_PAGES,
) -> list[dict]:
    """
    Convert an Office document (OOXML or legacy) to JPEG page images.
    Route: document → LibreOffice → PDF → pdf2image → JPEG images.
    """
    pdf_bytes, ok = libreoffice_to_pdf(data, src_ext)
    if not ok:
        return []
    log.info("LibreOffice rendered '%s' to PDF (%d bytes)", original_name, len(pdf_bytes))
    return pdf_to_images(pdf_bytes, dpi, original_name, conv_id, files_dir, max_pages)


# ---------------------------------------------------------------------------
# Image format normalisation
# ---------------------------------------------------------------------------

_NATIVE_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
_CONVERT_IMAGE_EXTS = {".bmp", ".tiff", ".tif"}


def normalize_image(data: bytes, ext: str) -> tuple[bytes, str, bool]:
    """
    Normalise an image to JPEG if necessary.
    Native formats (JPEG, PNG, GIF, WEBP) pass through unchanged.
    BMP and TIFF are converted to JPEG.
    Returns (output_bytes, output_mime, success).
    """
    if ext in _NATIVE_IMAGE_EXTS:
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".gif": "image/gif",
                ".webp": "image/webp"}.get(ext, "image/jpeg")
        return data, mime, True

    if ext in _CONVERT_IMAGE_EXTS:
        if not _PIL:
            log.warning("Pillow not installed — cannot convert %s to JPEG", ext)
            return data, "application/octet-stream", False
        try:
            img = _PILImage.open(io.BytesIO(data)).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=PDF_QUALITY)
            return buf.getvalue(), "image/jpeg", True
        except Exception as e:
            log.error("Image normalisation failed for %s: %s", ext, e)
            return data, "application/octet-stream", False

    return data, "application/octet-stream", False


# ---------------------------------------------------------------------------
# PDF metadata pre-analysis (used by /pdf_info endpoint)
# ---------------------------------------------------------------------------

def get_pdf_info(data: bytes) -> dict:
    """
    Quick analysis of a PDF before upload:
    page count, encryption status, embedded image detection, DPI recommendation.
    """
    result: dict = {"file_size_kb": round(len(data) / 1024, 1)}

    if not _PYPDF:
        result["error"] = "pypdf not installed"
        return result

    try:
        reader = _PdfReader(io.BytesIO(data), strict=False)

        result["encrypted"]  = reader.is_encrypted
        result["page_count"] = len(reader.pages)

        if reader.is_encrypted:
            result["recommended_dpi"] = None
            result["content_hint"]    = "encrypted — password required"
            return result

        has_images  = False
        kb_per_page = result["file_size_kb"] / max(1, result["page_count"])

        for page in reader.pages[:5]:
            resources = page.get("/Resources", {})
            if isinstance(resources, dict):
                xobjs = resources.get("/XObject", {})
                if isinstance(xobjs, dict):
                    for xref in xobjs.values():
                        try:
                            obj = xref.get_object() if hasattr(xref, "get_object") else xref
                            if isinstance(obj, dict) and obj.get("/Subtype") == "/Image":
                                has_images = True
                                break
                        except Exception:
                            pass
            if has_images:
                break

        result["has_embedded_images"] = has_images

        if has_images or kb_per_page > 150:
            dpi, hint = 600, "image-heavy document"
        elif kb_per_page > 50:
            dpi, hint = 300, "mixed text and graphics"
        else:
            dpi, hint = 150, "text-only document"

        result["recommended_dpi"] = dpi
        result["content_hint"]    = hint

    except Exception as e:
        result["error"] = str(e)

    return result
