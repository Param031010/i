"""Step 1: load uploaded invoices (PNG/JPG/WebP/PDF) into memory, losslessly."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from io import BytesIO

import pymupdf  # PyMuPDF
from PIL import Image

PDF_RENDER_DPI = 300  # PDF pages are rasterised once, at this fixed resolution


@dataclass(frozen=True)
class InvoiceDoc:
    """An uploaded file held in memory. Nothing here is resized or re-encoded."""
    name: str
    file_hash: str
    original_bytes: bytes          # exact bytes the user uploaded
    pages: list[Image.Image] = field(default_factory=list)  # full-resolution pixels
    source_type: str = "image"     # "image" | "pdf"


def _load_image(data: bytes) -> Image.Image:
    img = Image.open(BytesIO(data))
    img.load()  # decode fully; pixels stay exactly as stored in the file
    return img


def _render_pdf(data: bytes) -> list[Image.Image]:
    pages = []
    zoom = PDF_RENDER_DPI / 72
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        for page in pdf:
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    return pages


def load_invoice(name: str, data: bytes) -> InvoiceDoc:
    digest = hashlib.sha256(data).hexdigest()
    if name.lower().endswith(".pdf"):
        pages, kind = _render_pdf(data), "pdf"
    else:
        pages, kind = [_load_image(data)], "image"
    if not pages:
        raise ValueError("The file contains no pages.")
    return InvoiceDoc(name, digest, data, pages, kind)
