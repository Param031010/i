"""Step 7: verify the Step 6 JSON against the original image with a multimodal Claude model on Bedrock."""
from __future__ import annotations

import base64
import copy
import json
from dataclasses import dataclass
from io import BytesIO

from PIL import Image

from ingest import InvoiceDoc

DEFAULT_MODEL = "anthropic.claude-opus-5-5"
DEFAULT_REGION = "us-east-1"
MAX_IMAGE_BYTES = 3_500_000  # keeps base64 under Bedrock's ~5 MB per-image limit
MAX_EDGE = 8000
_MEDIA = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}

SYSTEM = """You verify OCR output for tables in an invoice image.
You get the invoice image and JSON produced by a handwriting OCR model (TrOCR). \
The JSON describes tables > rows > cells, each cell with its text and bounding boxes.

Rules:
- Compare each cell's "text" with what is visible in the image inside that cell.
- Change a cell's text ONLY when the OCR is clearly wrong (misread characters, stray symbols, \
split or merged words) or the cell is marked blank but visible text is clearly there.
- If you are unsure, or the writing is genuinely unreadable, keep the OCR text unchanged. \
Never guess, never infer values from other cells, never compute or "fix" totals, never invent data.
- Do not add, remove, merge, split or reorder tables, rows or cells.
- Return ONLY one JSON object, with no commentary and no code fences, in exactly this shape:
{"cells": [{"table": <int>, "row": <int>, "col": <int>, "text": "<final text>"}]}
List every cell from the input, in the same order, using the same table/row/col values."""


@dataclass
class LLMImage:
    media_type: str
    data: bytes
    note: str  # what was actually sent, for display


def prepare_image(doc: InvoiceDoc, page_idx: int) -> LLMImage:
    """Original bytes when they fit; otherwise a re-encoded copy used only for sending."""
    if doc.source_type == "image":
        fmt = Image.open(BytesIO(doc.original_bytes)).format
        img = doc.pages[0]
        if fmt in _MEDIA and len(doc.original_bytes) <= MAX_IMAGE_BYTES and max(img.size) <= MAX_EDGE:
            return LLMImage(_MEDIA[fmt], doc.original_bytes, "original file, unmodified")
    img = doc.pages[page_idx].convert("RGB")
    scale = min(1.0, MAX_EDGE / max(img.size))
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, "PNG")
    if len(buf.getvalue()) <= MAX_IMAGE_BYTES:
        return LLMImage("image/png", buf.getvalue(),
                        "lossless PNG of the page" + (" (downscaled to fit 8000px)" if scale < 1 else ""))
    for s in (1.0, 0.85, 0.7, 0.55, 0.4):  # transmission copy only; the stored original is untouched
        im = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS) if s < 1 else img
        buf = BytesIO()
        im.save(buf, "JPEG", quality=92)
        if len(buf.getvalue()) <= MAX_IMAGE_BYTES:
            return LLMImage("image/jpeg", buf.getvalue(), f"JPEG q92 at {im.width}x{im.height} (size limit)")
    raise ValueError("Image is too large to send even after downscaling.")


def merge_verified(original: dict, reply: dict, model: str) -> tuple[dict, list[dict]]:
    """Apply only cell `text` edits onto the original JSON. Structure and boxes always come from
    the original, so the model cannot add, drop or move anything."""
    edits = {}
    for c in reply.get("cells", []):
        try:
            edits[(int(c["table"]), int(c["row"]), int(c["col"]))] = c["text"]
        except (KeyError, TypeError, ValueError):
            continue

    result = copy.deepcopy(original)
    changes = []
    for t in result["tables"]:
        for r in t["rows"]:
            for c in r["cells"]:
                key = (t["table"], r["row"], c["col"])
                new = edits.get(key)
                if not isinstance(new, str):
                    continue
                new = new.strip()
                if new != c["text"]:
                    changes.append({"table": key[0], "row": key[1], "col": key[2],
                                    "ocr_text": c["text"], "verified_text": new})
                    c["ocr_text"] = c["text"]
                    c["text"] = new
                    c["is_blank"] = new == ""
                    c["corrected"] = True
    result["verification"] = {"model": model, "cells_corrected": len(changes)}
    return result, changes


def call_bedrock(doc: InvoiceDoc, page_idx: int, payload: dict, model: str = DEFAULT_MODEL,
                 region: str = DEFAULT_REGION, profile: str | None = None, client=None) -> tuple[str, str]:
    """Send image + JSON to Bedrock. Returns (raw reply text, note about the image sent).
    The reply is NOT parsed or trusted here - Step 8 validates it."""
    if client is None:
        from anthropic import AnthropicBedrockMantle
        client = AnthropicBedrockMantle(aws_region=region, **({"aws_profile": profile} if profile else {}))

    img = prepare_image(doc, page_idx)
    resp = client.messages.create(
        model=model,
        max_tokens=16000,
        system=SYSTEM,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": img.media_type,
                                         "data": base64.standard_b64encode(img.data).decode()}},
            {"type": "text", "text": "OCR JSON to verify:\n" + json.dumps(payload, ensure_ascii=False)},
        ]}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("The model declined this request.")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("The reply was cut off (max_tokens); the table is too large for one call.")
    return "".join(b.text for b in resp.content if b.type == "text"), img.note
