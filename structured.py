"""Step 6: assemble TrOCR cell results into one structured JSON document."""
from __future__ import annotations

from ingest import InvoiceDoc
from detect import DetectedTable
from ocr import CellText


def _box(b) -> dict:
    return {"x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3]}


def build_json(doc: InvoiceDoc, page_idx: int, tables: list[DetectedTable],
               results: list[CellText]) -> dict:
    page = doc.pages[page_idx]
    by_table: dict[int, dict[int, list[CellText]]] = {}
    for r in results:
        by_table.setdefault(r.table, {}).setdefault(r.row, []).append(r)

    out_tables = []
    for ti, t in enumerate(tables):
        rows_out = []
        rows = by_table.get(ti, {})
        for ri in sorted(rows):
            cells = sorted(rows[ri], key=lambda c: c.col)
            rows_out.append({
                "row": ri,
                "is_header": any(c.is_header for c in cells),
                "cells": [{
                    "col": c.col,
                    "text": c.text,
                    "is_blank": c.blank,
                    "box_in_table": _box(c.box),
                    "box_in_page": _box(c.page_box),
                } for c in cells],
            })
        out_tables.append({
            "table": ti,
            "detection_label": t.label,
            "detection_confidence": round(t.score, 4),
            "box_in_page": _box(t.box),
            "n_rows": len(rows_out),
            "n_cols": max((len(r["cells"]) for r in rows_out), default=0),
            "rows": rows_out,
        })

    return {
        "index_base": 0,  # table, row and col are 0-based
        "source": {
            "file_name": doc.name,
            "sha256": doc.file_hash,
            "page": page_idx,
            "page_width": page.width,
            "page_height": page.height,
        },
        "tables": out_tables,
    }
