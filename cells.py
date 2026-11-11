"""Step 4: build cell boxes from row x column intersections and crop them."""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from structure import TableStructure


@dataclass(frozen=True)
class Cell:
    row: int                          # 0-based, top to bottom
    col: int                          # 0-based, left to right
    box: tuple[int, int, int, int]    # (x0, y0, x1, y1) relative to the table crop
    crop: Image.Image                 # cut from the full-resolution table crop
    is_header: bool                   # row lies inside a detected column header


def _intersect(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _overlap_ratio(inner, outer) -> float:
    i = _intersect(inner, outer)
    if not i:
        return 0.0
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return ((i[2] - i[0]) * (i[3] - i[1])) / area if area else 0.0


def build_cells(table_crop: Image.Image, structure: TableStructure, padding: int = 2) -> list[Cell]:
    """One cell per (row, column) pair, in row-major order."""
    rows = sorted(structure.of("table row"), key=lambda e: e.box[1])
    cols = sorted(structure.of("table column"), key=lambda e: e.box[0])
    headers = [h.box for h in structure.of("table column header")]

    W, H = table_crop.size
    cells = []
    for r, row in enumerate(rows):
        is_header = any(_overlap_ratio(row.box, h) >= 0.5 for h in headers)
        for c, col in enumerate(cols):
            box = _intersect(row.box, col.box)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            box = (max(0, x0 - padding), max(0, y0 - padding), min(W, x1 + padding), min(H, y1 + padding))
            cells.append(Cell(r, c, box, table_crop.crop(box), is_header))
    return cells
