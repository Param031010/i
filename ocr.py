"""Step 5: handwritten OCR on cell crops with microsoft/trocr-base-handwritten."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image, ImageStat
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from cells import Cell
from detect import DetectedTable

MODEL_ID = "microsoft/trocr-base-handwritten"
BLANK_STD = 6.0  # grayscale std-dev below this = empty cell; TrOCR hallucinates on blanks


@dataclass(frozen=True)
class CellText:
    table: int                              # 0-based table index on the page
    row: int                                # 0-based
    col: int                                # 0-based
    box: tuple[int, int, int, int]          # relative to the table crop
    page_box: tuple[int, int, int, int]     # in original page-image pixels
    text: str
    blank: bool
    is_header: bool


def _is_blank(img: Image.Image) -> bool:
    return ImageStat.Stat(img.convert("L")).stddev[0] < BLANK_STD


class HandwritingOCR:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = TrOCRProcessor.from_pretrained(MODEL_ID)
        self.model = (
            VisionEncoderDecoderModel.from_pretrained(MODEL_ID).to(self.device).eval()
        )

    @torch.inference_mode()
    def read(self, images: list[Image.Image], batch_size: int = 8) -> list[str]:
        out: list[str] = []
        for i in range(0, len(images), batch_size):
            batch = [im.convert("RGB") for im in images[i : i + batch_size]]
            pixel = self.processor(images=batch, return_tensors="pt").pixel_values.to(self.device)
            ids = self.model.generate(pixel, max_new_tokens=64)
            out += [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]
        return out

    def read_cells(self, tables: list[DetectedTable], cells_per_table: list[list[Cell]],
                   progress=None) -> list[CellText]:
        jobs = [(ti, c) for ti, cells in enumerate(cells_per_table) for c in cells]
        blank = [_is_blank(c.crop) for _, c in jobs]
        todo = [i for i, b in enumerate(blank) if not b]
        texts = {}
        step = 8
        for s in range(0, len(todo), step):
            idx = todo[s : s + step]
            for i, t in zip(idx, self.read([jobs[i][1].crop for i in idx], step)):
                texts[i] = t
            if progress:
                progress(min(1.0, (s + step) / max(1, len(todo))))

        results = []
        for i, (ti, c) in enumerate(jobs):
            ox, oy = tables[ti].box[0], tables[ti].box[1]
            x0, y0, x1, y1 = c.box
            results.append(CellText(ti, c.row, c.col, c.box, (x0 + ox, y0 + oy, x1 + ox, y1 + oy),
                                    texts.get(i, ""), blank[i], c.is_header))
        return results
