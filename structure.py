"""Step 3: table structure recognition with microsoft/table-transformer-structure-recognition."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

MODEL_ID = "microsoft/table-transformer-structure-recognition"
DEFAULT_THRESHOLD = 0.6

# model label -> (short name, overlay colour)
ELEMENTS = {
    "table row": ("Row", (30, 100, 255)),
    "table column": ("Column", (230, 40, 40)),
    "table column header": ("Header", (0, 170, 60)),
    "table projected row header": ("Projected row header", (150, 50, 200)),
    "table spanning cell": ("Spanning cell", (255, 140, 0)),
}


@dataclass(frozen=True)
class Element:
    kind: str                         # key of ELEMENTS
    score: float
    box: tuple[int, int, int, int]    # (x0, y0, x1, y1) relative to the table crop


@dataclass
class TableStructure:
    elements: list[Element] = field(default_factory=list)

    def of(self, kind: str) -> list[Element]:
        return [e for e in self.elements if e.kind == kind]


class StructureRecognizer:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = (
            TableTransformerForObjectDetection.from_pretrained(MODEL_ID)
            .to(self.device)
            .eval()
        )

    @torch.inference_mode()
    def recognize(self, crop: Image.Image, threshold: float = DEFAULT_THRESHOLD) -> TableStructure:
        rgb = crop.convert("RGB")
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        res = self.processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=[(rgb.height, rgb.width)]
        )[0]

        elements = []
        for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
            kind = self.model.config.id2label[label.item()]
            if kind not in ELEMENTS:  # skip the generic "table" box / "no object"
                continue
            x0, y0, x1, y1 = (int(round(v)) for v in box.tolist())
            elements.append(Element(kind, float(score), (max(0, x0), max(0, y0), min(crop.width, x1), min(crop.height, y1))))

        # rows read top-to-bottom, everything else left-to-right
        elements.sort(key=lambda e: (e.kind, e.box[1] if e.kind == "table row" else e.box[0], e.box[0]))
        return TableStructure(elements)


def draw_structure(crop: Image.Image, structure: TableStructure, kinds: set[str]) -> Image.Image:
    """Overlay on a copy of the crop; the crop itself is left untouched."""
    out = crop.convert("RGB")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    width = max(2, round(max(out.size) / 400))
    for e in structure.elements:
        if e.kind not in kinds:
            continue
        colour = ELEMENTS[e.kind][1]
        d.rectangle(e.box, outline=colour + (255,), fill=colour + (35,), width=width)
    return Image.alpha_composite(out.convert("RGBA"), overlay).convert("RGB")
