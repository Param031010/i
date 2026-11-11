"""Step 2: table detection with microsoft/table-transformer-detection."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

MODEL_ID = "microsoft/table-transformer-detection"
DEFAULT_THRESHOLD = 0.7
CROP_PADDING = 20  # px added around each box so borders/headers aren't clipped


@dataclass(frozen=True)
class DetectedTable:
    label: str
    score: float
    box: tuple[int, int, int, int]  # (x0, y0, x1, y1) in original-image pixels
    crop: Image.Image               # cut from the original, full resolution


class TableDetector:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = (
            TableTransformerForObjectDetection.from_pretrained(MODEL_ID)
            .to(self.device)
            .eval()
        )

    @torch.inference_mode()
    def detect(
        self,
        image: Image.Image,
        threshold: float = DEFAULT_THRESHOLD,
        padding: int = CROP_PADDING,
    ) -> list[DetectedTable]:
        rgb = image.convert("RGB")  # model input only; `image` itself is untouched
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        res = self.processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=[(rgb.height, rgb.width)]
        )[0]

        tables = []
        for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
            x0, y0, x1, y1 = box.tolist()
            x0 = max(0, int(x0) - padding)
            y0 = max(0, int(y0) - padding)
            x1 = min(image.width, int(x1) + padding)
            y1 = min(image.height, int(y1) + padding)
            tables.append(
                DetectedTable(
                    label=self.model.config.id2label[label.item()],
                    score=float(score),
                    box=(x0, y0, x1, y1),
                    crop=image.crop((x0, y0, x1, y1)),
                )
            )
        tables.sort(key=lambda t: (t.box[1], t.box[0]))  # top-to-bottom, left-to-right
        return tables
