"""Step 9: write the validated verified JSON to an .xlsx file (one worksheet per table)."""
from __future__ import annotations

from io import BytesIO

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

_HEADER_FILL = PatternFill("solid", start_color="D9E1F2")
_THIN = Side(style="thin", color="BBBBBB")
_BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)


def build_workbook(verified: dict) -> bytes:
    """`verified` must be the output of Step 8 validation. Only table/row/col positions and the
    final cell text are written; TrOCR text, boxes and confidences are deliberately left out."""
    if "verification" not in verified:
        raise ValueError("Refusing to export: this JSON has not been through Bedrock verification.")
    tables = verified["tables"]
    if not tables:
        raise ValueError("No tables to export.")

    wb = Workbook()
    wb.remove(wb.active)
    for t in tables:
        ws = wb.create_sheet(f"Table {t['table'] + 1}")  # 'Table N' is well under Excel's 31-char limit
        widths: dict[int, int] = {}
        for r in t["rows"]:
            for c in r["cells"]:
                cell = ws.cell(row=r["row"] + 1, column=c["col"] + 1)
                # Store as text exactly as read: no number/date/formula coercion ("007", "=x", "1/2").
                cell.value = c["text"]
                cell.data_type = "s"
                cell.border = _BORDER
                cell.alignment = Alignment(vertical="top", wrap_text=True)
                if r["is_header"]:
                    cell.font = Font(bold=True)
                    cell.fill = _HEADER_FILL
                widths[c["col"]] = max(widths.get(c["col"], 0), len(c["text"]))
        for col, w in widths.items():
            ws.column_dimensions[get_column_letter(col + 1)].width = min(max(10, w + 2), 60)

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()
