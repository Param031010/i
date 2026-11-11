"""Step 8: validate the Bedrock reply before it is accepted as the verified JSON."""
from __future__ import annotations

import json
import re

from verify import merge_verified

# Fields a correction is allowed to touch on a cell. Everything else must equal the Step 6 JSON.
_CELL_EDITABLE = {"text", "is_blank", "ocr_text", "corrected"}


def parse_reply(raw: str) -> dict:
    """Parse the model's reply as JSON. Tolerates a code fence or surrounding prose; nothing else."""
    text = raw.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if m:
        text = m.group(1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        i, j = text.find("{"), text.rfind("}")
        if i == -1 or j <= i:
            raise ValueError("The reply contains no JSON object.")
        try:
            data = json.loads(text[i : j + 1])
        except json.JSONDecodeError as e:
            raise ValueError(f"The reply is not valid JSON: {e}")
    if not isinstance(data, dict):
        raise ValueError("The reply is JSON but not an object.")
    return data


def _original_cells(original: dict) -> list[tuple[int, int, int]]:
    return [(t["table"], r["row"], c["col"])
            for t in original["tables"] for r in t["rows"] for c in r["cells"]]


def validate_reply(original: dict, reply: dict) -> list[str]:
    """Check the model's reply covers exactly the cells detected in Step 6."""
    errors = []
    cells = reply.get("cells")
    if not isinstance(cells, list):
        return ['The reply has no "cells" list.']

    seen, order = {}, []
    for n, c in enumerate(cells):
        if not isinstance(c, dict):
            errors.append(f"cells[{n}] is not an object.")
            continue
        try:
            key = tuple(c[k] for k in ("table", "row", "col"))
        except KeyError as e:
            errors.append(f"cells[{n}] is missing {e}.")
            continue
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in key):
            errors.append(f"cells[{n}] has non-integer table/row/col: {key}.")
            continue
        if not isinstance(c.get("text"), str):
            errors.append(f"Cell {_label(key)} has no string \"text\".")
        if key in seen:
            errors.append(f"Cell {_label(key)} appears more than once.")
        seen[key] = c
        order.append(key)

    expected = _original_cells(original)
    missing = [k for k in expected if k not in seen]
    extra = [k for k in seen if k not in set(expected)]
    if missing:
        errors.append(f"{len(missing)} detected cell(s) missing from the reply, e.g. {_label(missing[0])}.")
    if extra:
        errors.append(f"{len(extra)} cell(s) in the reply were never detected (invented), e.g. {_label(extra[0])}.")

    # Cell values: a correction may fix text, but must not wipe out text OCR actually read.
    for t in original["tables"]:
        for r in t["rows"]:
            for c in r["cells"]:
                key = (t["table"], r["row"], c["col"])
                new = seen.get(key, {}).get("text")
                if isinstance(new, str) and c["text"].strip() and not new.strip():
                    errors.append(f"Cell {_label(key)}: OCR text {c['text']!r} was replaced with empty text.")
    return errors


def validate_output(original: dict, verified: dict) -> list[str]:
    """Compare the merged verified JSON to the Step 6 JSON: same tables, rows, columns, boxes."""
    errors = []
    try:
        json.loads(json.dumps(verified))
    except (TypeError, ValueError) as e:
        return [f"The verified JSON cannot be serialised: {e}"]

    for k in original:
        if k in ("tables", "verification"):
            continue
        if verified.get(k) != original[k]:
            errors.append(f'Top-level field "{k}" changed.')

    ot, vt = original["tables"], verified.get("tables")
    if not isinstance(vt, list) or len(vt) != len(ot):
        return errors + [f"Table count changed: {len(ot)} detected, {len(vt) if isinstance(vt, list) else 0} returned."]

    for a, b in zip(ot, vt):
        tl = f"Table {a['table'] + 1}"
        for k in a:
            if k != "rows" and a[k] != b.get(k):
                errors.append(f'{tl}: "{k}" changed.')
        if len(a["rows"]) != len(b["rows"]):
            errors.append(f"{tl}: row count changed ({len(a['rows'])} -> {len(b['rows'])}).")
            continue
        for ra, rb in zip(a["rows"], b["rows"]):
            rl = f"{tl}, row {ra['row'] + 1}"
            for k in ra:
                if k != "cells" and ra[k] != rb.get(k):
                    errors.append(f'{rl}: "{k}" changed.')
            if [c["col"] for c in ra["cells"]] != [c.get("col") for c in rb["cells"]]:
                errors.append(f"{rl}: columns changed or reordered.")
                continue
            for ca, cb in zip(ra["cells"], rb["cells"]):
                for k in ca:
                    if k not in _CELL_EDITABLE and ca[k] != cb.get(k):
                        errors.append(f'{rl}, col {ca["col"] + 1}: "{k}" changed.')
                if not isinstance(cb.get("text"), str):
                    errors.append(f'{rl}, col {ca["col"] + 1}: text is not a string.')
    return errors


def validate(original: dict, raw_reply: str, model: str):
    """Full Step 8 check. Returns (errors, verified_json_or_None, changes)."""
    try:
        reply = parse_reply(raw_reply)
    except ValueError as e:
        return [str(e)], None, []
    errors = validate_reply(original, reply)
    if errors:
        return errors, None, []
    verified, changes = merge_verified(original, reply, model)
    errors = validate_output(original, verified)
    if errors:
        return errors, None, []
    return [], verified, changes


def _label(key) -> str:
    t, r, c = key
    return f"(table {t + 1}, row {r + 1}, col {c + 1})"
