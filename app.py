import hashlib
import json

import streamlit as st

from detect import DEFAULT_THRESHOLD, DetectedTable, TableDetector
from structure import DEFAULT_THRESHOLD as STRUCT_THRESHOLD, ELEMENTS, StructureRecognizer, draw_structure
from cells import build_cells
from ocr import CellText, HandwritingOCR
from structured import build_json
from excel import build_workbook
from validate import validate
from verify import DEFAULT_MODEL, DEFAULT_REGION, call_bedrock
from ingest import InvoiceDoc, load_invoice

st.set_page_config(page_title="Invoice to Excel", layout="wide")
st.title("Invoice to Excel")
st.caption("Step 1 - upload  |  Step 2 - table detection  |  Step 3 - structure  |  Step 4 - cells  |  Step 5 - OCR  |  Step 6 - JSON  |  Step 7 - Bedrock  |  Step 8 - validation  |  Step 9 - Excel")


@st.cache_resource(show_spinner="Loading TrOCR model (first run downloads it)...")
def get_ocr() -> HandwritingOCR:
    return HandwritingOCR()


@st.cache_resource(show_spinner="Loading structure model (first run downloads it)...")
def get_recognizer() -> StructureRecognizer:
    return StructureRecognizer()


@st.cache_resource(show_spinner="Loading table detection model (first run downloads it)...")
def get_detector() -> TableDetector:
    return TableDetector()


uploaded = st.file_uploader(
    "Upload invoice", type=["png", "jpg", "jpeg", "webp", "pdf"]
)

if uploaded is None:
    st.session_state.pop("doc", None)
    st.info("Upload a file to begin.")
    st.stop()

data = uploaded.getvalue()
doc: InvoiceDoc | None = st.session_state.get("doc")

# Decode once per distinct file; reruns reuse the in-memory original.
if doc is None or doc.file_hash != hashlib.sha256(data).hexdigest():
    try:
        with st.spinner("Reading file..."):
            doc = load_invoice(uploaded.name, data)
    except Exception as e:
        st.session_state.pop("doc", None)
        st.error(f"Could not read this file: {e}")
        st.stop()
    st.session_state["doc"] = doc
    st.session_state["page_idx"] = 0

page_idx = 0
if len(doc.pages) > 1:
    page_idx = st.selectbox(
        "Page", range(len(doc.pages)), format_func=lambda i: f"Page {i + 1} of {len(doc.pages)}"
    )
st.session_state["page_idx"] = page_idx

img = doc.pages[page_idx]
c1, c2 = st.columns([3, 1])
with c1:
    st.image(img, caption=doc.name, width="stretch")  # display only; stored image untouched
with c2:
    st.metric("Width (px)", img.width)
    st.metric("Height (px)", img.height)
    st.write(f"**Type:** {doc.source_type.upper()}  \n**Mode:** {img.mode}  \n**Size:** {len(doc.original_bytes) / 1024:.1f} KB")
    st.caption(f"SHA-256: `{doc.file_hash[:12]}...`")

st.divider()
st.subheader("Detected tables")
threshold = st.slider("Confidence threshold", 0.3, 0.99, DEFAULT_THRESHOLD, 0.01)

# Cache detections per (file, page, threshold) so reruns don't re-run the model.
key = (doc.file_hash, page_idx, threshold)
if st.session_state.get("det_key") != key:
    with st.spinner("Detecting tables..."):
        try:
            st.session_state["tables"] = get_detector().detect(img, threshold)
        except Exception as e:
            st.session_state.pop("det_key", None)
            st.error(f"Table detection failed: {e}")
            st.stop()
    st.session_state["det_key"] = key

tables: list[DetectedTable] = st.session_state["tables"]
if not tables:
    st.warning("No tables found. Try lowering the confidence threshold.")
else:
    st.success(f"Found {len(tables)} table(s).")

    st.divider()
    st.subheader("Table structure")
    s_threshold = st.slider("Structure confidence threshold", 0.3, 0.99, STRUCT_THRESHOLD, 0.01)
    shown = st.multiselect(
        "Show on overlay",
        list(ELEMENTS),
        default=["table row", "table column", "table column header"],
        format_func=lambda k: ELEMENTS[k][0],
    )

    skey = (key, s_threshold)
    if st.session_state.get("struct_key") != skey:
        with st.spinner("Recognising rows and columns..."):
            try:
                rec = get_recognizer()
                st.session_state["structures"] = [rec.recognize(t.crop, s_threshold) for t in tables]
            except Exception as e:
                st.session_state.pop("struct_key", None)
                st.error(f"Structure recognition failed: {e}")
                st.stop()
        st.session_state["struct_key"] = skey
    structures = st.session_state["structures"]

    st.divider()
    st.subheader("Cells")
    cell_pad = st.slider("Cell padding (px)", 0, 10, 2, help="Extra pixels around each cell so strokes at the edge aren't cut.")

    cells_per_table = [build_cells(t.crop, stc, cell_pad) for t, stc in zip(tables, structures)]

    st.divider()
    st.subheader("Handwriting OCR")
    okey = (skey, cell_pad)
    n_cells = sum(len(c) for c in cells_per_table)
    if st.button(f"Run OCR on {n_cells} cells", disabled=n_cells == 0, type="primary"):
        bar = st.progress(0.0, text="Reading cells...")
        try:
            res = get_ocr().read_cells(tables, cells_per_table, progress=lambda f: bar.progress(f, text="Reading cells..."))
        except Exception as e:
            bar.empty()
            st.error(f"OCR failed: {e}")
            st.stop()
        bar.empty()
        st.session_state["ocr"] = res
        st.session_state["ocr_key"] = okey
    ocr_results: list[CellText] = st.session_state["ocr"] if st.session_state.get("ocr_key") == okey else []
    if not ocr_results and n_cells:
        st.caption("OCR is run on demand (it is slow on CPU). Results are cleared if the file, page, thresholds or padding change.")
    text_at = {(r.table, r.row, r.col): r for r in ocr_results}
    if ocr_results:
        st.dataframe(
            [{"table": r.table + 1, "row": r.row + 1, "col": r.col + 1, "text": r.text, "blank": r.blank,
              "header": r.is_header, "box (in table)": r.box, "box (in page)": r.page_box}
             for r in ocr_results],
            hide_index=True, width="stretch",
        )

    if ocr_results:
        st.divider()
        st.subheader("Structured OCR JSON")
        payload = build_json(doc, page_idx, tables, ocr_results)
        st.json(payload, expanded=2)
        st.download_button("Download JSON", json.dumps(payload, indent=2),
                           file_name=f"{doc.name.rsplit('.', 1)[0]}_ocr.json", mime="application/json")

        st.divider()
        st.subheader("Bedrock verification")
        st.caption("Sends the page image and the JSON above to Amazon Bedrock, using your AWS credentials "
                   "(environment variables, `aws configure`, or a named profile).")
        v1, v2, v3 = st.columns(3)
        model_id = v1.text_input("Bedrock model ID", DEFAULT_MODEL,
                                 help="A multimodal Claude model enabled in your AWS account.")
        region = v2.text_input("AWS region", DEFAULT_REGION)
        profile = v3.text_input("AWS profile (optional)", "")
        pkey = (hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), model_id, region)
        if st.button("Verify with Bedrock", type="primary"):
            with st.spinner("Verifying with Bedrock..."):
                try:
                    raw, img_note = call_bedrock(doc, page_idx, payload, model_id, region, profile or None)
                    st.session_state["bedrock"] = (raw, img_note)
                    st.session_state["bedrock_key"] = pkey
                except Exception as e:
                    st.session_state.pop("bedrock_key", None)
                    st.error(f"Bedrock call failed: {e}")

        if st.session_state.get("bedrock_key") == pkey:
            raw, img_note = st.session_state["bedrock"]

            st.divider()
            st.subheader("Validation")
            errors, verified, changes = validate(payload, raw, model_id)
            if errors:
                st.error("Validation failed. The Bedrock output was rejected and will not be used.")
                for e in errors:
                    st.markdown(f"- {e}")
                with st.expander("Raw Bedrock reply"):
                    st.code(raw, language="json")
            else:
                n_cells = sum(len(r["cells"]) for t in payload["tables"] for r in t["rows"])
                n_rows = sum(len(t["rows"]) for t in payload["tables"])
                st.success(
                    f"Validation passed: valid JSON; all {len(payload['tables'])} table(s), {n_rows} row(s) and "
                    f"{n_cells} cell(s) preserved with identical structure and boxes. "
                    f"{len(changes)} cell(s) corrected. Image sent: {img_note}."
                )
                if changes:
                    st.dataframe([{"table": c["table"] + 1, "row": c["row"] + 1, "col": c["col"] + 1,
                                   "TrOCR": c["ocr_text"], "verified": c["verified_text"]} for c in changes],
                                 hide_index=True, width="stretch")
                st.markdown("**Verified JSON**")
                st.json(verified, expanded=2)
                st.download_button("Download verified JSON", json.dumps(verified, indent=2, ensure_ascii=False),
                                   file_name=f"{doc.name.rsplit('.', 1)[0]}_verified.json", mime="application/json")

                st.divider()
                st.subheader("Excel export")
                try:
                    xlsx = build_workbook(verified)
                except Exception as e:
                    st.error(f"Excel export failed: {e}")
                else:
                    st.caption(f"{len(verified['tables'])} worksheet(s), one per detected table. "
                               "Only the validated text is written; the raw TrOCR output is not included.")
                    st.download_button(
                        "Download Excel (.xlsx)", xlsx,
                        file_name=f"{doc.name.rsplit('.', 1)[0]}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                    )

    for i, (t, stc) in enumerate(zip(tables, structures), 1):
        st.markdown(f"### Table {i} - detection confidence **{t.score:.1%}**")
        counts = " | ".join(f"{ELEMENTS[k][0]}s: **{len(stc.of(k))}**" for k in ELEMENTS)
        st.write(counts)
        left, right = st.columns([3, 2])
        with left:
            st.image(draw_structure(t.crop, stc, set(shown)), width="stretch",
                     caption=f"{t.crop.width}x{t.crop.height}px (original crop is unchanged)")
        with right:
            st.dataframe(
                [
                    {"type": ELEMENTS[e.kind][0], "confidence": round(e.score, 3),
                     "x0": e.box[0], "y0": e.box[1], "x1": e.box[2], "y1": e.box[3]}
                    for e in stc.elements if e.kind in shown
                ],
                hide_index=True, width="stretch",
            )
        with st.expander("Original crop"):
            st.image(t.crop, width="stretch")

        cells = cells_per_table[i - 1]
        if not cells:
            st.warning("No rows/columns detected for this table, so no cells could be built. Try lowering the structure threshold.")
            continue
        n_rows = max(c.row for c in cells) + 1
        n_cols = max(c.col for c in cells) + 1
        st.markdown(f"**Table {i} cells:** {n_rows} rows x {n_cols} columns = {len(cells)} cells (labels are 1-based)")
        by_pos = {(c.row, c.col): c for c in cells}
        with st.expander(f"Cell grid - Table {i}", expanded=True):
            for r in range(n_rows):
                grid = st.columns(n_cols)
                for c in range(n_cols):
                    cell = by_pos.get((r, c))
                    if cell is None:
                        continue
                    with grid[c]:
                        tag = " (header)" if cell.is_header else ""
                        st.image(cell.crop, width="stretch", caption=f"Row {r + 1}, Col {c + 1}{tag}")
                        res = text_at.get((i - 1, r, c))
                        if res is not None:
                            st.markdown(f"**{res.text}**" if res.text else "_(empty)_")
