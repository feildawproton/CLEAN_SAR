"""
audit/claude_code_opus_5/a12_sicd_standard_check.py

Consults the authoritative NGA SICD documents in references/ and the XSD in
schemas/ for the normative definitions of:
    Grid/Row|Col/ImpRespWid, ImpRespBW, WgtType/WindowName, WgtFunct
and reports every passage that constrains their relationship, so the DiffPFA
writer can be judged against the standard rather than against convention.
"""
import re
import sys
from pypdf import PdfReader

OUT = []


def log(s=""):
    print(s)
    OUT.append(str(s))


DOCS = {
    "DIDD (0024-1) Design & Impl Description Doc": "references/NGA.STND.0024-1_1.5_SICD_DIDD.pdf",
    "IPDD (0024-3) Image Products Description Doc": "references/NGA.STND.0024-3_1.5_SICD_IPDD.pdf",
}

TERMS = ["ImpRespWid", "ImpRespBW", "WgtFunct", "WgtType"]

for label, path in DOCS.items():
    log("=" * 90)
    log(f"{label}")
    log("=" * 90)
    reader = PdfReader(f"/home/feildaw/CLEAN_SAR/{path}")
    log(f"  {len(reader.pages)} pages")
    hits = 0
    for pno, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            continue
        norm = re.sub(r"[ \t]+", " ", txt)
        # keep pages that define the relationship, not every mention
        if not any(t in norm for t in TERMS):
            continue
        interesting = (
            ("ImpRespWid" in norm and "ImpRespBW" in norm)
            or ("broadening" in norm.lower())
            or ("0.886" in norm)
            or ("WgtFunct" in norm and "WgtType" in norm)
        )
        if not interesting:
            continue
        hits += 1
        if hits > 14:
            break
        log()
        log(f"  ---- page {pno + 1} ----")
        for line in norm.splitlines():
            ls = line.strip()
            if not ls:
                continue
            if any(t in ls for t in TERMS) or "0.886" in ls or "broaden" in ls.lower() \
               or "IPR" in ls or "impulse response" in ls.lower():
                log(f"    {ls[:150]}")
    log()

# ------------------------------------------------------------------ schema
log("=" * 90)
log("XSD SCHEMA: declared types / constraints for the four fields")
log("=" * 90)
for sch in ["schemas/NGA.STND.0024-4_1.5_Schema.xsd", "schemas/SICD_schema_V1.3.0_2021_11_30.xsd"]:
    log(f"  ---- {sch} ----")
    txt = open(f"/home/feildaw/CLEAN_SAR/{sch}").read()
    for t in TERMS:
        for m in re.finditer(rf'name="{t}"[^>]*', txt):
            log(f"    {m.group(0)[:170]}")
    log()

with open("/home/feildaw/CLEAN_SAR/audit/claude_code_opus_5/a12_sicd_standard_check.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
