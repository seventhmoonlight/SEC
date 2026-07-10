from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR


AUDIT_CSV = Path("outputs/tables/sec_sample_audit.csv")
TABLE_DIR = Path("outputs/tables")
DEBUG_DIR = Path("outputs/debug/ocr_crops")

FIELDS = {
    "start_min": (128, 108, 210, 138),
    "end_min": (344, 108, 426, 138),
    "peak_width_sec": (128, 142, 210, 172),
    "detection_threshold": (344, 142, 426, 172),
    "minimum_area": (128, 230, 210, 260),
    "minimum_height": (344, 230, 426, 260),
}


def parse_number(text: str) -> float | None:
    clean = text.strip()
    clean = clean.replace("O", "0").replace("o", "0")
    clean = clean.replace("l", "1").replace("I", "1")
    clean = clean.replace(" ", "")
    sci_missing_e = re.search(r"([-+]?\d+(?:\.\d+)?)\+(\d+)", clean)
    if sci_missing_e and "e" not in clean.lower():
        try:
            return float(sci_missing_e.group(1)) * (10 ** int(sci_missing_e.group(2)))
        except ValueError:
            pass
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", clean)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def result_center(item: list) -> tuple[float, float]:
    pts = item[0]
    return (
        sum(float(p[0]) for p in pts) / len(pts),
        sum(float(p[1]) for p in pts) / len(pts),
    )


def read_box_number(results: list, box: tuple[int, int, int, int]) -> tuple[float | None, str, float]:
    x1, y1, x2, y2 = box
    hits = []
    for item in results or []:
        cx, cy = result_center(item)
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            hits.append(item)
    if not hits:
        return None, "", 0.0
    hits.sort(key=lambda item: (result_center(item)[1], result_center(item)[0]))
    text = " ".join(str(item[1]) for item in hits)
    conf = max(float(item[2]) for item in hits)
    return parse_number(text), text, conf


def crop_hash(img: Image.Image) -> str:
    grey = img.convert("L").resize((96, 32))
    return hashlib.md5(grey.tobytes()).hexdigest()


def read_crop_number_rec_only(
    ocr: RapidOCR,
    crop: Image.Image,
    cache: dict[str, tuple[float | None, str, float]],
) -> tuple[float | None, str, float]:
    crop = crop.resize((crop.width * 4, crop.height * 4))
    h = crop_hash(crop)
    if h in cache:
        return cache[h]
    arr = np.asarray(crop.convert("RGB"))
    result, _ = ocr(arr, use_det=False, use_cls=False, use_rec=True)
    if not result:
        out = (None, "", 0.0)
    else:
        text = " ".join(str(item[0]) for item in result)
        conf = max(float(item[1]) for item in result)
        out = (parse_number(text), text, conf)
    cache[h] = out
    return out


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    ocr = RapidOCR()
    crop_cache: dict[str, tuple[float | None, str, float]] = {}
    rows = list(csv.DictReader(AUDIT_CSV.open("r", encoding="utf-8-sig")))
    out_rows: list[dict[str, object]] = []

    for row in rows:
        pngs = [p for p in row.get("parameter_pngs", "").split(";") if p]
        out = {
            "sample_dir": row["sample_dir"],
            "sample_name": row["sample_name"],
            "arw_path": row["arw_path"],
            "arw_size": row["arw_size"],
            "arw_md5": row["arw_md5"],
            "duplicate_group_size": row.get("duplicate_group_size", "1"),
            "parameter_png": pngs[0] if pngs else "",
            "label_status": "ok",
        }
        problems = []
        if not pngs:
            problems.append("missing_parameter_png")
        else:
            img = Image.open(pngs[0]).convert("RGB")
            for field, box in FIELDS.items():
                crop = img.crop(box)
                value, text, conf = read_crop_number_rec_only(ocr, crop, crop_cache)
                out[field] = "" if value is None else value
                out[f"{field}_ocr_text"] = text
                out[f"{field}_ocr_conf"] = conf
                if value is None and field in {"peak_width_sec", "detection_threshold"}:
                    problems.append(f"missing_{field}")
        if str(row.get("status", "")).find("bad_arw") >= 0:
            problems.append("bad_arw")
        if problems:
            out["label_status"] = "|".join(problems)
        out_rows.append(out)

    write_csv(TABLE_DIR / "sec_parameter_labels_ocr.csv", out_rows)
    print("rows", len(out_rows))
    print("ok", sum(1 for r in out_rows if r["label_status"] == "ok"))
    print("bad", sum(1 for r in out_rows if r["label_status"] != "ok"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
