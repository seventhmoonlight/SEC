from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from build_parameter_labels import FIELDS, read_crop_number_rec_only
from sec_data_audit import classify_png, md5_file


OUT_FIELDS = [
    "sample_dir",
    "arw_path",
    "arw_md5",
    "algorithm",
    "integration_start",
    "integration_end",
    "peak_width_sec",
    "detection_threshold",
    "minimum_height",
    "parameter_png",
    "label_status",
    "label_source",
]


def iter_sample_dirs(root: Path, start: int | None, end: int | None, names: list[str] | None = None) -> list[Path]:
    dirs = [p for p in root.iterdir() if p.is_dir()]
    selected = []
    seen: set[Path] = set()
    for path in dirs:
        if not path.name.isdigit():
            continue
        value = int(path.name)
        if start is not None and value < start:
            continue
        if end is not None and value > end:
            continue
        selected.append(path)
        seen.add(path)
    for name in names or []:
        path = root / name
        if path.is_dir() and path not in seen:
            selected.append(path)
            seen.add(path)
    return sorted(selected, key=sort_key)


def sort_key(path: Path) -> tuple[int, int | str]:
    if path.name.isdigit():
        return 0, int(path.name)
    return 1, path.name


def parameter_png(sample_dir: Path) -> Path | None:
    for path in sorted(sample_dir.glob("*.png")):
        if classify_png(path) == "parameter_screenshot":
            return path
    return None


def first_arw(sample_dir: Path) -> Path | None:
    arws = sorted(sample_dir.glob("*.arw"))
    return arws[0] if arws else None


def extract_row(ocr: RapidOCR, sample_dir: Path, cache: dict) -> dict[str, object]:
    arw = first_arw(sample_dir)
    png = parameter_png(sample_dir)
    row: dict[str, object] = {
        "sample_dir": str(sample_dir),
        "arw_path": "" if arw is None else str(arw),
        "arw_md5": "" if arw is None else md5_file(arw),
        "algorithm": "",
        "integration_start": "",
        "integration_end": "",
        "peak_width_sec": "",
        "detection_threshold": "",
        "minimum_height": "",
        "parameter_png": "" if png is None else str(png),
        "label_status": "ok",
        "label_source": "ocr_parameter_screenshot",
    }
    problems = []
    if arw is None:
        problems.append("missing_arw")
    if png is None:
        problems.append("missing_parameter_png")
    else:
        img = Image.open(png).convert("RGB")
        for field, box in FIELDS.items():
            crop = img.crop(box)
            value, text, conf = read_crop_number_rec_only(ocr, crop, cache)
            row[f"{field}_ocr_text"] = text
            row[f"{field}_ocr_conf"] = conf
            if field == "start_min":
                row["integration_start"] = "" if value is None else value
            elif field == "end_min":
                row["integration_end"] = "" if value is None else value
            elif field in row:
                row[field] = "" if value is None else value
            if value is None and field in {"peak_width_sec", "detection_threshold"}:
                problems.append(f"missing_{field}")
    if problems:
        row["label_status"] = "|".join(problems)
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys = OUT_FIELDS[:]
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract OCR training labels from sample parameter screenshots.")
    parser.add_argument("root")
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--names", nargs="*", help="Additional non-numeric sample directory names, for example S1 S2 S3.")
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    ocr = RapidOCR()
    cache: dict = {}
    rows = [extract_row(ocr, sample_dir, cache) for sample_dir in iter_sample_dirs(Path(args.root), args.start, args.end, args.names)]
    write_csv(Path(args.csv), rows)
    print({"rows": len(rows), "ok": sum(1 for row in rows if row["label_status"] == "ok")})


if __name__ == "__main__":
    main()
