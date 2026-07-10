from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


AUDIT_CSV = Path("outputs/tables/sec_sample_audit.csv")
DEBUG_DIR = Path("outputs/debug")
TABLE_DIR = Path("outputs/tables")

FIELDS = {
    "start_min": (128, 108, 210, 138),
    "end_min": (344, 108, 426, 138),
    "peak_width_sec": (128, 142, 210, 172),
    "detection_threshold": (344, 142, 426, 172),
    "minimum_area": (128, 230, 210, 260),
    "minimum_height": (344, 230, 426, 260),
}


def image_hash(img: Image.Image) -> str:
    grey = img.convert("L").resize((96, 32))
    return hashlib.md5(grey.tobytes()).hexdigest()


def crop_field(img: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    # Empower screenshots vary by a few pixels. Use fixed coordinates because the
    # LC Processing Method window content is anchored consistently.
    w, h = img.size
    x1, y1, x2, y2 = box
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    return img.crop((x1, y1, x2, y2))


def main() -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(AUDIT_CSV.open("r", encoding="utf-8-sig")))
    unique: dict[tuple[str, str], dict[str, object]] = {}
    assignments: list[dict[str, str]] = []

    for row in rows:
        pngs = [p for p in row.get("parameter_pngs", "").split(";") if p]
        if not pngs:
            continue
        png = Path(pngs[0])
        img = Image.open(png).convert("RGB")
        for field, box in FIELDS.items():
            crop = crop_field(img, box)
            h = image_hash(crop)
            key = (field, h)
            if key not in unique:
                crop_path = DEBUG_DIR / f"crop_{field}_{len(unique):03d}.png"
                crop.save(crop_path)
                unique[key] = {
                    "field": field,
                    "hash": h,
                    "crop_path": str(crop_path),
                    "example_sample_dir": row["sample_dir"],
                    "example_png": str(png),
                    "count": 0,
                }
            unique[key]["count"] = int(unique[key]["count"]) + 1
            assignments.append({
                "sample_dir": row["sample_dir"],
                "field": field,
                "hash": h,
                "crop_path": str(unique[key]["crop_path"]),
            })

    write_csv(TABLE_DIR / "parameter_crop_unique_values.csv", list(unique.values()))
    write_csv(TABLE_DIR / "parameter_crop_assignments.csv", assignments)
    make_contact_sheet(list(unique.values()), DEBUG_DIR / "parameter_crop_contact_sheet.png")
    print(f"unique field crops: {len(unique)}")
    print(f"contact sheet: {DEBUG_DIR / 'parameter_crop_contact_sheet.png'}")


def make_contact_sheet(rows: list[dict[str, object]], out: Path) -> None:
    thumb_w, thumb_h = 164, 60
    label_h = 44
    cols = 4
    rows_n = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, max(1, rows_n) * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for i, row in enumerate(rows):
        x = (i % cols) * thumb_w
        y = (i // cols) * (thumb_h + label_h)
        crop = Image.open(str(row["crop_path"])).convert("RGB")
        crop = crop.resize((thumb_w - 16, thumb_h - 12))
        sheet.paste(crop, (x + 8, y + 4))
        label = f"{i:03d} {row['field']} n={row['count']}"
        draw.text((x + 8, y + thumb_h), label, fill="black", font=font)
        draw.text((x + 8, y + thumb_h + 14), str(row["example_sample_dir"])[:28], fill="black", font=font)
    sheet.save(out)


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
