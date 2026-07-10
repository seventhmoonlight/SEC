from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


ROOTS = [Path("data"), Path("extra")]
OUTPUT_DIR = Path("outputs/reports")
TABLE_DIR = Path("outputs/tables")


@dataclass
class TracePeak:
    peak_id: int
    time: float | None
    start: float | None
    end: float | None
    height: float | None
    area_abs: float | None
    area_total: float | None
    width: float | None


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_sample_dirs() -> Iterable[Path]:
    for root in ROOTS:
        if not root.exists():
            continue
        yield from sorted(p for p in root.iterdir() if p.is_dir())


def first_file(files: list[Path]) -> str:
    return str(files[0]) if files else ""


def classify_png(path: Path) -> str:
    """Heuristic split for Empower method screenshots vs chromatogram plots."""
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return "bad_png"
    w, h = img.size
    arr = np.asarray(img)
    # Method window screenshots are usually tall grey UI windows with many borders.
    # Chromatogram plots are usually shorter and mostly white plot areas.
    grey = np.mean(np.abs(arr[:, :, 0].astype(float) - arr[:, :, 1].astype(float))) < 8
    light_grey_ratio = np.mean(
        (arr[:, :, 0] > 180) & (arr[:, :, 0] < 245) &
        (arr[:, :, 1] > 180) & (arr[:, :, 1] < 245) &
        (arr[:, :, 2] > 180) & (arr[:, :, 2] < 245)
    )
    dark_ratio = np.mean(np.all(arr < 80, axis=2))
    white_ratio = np.mean(np.all(arr > 245, axis=2))
    if h >= 380 and 420 <= w <= 540 and grey and light_grey_ratio > 0.35:
        return "parameter_screenshot"
    if 420 <= w <= 540 and grey and light_grey_ratio > 0.35 and dark_ratio > 0.02 and white_ratio > 0.05:
        return "parameter_screenshot"
    if h <= 330 or dark_ratio < 0.015:
        return "chromatogram_plot"
    return "unknown_png"


def parse_rpt(path: Path) -> tuple[np.ndarray, np.ndarray, list[TracePeak]]:
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    traces: list[tuple[float, float]] = []
    peaks: list[TracePeak] = []
    in_lc = False
    in_trace = False
    current_peak: dict[str, str] | None = None

    for line in text:
        s = line.strip()
        if s.startswith("Description\tLC"):
            in_lc = True
        elif s.startswith("[FUNCTION]") and in_lc:
            in_lc = False
        elif in_lc and s == "[TRACE]":
            in_trace = True
        elif in_trace and s == "}":
            in_trace = False
        elif in_trace and s and not s.startswith(";"):
            parts = s.split()
            if len(parts) >= 2:
                try:
                    traces.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
        elif in_lc and s == "[PEAK]":
            current_peak = {}
        elif current_peak is not None and s == "}":
            peaks.append(peak_from_dict(current_peak))
            current_peak = None
        elif current_peak is not None and s:
            key, _, value = s.partition("\t")
            current_peak[key] = value

    if traces:
        arr = np.asarray(traces, dtype=float)
        return arr[:, 0], arr[:, 1], peaks
    return np.asarray([], dtype=float), np.asarray([], dtype=float), peaks


def parse_arw(path: Path) -> tuple[np.ndarray, np.ndarray]:
    pairs: list[tuple[float, float]] = []
    if not path.exists() or path.stat().st_size < 100:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        try:
            pairs.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    if not pairs:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    arr = np.asarray(pairs, dtype=float)
    return arr[:, 0], arr[:, 1]


def peak_from_dict(d: dict[str, str]) -> TracePeak:
    def f(key: str) -> float | None:
        try:
            return float(d.get(key, ""))
        except ValueError:
            return None

    start = end = None
    if "Peak" in d:
        parts = d["Peak"].split()
        if len(parts) >= 2:
            try:
                start = float(parts[0])
                end = float(parts[1])
            except ValueError:
                pass
    return TracePeak(
        peak_id=int(f("Peak ID") or 0),
        time=f("Time"),
        start=start,
        end=end,
        height=f("Height"),
        area_abs=f("AreaAbs"),
        area_total=f("Area %Total"),
        width=f("Width"),
    )


def signal_features(t: np.ndarray, y: np.ndarray, peaks: list[TracePeak]) -> dict[str, float | int]:
    if len(t) == 0 or len(y) == 0:
        return {"trace_points": 0}
    y = y.astype(float)
    baseline = float(np.percentile(y, 5))
    ymax = float(np.max(y))
    ymin = float(np.min(y))
    peak_idx = int(np.argmax(y))
    peak_time = float(t[peak_idx])
    above_1 = y > baseline + 0.01 * (ymax - baseline)
    above_5 = y > baseline + 0.05 * (ymax - baseline)
    width_1 = span_width(t, above_1)
    width_5 = span_width(t, above_5)
    noise = robust_noise(y[: max(20, len(y) // 10)])
    peak_areas = [p.area_abs for p in peaks if p.area_abs is not None]
    peak_heights = [p.height for p in peaks if p.height is not None]
    return {
        "trace_points": int(len(t)),
        "time_min": float(np.min(t)),
        "time_max": float(np.max(t)),
        "signal_min": ymin,
        "signal_max": ymax,
        "signal_range": ymax - ymin,
        "baseline_p05": baseline,
        "noise_early_mad": noise,
        "main_peak_time": peak_time,
        "width_above_1pct": width_1,
        "width_above_5pct": width_5,
        "integrated_peak_count": int(len(peaks)),
        "integrated_area_sum": float(np.sum(peak_areas)) if peak_areas else 0.0,
        "integrated_max_height": float(np.max(peak_heights)) if peak_heights else 0.0,
    }


def span_width(t: np.ndarray, mask: np.ndarray) -> float:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return 0.0
    return float(t[idx[-1]] - t[idx[0]])


def robust_noise(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    med = np.median(y)
    return float(np.median(np.abs(y - med)))


def normalized_target_peaks(peaks: list[TracePeak]) -> dict[str, float | int]:
    usable = [p for p in peaks if p.area_abs is not None and p.time is not None]
    usable = [p for p in usable if 3.0 <= (p.time or 0) <= 7.0]
    usable.sort(key=lambda p: p.time or 0)
    areas = [p.area_abs or 0.0 for p in usable]
    total = sum(areas)
    out: dict[str, float | int] = {"target_peak_count": len(usable)}
    names = ["hmws", "monomer", "lmws"]
    for name, p in zip(names, usable[:3]):
        area = p.area_abs or 0.0
        out[f"{name}_time"] = p.time or math.nan
        out[f"{name}_height"] = p.height or math.nan
        out[f"{name}_area"] = area
        out[f"{name}_area_pct_norm"] = 100.0 * area / total if total else math.nan
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    hash_groups: dict[str, list[str]] = {}

    for d in iter_sample_dirs():
        files = list(d.iterdir())
        arws = sorted(f for f in files if f.suffix.lower() == ".arw")
        cdfs = sorted(f for f in files if f.suffix.lower() == ".cdf")
        rpts = sorted(f for f in files if f.suffix.lower() == ".rpt")
        pngs = sorted(f for f in files if f.suffix.lower() == ".png")
        arw_hash = md5_file(arws[0]) if arws else ""
        if arw_hash:
            hash_groups.setdefault(arw_hash, []).append(str(d))
        png_classes = {str(p): classify_png(p) for p in pngs}
        arw_t, arw_y = parse_arw(arws[0]) if arws else (np.asarray([]), np.asarray([]))
        rpt_t, rpt_y, peaks = parse_rpt(rpts[0]) if rpts else (np.asarray([]), np.asarray([]), [])
        features = {f"arw_{k}": v for k, v in signal_features(arw_t, arw_y, []).items()}
        features.update({f"rpt_{k}": v for k, v in signal_features(rpt_t, rpt_y, peaks).items()})
        peak_summary = normalized_target_peaks(peaks)

        row = {
            "sample_dir": str(d),
            "group_root": d.parts[0],
            "sample_name": d.name,
            "arw_path": first_file(arws),
            "arw_size": arws[0].stat().st_size if arws else 0,
            "arw_md5": arw_hash,
            "cdf_path": first_file(cdfs),
            "cdf_size": cdfs[0].stat().st_size if cdfs else 0,
            "rpt_path": first_file(rpts),
            "rpt_size": rpts[0].stat().st_size if rpts else 0,
            "png_count": len(pngs),
            "parameter_pngs": ";".join(p for p, c in png_classes.items() if c == "parameter_screenshot"),
            "plot_pngs": ";".join(p for p, c in png_classes.items() if c == "chromatogram_plot"),
            "unknown_pngs": ";".join(p for p, c in png_classes.items() if c == "unknown_png"),
            "status": "ok",
        }
        problems = []
        if not arws:
            problems.append("missing_arw")
        elif arws[0].stat().st_size < 100:
            problems.append("bad_arw")
        if not rpts:
            problems.append("missing_rpt")
        if not pngs:
            problems.append("missing_png")
        if not row["parameter_pngs"]:
            problems.append("no_parameter_png_detected")
        if problems:
            row["status"] = "|".join(problems)
        rows.append(row)

        feature_rows.append({**row, **features, **peak_summary})

    duplicate_dirs = {h: dirs for h, dirs in hash_groups.items() if len(dirs) > 1}
    for row in rows:
        dirs = duplicate_dirs.get(str(row["arw_md5"]), [])
        row["duplicate_group_size"] = len(dirs) if dirs else 1
        row["duplicate_dirs"] = ";".join(dirs)
    for row in feature_rows:
        dirs = duplicate_dirs.get(str(row["arw_md5"]), [])
        row["duplicate_group_size"] = len(dirs) if dirs else 1
        row["duplicate_dirs"] = ";".join(dirs)

    write_csv(TABLE_DIR / "sec_sample_audit.csv", rows)
    write_csv(TABLE_DIR / "sec_trace_features.csv", feature_rows)
    report = {
        "sample_dirs": len(rows),
        "ok_rows": sum(1 for r in rows if r["status"] == "ok"),
        "bad_arw_rows": sum(1 for r in rows if "bad_arw" in str(r["status"])),
        "missing_parameter_png_rows": sum(1 for r in rows if "no_parameter_png_detected" in str(r["status"])),
        "duplicate_arw_groups": len(duplicate_dirs),
        "duplicate_arw_rows": sum(len(v) for v in duplicate_dirs.values()),
    }
    (OUTPUT_DIR / "sec_data_audit_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


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
