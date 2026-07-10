from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAINING_CSV = ROOT / "outputs" / "tables" / "sec_training_dataset_clean.csv"
OUT_DIR = ROOT / "outputs" / "tables"


def source_bucket(row: pd.Series) -> str:
    status = str(row.get("label_status", ""))
    source = str(row.get("label_source", ""))
    if status == "manual_confirmed":
        return "manual_confirmed"
    if status == "ocr_training" and source.startswith("user_confirmed"):
        return "user_confirmed_ocr_training"
    if status == "ocr_training":
        return "ocr_training_reviewed"
    if "legacy_ocr" in source:
        return "legacy_ocr_review"
    return status or source or "unknown"


def profile_id(row: pd.Series) -> str:
    values = [row["Peak Width"], row["Detection Threshold"], row["Minimum Height"]]
    if any(pd.isna(value) for value in values):
        return ""
    return "{:.4f}|{:.4f}|{:.4f}".format(*(float(value) for value in values))


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TRAINING_CSV)

    ensure_columns(
        df,
        [
            "sample_dir",
            "sample_name",
            "arw_path",
            "arw_md5",
            "label_status",
            "label_source",
            "base_label_csv",
            "duplicate_dirs",
            "cdf_path",
            "rpt_path",
            "peak_width_sec",
            "detection_threshold",
            "minimum_height",
            "peak_width_clean",
            "threshold_clean",
            "minimum_height_clean",
            "duplicate_group_size",
            "arw_size",
            "hmws_area_pct_norm",
            "monomer_area_pct_norm",
            "lmws_area_pct_norm",
        ],
    )

    df["training_source_bucket"] = df.apply(source_bucket, axis=1)
    df["Peak Width"] = pd.to_numeric(df["peak_width_clean"], errors="coerce")
    df["Detection Threshold"] = pd.to_numeric(df["threshold_clean"], errors="coerce")
    df["Minimum Height"] = pd.to_numeric(df["minimum_height_clean"], errors="coerce")
    df["profile_id"] = df.apply(profile_id, axis=1)
    profile_counts = df["profile_id"].value_counts(dropna=False).to_dict()
    df["profile_training_count"] = df["profile_id"].map(profile_counts)

    master_cols = [
        "sample_dir",
        "sample_name",
        "training_source_bucket",
        "label_status",
        "label_source",
        "base_label_csv",
        "arw_path",
        "arw_md5",
        "arw_size",
        "cdf_path",
        "rpt_path",
        "peak_width_sec",
        "detection_threshold",
        "minimum_height",
        "Peak Width",
        "Detection Threshold",
        "Minimum Height",
        "profile_id",
        "profile_training_count",
        "duplicate_group_size",
        "duplicate_dirs",
        "hmws_area_pct_norm",
        "monomer_area_pct_norm",
        "lmws_area_pct_norm",
    ]
    master = (
        df[master_cols]
        .copy()
        .sort_values(["training_source_bucket", "sample_dir", "arw_md5"], kind="stable")
        .reset_index(drop=True)
    )
    master.insert(0, "training_row", range(1, len(master) + 1))

    source_summary = (
        master.groupby(["training_source_bucket", "label_status", "label_source"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["training_source_bucket", "label_status", "label_source"])
    )
    profile_summary = (
        master.groupby(["profile_id", "Peak Width", "Detection Threshold", "Minimum Height"], dropna=False)
        .agg(
            rows=("training_row", "count"),
            sources=("training_source_bucket", lambda s: ";".join(sorted(set(map(str, s))))),
            examples=("sample_dir", lambda s: ";".join(map(str, list(s)[:8]))),
        )
        .reset_index()
        .sort_values(["Peak Width", "Detection Threshold", "Minimum Height"], na_position="last")
    )

    master_path = OUT_DIR / "sec_training_data_master.csv"
    source_summary_path = OUT_DIR / "sec_training_data_master_source_summary.csv"
    profile_summary_path = OUT_DIR / "sec_training_data_master_profile_summary.csv"
    xlsx_path = OUT_DIR / "sec_training_data_master.xlsx"

    master.to_csv(master_path, index=False, encoding="utf-8-sig")
    source_summary.to_csv(source_summary_path, index=False, encoding="utf-8-sig")
    profile_summary.to_csv(profile_summary_path, index=False, encoding="utf-8-sig")

    xlsx_status = str(xlsx_path)
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            master.to_excel(writer, sheet_name="training_rows", index=False)
            source_summary.to_excel(writer, sheet_name="source_summary", index=False)
            profile_summary.to_excel(writer, sheet_name="profile_summary", index=False)
    except Exception as exc:
        try:
            with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
                master.to_excel(writer, sheet_name="training_rows", index=False)
                source_summary.to_excel(writer, sheet_name="source_summary", index=False)
                profile_summary.to_excel(writer, sheet_name="profile_summary", index=False)
        except Exception as fallback_exc:
            xlsx_status = f"not written: {exc}; fallback failed: {fallback_exc}"

    print(
        json.dumps(
            {
                "master_csv": str(master_path),
                "source_summary_csv": str(source_summary_path),
                "profile_summary_csv": str(profile_summary_path),
                "xlsx": xlsx_status,
                "rows": int(len(master)),
                "profiles": int(master["profile_id"].nunique()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
