from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from predict_sec_parameters import feature_frame, predict_core_parameters


MODEL_PATH = Path("outputs/models/sec_parameter_model.pkl")
TRAIN_CSV = Path("outputs/tables/sec_training_dataset_clean.csv")
TABLE_DIR = Path("outputs/tables")
REPORT_DIR = Path("outputs/reports")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("rb") as f:
        package = pickle.load(f)
    df = pd.read_csv(TRAIN_CSV)
    rows = []
    for _, row in df.iterrows():
        X = feature_frame(Path(row["arw_path"]), package["feature_cols"])
        peak_width, threshold, mh_pred, details = predict_core_parameters(package, X)
        mh_prob = 1.0 if package.get("core_target_models") else 0.0
        warnings = []
        if isinstance(details, dict):
            for detail in details.values():
                if isinstance(detail, dict):
                    warnings.extend(detail.get("warnings", []))
        rows.append(
            {
                "sample_dir": row["sample_dir"],
                "true_peak_width": row["peak_width_clean"],
                "pred_peak_width": peak_width,
                "peak_width_abs_err": abs(float(row["peak_width_clean"]) - peak_width),
                "true_threshold": row["threshold_clean"],
                "pred_threshold": threshold,
                "threshold_abs_err": abs(float(row["threshold_clean"]) - threshold),
                "true_minimum_height": row.get("minimum_height_clean", np.nan),
                "pred_minimum_height": "" if mh_pred is None else mh_pred,
                "minimum_height_abs_err": "" if mh_pred is None or pd.isna(row.get("minimum_height_clean", np.nan)) else abs(float(row["minimum_height_clean"]) - mh_pred),
                "minimum_height_probability": mh_prob,
                "prediction_warnings": " | ".join(warnings),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "sec_training_prediction_check.csv", index=False, encoding="utf-8-sig")
    report = {
        "rows": len(out),
        "peak_width_mae_training": float(out["peak_width_abs_err"].mean()),
        "threshold_mae_training": float(out["threshold_abs_err"].mean()),
        "minimum_height_mae_training": float(pd.to_numeric(out["minimum_height_abs_err"], errors="coerce").mean()),
        "threshold_large_error_rows": int((out["threshold_abs_err"] > 5).sum()),
    }
    (REPORT_DIR / "sec_training_prediction_check_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
