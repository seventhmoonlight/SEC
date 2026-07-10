from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from predict_sec_parameters import MODEL_PATH, predict


PASS_LIMITS = {
    "peak_width": 10.0,
    "threshold": 5.0,
    "minimum_height": 5.0,
}


def as_float(value: object) -> float:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return 0.0
    return float(value)


def evaluate_row(row: pd.Series, model_path: Path, use_manual_labels: bool) -> dict[str, object]:
    result = predict(
        Path(str(row["input"])),
        use_manual_labels=use_manual_labels,
        model_path=model_path,
        snap_parameters=None,
    )
    true_peak_width = as_float(row["Peak Width"])
    true_threshold = as_float(row["Detection Threshold"])
    true_minimum_height = as_float(row["Minimum Height"])
    pred_peak_width = as_float(result["Peak Width"])
    pred_threshold = as_float(result["Detection Threshold"])
    pred_minimum_height = as_float(result["Minimum Height"])

    peak_width_err = abs(pred_peak_width - true_peak_width)
    threshold_err = abs(pred_threshold - true_threshold)
    minimum_height_err = abs(pred_minimum_height - true_minimum_height)
    peak_width_pass = peak_width_err <= PASS_LIMITS["peak_width"]
    threshold_pass = threshold_err <= PASS_LIMITS["threshold"]
    minimum_height_pass = minimum_height_err <= PASS_LIMITS["minimum_height"]
    all_params_pass = peak_width_pass and threshold_pass and minimum_height_pass

    return {
        "input": row["input"],
        "arw_used": result.get("arw_used", ""),
        "true_peak_width": true_peak_width,
        "pred_peak_width": pred_peak_width,
        "peak_width_abs_err": round(peak_width_err, 6),
        "peak_width_pass": peak_width_pass,
        "true_threshold": true_threshold,
        "pred_threshold": pred_threshold,
        "threshold_abs_err": round(threshold_err, 6),
        "threshold_pass": threshold_pass,
        "true_minimum_height": true_minimum_height,
        "pred_minimum_height": pred_minimum_height,
        "minimum_height_abs_err": round(minimum_height_err, 6),
        "minimum_height_pass": minimum_height_pass,
        "all_params_pass": all_params_pass,
        "risk_level": result.get("risk_level", ""),
        "auto_apply": bool(result.get("auto_apply", False)),
        "auto_apply_correct": bool(result.get("auto_apply", False)) and all_params_pass,
        "prediction_warnings": " | ".join(str(item) for item in result.get("Prediction Warnings", [])),
        "risk_reasons": json.dumps(result.get("risk_reasons", []), ensure_ascii=False, separators=(",", ":")),
    }


def rate(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return round(float(series.mean()), 6)


def build_summary(out: pd.DataFrame) -> dict[str, object]:
    auto_apply_rows = out[out["auto_apply"]]
    return {
        "rows": int(len(out)),
        "evaluation_type": "blind_test_holdout",
        "pass_limits": {
            "peak_width_abs_err_max": PASS_LIMITS["peak_width"],
            "threshold_abs_err_max": PASS_LIMITS["threshold"],
            "minimum_height_abs_err_max": PASS_LIMITS["minimum_height"],
        },
        "peak_width_mae": round(float(out["peak_width_abs_err"].mean()), 6) if len(out) else 0.0,
        "peak_width_pass_rate": rate(out["peak_width_pass"]),
        "threshold_mae": round(float(out["threshold_abs_err"].mean()), 6) if len(out) else 0.0,
        "threshold_pass_rate": rate(out["threshold_pass"]),
        "minimum_height_mae": round(float(out["minimum_height_abs_err"].mean()), 6) if len(out) else 0.0,
        "minimum_height_pass_rate": rate(out["minimum_height_pass"]),
        "all_params_pass_rate": rate(out["all_params_pass"]),
        "auto_apply_rate": rate(out["auto_apply"]),
        "auto_apply_rows": int(len(auto_apply_rows)),
        "auto_apply_precision": rate(auto_apply_rows["all_params_pass"]) if len(auto_apply_rows) else 0.0,
        "risk_level_counts": out["risk_level"].value_counts(dropna=False).to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SEC parameter predictions on a confirmed blind-test truth table.")
    parser.add_argument("--truth", required=True, help="Confirmed blind-test CSV with input, Peak Width, Detection Threshold, Minimum Height.")
    parser.add_argument("--csv", required=True, help="Output per-row error CSV path.")
    parser.add_argument("--report", help="Optional summary JSON path. Defaults to <csv stem>_summary.json.")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Model pickle path.")
    parser.add_argument("--use-manual-labels", action="store_true", help="Allow manual label lookup. Keep off for true blind evaluation.")
    args = parser.parse_args()

    truth = pd.read_csv(args.truth)
    required = {"input", "Peak Width", "Detection Threshold", "Minimum Height"}
    missing = sorted(required - set(truth.columns))
    if missing:
        raise ValueError(f"Blind truth CSV is missing required columns: {missing}")

    rows = [evaluate_row(row, Path(args.model), args.use_manual_labels) for _, row in truth.iterrows()]
    out = pd.DataFrame(rows)
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary = build_summary(out)
    report_path = Path(args.report) if args.report else csv_path.with_name(f"{csv_path.stem}_summary.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
