from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_sec_parameter_model import build_dataset


REPORT_DIR = Path("outputs/reports")
TABLE_DIR = Path("outputs/tables")


def family_group(sample_dir: object) -> str:
    path = str(sample_dir).replace("/", "\\")
    parts = path.split("\\")
    root = parts[0] if parts else "unknown"
    name = parts[-1] if parts else "unknown"
    letters = "".join(ch for ch in name if ch.isalpha())
    if not letters:
        letters = "numbered"
    if letters.upper().startswith("HT"):
        letters = "HT"
    return f"{root}\\{letters}"


def numeric_cols(df: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    cols = []
    for col in df.columns:
        if col.startswith(prefixes):
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().any():
                cols.append(col)
    return cols


def feature_sets(df: pd.DataFrame, base_feature_cols: list[str]) -> dict[str, list[str]]:
    arw_summary = [c for c in base_feature_cols if c.startswith("arw_")]
    trace = [c for c in base_feature_cols if c.startswith("trace_")]
    rpt = numeric_cols(df, ("rpt_", "target_", "hmws_", "monomer_", "lmws_"))
    return {
        "arw_summary": arw_summary,
        "arw_summary_trace": arw_summary + trace,
        "arw_rpt_summary_trace": arw_summary + rpt + trace,
    }


def candidate_models(n_rows: int) -> dict[str, object]:
    return {
        "extra_trees_regressor": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(n_estimators=300, random_state=7, min_samples_leaf=2),
        ),
        "random_forest_regressor": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=300, random_state=9, min_samples_leaf=2),
        ),
        "knn_regressor": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=min(5, max(1, n_rows // 10)), weights="distance"),
        ),
        "extra_trees_combo_classifier": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(n_estimators=400, random_state=17, min_samples_leaf=1),
        ),
    }


def trusted_combos(y_train: pd.DataFrame) -> list[tuple[float, float]]:
    counts = y_train.groupby(["peak_width_clean", "threshold_clean"]).size().reset_index(name="count")
    trusted = counts[counts["count"] >= 2]
    if trusted.empty:
        trusted = counts
    return [(float(row["peak_width_clean"]), float(row["threshold_clean"])) for _, row in trusted.iterrows()]


def snap_to_training_combos(preds: np.ndarray, combos: list[tuple[float, float]]) -> np.ndarray:
    if not combos:
        return preds
    snapped = np.zeros_like(preds, dtype=float)
    for i, (peak_width, threshold) in enumerate(preds):
        best = min(
            combos,
            key=lambda combo: ((peak_width - combo[0]) / 8.0) ** 2 + ((threshold - combo[1]) / 3.0) ** 2,
        )
        snapped[i] = best
    return snapped


def evaluate_split(df: pd.DataFrame, cols: list[str], group_col: str) -> list[dict[str, object]]:
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    y = df[["peak_width_clean", "threshold_clean"]].astype(float)
    groups = df[group_col].astype(str)
    n_splits = min(5, groups.nunique(), len(df))
    if n_splits < 2:
        return []

    rows = []
    for model_name, model in candidate_models(len(df)).items():
        pred = np.zeros((len(df), 2), dtype=float)
        pred_snapped = np.zeros((len(df), 2), dtype=float)
        fold_id = 0
        for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups=groups):
            fold_id += 1
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            if model_name.endswith("classifier"):
                model.fit(X_train, y_train.astype(str))
                fold_pred = model.predict(X_test).astype(float)
            else:
                model.fit(X_train, y_train)
                fold_pred = model.predict(X_test).astype(float)
            pred[test_idx] = fold_pred
            pred_snapped[test_idx] = snap_to_training_combos(fold_pred, trusted_combos(y_train))

        rows.append(metric_row(model_name, "raw", group_col, y.to_numpy(dtype=float), pred))
        rows.append(metric_row(model_name, "snapped_to_train_combos", group_col, y.to_numpy(dtype=float), pred_snapped))
    return rows


def metric_row(model_name: str, output_mode: str, group_col: str, true: np.ndarray, pred: np.ndarray) -> dict[str, object]:
    peak_err = np.abs(true[:, 0] - pred[:, 0])
    threshold_err = np.abs(true[:, 1] - pred[:, 1])
    return {
        "model": model_name,
        "output_mode": output_mode,
        "split": group_col,
        "rows": int(len(true)),
        "peak_width_mae": round(float(mean_absolute_error(true[:, 0], pred[:, 0])), 6),
        "peak_width_within_5sec": round(float(np.mean(peak_err <= 5.0)), 6),
        "peak_width_within_10sec": round(float(np.mean(peak_err <= 10.0)), 6),
        "threshold_mae": round(float(mean_absolute_error(true[:, 1], pred[:, 1])), 6),
        "threshold_within_2": round(float(np.mean(threshold_err <= 2.0)), 6),
        "threshold_within_5": round(float(np.mean(threshold_err <= 5.0)), 6),
        "core_combo_exact": round(float(np.mean((peak_err == 0.0) & (threshold_err == 0.0))), 6),
        "priority_score": round(float(np.mean(peak_err / 10.0 + threshold_err / 5.0)), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SEC parameter models without using test2 as the optimization target.")
    parser.add_argument("--include-manual-labels", action="store_true", help="Include datasets/manual_parameter_labels.csv in the benchmark dataset.")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df, base_feature_cols = build_dataset(include_manual_labels=args.include_manual_labels)
    df = df.copy()
    df["family_group"] = df["sample_dir"].map(family_group)
    df = df[~df["sample_dir"].astype(str).str.startswith("test2\\")].copy()

    all_rows = []
    for feature_name, cols in feature_sets(df, base_feature_cols).items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        for split in ["arw_md5", "family_group"]:
            for row in evaluate_split(df, cols, split):
                row["feature_set"] = feature_name
                all_rows.append(row)

    out = pd.DataFrame(all_rows).sort_values(["split", "priority_score", "peak_width_mae", "threshold_mae"])
    out_path = TABLE_DIR / "sec_generalization_benchmark.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    best = out.iloc[0].to_dict() if not out.empty else {}
    report = {
        "meaning": "This benchmark excludes test2 and evaluates on historical data with group splits, so it is closer to real unknown-sample behavior than test2 replay.",
        "rows": int(len(df)),
        "manual_labels_included": bool(args.include_manual_labels),
        "best_overall": best,
        "best_by_split": {
            split: out[out["split"] == split].iloc[0].to_dict()
            for split in sorted(out["split"].unique())
        } if not out.empty else {},
    }
    report_path = REPORT_DIR / "sec_generalization_benchmark_summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
