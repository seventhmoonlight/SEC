from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, savgol_filter
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
MASTER_CSV = ROOT / "DA" / "sec_training_data_master.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
TABLE_DIR = OUTPUT_DIR / "tables"

TARGET_COLUMNS = ["peak_width_sec", "detection_threshold", "minimum_height"]
OUTPUT_COLUMNS = ["Peak Width", "Detection Threshold", "Minimum Height"]

PARAMETER_BOUNDS = {
    "peak_width_sec": (15.0, 300.0),
    "detection_threshold": (0.0, 60.0),
    # Temporary practical bound from reviewed training data; update once the method range is confirmed.
    "minimum_height": (0.0, None),
}


@dataclass
class Chromatogram:
    time: np.ndarray
    signal: np.ndarray


def read_arw(path: Path) -> Chromatogram:
    rows: list[tuple[float, float]] = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.replace("\r", "\n").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue

    if len(rows) < 10:
        raise ValueError(f"Not enough numeric ARW points in {path}")

    arr = np.asarray(rows, dtype=float)
    order = np.argsort(arr[:, 0])
    return Chromatogram(time=arr[order, 0], signal=arr[order, 1])


def safe_stat(values: Iterable[float], fn, default: float = 0.0) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(fn(arr))


def robust_noise(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.0
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return float(np.std(values)), float(1.4826 * mad)


def smooth_signal(y: np.ndarray) -> np.ndarray:
    n = y.size
    if n < 11:
        return y.copy()
    window = min(51, n // 10 * 2 + 1)
    window = max(11, window)
    if window >= n:
        window = n - 1 if n % 2 == 0 else n
    if window < 5:
        return y.copy()
    return savgol_filter(y, window_length=window, polyorder=3, mode="interp")


def region(y: np.ndarray, start_frac: float, end_frac: float) -> np.ndarray:
    n = y.size
    start = max(0, min(n, int(round(n * start_frac))))
    end = max(start + 1, min(n, int(round(n * end_frac))))
    return y[start:end]


def extract_features(chrom: Chromatogram) -> dict[str, float]:
    t = chrom.time.astype(float)
    y = chrom.signal.astype(float)
    finite = np.isfinite(t) & np.isfinite(y)
    t = t[finite]
    y = y[finite]
    if t.size < 10:
        raise ValueError("Chromatogram has fewer than 10 finite points")

    y_s = smooth_signal(y)
    dt = np.diff(t)
    dy = np.gradient(y_s)
    ddy = np.gradient(dy)

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_range = max(y_max - y_min, 1e-12)
    y_norm = (y_s - y_min) / y_range

    edge_values = np.concatenate([region(y_s, 0.0, 0.1), region(y_s, 0.9, 1.0)])
    noise_std, noise_mad = robust_noise(edge_values)
    signal_noise = y_range / max(noise_mad, noise_std, 1e-12)

    prominence = max(0.01, 0.05 * np.std(y_norm))
    peaks, peak_props = find_peaks(y_norm, prominence=prominence, distance=max(1, y.size // 200))

    peak_count = int(peaks.size)
    if peak_count:
        prominences = peak_props.get("prominences", np.zeros(peak_count))
        main_idx = int(peaks[int(np.argmax(prominences))])
        widths_half = peak_widths(y_norm, [main_idx], rel_height=0.5)[0]
        widths_base = peak_widths(y_norm, [main_idx], rel_height=0.95)[0]
        median_dt = float(np.median(dt)) if dt.size else 0.0
        main_width_half_time = float(widths_half[0] * median_dt)
        main_width_base_time = float(widths_base[0] * median_dt)
        main_rt = float(t[main_idx])
        main_height = float(y_s[main_idx] - np.percentile(edge_values, 50))
        peak_prominence_max = float(np.max(prominences))
        peak_prominence_mean = float(np.mean(prominences))
    else:
        main_idx = int(np.argmax(y_s))
        main_width_half_time = 0.0
        main_width_base_time = 0.0
        main_rt = float(t[main_idx])
        main_height = float(y_s[main_idx] - np.percentile(edge_values, 50))
        peak_prominence_max = 0.0
        peak_prominence_mean = 0.0

    total_area = float(np.trapezoid(y_s - np.min(y_s), t))
    early_area = float(np.trapezoid(region(y_s - np.min(y_s), 0.0, 0.25), region(t, 0.0, 0.25)))
    mid_area = float(np.trapezoid(region(y_s - np.min(y_s), 0.25, 0.75), region(t, 0.25, 0.75)))
    late_area = float(np.trapezoid(region(y_s - np.min(y_s), 0.75, 1.0), region(t, 0.75, 1.0)))
    area_den = max(total_area, 1e-12)

    q = np.percentile(y, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    d_q = np.percentile(dy, [1, 5, 25, 50, 75, 95, 99])
    dd_q = np.percentile(ddy, [1, 5, 25, 50, 75, 95, 99])

    return {
        "point_count": float(t.size),
        "run_start": float(t[0]),
        "run_end": float(t[-1]),
        "run_length": float(t[-1] - t[0]),
        "sampling_interval_median": safe_stat(dt, np.median),
        "sampling_interval_mean": safe_stat(dt, np.mean),
        "sampling_interval_std": safe_stat(dt, np.std),
        "signal_min": y_min,
        "signal_max": y_max,
        "signal_range": y_range,
        "signal_mean": float(np.mean(y)),
        "signal_std": float(np.std(y)),
        "signal_median": float(np.median(y)),
        "signal_iqr": float(q[5] - q[3]),
        "signal_q01": float(q[0]),
        "signal_q05": float(q[1]),
        "signal_q10": float(q[2]),
        "signal_q25": float(q[3]),
        "signal_q75": float(q[5]),
        "signal_q90": float(q[6]),
        "signal_q95": float(q[7]),
        "signal_q99": float(q[8]),
        "edge_noise_std": noise_std,
        "edge_noise_mad": noise_mad,
        "signal_to_noise": float(signal_noise),
        "baseline_start_mean": float(np.mean(region(y_s, 0.0, 0.1))),
        "baseline_end_mean": float(np.mean(region(y_s, 0.9, 1.0))),
        "baseline_drift": float(np.mean(region(y_s, 0.9, 1.0)) - np.mean(region(y_s, 0.0, 0.1))),
        "derivative_std": float(np.std(dy)),
        "derivative_abs_mean": float(np.mean(np.abs(dy))),
        "derivative_q01": float(d_q[0]),
        "derivative_q05": float(d_q[1]),
        "derivative_q25": float(d_q[2]),
        "derivative_q50": float(d_q[3]),
        "derivative_q75": float(d_q[4]),
        "derivative_q95": float(d_q[5]),
        "derivative_q99": float(d_q[6]),
        "second_derivative_std": float(np.std(ddy)),
        "second_derivative_abs_mean": float(np.mean(np.abs(ddy))),
        "second_derivative_q01": float(dd_q[0]),
        "second_derivative_q05": float(dd_q[1]),
        "second_derivative_q25": float(dd_q[2]),
        "second_derivative_q50": float(dd_q[3]),
        "second_derivative_q75": float(dd_q[4]),
        "second_derivative_q95": float(dd_q[5]),
        "second_derivative_q99": float(dd_q[6]),
        "rough_peak_count": float(peak_count),
        "main_peak_rt": main_rt,
        "main_peak_rt_frac": float((main_rt - t[0]) / max(t[-1] - t[0], 1e-12)),
        "main_peak_height": main_height,
        "main_peak_height_frac": float(main_height / y_range),
        "main_peak_width_half_time": main_width_half_time,
        "main_peak_width_base_time": main_width_base_time,
        "peak_prominence_max": peak_prominence_max,
        "peak_prominence_mean": peak_prominence_mean,
        "total_area_shifted": total_area,
        "early_area_frac": early_area / area_den,
        "middle_area_frac": mid_area / area_den,
        "late_area_frac": late_area / area_den,
    }


def load_feature_table(master: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for _, row in master.iterrows():
        arw_path = ROOT / str(row["arw_path"])
        features = extract_features(read_arw(arw_path))
        features["training_row"] = float(row["training_row"])
        rows.append(features)
    return pd.DataFrame(rows)


def clip_predictions(pred: np.ndarray) -> np.ndarray:
    clipped = pred.copy().astype(float)
    for idx, col in enumerate(TARGET_COLUMNS):
        low, high = PARAMETER_BOUNDS[col]
        if low is not None:
            clipped[:, idx] = np.maximum(clipped[:, idx], low)
        if high is not None:
            clipped[:, idx] = np.minimum(clipped[:, idx], high)
    return clipped


def build_models() -> dict[str, Pipeline]:
    return {
        "extra_trees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=600,
                        random_state=42,
                        min_samples_leaf=2,
                        max_features=0.75,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=600,
                        random_state=42,
                        min_samples_leaf=2,
                        max_features=0.75,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(name: str, model: Pipeline, x: pd.DataFrame, y: pd.DataFrame, groups: pd.Series) -> tuple[pd.DataFrame, dict]:
    unique_groups = groups.nunique()
    if unique_groups >= 5:
        splitter = GroupKFold(n_splits=5)
        splits = splitter.split(x, y, groups)
        split_name = "GroupKFold(arw_md5, n_splits=5)"
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        splits = splitter.split(x, y)
        split_name = "KFold(n_splits=5, shuffle=True)"

    pred = np.zeros((len(x), len(TARGET_COLUMNS)), dtype=float)
    fold_ids = np.zeros(len(x), dtype=int)
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        fold_pred = model.predict(x.iloc[test_idx])
        pred[test_idx] = clip_predictions(fold_pred)
        fold_ids[test_idx] = fold

    pred_df = pd.DataFrame(pred, columns=[f"pred_{c}" for c in TARGET_COLUMNS])
    pred_df["cv_fold"] = fold_ids

    metrics: dict[str, object] = {
        "model": name,
        "cv": split_name,
        "rows": int(len(x)),
        "features": int(x.shape[1]),
    }
    for i, col in enumerate(TARGET_COLUMNS):
        true = y[col].to_numpy(dtype=float)
        p = pred[:, i]
        metrics[f"{col}_mae"] = float(mean_absolute_error(true, p))
        metrics[f"{col}_median_abs_error"] = float(np.median(np.abs(true - p)))
        metrics[f"{col}_r2"] = float(r2_score(true, p))
        metrics[f"{col}_within_10pct_or_5abs"] = float(np.mean(np.abs(true - p) <= np.maximum(5.0, np.abs(true) * 0.10)))
    metrics["mean_target_mae"] = float(np.mean([metrics[f"{c}_mae"] for c in TARGET_COLUMNS]))
    return pred_df, metrics


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(MASTER_CSV)
    # DA is a reviewed training table; label_status is retained for audit only.
    # Use every row with complete parameter labels.
    master = master[master[TARGET_COLUMNS].notna().all(axis=1)].copy()

    features = load_feature_table(master)
    feature_cols = [c for c in features.columns if c != "training_row"]
    x = features[feature_cols]
    y = master[TARGET_COLUMNS].astype(float).reset_index(drop=True)
    groups = master["arw_md5"].fillna(master["sample_dir"]).reset_index(drop=True)

    models = build_models()
    all_metrics = []
    all_predictions = []
    for name, model in models.items():
        pred_df, metrics = evaluate_model(name, model, x, y, groups)
        all_metrics.append(metrics)
        out = master.reset_index(drop=True)[
            [
                "training_row",
                "sample_dir",
                "sample_name",
                "arw_path",
                "arw_md5",
                *TARGET_COLUMNS,
                "hmws_area_pct_norm",
                "monomer_area_pct_norm",
                "lmws_area_pct_norm",
            ]
        ].copy()
        out.insert(0, "model", name)
        out = pd.concat([out, pred_df], axis=1)
        for target in TARGET_COLUMNS:
            out[f"abs_error_{target}"] = (out[target] - out[f"pred_{target}"]).abs()
        all_predictions.append(out)

    metrics_df = pd.DataFrame(all_metrics).sort_values("mean_target_mae")
    best_name = str(metrics_df.iloc[0]["model"])
    best_model = models[best_name]
    best_model.fit(x, y)

    bundle = {
        "model": best_model,
        "model_name": best_name,
        "feature_columns": feature_cols,
        "target_columns": TARGET_COLUMNS,
        "output_columns": OUTPUT_COLUMNS,
        "parameter_bounds": PARAMETER_BOUNDS,
        "training_rows": int(len(master)),
        "training_source": str(MASTER_CSV.relative_to(ROOT)),
        "notes": [
            "Continuous regression only.",
            "Profile/template fields are excluded from training and prediction.",
            "Predictions are clipped to Empower parameter bounds.",
        ],
    }

    with (MODEL_DIR / "sec_parameter_model.pkl").open("wb") as f:
        pickle.dump(bundle, f)

    pd.concat(all_predictions, ignore_index=True).to_csv(TABLE_DIR / "sec_parameter_cv_predictions.csv", index=False)
    metrics_df.to_csv(TABLE_DIR / "sec_parameter_cv_metrics.csv", index=False)
    (MODEL_DIR / "sec_parameter_model_metrics.json").write_text(
        json.dumps(all_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(metrics_df.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Saved model: {MODEL_DIR / 'sec_parameter_model.pkl'}")
    print(f"Saved CV predictions: {TABLE_DIR / 'sec_parameter_cv_predictions.csv'}")


if __name__ == "__main__":
    main()
