from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from sec_data_audit import md5_file, parse_arw, signal_features

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover - optional production dependency
    CatBoostRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover - optional production dependency
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - optional production dependency
    XGBRegressor = None


LABEL_CSV = Path("outputs/tables/sec_parameter_labels_ocr.csv")
REVIEW_LABEL_CSV = Path("datasets/sec_parameter_labels_ocr_review.csv")
FEATURE_CSV = Path("outputs/tables/sec_trace_features.csv")
MANUAL_LABEL_CSV = Path("datasets/manual_parameter_labels.csv")
OCR_TRAINING_LABEL_CSV = Path("datasets/training_parameter_labels_ocr.csv")
MODEL_DIR = Path("outputs/models")
REPORT_DIR = Path("outputs/reports")
TABLE_DIR = Path("outputs/tables")

TRACE_WINDOWS = {
    "trace_full": {"n": 320, "start": None, "end": None},
    "trace_03_07": {"n": 160, "start": 3.0, "end": 7.0},
    "trace_10_23": {"n": 220, "start": 10.0, "end": 23.0},
}

CORE_TARGETS = {
    "peak_width": {
        "column": "peak_width_clean",
        "clip": (0.0, 300.0),
        "within": 10.0,
        "weight_floor": 2.0,
    },
    "threshold": {
        "column": "threshold_clean",
        "clip": (0.0, 200.0),
        "within": 5.0,
        "weight_floor": 1.0,
    },
    "minimum_height": {
        "column": "minimum_height_clean",
        "clip": (0.0, 200.0),
        "within": 5.0,
        "weight_floor": 1.0,
    },
}


def clean_peak_width(v: object) -> float | None:
    x = to_float(v)
    if x is None:
        return None
    if 2.0 <= x < 10.0:
        x *= 10.0
    if 20.0 <= x <= 150.0:
        return round(x, 3)
    return None


def clean_threshold(v: object) -> float | None:
    x = to_float(v)
    if x is None:
        return None
    # Common OCR failure: "1.000e+01" loses punctuation and becomes 1000.
    # Valid reviewed methods can use higher values such as 50, so do not
    # collapse or reject values merely because they are above the old range.
    if x > 200.0:
        x /= 100.0
    if 0.0 <= x <= 200.0:
        return round(x, 4)
    return None


def clean_min_height(v: object) -> float | None:
    x = to_float(v)
    if x is None:
        return None
    if x < 0:
        return None
    return round(x, 4)


def clean_ocr_training_min_height(v: object) -> float:
    x = to_float(v)
    if x is None:
        return 0.0
    if x in {1.0, 3.0}:
        x *= 10.0
    if x < 0:
        return 0.0
    return round(x, 4)


def to_float(v: object) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return x


def trace_vector(arw_path: str, n: int = 320, start: float | None = None, end: float | None = None) -> np.ndarray:
    t, y = parse_arw(Path(arw_path))
    if len(t) == 0:
        return np.full(n, np.nan)
    mask = np.isfinite(t) & np.isfinite(y)
    if start is not None:
        mask &= t >= start
    if end is not None:
        mask &= t <= end
    if mask.sum() < 10:
        mask = np.isfinite(t) & np.isfinite(y)
    if mask.sum() < 10:
        return np.full(n, np.nan)
    tt = t[mask]
    yy = y[mask].astype(float)
    grid = np.linspace(float(tt.min()), float(tt.max()), n)
    vec = np.interp(grid, tt, yy)
    lo = np.percentile(vec, 5)
    hi = np.percentile(vec, 99)
    scale = hi - lo
    if scale <= 1e-12:
        return np.zeros(n)
    return (vec - lo) / scale


def trace_feature_values(arw_path: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for prefix, spec in TRACE_WINDOWS.items():
        vec = trace_vector(
            arw_path,
            n=int(spec["n"]),
            start=spec["start"],
            end=spec["end"],
        )
        for i, value in enumerate(vec):
            values[f"{prefix}_{i:03d}"] = float(value)
    return values


def manual_label_rows() -> pd.DataFrame:
    if not MANUAL_LABEL_CSV.exists():
        return pd.DataFrame()
    labels = pd.read_csv(MANUAL_LABEL_CSV)
    rows: list[dict[str, object]] = []
    for _, label in labels.iterrows():
        arw_path = Path(str(label.get("arw_path", "")))
        t, y = parse_arw(arw_path)
        features = {f"arw_{k}": v for k, v in signal_features(t, y, []).items()}
        rows.append(
            {
                "sample_dir": label["sample_dir"],
                "sample_name": Path(str(label["sample_dir"])).name,
                "arw_path": str(arw_path),
                "arw_size": arw_path.stat().st_size if arw_path.exists() else 0,
                "arw_md5": md5_file(arw_path) if arw_path.exists() else "",
                "label_status": "manual_confirmed",
                "peak_width_sec": label.get("peak_width_sec"),
                "detection_threshold": label.get("detection_threshold"),
                "minimum_height": label.get("minimum_height"),
                "label_source": label.get("label_source", "manual"),
                **features,
            }
        )
    return pd.DataFrame(rows)


def ocr_training_label_rows() -> pd.DataFrame:
    if not OCR_TRAINING_LABEL_CSV.exists():
        return pd.DataFrame()
    labels = pd.read_csv(OCR_TRAINING_LABEL_CSV)
    labels = labels[labels["label_status"].astype(str) == "ok"].copy()
    rows: list[dict[str, object]] = []
    for _, label in labels.iterrows():
        arw_path = Path(str(label.get("arw_path", "")))
        t, y = parse_arw(arw_path)
        features = {f"arw_{k}": v for k, v in signal_features(t, y, []).items()}
        rows.append(
            {
                "sample_dir": label["sample_dir"],
                "sample_name": Path(str(label["sample_dir"])).name,
                "arw_path": str(arw_path),
                "arw_size": arw_path.stat().st_size if arw_path.exists() else 0,
                "arw_md5": md5_file(arw_path) if arw_path.exists() else "",
                "label_status": "ocr_training",
                "peak_width_sec": label.get("peak_width_sec"),
                "detection_threshold": label.get("detection_threshold"),
                "minimum_height": clean_ocr_training_min_height(label.get("minimum_height")),
                "label_source": label.get("label_source", "ocr_training"),
                **features,
            }
        )
    return pd.DataFrame(rows)


def build_dataset(
    include_manual_labels: bool = True,
    include_ocr_training_labels: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    label_csv = REVIEW_LABEL_CSV if REVIEW_LABEL_CSV.exists() else LABEL_CSV
    labels = pd.read_csv(label_csv)
    features = pd.read_csv(FEATURE_CSV)
    label_cols = [
        "sample_dir",
        "label_status",
        "peak_width_sec",
        "detection_threshold",
        "minimum_height",
    ]
    if "label_source" in labels.columns:
        label_cols.append("label_source")
    df = features.merge(
        labels[label_cols],
        on="sample_dir",
        how="left",
    )
    df["base_label_csv"] = str(label_csv)
    if include_manual_labels:
        manual = manual_label_rows()
        if not manual.empty:
            df = pd.concat([df, manual], ignore_index=True, sort=False)
    if include_ocr_training_labels:
        ocr_training = ocr_training_label_rows()
        if not ocr_training.empty:
            df = pd.concat([df, ocr_training], ignore_index=True, sort=False)
    df["peak_width_clean"] = df["peak_width_sec"].map(clean_peak_width)
    df["threshold_clean"] = df["detection_threshold"].map(clean_threshold)
    df["minimum_height_clean"] = df["minimum_height"].map(clean_min_height)
    df["usable_for_core"] = (
        (pd.to_numeric(df["arw_size"], errors="coerce") >= 100)
        & df["peak_width_clean"].notna()
        & df["threshold_clean"].notna()
    )
    df = df[df["usable_for_core"]].copy()

    numeric_feature_cols = [
        "arw_trace_points",
        "arw_time_min",
        "arw_time_max",
        "arw_signal_min",
        "arw_signal_max",
        "arw_signal_range",
        "arw_baseline_p05",
        "arw_noise_early_mad",
        "arw_main_peak_time",
        "arw_width_above_1pct",
        "arw_width_above_5pct",
    ]
    numeric_feature_cols = [c for c in numeric_feature_cols if c in df.columns]

    trace_rows = [trace_feature_values(str(arw)) for arw in df["arw_path"].fillna("")]
    trace_df = pd.DataFrame(trace_rows)
    df = pd.concat([df.reset_index(drop=True), trace_df.reset_index(drop=True)], axis=1)
    numeric_feature_cols.extend(list(trace_df.columns))

    # Collapse exact duplicate source files. If labels conflict across duplicates,
    # the median keeps the consensus and the conflict is reported separately.
    agg = {c: "first" for c in df.columns if c not in {"peak_width_clean", "threshold_clean", "minimum_height_clean"}}
    agg["peak_width_clean"] = "median"
    agg["threshold_clean"] = "median"
    agg["minimum_height_clean"] = "median"
    df_unique = df.groupby("arw_md5", as_index=False).agg(agg)
    return df_unique, numeric_feature_cols


def select_feature_cols(feature_cols: list[str], feature_mode: str) -> list[str]:
    if feature_mode == "arw-summary":
        selected = [c for c in feature_cols if c.startswith("arw_")]
    elif feature_mode == "arw-summary-trace":
        selected = [c for c in feature_cols if c.startswith("arw_") or c.startswith("trace_")]
    else:
        raise ValueError(f"Unknown feature mode: {feature_mode}")
    if not selected:
        raise ValueError(f"No feature columns selected for mode: {feature_mode}")
    return selected


def continuous_candidates(n_rows: int) -> dict[str, object]:
    neighbors = min(7, max(1, n_rows // 10))
    candidates = {
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-3, 4, 24)),
        ),
        "svr_rbf": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SVR(C=20.0, epsilon=0.5, gamma="scale"),
        ),
        "knn": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=neighbors, weights="distance"),
        ),
        "gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            GradientBoostingRegressor(n_estimators=250, random_state=7, max_depth=2, min_samples_leaf=3),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(n_estimators=400, random_state=7, min_samples_leaf=2),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=400, random_state=7, min_samples_leaf=2),
        ),
    }
    if CatBoostRegressor is not None:
        candidates["catboost"] = make_pipeline(
            SimpleImputer(strategy="median"),
            CatBoostRegressor(
                iterations=350,
                depth=4,
                learning_rate=0.05,
                loss_function="MAE",
                eval_metric="MAE",
                l2_leaf_reg=6.0,
                random_seed=7,
                verbose=False,
                allow_writing_files=False,
            ),
        )
    if LGBMRegressor is not None:
        candidates["lightgbm"] = make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMRegressor(
                objective="regression_l1",
                n_estimators=350,
                learning_rate=0.05,
                num_leaves=15,
                min_child_samples=4,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=2.0,
                random_state=7,
                n_jobs=-1,
                verbose=-1,
            ),
        )
    if XGBRegressor is not None:
        candidates["xgboost"] = make_pipeline(
            SimpleImputer(strategy="median"),
            XGBRegressor(
                objective="reg:absoluteerror",
                n_estimators=350,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.05,
                reg_lambda=2.0,
                random_state=7,
                n_jobs=-1,
                verbosity=0,
            ),
        )
    return candidates


def target_cv_splits(groups: pd.Series, n_rows: int) -> GroupKFold:
    return GroupKFold(n_splits=min(5, max(2, min(n_rows, groups.nunique()))))


def train_core_target(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_name: str,
    target_col: str,
) -> tuple[dict[str, object], dict[str, object]]:
    target_df = df[df[target_col].notna()].copy()
    X = target_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = target_df[target_col].astype(float)
    groups = target_df["arw_md5"].astype(str)
    candidates = continuous_candidates(len(target_df))
    cv = target_cv_splits(groups, len(target_df))

    rows: list[dict[str, object]] = []
    fitted_models: dict[str, object] = {}
    best_name = ""
    best_mae = float("inf")
    for name, base_model in candidates.items():
        preds = np.zeros(len(target_df), dtype=float)
        for train_idx, test_idx in cv.split(X, y, groups=groups):
            model = clone(base_model)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                preds[test_idx] = model.predict(X.iloc[test_idx]).astype(float)
        mae = float(mean_absolute_error(y, preds))
        within = float(np.mean(np.abs(y.to_numpy() - preds) <= float(CORE_TARGETS[target_name]["within"])))
        rows.append(
            {
                "target": target_name,
                "model": name,
                "mae": mae,
                "within_tolerance": within,
                "pred_min_cv": float(np.min(preds)),
                "pred_max_cv": float(np.max(preds)),
            }
        )
        if mae < best_mae:
            best_mae = mae
            best_name = name
        fitted = clone(base_model)
        fitted.fit(X, y)
        fitted_models[name] = fitted

    target_min = float(y.min())
    target_max = float(y.max())
    model_package = {
        "target": target_name,
        "column": target_col,
        "prediction_mode": "best_cv_open_source_tabular_regressor",
        "selected_model": best_name,
        "models": fitted_models,
        "weights": {best_name: 1.0},
        "target_min": target_min,
        "target_max": target_max,
        "clip_min": float(CORE_TARGETS[target_name]["clip"][0]),
        "clip_max": float(CORE_TARGETS[target_name]["clip"][1]),
        "cv": rows,
    }
    report = {
        "target": target_name,
        "rows": len(target_df),
        "best_model": best_name,
        "best_mae": best_mae,
        "target_min": target_min,
        "target_max": target_max,
        "models": rows,
        "selected_model_weight": {best_name: 1.0},
    }
    return report, model_package


def train_core_models(df: pd.DataFrame, feature_cols: list[str]) -> tuple[dict[str, object], dict[str, object]]:
    target_reports = []
    target_models = {}
    for target_name, spec in CORE_TARGETS.items():
        report, model = train_core_target(df, feature_cols, target_name, str(spec["column"]))
        target_reports.append(report)
        target_models[target_name] = model
    summary = {
        "prediction_mode": "per_target_best_cv_open_source_tabular_regressor",
        "core_target_reports": target_reports,
        "best_core_models": {row["target"]: row["best_model"] for row in target_reports},
        "core_cv": target_reports,
    }
    return summary, target_models


def snap_to_values(value: float, values: list[float]) -> float:
    if not values:
        return value
    return float(min(values, key=lambda option: abs(option - value)))


def train_primary_parameter_model(df: pd.DataFrame, feature_cols: list[str]) -> tuple[dict[str, object], object]:
    target_cols = ["peak_width_clean", "threshold_clean", "minimum_height_clean"]
    target_names = ["peak_width", "threshold", "minimum_height"]
    model_df = df.dropna(subset=target_cols).copy()
    X = model_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = model_df[target_cols].astype(float)
    groups = model_df["arw_md5"].astype(str)
    target_values = {
        name: sorted(float(v) for v in model_df[col].dropna().unique())
        for name, col in zip(target_names, target_cols)
    }
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesRegressor(
            n_estimators=400,
            random_state=31,
            min_samples_leaf=1,
            max_features=0.65,
            n_jobs=-1,
        ),
    )
    cv = target_cv_splits(groups, len(model_df))
    raw_preds = np.zeros_like(y.to_numpy(dtype=float))
    for train_idx, test_idx in cv.split(X, y, groups=groups):
        fold_model = clone(model)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            raw_preds[test_idx] = fold_model.predict(X.iloc[test_idx])

    snapped_preds = raw_preds.copy()
    for i, name in enumerate(target_names):
        snapped_preds[:, i] = [snap_to_values(float(value), target_values[name]) for value in raw_preds[:, i]]

    true_values = y.to_numpy(dtype=float)
    raw_abs_err = np.abs(true_values - raw_preds)
    snapped_abs_err = np.abs(true_values - snapped_preds)
    tolerances = np.array([CORE_TARGETS[name]["within"] for name in target_names], dtype=float)
    raw_all_params_pass = np.all(raw_abs_err <= tolerances, axis=1)
    snapped_all_params_pass = np.all(snapped_abs_err <= tolerances, axis=1)

    fitted = clone(model)
    fitted.fit(X, y)

    per_target = {}
    for i, name in enumerate(target_names):
        per_target[name] = {
            "mae_raw_cv": float(raw_abs_err[:, i].mean()),
            "mae_single_parameter_snapped_cv": float(snapped_abs_err[:, i].mean()),
            "within_tolerance_raw_cv": float(np.mean(raw_abs_err[:, i] <= tolerances[i])),
            "within_tolerance_single_parameter_snapped_cv": float(np.mean(snapped_abs_err[:, i] <= tolerances[i])),
            "target_values": target_values[name],
        }

    report = {
        "model": "extra_trees_multi_output_regressor",
        "prediction_mode": "multi_output_parameter_regressor_with_single_parameter_snapping",
        "rows": int(len(model_df)),
        "features": int(len(feature_cols)),
        "targets": target_names,
        "per_target": per_target,
        "all_params_pass_rate_raw_cv": float(np.mean(raw_all_params_pass)),
        "all_params_pass_rate_single_parameter_snapped_cv": float(np.mean(snapped_all_params_pass)),
        "note": (
            "Primary model predicts the three parameters together and may compose parameter combinations "
            "that were not present as complete profiles in training. Snapping is per parameter only."
        ),
    }
    return report, fitted


def evaluate_models(df: pd.DataFrame, feature_cols: list[str]) -> tuple[dict[str, object], object, object]:
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = df[["peak_width_clean", "threshold_clean"]].astype(float)
    groups = df["arw_md5"].astype(str)
    n_splits = min(5, len(df))
    cv = GroupKFold(n_splits=n_splits)
    candidates = {
        "knn": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), KNeighborsRegressor(n_neighbors=min(5, max(1, len(df) // 8)), weights="distance")),
        "extra_trees": make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(n_estimators=300, random_state=7, min_samples_leaf=2)),
        "random_forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(n_estimators=300, random_state=7, min_samples_leaf=2)),
    }

    rows = []
    best_name = ""
    best_score = float("inf")
    best_model = None
    for name, model in candidates.items():
        preds = np.zeros_like(y.to_numpy(dtype=float))
        for train_idx, test_idx in cv.split(X, y, groups=groups):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds[test_idx] = model.predict(X.iloc[test_idx])
        pw_mae = mean_absolute_error(y.iloc[:, 0], preds[:, 0])
        th_mae = mean_absolute_error(y.iloc[:, 1], preds[:, 1])
        score = pw_mae / 40.0 + th_mae / 10.0
        rows.append(
            {
                "model": name,
                "peak_width_mae": pw_mae,
                "threshold_mae": th_mae,
                "within_pw_10sec": float(np.mean(np.abs(y.iloc[:, 0].to_numpy() - preds[:, 0]) <= 10.0)),
                "within_threshold_5": float(np.mean(np.abs(y.iloc[:, 1].to_numpy() - preds[:, 1]) <= 5.0)),
                "score": score,
            }
        )
        if score < best_score:
            best_score = score
            best_name = name
            best_model = candidates[name]

    assert best_model is not None
    best_model.fit(X, y)

    y_class = y.astype(str)
    clf = make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(n_estimators=400, random_state=17, min_samples_leaf=1))
    clf_preds = np.zeros_like(y.to_numpy(dtype=float))
    for train_idx, test_idx in cv.split(X, y, groups=groups):
        clf.fit(X.iloc[train_idx], y_class.iloc[train_idx])
        clf_preds[test_idx] = clf.predict(X.iloc[test_idx]).astype(float)
    clf_pw_mae = mean_absolute_error(y.iloc[:, 0], clf_preds[:, 0])
    clf_th_mae = mean_absolute_error(y.iloc[:, 1], clf_preds[:, 1])
    rows.append(
        {
            "model": "extra_trees_classifier_discrete",
            "peak_width_mae": clf_pw_mae,
            "threshold_mae": clf_th_mae,
            "within_pw_10sec": float(np.mean(np.abs(y.iloc[:, 0].to_numpy() - clf_preds[:, 0]) <= 10.0)),
            "within_threshold_5": float(np.mean(np.abs(y.iloc[:, 1].to_numpy() - clf_preds[:, 1]) <= 5.0)),
            "score": clf_pw_mae / 40.0 + clf_th_mae / 10.0,
        }
    )
    clf.fit(X, y_class)
    return {"best_regression_model": best_name, "cv": rows}, best_model, clf


def train_min_height(df: pd.DataFrame, feature_cols: list[str]) -> tuple[object, object | None]:
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    present = df["minimum_height_clean"].notna().astype(int)
    clf = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=200, random_state=11, min_samples_leaf=2))
    clf.fit(X, present)
    reg = None
    if present.sum() >= 5:
        reg = make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(n_estimators=200, random_state=13, min_samples_leaf=2))
        reg.fit(X[present == 1], df.loc[present == 1, "minimum_height_clean"].astype(float))
    return clf, reg


def mode_float(values: pd.Series) -> float | None:
    clean = values.dropna().astype(float).round(4)
    if clean.empty:
        return None
    return float(clean.value_counts().sort_values(ascending=False).index[0])


def profile_id(peak_width: object, threshold: object, minimum_height: object) -> str:
    return f"{float(peak_width):.4f}|{float(threshold):.4f}|{float(minimum_height):.4f}"


def profile_values(profile: str) -> tuple[float, float, float]:
    peak_width, threshold, minimum_height = profile.split("|")
    return float(peak_width), float(threshold), float(minimum_height)


def train_profile_model(df: pd.DataFrame, feature_cols: list[str]) -> tuple[dict[str, object], object, dict[str, dict[str, object]]]:
    profile_df = df.dropna(subset=["peak_width_clean", "threshold_clean", "minimum_height_clean"]).copy()
    profile_df["profile_id"] = [
        profile_id(row["peak_width_clean"], row["threshold_clean"], row["minimum_height_clean"])
        for _, row in profile_df.iterrows()
    ]
    X = profile_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = profile_df["profile_id"].astype(str)
    groups = profile_df["arw_md5"].astype(str)

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesClassifier(
            n_estimators=400,
            random_state=29,
            min_samples_leaf=1,
            class_weight="balanced",
        ),
    )
    cv = target_cv_splits(groups, len(profile_df))
    pred_profiles = np.empty(len(profile_df), dtype=object)
    for train_idx, test_idx in cv.split(X, y, groups=groups):
        fold_model = clone(model)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred_profiles[test_idx] = fold_model.predict(X.iloc[test_idx])

    pred_values = np.array([profile_values(str(value)) for value in pred_profiles], dtype=float)
    true_values = profile_df[["peak_width_clean", "threshold_clean", "minimum_height_clean"]].to_numpy(dtype=float)
    abs_err = np.abs(true_values - pred_values)
    all_params_pass = (
        (abs_err[:, 0] <= CORE_TARGETS["peak_width"]["within"])
        & (abs_err[:, 1] <= CORE_TARGETS["threshold"]["within"])
        & (abs_err[:, 2] <= CORE_TARGETS["minimum_height"]["within"])
    )

    fitted = clone(model)
    fitted.fit(X, y)

    counts = profile_df["profile_id"].value_counts().to_dict()
    lookup: dict[str, dict[str, object]] = {}
    for profile, count in counts.items():
        peak_width, threshold, minimum_height = profile_values(str(profile))
        lookup[str(profile)] = {
            "peak_width": peak_width,
            "threshold": threshold,
            "minimum_height": minimum_height,
            "count": int(count),
        }

    report = {
        "model": "extra_trees_profile_classifier",
        "rows": int(len(profile_df)),
        "classes": int(y.nunique()),
        "profile_exact_accuracy_cv": float(np.mean(pred_profiles == y.to_numpy())),
        "all_params_pass_rate_cv": float(np.mean(all_params_pass)),
        "peak_width_mae_cv": float(abs_err[:, 0].mean()),
        "threshold_mae_cv": float(abs_err[:, 1].mean()),
        "minimum_height_mae_cv": float(abs_err[:, 2].mean()),
        "profile_counts": {str(key): int(value) for key, value in sorted(counts.items())},
    }
    return report, fitted, lookup


def target_metadata(df: pd.DataFrame) -> dict[str, object]:
    combo_counts = (
        df.groupby(["peak_width_clean", "threshold_clean"])
        .size()
        .reset_index(name="count")
        .sort_values(["peak_width_clean", "threshold_clean"])
    )
    trusted_combos = combo_counts[combo_counts["count"] >= 2].copy()
    if trusted_combos.empty:
        trusted_combos = combo_counts.copy()
    min_height_by_combo: dict[str, float] = {}
    for (peak_width, threshold), group in df.groupby(["peak_width_clean", "threshold_clean"]):
        value = mode_float(group["minimum_height_clean"])
        if value is not None:
            min_height_by_combo[f"{float(peak_width):.4f}|{float(threshold):.4f}"] = value
    return {
        "peak_width_values": sorted(float(v) for v in df["peak_width_clean"].dropna().unique()),
        "threshold_values": sorted(float(v) for v in df["threshold_clean"].dropna().unique()),
        "minimum_height_values": sorted(float(v) for v in df["minimum_height_clean"].dropna().unique()),
        "core_target_combos": [
            {"peak_width": float(row["peak_width_clean"]), "threshold": float(row["threshold_clean"]), "count": int(row["count"])}
            for _, row in trusted_combos.iterrows()
        ],
        "min_height_by_combo": min_height_by_combo,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SEC parameter prediction model.")
    parser.add_argument("--no-manual-labels", action="store_true", help="Exclude datasets/manual_parameter_labels.csv from training.")
    parser.add_argument("--include-ocr-training-labels", action="store_true", help="Include reviewed datasets/training_parameter_labels_ocr.csv in training.")
    parser.add_argument("--model-out", default=str(MODEL_DIR / "sec_parameter_model.pkl"), help="Output model pickle path.")
    parser.add_argument("--dataset-out", default=str(TABLE_DIR / "sec_training_dataset_clean.csv"), help="Output cleaned training CSV path.")
    parser.add_argument("--report-out", default=str(REPORT_DIR / "sec_model_training_report.json"), help="Output training report JSON path.")
    parser.add_argument(
        "--feature-mode",
        choices=["arw-summary", "arw-summary-trace"],
        default="arw-summary",
        help="Feature set used by the production model. arw-summary is the more conservative generalization default.",
    )
    parser.add_argument("--snap-parameters-default", action="store_true", help="Make prediction snap raw outputs to known parameter combinations by default.")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df, all_feature_cols = build_dataset(
        include_manual_labels=not args.no_manual_labels,
        include_ocr_training_labels=args.include_ocr_training_labels,
    )
    feature_cols = select_feature_cols(all_feature_cols, args.feature_mode)
    df["profile_id"] = [
        profile_id(row["peak_width_clean"], row["threshold_clean"], row["minimum_height_clean"])
        for _, row in df.iterrows()
    ]
    primary_report, primary_model = train_primary_parameter_model(df, feature_cols)
    profile_report, profile_model, profile_lookup = train_profile_model(df, feature_cols)
    report = {
        "prediction_mode": "multi_output_parameter_regressor_with_single_parameter_snapping",
        "core_target_reports": [],
        "core_cv": [],
        "deprecated_core_target_models_trained": False,
    }
    core_target_models = {}
    targets = target_metadata(df)
    base_label_csv = str(df["base_label_csv"].iloc[0]) if "base_label_csv" in df.columns and len(df) else ""
    model_package = {
        "prediction_mode": "multi_output_parameter_regressor_with_single_parameter_snapping",
        "primary_parameter_model": primary_model,
        "primary_parameter_report": primary_report,
        "profile_model": profile_model,
        "profile_lookup": profile_lookup,
        "profile_report": profile_report,
        "core_target_models": core_target_models,
        "core_regression_model": None,
        "core_classifier_model": None,
        "min_height_classifier": None,
        "min_height_regressor": None,
        "feature_cols": feature_cols,
        "feature_mode": args.feature_mode,
        "base_label_csv": base_label_csv,
        "training_rows_unique": len(df),
        "manual_labels_included": not args.no_manual_labels,
        "ocr_training_labels_included": args.include_ocr_training_labels,
        "snap_parameters_default": True,
        **targets,
        "sop_bounds": {
            "integration_algorithm": "ApexTrack",
            "integration_start_min_default": 3.5,
            "integration_end_min_default": 6.6,
            "peak_width_initial": 36.0,
            "peak_width_recommended_range": [20.0, 40.0],
            "threshold_initial": 0.0,
            "threshold_recommended_range": [0.0, 5.0],
        },
    }
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    with model_out.open("wb") as f:
        pickle.dump(model_package, f)
    dataset_out = Path(args.dataset_out)
    dataset_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dataset_out, index=False, encoding="utf-8-sig")
    report["prediction_mode"] = "multi_output_parameter_regressor_with_single_parameter_snapping"
    report["primary_parameter_model"] = primary_report
    report["profile_classifier_auxiliary"] = profile_report
    report["training_rows_unique"] = len(df)
    report["manual_labels_included"] = not args.no_manual_labels
    report["ocr_training_labels_included"] = args.include_ocr_training_labels
    report["feature_mode"] = args.feature_mode
    report["base_label_csv"] = base_label_csv
    report["snap_parameters_default"] = True
    report["target_distribution"] = {
        "peak_width": df["peak_width_clean"].value_counts().sort_index().to_dict(),
        "threshold": df["threshold_clean"].value_counts().sort_index().to_dict(),
        "minimum_height": df["minimum_height_clean"].value_counts().sort_index().to_dict(),
        "minimum_height_non_null": int(df["minimum_height_clean"].notna().sum()),
    }
    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
