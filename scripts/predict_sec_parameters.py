from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from train_sec_parameter_model import ROOT, extract_features, read_arw, clip_predictions


MODEL_PATH = ROOT / "outputs" / "models" / "sec_parameter_model.pkl"
SHORT_RT_CUTOFF_MIN = 10.0
SHORT_RT_INTEGRATION_WINDOW = (2.5, 7.2)
LONG_RT_INTEGRATION_WINDOW = (9.0, 23.0)


def resolve_arw_path(input_path: str) -> Path:
    path = Path(input_path)
    if not path.is_absolute():
        path = ROOT / path
    if path.is_dir():
        candidates = sorted(p for p in path.glob("*.arw") if p.stat().st_size > 0)
        if not candidates:
            raise FileNotFoundError(f"No non-empty .arw file found in {path}")
        return candidates[0]
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def estimate_confidence(model, x: pd.DataFrame, pred: np.ndarray, bounds: dict) -> tuple[float, dict[str, float]]:
    estimator = model.named_steps.get("model")
    imputer = model.named_steps.get("imputer")
    if estimator is None or imputer is None or not hasattr(estimator, "estimators_"):
        return 0.5, {}

    x_imp = imputer.transform(x)
    tree_preds = np.asarray([tree.predict(x_imp)[0] for tree in estimator.estimators_], dtype=float)
    stds = tree_preds.std(axis=0)

    details: dict[str, float] = {}
    scores = []
    for idx, col in enumerate(["peak_width_sec", "detection_threshold", "minimum_height"]):
        low, high = bounds.get(col, (None, None))
        if high is not None and low is not None and high > low:
            scale = high - low
        else:
            scale = max(abs(float(pred[0, idx])), 1.0)
        uncertainty = float(stds[idx] / max(scale, 1e-12))
        score = float(max(0.0, min(1.0, 1.0 - 4.0 * uncertainty)))
        details[f"{col}_tree_std"] = float(stds[idx])
        details[f"{col}_confidence"] = score
        scores.append(score)

    boundary_penalty = 0.0
    for idx, col in enumerate(["peak_width_sec", "detection_threshold", "minimum_height"]):
        low, high = bounds.get(col, (None, None))
        value = float(pred[0, idx])
        if low is not None and abs(value - low) < 1e-9:
            boundary_penalty += 0.1
        if high is not None and abs(value - high) < 1e-9:
            boundary_penalty += 0.1

    confidence = float(max(0.0, min(1.0, np.mean(scores) - boundary_penalty)))
    return confidence, details


def choose_integration_window(main_peak_rt: float) -> tuple[float, float, str]:
    if main_peak_rt < SHORT_RT_CUTOFF_MIN:
        start, end = SHORT_RT_INTEGRATION_WINDOW
        return start, end, "short_rt"
    start, end = LONG_RT_INTEGRATION_WINDOW
    return start, end, "long_rt"


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        raise SystemExit("Usage: python scripts\\predict_sec_parameters.py <arw-file-or-sample-dir>")

    arw_path = resolve_arw_path(argv[1])
    with MODEL_PATH.open("rb") as f:
        bundle = pickle.load(f)
    estimator = bundle["model"].named_steps.get("model")
    if estimator is not None and hasattr(estimator, "n_jobs"):
        estimator.n_jobs = 1

    features = extract_features(read_arw(arw_path))
    x = pd.DataFrame([{col: features.get(col, np.nan) for col in bundle["feature_columns"]}])
    main_peak_rt = float(features["main_peak_rt"])
    integration_start, integration_end, integration_window_type = choose_integration_window(main_peak_rt)

    raw_pred = np.asarray(bundle["model"].predict(x), dtype=float)
    pred = clip_predictions(raw_pred)
    confidence, confidence_details = estimate_confidence(
        bundle["model"],
        x,
        pred,
        bundle["parameter_bounds"],
    )

    result = {
        "input": str(Path(argv[1])),
        "arw_used": str(arw_path.relative_to(ROOT) if arw_path.is_relative_to(ROOT) else arw_path),
        "Algorithm": "ApexTrack",
        "Integration Start": round(integration_start, 6),
        "Integration End": round(integration_end, 6),
        "main_peak_rt": round(main_peak_rt, 6),
        "integration_window_type": integration_window_type,
        "Peak Width": round(float(pred[0, 0]), 6),
        "Detection Threshold": round(float(pred[0, 1]), 6),
        "Minimum Height": round(float(pred[0, 2]), 6),
        "confidence": round(confidence, 6),
        "model_name": bundle["model_name"],
        "training_rows": bundle["training_rows"],
        "confidence_details": confidence_details,
        "model_note": "Continuous regression; profile/template fields were not used.",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv)
