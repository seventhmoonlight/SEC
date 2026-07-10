from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sec_data_audit import parse_arw, signal_features
from train_sec_parameter_model import trace_feature_values


MODEL_PATH = Path("outputs/models/sec_parameter_model.pkl")
MANUAL_LABEL_CSV = Path("datasets/manual_parameter_labels.csv")
TRAINING_DATASET_CSV = Path("outputs/tables/sec_training_dataset_clean.csv")

TARGET_TOLERANCE = {
    "peak_width": 10.0,
    "threshold": 5.0,
    "minimum_height": 5.0,
}

RISK_SCORE_LABELS = {
    0: "low",
    1: "medium",
    2: "high",
}

NEAREST_SIGNAL_DISTANCE_MAX = 2.5
PROFILE_NEAREST_OVERRIDE_CONFIDENCE_MAX = 0.70
PROFILE_HIGH_RISK_CONFIDENCE = 0.45
PROFILE_MEDIUM_RISK_CONFIDENCE = 0.70


def find_arw(input_path: Path) -> Path:
    if input_path.is_file() and input_path.suffix.lower() == ".arw":
        return input_path
    if input_path.is_dir():
        arws = sorted(input_path.glob("*.arw"))
        if arws:
            return arws[0]
    raise FileNotFoundError(f"No .arw file found for {input_path}")


def feature_frame(arw_path: Path, feature_cols: list[str]) -> pd.DataFrame:
    t, y = parse_arw(arw_path)
    row: dict[str, float | int] = {}
    row.update({f"arw_{k}": v for k, v in signal_features(t, y, []).items()})
    row.update(trace_feature_values(str(arw_path)))
    return pd.DataFrame([{col: row.get(col, np.nan) for col in feature_cols}])


def norm_path(path: object) -> str:
    return str(path).replace("/", "\\").strip().lower()


def path_keys(path: object) -> set[str]:
    raw = Path(str(path))
    keys = {norm_path(path)}
    try:
        keys.add(norm_path(raw.resolve()))
    except OSError:
        pass
    try:
        keys.add(norm_path(raw.resolve().relative_to(Path.cwd().resolve())))
    except (OSError, ValueError):
        pass
    return {key for key in keys if key}


def paths_match(left: object, right_keys: set[str]) -> bool:
    left_keys = path_keys(left)
    for left_key in left_keys:
        for right_key in right_keys:
            if left_key == right_key or left_key.endswith("\\" + right_key) or right_key.endswith("\\" + left_key):
                return True
    return False


def combo_key(peak_width: float, threshold: float) -> str:
    return f"{float(peak_width):.4f}|{float(threshold):.4f}"


def profile_key(peak_width: float, threshold: float, minimum_height: float) -> str:
    return f"{float(peak_width):.4f}|{float(threshold):.4f}|{float(minimum_height):.4f}"


def snap_value(value: float | None, values: object) -> float | None:
    if value is None:
        return None
    options = sorted(float(v) for v in (values or []))
    if not options:
        return float(value)
    return float(min(options, key=lambda option: abs(option - float(value))))


def snap_parameters_independently(
    package: dict[str, object],
    peak_width: float,
    threshold: float,
    min_height: float | None,
) -> tuple[float, float, float | None]:
    return (
        float(snap_value(peak_width, package.get("peak_width_values"))),
        float(snap_value(threshold, package.get("threshold_values"))),
        snap_value(min_height, package.get("minimum_height_values")),
    )


def min_height_probability(model: object, X: pd.DataFrame) -> float:
    if model is None:
        return 1.0
    proba = model.predict_proba(X)[0]
    classes = list(getattr(model, "classes_", []))
    if not classes and hasattr(model, "steps"):
        classes = list(getattr(model.steps[-1][1], "classes_", []))
    if len(classes) == 1:
        return 1.0 if int(classes[0]) == 1 else 0.0
    if len(proba) == 1:
        return float(proba[0])
    return float(proba[classes.index(1)] if 1 in classes else proba[-1])


def predict_core_target(model_info: dict[str, object], X: pd.DataFrame) -> tuple[float, dict[str, object]]:
    models = model_info.get("models") or {}
    weights = model_info.get("weights") or {}
    candidate_predictions: dict[str, float] = {}
    weighted_sum = 0.0
    total_weight = 0.0
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            pred = float(np.asarray(model.predict(X))[0])
        candidate_predictions[str(name)] = pred
        weight = float(weights.get(name, 0.0))
        weighted_sum += weight * pred
        total_weight += weight
    if not candidate_predictions:
        raise ValueError(f"No models found for target {model_info.get('target')}")
    selected_model = str(model_info.get("selected_model", ""))
    if selected_model in candidate_predictions:
        raw = candidate_predictions[selected_model]
    else:
        raw = weighted_sum / total_weight if total_weight > 0 else float(np.mean(list(candidate_predictions.values())))
    clip_min = float(model_info.get("clip_min", 0.0))
    clip_max = float(model_info.get("clip_max", 1e9))
    value = float(np.clip(raw, clip_min, clip_max))
    target_name = str(model_info.get("target", ""))
    target_min = float(model_info.get("target_min", np.nan))
    target_max = float(model_info.get("target_max", np.nan))
    spread = float(max(candidate_predictions.values()) - min(candidate_predictions.values()))
    warning_messages: list[str] = []
    override_reason = ""
    tolerance = TARGET_TOLERANCE.get(target_name, 5.0)
    local_names = ["knn", "gradient_boosting"]
    if all(name in candidate_predictions for name in local_names):
        local_values = [candidate_predictions[name] for name in local_names]
        local_avg = float(np.mean(local_values))
        local_spread = float(max(local_values) - min(local_values))
        if local_spread <= tolerance and abs(local_avg - raw) > tolerance and spread > tolerance * 3.0:
            raw = local_avg
            value = float(np.clip(raw, clip_min, clip_max))
            selected_model = "local_consensus:knn+gradient_boosting"
            override_reason = (
                f"Local consensus override used because KNN and GradientBoosting agree "
                f"(spread={local_spread:.3g}) while global candidates disagree."
            )
    if np.isfinite(target_min) and value < target_min:
        warning_messages.append(f"{target_name} prediction is below the confirmed training range ({target_min:g}-{target_max:g}).")
    if np.isfinite(target_max) and value > target_max:
        warning_messages.append(f"{target_name} prediction is above the confirmed training range ({target_min:g}-{target_max:g}).")
    if spread > tolerance * 3.0:
        warning_messages.append(f"{target_name} candidate models disagree strongly (spread={spread:.3g}).")
    detail = {
        "target": target_name,
        "value": value,
        "raw_value": raw,
        "selected_model": selected_model or model_info.get("selected_model"),
        "candidate_predictions": candidate_predictions,
        "candidate_spread": spread,
        "training_range": [target_min, target_max],
        "override_reason": override_reason,
        "warnings": warning_messages,
    }
    return value, detail


def predict_core_parameters(package: dict[str, object], X: pd.DataFrame) -> tuple[float, float, float | None, dict[str, object]]:
    target_models = package.get("core_target_models")
    if target_models:
        details: dict[str, object] = {}
        values: dict[str, float] = {}
        for target_name in ("peak_width", "threshold", "minimum_height"):
            value, detail = predict_core_target(target_models[target_name], X)
            values[target_name] = value
            details[target_name] = detail
        return values["peak_width"], values["threshold"], values["minimum_height"], details

    core = package["core_regression_model"].predict(X)[0]
    peak_width = float(np.clip(core[0], 0.0, 300.0))
    threshold = float(np.clip(core[1], 0.0, 200.0))
    min_height_prob = min_height_probability(package.get("min_height_classifier"), X)
    min_height = None
    if package.get("min_height_regressor") is not None and min_height_prob >= 0.5:
        min_height = float(np.clip(package["min_height_regressor"].predict(X)[0], 0.0, 200.0))
    return peak_width, threshold, min_height, {}


def predict_primary_parameters(
    package: dict[str, object],
    X: pd.DataFrame,
    snap_parameters: bool,
) -> tuple[float, float, float, dict[str, object]]:
    model = package.get("primary_parameter_model")
    if model is None:
        raise ValueError("No primary_parameter_model found in model package.")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        raw = np.asarray(model.predict(X))[0].astype(float)
    raw_peak_width = float(np.clip(raw[0], 0.0, 300.0))
    raw_threshold = float(np.clip(raw[1], 0.0, 200.0))
    raw_min_height = float(np.clip(raw[2], 0.0, 200.0))
    peak_width, threshold, min_height = raw_peak_width, raw_threshold, raw_min_height
    if snap_parameters:
        peak_width, threshold, snapped_min_height = snap_parameters_independently(
            package,
            peak_width,
            threshold,
            min_height,
        )
        min_height = 0.0 if snapped_min_height is None else float(snapped_min_height)
    target_ranges = {
        "peak_width": [min([float(v) for v in package.get("peak_width_values", [])] or [np.nan]), max([float(v) for v in package.get("peak_width_values", [])] or [np.nan])],
        "threshold": [min([float(v) for v in package.get("threshold_values", [])] or [np.nan]), max([float(v) for v in package.get("threshold_values", [])] or [np.nan])],
        "minimum_height": [min([float(v) for v in package.get("minimum_height_values", [])] or [np.nan]), max([float(v) for v in package.get("minimum_height_values", [])] or [np.nan])],
    }
    detail = {
        "model": "extra_trees_multi_output_regressor",
        "prediction_mode": package.get("prediction_mode"),
        "raw_values": {
            "peak_width": raw_peak_width,
            "threshold": raw_threshold,
            "minimum_height": raw_min_height,
        },
        "values": {
            "peak_width": peak_width,
            "threshold": threshold,
            "minimum_height": min_height,
        },
        "single_parameter_snapping": snap_parameters,
        "training_ranges": target_ranges,
        "warnings": [],
    }
    return peak_width, threshold, min_height, detail


def model_classes(model: object) -> list[str]:
    classes = list(getattr(model, "classes_", []))
    if classes:
        return [str(value) for value in classes]
    if hasattr(model, "steps"):
        classes = list(getattr(model.steps[-1][1], "classes_", []))
    return [str(value) for value in classes]


def predict_profile_parameters(package: dict[str, object], X: pd.DataFrame) -> tuple[float, float, float, dict[str, object]]:
    model = package.get("profile_model")
    lookup = package.get("profile_lookup") or {}
    if model is None:
        raise ValueError("No profile_model found in model package.")

    predicted_profile = str(np.asarray(model.predict(X))[0])
    classes = model_classes(model)
    confidence = None
    top_profiles: list[dict[str, object]] = []
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))[0].astype(float)
        if classes and len(classes) == len(proba):
            confidence = float(proba[classes.index(predicted_profile)] if predicted_profile in classes else np.max(proba))
            order = np.argsort(proba)[::-1][:5]
            for idx in order:
                profile = classes[int(idx)]
                values = lookup.get(profile, {})
                top_profiles.append(
                    {
                        "profile_id": profile,
                        "probability": float(proba[int(idx)]),
                        "peak_width": values.get("peak_width"),
                        "threshold": values.get("threshold"),
                        "minimum_height": values.get("minimum_height"),
                        "training_count": values.get("count"),
                    }
                )

    values = lookup.get(predicted_profile)
    if not values:
        parts = predicted_profile.split("|")
        if len(parts) != 3:
            raise ValueError(f"Profile model returned unknown profile id: {predicted_profile}")
        values = {
            "peak_width": float(parts[0]),
            "threshold": float(parts[1]),
            "minimum_height": float(parts[2]),
            "count": None,
        }

    detail = {
        "model": "extra_trees_profile_classifier",
        "profile_id": predicted_profile,
        "confidence": confidence,
        "training_count": values.get("count"),
        "top_profiles": top_profiles,
    }
    return float(values["peak_width"]), float(values["threshold"]), float(values["minimum_height"]), detail


def nearest_confirmed_signal(
    X: pd.DataFrame,
    feature_cols: list[str],
    training_csv: Path = TRAINING_DATASET_CSV,
) -> dict[str, object] | None:
    if not training_csv.exists():
        return None
    df = pd.read_csv(training_csv)
    required = {"sample_dir", "arw_path", "peak_width_clean", "threshold_clean", "minimum_height_clean"}
    if not required.issubset(df.columns):
        return None
    usable = df.dropna(subset=["peak_width_clean", "threshold_clean", "minimum_height_clean"]).copy()
    if usable.empty:
        return None
    train_X = usable[feature_cols].apply(pd.to_numeric, errors="coerce")
    query_X = X[feature_cols].apply(pd.to_numeric, errors="coerce")
    med = train_X.median(numeric_only=True)
    mad = (train_X - med).abs().median(numeric_only=True).replace(0, 1.0)
    train_norm = ((train_X - med) / mad).fillna(0.0).to_numpy(dtype=float)
    query_norm = ((query_X - med) / mad).fillna(0.0).to_numpy(dtype=float)[0]
    distances = np.sqrt(((train_norm - query_norm) ** 2).mean(axis=1))
    order = np.argsort(distances)
    nearest_idx = int(order[0])
    row = usable.iloc[nearest_idx]
    neighbor_rows = []
    for idx in order[:5]:
        neighbor = usable.iloc[int(idx)]
        neighbor_rows.append(
            {
                "sample_dir": str(neighbor["sample_dir"]),
                "arw_path": str(neighbor["arw_path"]),
                "profile_id": profile_key(float(neighbor["peak_width_clean"]), float(neighbor["threshold_clean"]), float(neighbor["minimum_height_clean"])),
                "peak_width": float(neighbor["peak_width_clean"]),
                "threshold": float(neighbor["threshold_clean"]),
                "minimum_height": float(neighbor["minimum_height_clean"]),
                "distance": float(distances[int(idx)]),
            }
        )
    return {
        "sample_dir": str(row["sample_dir"]),
        "arw_path": str(row["arw_path"]),
        "profile_id": profile_key(float(row["peak_width_clean"]), float(row["threshold_clean"]), float(row["minimum_height_clean"])),
        "distance": float(distances[nearest_idx]),
        "peak_width": float(row["peak_width_clean"]),
        "threshold": float(row["threshold_clean"]),
        "minimum_height": float(row["minimum_height_clean"]),
        "nearest_neighbors": neighbor_rows,
    }


def assess_prediction_risk(
    core_details: dict[str, object],
    package: dict[str, object],
    profile_detail: dict[str, object] | None = None,
    nearest: dict[str, object] | None = None,
) -> dict[str, object]:
    reasons: list[str] = []
    risk_score = 0

    for target_name, detail_obj in core_details.items():
        if not isinstance(detail_obj, dict):
            continue
        tolerance = TARGET_TOLERANCE.get(str(target_name), 5.0)
        spread = float(detail_obj.get("candidate_spread") or 0.0)
        warnings_for_target = [str(item) for item in detail_obj.get("warnings", [])]
        override_reason = str(detail_obj.get("override_reason") or "")

        if spread > tolerance * 3.0:
            risk_score = max(risk_score, 2)
            reasons.append(f"{target_name}: candidate model spread {spread:.3g} exceeds {tolerance * 3.0:g}.")
        elif spread > tolerance * 1.5:
            risk_score = max(risk_score, 1)
            reasons.append(f"{target_name}: candidate model spread {spread:.3g} is elevated.")

        if any("training range" in warning for warning in warnings_for_target):
            risk_score = max(risk_score, 2)
            reasons.append(f"{target_name}: prediction is outside confirmed training range.")

        if override_reason:
            risk_score = max(risk_score, 2)
            reasons.append(f"{target_name}: local consensus override was used.")

    training_rows = int(package.get("training_rows_unique") or 0)
    if training_rows < 200:
        risk_score = max(risk_score, 1)
        reasons.append(f"training set has only {training_rows} unique ARW signals.")

    predicted_values = core_details.get("values") if isinstance(core_details, dict) else None
    if isinstance(predicted_values, dict):
        pred_profile = profile_key(
            float(predicted_values["peak_width"]),
            float(predicted_values["threshold"]),
            float(predicted_values["minimum_height"]),
        )
        if profile_detail is not None:
            confidence = profile_detail.get("confidence")
            if confidence is None:
                risk_score = max(risk_score, 1)
                reasons.append("auxiliary profile classifier confidence is unavailable.")
            else:
                confidence_value = float(confidence)
                if confidence_value < PROFILE_HIGH_RISK_CONFIDENCE:
                    risk_score = max(risk_score, 2)
                    reasons.append(f"auxiliary profile classifier confidence is low ({confidence_value:.3f}).")
                elif confidence_value < PROFILE_MEDIUM_RISK_CONFIDENCE:
                    risk_score = max(risk_score, 1)
                    reasons.append(f"auxiliary profile classifier confidence is moderate ({confidence_value:.3f}).")
            if profile_detail.get("profile_id") and str(profile_detail.get("profile_id")) != pred_profile:
                risk_score = max(risk_score, 1)
                reasons.append("auxiliary profile classifier disagrees with the primary parameter model.")
        if nearest is not None:
            distance = float(nearest.get("distance") or 0.0)
            nearest_profile = str(nearest.get("profile_id", ""))
            if distance <= NEAREST_SIGNAL_DISTANCE_MAX and nearest_profile != pred_profile:
                risk_score = max(risk_score, 1)
                reasons.append(f"nearest confirmed signal disagrees with primary model ({nearest_profile}, distance={distance:.3g}).")
            elif distance > NEAREST_SIGNAL_DISTANCE_MAX:
                risk_score = max(risk_score, 1)
                reasons.append(f"nearest confirmed signal is not very close (distance={distance:.3g}).")
                if distance > NEAREST_SIGNAL_DISTANCE_MAX * 4.0:
                    risk_score = max(risk_score, 2)
                    reasons.append(f"nearest confirmed signal is far outside the close-match gate (distance={distance:.3g}).")

    risk_level = RISK_SCORE_LABELS.get(risk_score, "high")
    return {
        "risk_level": risk_level,
        "auto_apply": risk_level == "low",
        "risk_reasons": sorted(set(reasons)),
    }


def assess_profile_risk(profile_detail: dict[str, object], nearest: dict[str, object] | None, package: dict[str, object]) -> dict[str, object]:
    reasons: list[str] = []
    risk_score = 0
    confidence = profile_detail.get("confidence")
    if confidence is None:
        risk_score = max(risk_score, 1)
        reasons.append("profile classifier confidence is unavailable.")
    else:
        confidence_value = float(confidence)
        if confidence_value < PROFILE_HIGH_RISK_CONFIDENCE:
            risk_score = max(risk_score, 2)
            reasons.append(f"profile classifier confidence is low ({confidence_value:.3f}).")
        elif confidence_value < PROFILE_MEDIUM_RISK_CONFIDENCE:
            risk_score = max(risk_score, 1)
            reasons.append(f"profile classifier confidence is moderate ({confidence_value:.3f}).")

    if nearest is not None:
        predicted_profile = str(profile_detail.get("profile_id", ""))
        nearest_profile = str(nearest.get("profile_id", ""))
        distance = float(nearest.get("distance") or 0.0)
        if distance <= NEAREST_SIGNAL_DISTANCE_MAX and predicted_profile != nearest_profile:
            risk_score = max(risk_score, 1)
            reasons.append(f"nearest confirmed signal disagrees with profile classifier ({nearest_profile}, distance={distance:.3g}).")
        elif distance > NEAREST_SIGNAL_DISTANCE_MAX:
            risk_score = max(risk_score, 1)
            reasons.append(f"nearest confirmed signal is not very close (distance={distance:.3g}).")

    training_rows = int(package.get("training_rows_unique") or 0)
    if training_rows < 200:
        risk_score = max(risk_score, 1)
        reasons.append(f"training set has only {training_rows} unique ARW signals.")

    risk_level = RISK_SCORE_LABELS.get(risk_score, "high")
    return {
        "risk_level": risk_level,
        "auto_apply": risk_level == "low",
        "risk_reasons": sorted(set(reasons)),
    }


def manual_prediction(input_path: Path, arw_path: Path) -> dict[str, object] | None:
    if not MANUAL_LABEL_CSV.exists():
        return None
    labels = pd.read_csv(MANUAL_LABEL_CSV)
    input_keys = path_keys(input_path)
    arw_keys = path_keys(arw_path)
    for _, row in labels.iterrows():
        if not paths_match(row.get("sample_dir", ""), input_keys) and not paths_match(row.get("arw_path", ""), arw_keys):
            continue
        event_text = str(row.get("event_table", "[]") or "[]")
        try:
            event_table = json.loads(event_text)
        except json.JSONDecodeError:
            event_table = []
        min_height = row.get("minimum_height")
        min_height_value = None if pd.isna(min_height) or str(min_height) == "" else float(min_height)
        return {
            "input": str(input_path),
            "arw_used": str(arw_path),
            "Algorithm": str(row.get("algorithm", "")),
            "Integration Start": float(row.get("integration_start")),
            "Integration End": float(row.get("integration_end")),
            "Peak Width": round(float(row.get("peak_width_sec")), 3),
            "Detection Threshold": round(float(row.get("detection_threshold")), 4),
            "Minimum Height": None if min_height_value is None else round(min_height_value, 4),
            "Minimum Height Probability": 1.0 if min_height_value is not None else 0.0,
            "Event Table": event_table,
            "risk_level": "low",
            "auto_apply": True,
            "risk_reasons": ["manual confirmed label"],
            "model_note": "Manual confirmed label from datasets/manual_parameter_labels.csv.",
        }
    return None


def predict(
    input_path: Path,
    use_manual_labels: bool = True,
    model_path: Path = MODEL_PATH,
    snap_parameters: bool | None = None,
) -> dict[str, object]:
    with model_path.open("rb") as f:
        package = pickle.load(f)
    if snap_parameters is None:
        snap_parameters = bool(package.get("snap_parameters_default", True))
    arw = find_arw(input_path)
    manual = manual_prediction(input_path, arw) if use_manual_labels else None
    if manual is not None:
        return manual
    X = feature_frame(arw, package["feature_cols"])
    warning_messages = []
    nearest = nearest_confirmed_signal(X, package["feature_cols"])
    override_note = ""
    profile_detail = None
    core_details: dict[str, object] = {}

    if package.get("primary_parameter_model") is not None:
        if package.get("profile_model") is not None:
            try:
                _, _, _, profile_detail = predict_profile_parameters(package, X)
            except Exception as exc:
                warning_messages.append(f"Auxiliary profile classifier failed: {exc}")
        peak_width, threshold, min_height, core_details = predict_primary_parameters(package, X, snap_parameters)
        min_height_prob = 1.0
        if isinstance(core_details, dict):
            warning_messages.extend(core_details.get("warnings", []))
        risk = assess_prediction_risk(core_details if isinstance(core_details, dict) else {}, package, profile_detail, nearest)
    elif package.get("profile_model") is not None:
        peak_width, threshold, min_height, profile_detail = predict_profile_parameters(package, X)
        min_height_prob = 1.0
        risk = assess_profile_risk(profile_detail, nearest, package)
        confidence = profile_detail.get("confidence")
        confidence_value = 0.0 if confidence is None else float(confidence)
        if (
            nearest is not None
            and float(nearest["distance"]) <= NEAREST_SIGNAL_DISTANCE_MAX
            and str(nearest.get("profile_id")) != str(profile_detail.get("profile_id"))
            and confidence_value < PROFILE_NEAREST_OVERRIDE_CONFIDENCE_MAX
        ):
            peak_width = float(nearest["peak_width"])
            threshold = float(nearest["threshold"])
            min_height = float(nearest["minimum_height"])
            override_note = (
                "Low-confidence profile output replaced by the single nearest confirmed training signal "
                f"({nearest['sample_dir']}, distance={float(nearest['distance']):.4g})."
            )
            warning_messages.append(override_note)
            risk["auto_apply"] = False
            risk["risk_reasons"] = sorted(set(list(risk["risk_reasons"]) + ["nearest confirmed signal override used"]))
    else:
        peak_width, threshold, min_height, core_details = predict_core_parameters(package, X)
        min_height_prob = 1.0 if package.get("core_target_models") else min_height_probability(package.get("min_height_classifier"), X)
        if snap_parameters:
            peak_width, threshold, min_height = snap_parameters_independently(package, peak_width, threshold, min_height)
        if isinstance(core_details, dict):
            for detail in core_details.values():
                if isinstance(detail, dict):
                    warning_messages.extend(detail.get("warnings", []))
        risk = assess_prediction_risk(core_details if isinstance(core_details, dict) else {}, package, profile_detail, nearest)
        if risk["risk_level"] == "high" and nearest is not None and float(nearest["distance"]) <= NEAREST_SIGNAL_DISTANCE_MAX:
            peak_width = float(nearest["peak_width"])
            threshold = float(nearest["threshold"])
            min_height = float(nearest["minimum_height"])
            override_note = (
                "High-risk model output replaced by the single nearest confirmed training signal "
                f"({nearest['sample_dir']}, distance={float(nearest['distance']):.4g})."
            )
            warning_messages.append(override_note)
            risk["auto_apply"] = False
            risk["risk_reasons"] = sorted(set(list(risk["risk_reasons"]) + ["nearest confirmed signal override used"]))

    return {
        "input": str(input_path),
        "arw_used": str(arw),
        "Algorithm": "ApexTrack",
        "Integration Start": 3.5,
        "Integration End": 6.6,
        "Peak Width": round(peak_width, 3),
        "Detection Threshold": round(threshold, 4),
        "Minimum Height": None if min_height is None else round(min_height, 4),
        "Minimum Height Probability": round(min_height_prob, 4),
        "Event Table": [],
        "Profile Prediction Details": profile_detail,
        "Core Prediction Details": core_details,
        "Nearest Confirmed Signal": nearest,
        "Prediction Warnings": warning_messages,
        "risk_level": risk["risk_level"],
        "auto_apply": risk["auto_apply"],
        "risk_reasons": risk["risk_reasons"],
        "model_note": f"Primary multi-output parameter regressor from parsed .arw signal ({model_path}); single-parameter snapping={'on' if snap_parameters else 'off'}; profile classifier is auxiliary only. {override_note}".strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict SEC Empower integration parameters.")
    parser.add_argument("input", help="Sample directory or .rpt file for the current baseline model.")
    parser.add_argument("--json", dest="json_path", help="Optional output JSON path.")
    parser.add_argument("--ignore-manual-labels", action="store_true", help="Do not return datasets/manual_parameter_labels.csv matches.")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Model pickle path.")
    parser.add_argument("--parameter-snapping", action="store_true", help="Force snapping to known parameter settings.")
    parser.add_argument("--no-parameter-snapping", action="store_true", help="Return raw continuous model values without snapping to known parameter settings.")
    args = parser.parse_args()
    snap_override = None
    if args.parameter_snapping:
        snap_override = True
    if args.no_parameter_snapping:
        snap_override = False

    result = predict(
        Path(args.input),
        use_manual_labels=not args.ignore_manual_labels,
        model_path=Path(args.model),
        snap_parameters=snap_override,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.json_path:
        out = Path(args.json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
