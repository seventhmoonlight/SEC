from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "input",
    "algorithm_ok",
    "start_err",
    "end_err",
    "peak_width_err",
    "threshold_err",
    "min_height_err",
    "event_table_ok",
]


def as_float(value: object) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def as_event(value: object) -> object:
    if value is None or value == "":
        return []
    return json.loads(str(value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SEC parameter predictions with confirmed truth.")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--report", help="Optional summary JSON path. Defaults to <csv stem>_summary.json.")
    args = parser.parse_args()

    pred_rows = {row["input"]: row for row in read_csv(Path(args.pred))}
    rows: list[dict[str, object]] = []
    for truth in read_csv(Path(args.truth)):
        pred = pred_rows[truth["input"]]
        rows.append(
            {
                "input": truth["input"],
                "algorithm_ok": pred["Algorithm"] == truth["Algorithm"],
                "start_err": round(abs(as_float(pred["Integration Start"]) - as_float(truth["Integration Start"])), 6),
                "end_err": round(abs(as_float(pred["Integration End"]) - as_float(truth["Integration End"])), 6),
                "peak_width_err": round(abs(as_float(pred["Peak Width"]) - as_float(truth["Peak Width"])), 6),
                "threshold_err": round(abs(as_float(pred["Detection Threshold"]) - as_float(truth["Detection Threshold"])), 6),
                "min_height_err": round(abs(as_float(pred["Minimum Height"]) - as_float(truth["Minimum Height"])), 6),
                "event_table_ok": as_event(pred["Event Table"]) == as_event(truth["Event Table"]),
            }
        )

    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary_obj = summary(rows)
    report = Path(args.report) if args.report else out.with_name(f"{out.stem}_summary.json")
    report.write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary_obj, ensure_ascii=False, indent=2))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def summary(rows: list[dict[str, object]]) -> dict[str, object]:
    peak_width_mae = mean(rows, "peak_width_err")
    threshold_mae = mean(rows, "threshold_err")
    min_height_mae = mean(rows, "min_height_err")
    return {
        "rows": len(rows),
        "priority": "Peak Width > Detection Threshold > Minimum Height; other fields are reported as secondary diagnostics.",
        "priority_parameter_score": round(peak_width_mae / 10.0 + threshold_mae / 5.0 + min_height_mae / 30.0, 6),
        "peak_width_mae": peak_width_mae,
        "peak_width_within_10sec": within(rows, "peak_width_err", 10.0),
        "threshold_mae": threshold_mae,
        "threshold_within_5": within(rows, "threshold_err", 5.0),
        "min_height_mae": min_height_mae,
        "min_height_within_10": within(rows, "min_height_err", 10.0),
        "algorithm_errors": sum(1 for row in rows if not row["algorithm_ok"]),
        "event_table_errors": sum(1 for row in rows if not row["event_table_ok"]),
        "start_mae": mean(rows, "start_err"),
        "end_mae": mean(rows, "end_err"),
    }


def mean(rows: list[dict[str, object]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(float(row[key]) for row in rows) / len(rows), 6)


def within(rows: list[dict[str, object]], key: str, limit: float) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for row in rows if float(row[key]) <= limit) / len(rows), 6)


if __name__ == "__main__":
    main()
