from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from predict_sec_parameters import MODEL_PATH, predict


FIELDS = [
    "input",
    "arw_used",
    "Algorithm",
    "Integration Start",
    "Integration End",
    "Peak Width",
    "Detection Threshold",
    "Minimum Height",
    "Minimum Height Probability",
    "Event Table",
    "risk_level",
    "auto_apply",
    "risk_reasons",
    "model_note",
]


def iter_inputs(path: Path) -> list[Path]:
    if path.is_dir():
        children = sorted([p for p in path.iterdir() if p.is_dir()], key=sort_key)
        if children:
            return children
    return [path]


def sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.name), path.name
    except ValueError:
        return 10**9, path.name


def row_for(result: dict[str, object]) -> dict[str, object]:
    row = {field: result.get(field, "") for field in FIELDS}
    row["Event Table"] = json.dumps(result.get("Event Table", []), ensure_ascii=False, separators=(",", ":"))
    row["risk_reasons"] = json.dumps(result.get("risk_reasons", []), ensure_ascii=False, separators=(",", ":"))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch predict SEC Empower integration parameters.")
    parser.add_argument("input", help="Sample directory, .arw file, or directory containing sample subdirectories.")
    parser.add_argument("--csv", required=True, help="Output CSV path.")
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

    rows = [
        row_for(
            predict(
                path,
                use_manual_labels=not args.ignore_manual_labels,
                model_path=Path(args.model),
                snap_parameters=snap_override,
            )
        )
        for path in iter_inputs(Path(args.input))
    ]
    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
