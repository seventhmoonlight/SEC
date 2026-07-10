from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from train_sec_parameter_model import ROOT, read_arw, smooth_signal


OUTPUT_DIR = ROOT / "outputs" / "plots"


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


def safe_output_name(arw_path: Path) -> str:
    try:
        rel = arw_path.relative_to(ROOT)
        stem = "__".join(rel.with_suffix("").parts)
    except ValueError:
        stem = arw_path.stem
    return f"{stem}.png"


def plot_arw(input_path: str) -> Path:
    arw_path = resolve_arw_path(input_path)
    chrom = read_arw(arw_path)
    time = chrom.time
    signal = chrom.signal
    smoothed = smooth_signal(signal)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / safe_output_name(arw_path)

    fig, ax = plt.subplots(figsize=(11, 5), dpi=160)
    ax.plot(time, signal, color="black", linewidth=0.75, label="Raw")
    ax.plot(time, smoothed, color="#d95f02", linewidth=1.0, alpha=0.85, label="Smoothed")

    main_idx = int(np.nanargmax(smoothed))
    ax.axvline(time[main_idx], color="#1b9e77", linewidth=0.9, linestyle="--", alpha=0.8)
    ax.text(
        time[main_idx],
        smoothed[main_idx],
        f"  RT {time[main_idx]:.3f}",
        color="#1b9e77",
        fontsize=9,
        va="bottom",
    )

    ax.set_title(str(arw_path.relative_to(ROOT) if arw_path.is_relative_to(ROOT) else arw_path))
    ax.set_xlabel("Minutes")
    ax.set_ylabel("Response")
    ax.grid(True, color="#d9d9d9", linewidth=0.5, alpha=0.7)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        raise SystemExit("Usage: python scripts\\plot_arw.py <arw-file-or-sample-dir>")
    output_path = plot_arw(argv[1])
    print(output_path)


if __name__ == "__main__":
    main(sys.argv)
