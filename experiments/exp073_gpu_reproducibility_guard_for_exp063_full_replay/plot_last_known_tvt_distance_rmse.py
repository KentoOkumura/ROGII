from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PREDICTIONS = Path(
    "/tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/"
    "train_v2/artifacts/exp063_full_replay_repro_guard_predictions.csv.gz"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"

ABS_BINS = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, np.inf]
ABS_LABELS = [
    "000_001",
    "001_002",
    "002_005",
    "005_010",
    "010_020",
    "020_040",
    "040_060",
    "060_080",
    "080_plus",
]
SIGNED_BINS = [
    -np.inf,
    -80.0,
    -60.0,
    -40.0,
    -20.0,
    -10.0,
    -5.0,
    -2.0,
    -1.0,
    0.0,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    40.0,
    60.0,
    80.0,
    np.inf,
]
SIGNED_LABELS = [
    "lt_m080",
    "m080_m060",
    "m060_m040",
    "m040_m020",
    "m020_m010",
    "m010_m005",
    "m005_m002",
    "m002_m001",
    "m001_000",
    "000_001",
    "001_002",
    "002_005",
    "005_010",
    "010_020",
    "020_040",
    "040_060",
    "060_080",
    "080_plus",
]


def rmse(values: pd.Series) -> float:
    array = values.to_numpy(np.float64)
    return float(np.sqrt(np.mean(np.square(array))))


def summarize(frame: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    summary = (
        frame.groupby(bucket_col, observed=False)
        .agg(
            rows=("id", "size"),
            wells=("well", "nunique"),
            distance_min=("distance_tvt", "min"),
            distance_max=("distance_tvt", "max"),
            distance_abs_mean=("distance_abs_tvt", "mean"),
            error_mean=("error_tvt", "mean"),
            mae_tvt=("error_tvt", lambda value: float(np.mean(np.abs(value)))),
            rmse_tvt=("error_tvt", rmse),
        )
        .reset_index()
    )
    return summary[summary["rows"] > 0].copy()


def write_plot(summary: pd.DataFrame, output_path: Path, *, overall_rmse: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = summary["distance_abs_bucket"].astype(str).to_list()
    rmse_values = summary["rmse_tvt"].to_numpy(np.float64)
    row_counts = summary["rows"].to_numpy(np.int64)

    positions = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    bars = ax.bar(positions, rmse_values, color="#2f6f8f", width=0.72)
    ax.axhline(
        overall_rmse,
        color="#9b2c2c",
        linestyle="--",
        linewidth=1.3,
        label=f"overall RMSE {overall_rmse:.4f}",
    )
    ax.set_title("exp073 lgb_mean RMSE by distance from last_known_tvt")
    ax.set_xlabel("|target_tvt - last_known_tvt| bucket")
    ax.set_ylabel("RMSE TVT")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")

    for bar, value in zip(bars, rmse_values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax2 = ax.twinx()
    ax2.plot(positions, row_counts, color="#555555", marker="o", linewidth=1.2, label="rows")
    ax2.set_yscale("log")
    ax2.set_ylabel("Rows (log scale)")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def build_outputs(predictions_path: Path, output_dir: Path, model: str) -> dict[str, Path]:
    if not predictions_path.exists() or predictions_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing non-empty predictions file: {predictions_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = ["id", "well", "model", "target_tvt", "last_known_tvt", "pred_tvt"]
    frame = pd.read_csv(predictions_path, usecols=usecols)
    frame = frame[frame["model"].astype(str).eq(model)].copy()
    if frame.empty:
        raise ValueError(f"No rows found for model={model}")

    frame["distance_tvt"] = frame["target_tvt"].to_numpy(np.float64) - frame["last_known_tvt"].to_numpy(np.float64)
    frame["distance_abs_tvt"] = np.abs(frame["distance_tvt"].to_numpy(np.float64))
    frame["error_tvt"] = frame["pred_tvt"].to_numpy(np.float64) - frame["target_tvt"].to_numpy(np.float64)
    frame["distance_abs_bucket"] = pd.cut(
        frame["distance_abs_tvt"],
        bins=ABS_BINS,
        labels=ABS_LABELS,
        include_lowest=True,
        right=False,
    )
    frame["distance_signed_bucket"] = pd.cut(
        frame["distance_tvt"],
        bins=SIGNED_BINS,
        labels=SIGNED_LABELS,
        include_lowest=True,
        right=False,
    )

    abs_summary = summarize(frame, "distance_abs_bucket")
    signed_summary = summarize(frame, "distance_signed_bucket")
    overall_rmse = float(np.sqrt(np.mean(np.square(frame["error_tvt"].to_numpy(np.float64)))))

    abs_csv = output_dir / f"exp073_{model}_rmse_by_last_known_tvt_abs_distance.csv"
    signed_csv = output_dir / f"exp073_{model}_rmse_by_last_known_tvt_signed_distance.csv"
    png = output_dir / f"exp073_{model}_rmse_by_last_known_tvt_abs_distance.png"
    abs_summary.to_csv(abs_csv, index=False)
    signed_summary.to_csv(signed_csv, index=False)
    write_plot(abs_summary, png, overall_rmse=overall_rmse)

    return {"abs_csv": abs_csv, "signed_csv": signed_csv, "png": png}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="lgb_mean")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_outputs(args.predictions, args.output_dir, args.model)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
