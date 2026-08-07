from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PREDICTIONS = Path(
    "/tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/"
    "train_v2/artifacts/exp063_full_replay_repro_guard_predictions.csv.gz"
)
DEFAULT_TRAIN_DIR = Path("data/raw/train")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"

STEP_BINS = [1, 11, 26, 51, 101, 251, 501, 1001, 1501, np.inf]
STEP_LABELS = [
    "001_010",
    "011_025",
    "026_050",
    "051_100",
    "101_250",
    "251_500",
    "501_1000",
    "1001_1500",
    "1501_plus",
]


def rmse(values: pd.Series) -> float:
    array = values.to_numpy(np.float64)
    return float(np.sqrt(np.mean(np.square(array))))


def load_last_known_index(train_dir: Path, wells: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, int | str]] = []
    for well in sorted(wells.astype(str).unique()):
        path = train_dir / f"{well}__horizontal_well.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing train horizontal well file: {path}")
        tvt_input = pd.read_csv(path, usecols=["TVT_input"])["TVT_input"]
        known = pd.to_numeric(tvt_input, errors="coerce").dropna()
        if known.empty:
            raise ValueError(f"No known TVT_input prefix rows for well={well}")
        rows.append(
            {
                "well": well,
                "last_known_index": int(known.index[-1]),
                "known_prefix_rows": int(len(known)),
                "well_rows": int(len(tvt_input)),
            }
        )
    return pd.DataFrame(rows)


def add_step_columns(frame: pd.DataFrame, train_dir: Path) -> pd.DataFrame:
    row_index = frame["id"].astype(str).str.rsplit("_", n=1).str[-1]
    frame = frame.copy()
    frame["row_index"] = pd.to_numeric(row_index, errors="raise").astype(np.int32)
    anchors = load_last_known_index(train_dir, frame["well"])
    frame = frame.merge(anchors, on="well", how="left", validate="many_to_one")
    if frame["last_known_index"].isna().any():
        raise ValueError("Missing last_known_index after raw train merge")
    frame["step_from_last_known"] = frame["row_index"] - frame["last_known_index"]
    if (frame["step_from_last_known"] <= 0).any():
        bad = frame.loc[frame["step_from_last_known"] <= 0, ["id", "well", "row_index", "last_known_index"]].head()
        raise ValueError(f"Non-positive step_from_last_known rows found:\n{bad}")
    return frame


def summarize_by_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby("step_bucket", observed=False)
        .agg(
            rows=("id", "size"),
            wells=("well", "nunique"),
            step_min=("step_from_last_known", "min"),
            step_max=("step_from_last_known", "max"),
            step_mean=("step_from_last_known", "mean"),
            error_mean=("error_tvt", "mean"),
            mae_tvt=("error_tvt", lambda value: float(np.mean(np.abs(value)))),
            rmse_tvt=("error_tvt", rmse),
        )
        .reset_index()
    )
    return summary[summary["rows"] > 0].copy()


def summarize_by_well(frame: pd.DataFrame, threshold: int) -> pd.DataFrame:
    tail = frame[frame["step_from_last_known"] >= threshold].copy()
    total_sse = float(frame["sq_error"].sum())
    tail_sse = float(tail["sq_error"].sum())
    by_well = (
        tail.groupby("well", as_index=False)
        .agg(
            rows=("id", "size"),
            step_min=("step_from_last_known", "min"),
            step_max=("step_from_last_known", "max"),
            step_mean=("step_from_last_known", "mean"),
            target_delta_mean=("target_delta", "mean"),
            error_mean=("error_tvt", "mean"),
            mae_tvt=("error_tvt", lambda value: float(np.mean(np.abs(value)))),
            rmse_tvt=("error_tvt", rmse),
            sse=("sq_error", "sum"),
        )
        .sort_values(["sse", "rows"], ascending=[False, False])
    )
    by_well["tail_mse_share_pct"] = by_well["sse"] / tail_sse * 100.0
    by_well["overall_mse_share_pct"] = by_well["sse"] / total_sse * 100.0
    by_well.insert(1, "threshold_step", int(threshold))
    return by_well


def write_plot(summary: pd.DataFrame, output_path: Path, *, overall_rmse: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = summary["step_bucket"].astype(str).to_list()
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
    ax.set_title("exp073 lgb_mean RMSE by row steps from last known TVT")
    ax.set_xlabel("row steps from last known TVT_input")
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


def build_outputs(
    predictions_path: Path,
    train_dir: Path,
    output_dir: Path,
    model: str,
    tail_thresholds: list[int],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usecols = ["id", "well", "model", "target_tvt", "last_known_tvt", "target_delta", "pred_tvt"]
    frame = pd.read_csv(predictions_path, usecols=usecols, dtype={"id": str, "well": str})
    frame = frame[frame["model"].astype(str).eq(model)].copy()
    if frame.empty:
        raise ValueError(f"No rows found for model={model}")
    frame = add_step_columns(frame, train_dir)
    frame["error_tvt"] = frame["pred_tvt"].to_numpy(np.float64) - frame["target_tvt"].to_numpy(np.float64)
    frame["sq_error"] = np.square(frame["error_tvt"].to_numpy(np.float64))
    frame["step_bucket"] = pd.cut(
        frame["step_from_last_known"],
        bins=STEP_BINS,
        labels=STEP_LABELS,
        include_lowest=True,
        right=False,
    )

    overall_rmse = float(np.sqrt(frame["sq_error"].mean()))
    summary = summarize_by_bucket(frame)
    summary["row_share_pct"] = summary["rows"] / len(frame) * 100.0
    summary["mse_share_pct"] = summary["rmse_tvt"] ** 2 * summary["rows"] / frame["sq_error"].sum() * 100.0

    bucket_csv = output_dir / f"exp073_{model}_rmse_by_last_known_step.csv"
    png = output_dir / f"exp073_{model}_rmse_by_last_known_step.png"
    summary.to_csv(bucket_csv, index=False)
    write_plot(summary, png, overall_rmse=overall_rmse)

    outputs = {"bucket_csv": bucket_csv, "png": png}
    for threshold in tail_thresholds:
        by_well = summarize_by_well(frame, threshold)
        path = output_dir / f"exp073_{model}_tail_step_ge{threshold}_by_well.csv"
        by_well.to_csv(path, index=False)
        outputs[f"tail_step_ge{threshold}_by_well_csv"] = path
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="lgb_mean")
    parser.add_argument("--tail-thresholds", type=int, nargs="+", default=[500, 1000, 1500])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_outputs(
        args.predictions,
        args.train_dir,
        args.output_dir,
        args.model,
        args.tail_thresholds,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
