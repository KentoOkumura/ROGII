from __future__ import annotations

import argparse
import json
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from public_notebook_replay_audit import (
    build_well,
    configure_public_runtime,
    init_imputers,
    lik_pf,
    stable_seed,
)
from settings import ExperimentPaths, get_nested, load_config
from tvt_input_pfbeam_offset_calibration import (
    candidate_abs_from_replay,
    row_indices_from_ids,
    sha256_path,
    to_jsonable,
)

OUTPUT_PREFIX = "exp169_tvt_input_pfbeam_offset_calibration"
CANDIDATE_COLUMNS = ["pf_ancc", "beam_mean", "likpf_mean", "pf_z"]


@dataclass(frozen=True)
class ReplayMode:
    name: str
    anchor_idx: int
    mask_known_idx: np.ndarray
    description: str


def finite_rmse(pred: np.ndarray, true: np.ndarray) -> float | None:
    finite = np.isfinite(pred) & np.isfinite(true)
    if not finite.any():
        return None
    err = pred[finite].astype(np.float64) - true[finite].astype(np.float64)
    return float(np.sqrt(np.mean(err * err)))


def finite_mae(pred: np.ndarray, true: np.ndarray) -> float | None:
    finite = np.isfinite(pred) & np.isfinite(true)
    if not finite.any():
        return None
    return float(np.mean(np.abs(pred[finite].astype(np.float64) - true[finite].astype(np.float64))))


def choose_replay_mode(
    name: str,
    horizontal: pd.DataFrame,
    config: dict[str, Any],
) -> ReplayMode:
    known = np.flatnonzero(pd.to_numeric(horizontal["TVT_input"], errors="coerce").notna().to_numpy())
    if len(known) < 2:
        raise ValueError("well has fewer than two known TVT_input rows")

    min_known = int(get_nested(config, "model.offset_calibration.min_known_prefix_rows") or 80)
    holdout_rows = int(get_nested(config, "model.offset_calibration.prefix_holdout_rows") or 256)
    full_anchor_rows = int(get_nested(config, "visualization.full_known_anchor_rows") or min_known)

    if name == "exp169_holdout_tail":
        start = max(min_known, len(known) - holdout_rows)
        description = (
            "exp169-style replay: mask the last known-prefix holdout rows plus the original tail."
        )
    elif name == "full_known_backtest":
        start = max(1, min(full_anchor_rows, len(known) - 1))
        description = (
            "all-interval backtest: keep only an early known prefix, then replay across "
            "the remaining known interval and original tail."
        )
    else:
        raise ValueError(f"unsupported visualization replay mode: {name}")

    if start >= len(known):
        raise ValueError(f"replay start exceeds known prefix length: start={start}, known={len(known)}")
    return ReplayMode(
        name=name,
        anchor_idx=int(known[start - 1]),
        mask_known_idx=known[start:].astype(np.int64),
        description=description,
    )


def materialize_replay_candidates(replay: pd.DataFrame) -> pd.DataFrame:
    replay = replay.copy()
    for candidate in CANDIDATE_COLUMNS:
        values = candidate_abs_from_replay(replay, candidate)
        if values is not None:
            replay[candidate] = values.astype(np.float32)
    return replay


def replay_well_interval(
    well: str,
    mode_name: str,
    paths: ExperimentPaths,
    config: dict[str, Any],
    temp_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    hw_path = paths.train_data_dir / f"{well}__horizontal_well.csv"
    tw_path = paths.train_data_dir / f"{well}__typewell.csv"
    if not hw_path.exists() or not tw_path.exists():
        raise FileNotFoundError(f"missing raw train files for well={well}")

    horizontal = pd.read_csv(hw_path, low_memory=False)
    mode = choose_replay_mode(mode_name, horizontal, config)
    masked = horizontal.copy()
    masked.loc[mode.mask_known_idx, "TVT_input"] = np.nan

    temp_hw = temp_root / f"{well}__horizontal_well.csv"
    masked.to_csv(temp_hw, index=False)
    replay = build_well(temp_hw, tw_path, is_train=True)
    if replay is None or len(replay) == 0:
        raise RuntimeError(f"replay failed for well={well} mode={mode_name}")
    replay = replay.copy()
    replay["row_idx"] = row_indices_from_ids(replay["id"])

    if bool(get_nested(config, "visualization.replay_runtime.use_likpf")):
        tw = pd.read_csv(tw_path).sort_values("TVT")
        seed_key = "exp169_likpf_prefix" if mode_name == "exp169_holdout_tail" else "exp169_viz_likpf"
        out, ev_index, _ = lik_pf(masked, tw, seed_base=stable_seed(seed_key, well))
        if len(ev_index) == len(replay) and "pf_mean" in out:
            replay["likpf_mean"] = np.asarray(out["pf_mean"], dtype=np.float32)

    replay = materialize_replay_candidates(replay)
    row_idx = np.arange(len(horizontal), dtype=np.int64)
    plot_frame = horizontal[["MD", "TVT", "TVT_input", "GR", "Z"]].copy()
    plot_frame["row_idx"] = row_idx
    plot_frame["well"] = well
    plot_frame["mode"] = mode.name
    plot_frame["region"] = np.where(
        pd.to_numeric(plot_frame["TVT_input"], errors="coerce").notna(),
        "observed_tvt_input",
        "original_hidden_tail",
    )
    replay_cols = ["row_idx", *[c for c in CANDIDATE_COLUMNS if c in replay]]
    plot_frame = plot_frame.merge(replay[replay_cols], on="row_idx", how="left", validate="one_to_one")

    original_known = pd.to_numeric(horizontal["TVT_input"], errors="coerce").notna().to_numpy()
    eval_mask = np.isfinite(plot_frame[[c for c in CANDIDATE_COLUMNS if c in plot_frame]].to_numpy()).any(axis=1)
    known_eval = original_known & eval_mask
    hidden_eval = (~original_known) & eval_mask

    metrics: dict[str, Any] = {
        "well": well,
        "mode": mode.name,
        "description": mode.description,
        "anchor_idx": int(mode.anchor_idx),
        "anchor_md": float(horizontal.loc[mode.anchor_idx, "MD"]),
        "anchor_tvt_input": float(horizontal.loc[mode.anchor_idx, "TVT_input"]),
        "known_holdout_rows": int(len(mode.mask_known_idx)),
        "plot_rows": int(len(plot_frame)),
        "prediction_rows": int(eval_mask.sum()),
        "known_prediction_rows": int(known_eval.sum()),
        "hidden_prediction_rows": int(hidden_eval.sum()),
    }
    true = pd.to_numeric(plot_frame["TVT"], errors="coerce").to_numpy(np.float32)
    for candidate in CANDIDATE_COLUMNS:
        if candidate not in plot_frame:
            continue
        pred = pd.to_numeric(plot_frame[candidate], errors="coerce").to_numpy(np.float32)
        metrics[f"{candidate}_rmse_all_pred"] = finite_rmse(pred[eval_mask], true[eval_mask])
        metrics[f"{candidate}_mae_all_pred"] = finite_mae(pred[eval_mask], true[eval_mask])
        metrics[f"{candidate}_rmse_known_pred"] = finite_rmse(pred[known_eval], true[known_eval])
        metrics[f"{candidate}_mae_known_pred"] = finite_mae(pred[known_eval], true[known_eval])
        metrics[f"{candidate}_rmse_hidden_tail"] = finite_rmse(pred[hidden_eval], true[hidden_eval])
        metrics[f"{candidate}_mae_hidden_tail"] = finite_mae(pred[hidden_eval], true[hidden_eval])
    return plot_frame, metrics


def downsample_frame(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    idx = np.linspace(0, len(frame) - 1, max_points).round().astype(int)
    return frame.iloc[np.unique(idx)].copy()


def plot_all_interval(
    frame: pd.DataFrame,
    metrics: dict[str, Any],
    output_path: Path,
    config: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    max_points = int(get_nested(config, "visualization.max_points_per_plot") or 6000)
    frame = downsample_frame(frame.sort_values("MD"), max_points)
    x = pd.to_numeric(frame["MD"], errors="coerce")
    true = pd.to_numeric(frame["TVT"], errors="coerce")
    tvt_input = pd.to_numeric(frame["TVT_input"], errors="coerce")
    gr = pd.to_numeric(frame["GR"], errors="coerce")
    z = pd.to_numeric(frame["Z"], errors="coerce")

    colors = {
        "pf_ancc": "#dc2626",
        "beam_mean": "#f59e0b",
        "likpf_mean": "#7c3aed",
        "pf_z": "#059669",
    }
    labels = {
        "pf_ancc": "PF ANCC",
        "beam_mean": "Beam mean",
        "likpf_mean": "likPF mean",
        "pf_z": "PF-Z",
    }

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15.0, 10.5),
        dpi=140,
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.4, 1.3]},
    )
    ax_path, ax_err, ax_geo = axes

    last_known_rows = frame[pd.to_numeric(frame["TVT_input"], errors="coerce").notna()]
    if not last_known_rows.empty:
        last_known_md = float(last_known_rows["MD"].max())
        ax_path.axvspan(float(x.min()), last_known_md, color="#dbeafe", alpha=0.22, label="observed TVT_input region")
        ax_path.axvline(last_known_md, color="#1d4ed8", linestyle="--", linewidth=1.1, alpha=0.75, label="original prediction start")
        ax_err.axvline(last_known_md, color="#1d4ed8", linestyle="--", linewidth=1.0, alpha=0.6)
    ax_path.axvline(float(metrics["anchor_md"]), color="#111827", linestyle=":", linewidth=1.4, label="replay anchor")
    ax_err.axvline(float(metrics["anchor_md"]), color="#111827", linestyle=":", linewidth=1.2)

    ax_path.plot(x, true, color="black", linewidth=2.1, label="true TVT", zorder=4)
    ax_path.plot(x, tvt_input, color="#2563eb", linewidth=1.4, alpha=0.85, label="TVT_input")

    for candidate in CANDIDATE_COLUMNS:
        if candidate not in frame:
            continue
        values = pd.to_numeric(frame[candidate], errors="coerce")
        ax_path.plot(
            x,
            values,
            color=colors.get(candidate, None),
            linewidth=1.25,
            alpha=0.88,
            label=labels.get(candidate, candidate),
        )
        ax_err.plot(
            x,
            values - true,
            color=colors.get(candidate, None),
            linewidth=1.05,
            alpha=0.78,
            label=labels.get(candidate, candidate),
        )

    ax_err.axhline(0.0, color="#374151", linewidth=0.9, alpha=0.75)
    ax_err.axhline(10.0, color="#9ca3af", linestyle="--", linewidth=0.8)
    ax_err.axhline(-10.0, color="#9ca3af", linestyle="--", linewidth=0.8)

    ax_geo.plot(x, gr, color="#16a34a", linewidth=0.9, alpha=0.85, label="GR")
    ax_geo_twin = ax_geo.twinx()
    ax_geo_twin.plot(x, z, color="#64748b", linewidth=1.0, alpha=0.75, label="Z")

    title = (
        f"{metrics['well']} / {metrics['mode']} | "
        f"anchor row={metrics['anchor_idx']} MD={metrics['anchor_md']:.1f} | "
        f"known pred={metrics['known_prediction_rows']:,} hidden pred={metrics['hidden_prediction_rows']:,}"
    )
    ax_path.set_title(title)
    ax_path.set_ylabel("TVT")
    ax_err.set_ylabel("candidate - true TVT")
    ax_geo.set_ylabel("GR")
    ax_geo_twin.set_ylabel("Z")
    ax_geo.set_xlabel("MD")

    for ax in (ax_path, ax_err, ax_geo):
        ax.grid(True, color="#e5e7eb", linewidth=0.7, alpha=0.8)
    ax_path.legend(loc="best", fontsize=8, ncol=3)
    ax_err.legend(loc="best", fontsize=8, ncol=4)
    lines, line_labels = ax_geo.get_legend_handles_labels()
    twin_lines, twin_labels = ax_geo_twin.get_legend_handles_labels()
    ax_geo.legend(lines + twin_lines, line_labels + twin_labels, loc="best", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def write_html_index(manifest: pd.DataFrame, html_path: Path, title: str) -> None:
    rows = []
    for item in manifest.to_dict("records"):
        rel = Path(str(item["plot_path"])).name
        rows.append(
            "<section>"
            f"<h2>{item['well']} / {item['mode']}</h2>"
            f"<p>anchor row {item['anchor_idx']} | known pred {item['known_prediction_rows']} | "
            f"hidden pred {item['hidden_prediction_rows']}</p>"
            f"<img src='all_interval_pfbeam_plots/{rel}' style='max-width:100%;height:auto'>"
            "</section>"
        )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:24px;}"
        "section{margin-bottom:32px;} img{border:1px solid #ddd;}</style>"
        "</head><body>"
        f"<h1>{title}</h1>"
        + "\n".join(rows)
        + "</body></html>"
    )
    html_path.write_text(html)


def run_visualization(
    config: dict[str, Any] | None = None,
    paths: ExperimentPaths | None = None,
    *,
    well_ids: list[str] | None = None,
) -> dict[str, Any]:
    start = time.time()
    config = load_config() if config is None else config
    paths = ExperimentPaths() if paths is None else paths
    paths.ensure_output_dirs()

    replay_cfg = get_nested(config, "visualization.replay_runtime") or {}
    configure_public_runtime(
        data_dir=paths.raw_data_dir,
        output_dir=paths.artifacts_dir,
        n_jobs=int(replay_cfg.get("n_jobs", 2)),
        pf_seeds=int(replay_cfg.get("pf_seeds", 32)),
        pf_particles=int(replay_cfg.get("pf_particles", 300)),
    )
    train_wells = sorted(
        path.stem.replace("__horizontal_well", "")
        for path in paths.train_data_dir.glob("*__horizontal_well.csv")
    )
    init_imputers(train_wells)

    wells = well_ids or [str(well) for well in get_nested(config, "visualization.well_ids") or []]
    if not wells:
        raise ValueError("visualization.well_ids is empty")
    modes = [str(mode) for mode in get_nested(config, "visualization.modes") or ["exp169_holdout_tail"]]

    plots_dir = paths.artifacts_dir / "all_interval_pfbeam_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="exp169_all_interval_viz_") as tmp:
        temp_root = Path(tmp)
        for well in wells:
            for mode in modes:
                frame, metrics = replay_well_interval(well, mode, paths, config, temp_root)
                output_path = plots_dir / f"{mode}__{well}.png"
                plot_all_interval(frame, metrics, output_path, config)
                manifest_rows.append(
                    {
                        **metrics,
                        "plot_path": str(output_path.relative_to(paths.artifacts_dir)),
                        "plot_sha256": sha256_path(output_path),
                    }
                )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_all_interval_plot_manifest.csv"
    html_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_all_interval_plot_index.html"
    summary_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_all_interval_plot_summary.json"
    zip_path = paths.artifacts_dir / f"{OUTPUT_PREFIX}_all_interval_plots.zip"
    manifest.to_csv(manifest_path, index=False)
    write_html_index(manifest, html_path, "exp169 all-interval PF/Beam replay plots")
    if bool(get_nested(config, "visualization.zip_plots")):
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for png_path in sorted(plots_dir.glob("*.png")):
                zf.write(png_path, arcname=png_path.name)

    summary = {
        "experiment": OUTPUT_PREFIX,
        "mode": "all_interval_pfbeam_replay_plots",
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": time.time() - start,
        "wells": wells,
        "modes": modes,
        "plot_count": int(len(manifest)),
        "artifacts": {
            "manifest": str(manifest_path),
            "html_index": str(html_path),
            "plots_dir": str(plots_dir),
            "plots_zip": str(zip_path) if zip_path.exists() else None,
        },
        "artifact_sha256": {
            "manifest": sha256_path(manifest_path),
            "html_index": sha256_path(html_path),
            "plots_zip": sha256_path(zip_path) if zip_path.exists() else None,
        },
    }
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wells", nargs="*", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_visualization(well_ids=args.wells)
    print(json.dumps(to_jsonable(result), indent=2, sort_keys=True))
