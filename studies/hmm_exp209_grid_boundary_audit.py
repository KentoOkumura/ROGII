#!/usr/bin/env python3
"""Audit exp209 exact-HMM grid support and boundary-related error.

This is a readout-only study.  It reconstructs the exact per-well TVT grid
used by exp209, joins it to the saved exp270 posterior-mean candidate, and
measures whether large or persistent offsets are explained by:

1. true TVT leaving the fixed grid,
2. the posterior mean sticking near a grid edge, or
3. neither of the above.

No prediction, model, threshold, or submission artifact is changed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_CANDIDATES = Path(
    "experiments/exp270_exact_hmm_posterior_mode_candidate_audit/"
    "kaggle/output/aggregate_v4/artifacts/"
    "exp270_exact_hmm_posterior_mode_candidate_audit_candidates.csv.gz"
)
DEFAULT_OUTPUT = Path("studies/hmm_exp209_grid_boundary_audit_20260725")
STEP_FT = 0.35
BAND_PAD_FT = 100.0
TYPEWELL_PAD_FT = 40.0
EDGE_WIDTH_FT = 3.5
N_RATES = 41
RATE_SPAN_FLOOR = 0.10
SIG_R = 0.002
MOMENTUM = 0.998


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunksize", type=int, default=400_000)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def exact_grid_bounds(
    last_tvt: float,
    typewell_min: float,
    typewell_max: float,
) -> tuple[float, float, int, str, str]:
    grid_min = max(typewell_min - TYPEWELL_PAD_FT, last_tvt - BAND_PAD_FT)
    requested_max = min(typewell_max + TYPEWELL_PAD_FT, last_tvt + BAND_PAD_FT)
    grid = np.arange(grid_min, requested_max + STEP_FT, STEP_FT, dtype=np.float64)
    low_source = (
        "typewell_min_minus_40"
        if typewell_min - TYPEWELL_PAD_FT >= last_tvt - BAND_PAD_FT
        else "last_tvt_minus_100"
    )
    high_source = (
        "typewell_max_plus_40"
        if typewell_max + TYPEWELL_PAD_FT <= last_tvt + BAND_PAD_FT
        else "last_tvt_plus_100"
    )
    return float(grid[0]), float(grid[-1]), int(grid.size), low_source, high_source


def build_grid_ledger(train_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizontal_path in sorted(train_dir.glob("*__horizontal_well.csv")):
        well = horizontal_path.name.replace("__horizontal_well.csv", "")
        typewell_path = train_dir / f"{well}__typewell.csv"
        if not typewell_path.exists():
            continue
        horizontal = pd.read_csv(
            horizontal_path,
            usecols=["MD", "Z", "TVT", "TVT_input"],
        )
        known_mask = horizontal["TVT_input"].notna().to_numpy()
        known = pd.to_numeric(
            horizontal.loc[known_mask, "TVT_input"], errors="coerce"
        ).dropna()
        if known.empty:
            continue
        typewell = pd.read_csv(typewell_path, usecols=["TVT"])
        typewell_tvt = pd.to_numeric(typewell["TVT"], errors="coerce").dropna()
        if typewell_tvt.empty:
            continue
        last_tvt = float(known.iloc[-1])
        typewell_min = float(typewell_tvt.min())
        typewell_max = float(typewell_tvt.max())
        grid_min, grid_max, grid_size, low_source, high_source = exact_grid_bounds(
            last_tvt,
            typewell_min,
            typewell_max,
        )
        known_frame = horizontal.loc[known_mask]
        tail = known_frame.tail(30)
        tail_tvt = pd.to_numeric(tail["TVT_input"], errors="coerce").to_numpy(
            np.float64
        )
        tail_z = pd.to_numeric(tail["Z"], errors="coerce").to_numpy(np.float64)
        tail_md = pd.to_numeric(tail["MD"], errors="coerce").to_numpy(np.float64)
        tail_dmd = np.diff(tail_md)
        tail_rate = (np.diff(tail_tvt) + np.diff(tail_z)) / tail_dmd
        tail_valid = np.isfinite(tail_rate) & (tail_dmd > 0)
        init_rate = (
            float(np.median(tail_rate[tail_valid]))
            if int(tail_valid.sum()) >= 3
            else 0.0
        )
        rate_span = max(RATE_SPAN_FLOOR, abs(init_rate) + 0.04)
        rate_step = 2.0 * rate_span / (N_RATES - 1)

        eval_frame = horizontal.loc[~known_mask]
        eval_tvt = pd.to_numeric(eval_frame["TVT"], errors="coerce").to_numpy(
            np.float64
        )
        eval_z = pd.to_numeric(eval_frame["Z"], errors="coerce").to_numpy(np.float64)
        eval_md = pd.to_numeric(eval_frame["MD"], errors="coerce").to_numpy(
            np.float64
        )
        last_known = known_frame.iloc[-1]
        true_rate = (
            np.diff(
                np.concatenate(
                    [
                        [float(last_known["TVT_input"]) + float(last_known["Z"])],
                        eval_tvt + eval_z,
                    ]
                )
            )
            / np.diff(
                np.concatenate([[float(last_known["MD"])], eval_md])
            )
        )
        true_rate = true_rate[np.isfinite(true_rate)]
        rate_outside_fraction = (
            float((np.abs(true_rate) > rate_span).mean()) if true_rate.size else np.nan
        )

        raw_dm = np.diff(np.concatenate([[float(last_known["MD"])], eval_md]))
        dm = np.maximum(raw_dm, 1.0)
        rate_var_cells = (SIG_R * np.sqrt(dm) / rate_step) ** 2
        lower_mean_move = -(1.0 - MOMENTUM) * (-rate_span) * dm / rate_step
        upper_mean_move = -(1.0 - MOMENTUM) * rate_span * dm / rate_step
        lost_lower = np.maximum(0.5 * (rate_var_cells - lower_mean_move), 1e-12)
        lost_upper = np.maximum(0.5 * (rate_var_cells + upper_mean_move), 1e-12)
        total_lower = lost_lower + np.maximum(
            0.5 * (rate_var_cells + lower_mean_move), 1e-12
        )
        total_upper = lost_upper + np.maximum(
            0.5 * (rate_var_cells - upper_mean_move), 1e-12
        )
        lower_scale = np.minimum(1.0, 0.9 / np.maximum(total_lower, 0.9))
        upper_scale = np.minimum(1.0, 0.9 / np.maximum(total_upper, 0.9))
        lost_lower *= lower_scale
        lost_upper *= upper_scale
        rows.append(
            {
                "well": well,
                "last_tvt_grid_ledger": last_tvt,
                "typewell_tvt_min": typewell_min,
                "typewell_tvt_max": typewell_max,
                "grid_min": grid_min,
                "grid_max": grid_max,
                "grid_size": grid_size,
                "grid_span_ft": grid_max - grid_min,
                "grid_low_source": low_source,
                "grid_high_source": high_source,
                "init_rate": init_rate,
                "rate_span": rate_span,
                "rate_step": rate_step,
                "true_rate_abs_p95": (
                    float(np.quantile(np.abs(true_rate), 0.95))
                    if true_rate.size
                    else np.nan
                ),
                "true_rate_abs_max": (
                    float(np.max(np.abs(true_rate))) if true_rate.size else np.nan
                ),
                "true_rate_outside_fraction": rate_outside_fraction,
                "raw_md_step_min": float(np.min(raw_dm)),
                "raw_md_step_median": float(np.median(raw_dm)),
                "raw_md_step_max": float(np.max(raw_dm)),
                "raw_md_step_below_one_fraction": float((raw_dm < 1.0).mean()),
                "raw_md_step_nonpositive_fraction": float((raw_dm <= 0.0).mean()),
                "rate_lower_edge_lost_mass_mean": float(np.mean(lost_lower)),
                "rate_lower_edge_lost_mass_max": float(np.max(lost_lower)),
                "rate_upper_edge_lost_mass_mean": float(np.mean(lost_upper)),
                "rate_upper_edge_lost_mass_max": float(np.max(lost_upper)),
            }
        )
    ledger = pd.DataFrame(rows).sort_values("well", kind="mergesort").reset_index(drop=True)
    if len(ledger) != 773 or ledger["well"].nunique() != 773:
        raise ValueError(f"expected 773 grid ledgers, got {len(ledger)}")
    return ledger


def row_scope(
    truth: np.ndarray,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
) -> np.ndarray:
    scope = np.full(truth.shape, "interior", dtype=object)
    scope[truth < grid_min] = "truth_below_grid"
    scope[truth > grid_max] = "truth_above_grid"
    inside = (truth >= grid_min) & (truth <= grid_max)
    scope[inside & (truth - grid_min <= EDGE_WIDTH_FT)] = "truth_lower_edge"
    scope[inside & (grid_max - truth <= EDGE_WIDTH_FT)] = "truth_upper_edge"
    return scope


def update_sum(
    target: dict[str, dict[str, float]],
    key: str,
    truth: np.ndarray,
    pred: np.ndarray,
    pred_edge: np.ndarray,
) -> None:
    item = target.setdefault(
        key,
        {
            "rows": 0.0,
            "sum_error": 0.0,
            "sum_abs_error": 0.0,
            "sum_sq_error": 0.0,
            "pred_edge_rows": 0.0,
        },
    )
    error = pred - truth
    item["rows"] += float(error.size)
    item["sum_error"] += float(error.sum())
    item["sum_abs_error"] += float(np.abs(error).sum())
    item["sum_sq_error"] += float(np.dot(error, error))
    item["pred_edge_rows"] += float(pred_edge.sum())


def finalize_sums(acc: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for scope, item in sorted(acc.items()):
        n = int(item["rows"])
        rows.append(
            {
                "scope": scope,
                "rows": n,
                "bias_ft": item["sum_error"] / n if n else np.nan,
                "mae_ft": item["sum_abs_error"] / n if n else np.nan,
                "rmse_ft": math.sqrt(item["sum_sq_error"] / n) if n else np.nan,
                "sse": item["sum_sq_error"],
                "prediction_edge_fraction": item["pred_edge_rows"] / n if n else np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    total_sse = float(frame.loc[frame["scope"] == "all", "sse"].iloc[0])
    frame["sse_fraction_of_all"] = frame["sse"] / total_sse
    return frame


def spearman(left: pd.Series, right: pd.Series) -> float | None:
    pair = pd.DataFrame({"left": left, "right": right}).dropna()
    if (
        len(pair) < 3
        or pair["left"].nunique() < 2
        or pair["right"].nunique() < 2
    ):
        return None
    value = float(pair["left"].rank().corr(pair["right"].rank()))
    return value if math.isfinite(value) else None


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    train_dir = require_file(root / "data/raw/train")
    candidates = require_file(root / args.candidates)
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)

    ledger = build_grid_ledger(train_dir)
    ledger_map = ledger.set_index("well")
    scope_acc: dict[str, dict[str, float]] = {}
    well_acc: dict[str, dict[str, float]] = {}
    usecols = [
        "well",
        "true_tvt_readout_only",
        "last_known_tvt",
        "posterior_mean",
        "posterior_std",
    ]

    total_rows = 0
    last_tvt_max_abs_diff = 0.0
    for chunk in pd.read_csv(candidates, usecols=usecols, chunksize=args.chunksize):
        chunk["well"] = chunk["well"].astype(str)
        joined = chunk.join(
            ledger_map[
                [
                    "last_tvt_grid_ledger",
                    "grid_min",
                    "grid_max",
                    "typewell_tvt_min",
                    "typewell_tvt_max",
                ]
            ],
            on="well",
            how="left",
            validate="many_to_one",
        )
        if joined[["grid_min", "grid_max"]].isna().any().any():
            raise ValueError("candidate rows missing grid ledger")

        truth = pd.to_numeric(
            joined["true_tvt_readout_only"], errors="raise"
        ).to_numpy(np.float64)
        pred = pd.to_numeric(joined["posterior_mean"], errors="raise").to_numpy(np.float64)
        std = pd.to_numeric(joined["posterior_std"], errors="raise").to_numpy(np.float64)
        grid_min = joined["grid_min"].to_numpy(np.float64)
        grid_max = joined["grid_max"].to_numpy(np.float64)
        typewell_min = joined["typewell_tvt_min"].to_numpy(np.float64)
        typewell_max = joined["typewell_tvt_max"].to_numpy(np.float64)
        saved_last = pd.to_numeric(joined["last_known_tvt"], errors="raise").to_numpy(
            np.float64
        )
        ledger_last = joined["last_tvt_grid_ledger"].to_numpy(np.float64)
        last_tvt_max_abs_diff = max(
            last_tvt_max_abs_diff,
            float(np.max(np.abs(saved_last - ledger_last))),
        )
        if not (
            np.isfinite(truth).all()
            and np.isfinite(pred).all()
            and np.isfinite(std).all()
        ):
            raise ValueError("non-finite candidate readout")

        scopes = row_scope(truth, grid_min, grid_max)
        truth_outside_typewell = (truth < typewell_min) | (truth > typewell_max)
        pred_outside_typewell = (pred < typewell_min) | (pred > typewell_max)
        pred_edge = ((pred - grid_min) <= EDGE_WIDTH_FT) | (
            (grid_max - pred) <= EDGE_WIDTH_FT
        )
        update_sum(scope_acc, "all", truth, pred, pred_edge)
        if truth_outside_typewell.any():
            update_sum(
                scope_acc,
                "truth_outside_typewell",
                truth[truth_outside_typewell],
                pred[truth_outside_typewell],
                pred_edge[truth_outside_typewell],
            )
        if pred_outside_typewell.any():
            update_sum(
                scope_acc,
                "prediction_outside_typewell",
                truth[pred_outside_typewell],
                pred[pred_outside_typewell],
                pred_edge[pred_outside_typewell],
            )
        for scope in np.unique(scopes):
            mask = scopes == scope
            update_sum(scope_acc, str(scope), truth[mask], pred[mask], pred_edge[mask])

        work = pd.DataFrame(
            {
                "well": joined["well"].to_numpy(),
                "error": pred - truth,
                "abs_error": np.abs(pred - truth),
                "sq_error": (pred - truth) ** 2,
                "outside": (truth < grid_min) | (truth > grid_max),
                "truth_outside_typewell": truth_outside_typewell,
                "pred_outside_typewell": pred_outside_typewell,
                "truth_edge": ((truth - grid_min) <= EDGE_WIDTH_FT)
                | ((grid_max - truth) <= EDGE_WIDTH_FT),
                "pred_edge": pred_edge,
                "posterior_std": std,
            }
        )
        grouped = work.groupby("well", sort=False).agg(
            rows=("error", "size"),
            sum_error=("error", "sum"),
            sum_abs_error=("abs_error", "sum"),
            sum_sq_error=("sq_error", "sum"),
            outside_rows=("outside", "sum"),
            truth_outside_typewell_rows=("truth_outside_typewell", "sum"),
            pred_outside_typewell_rows=("pred_outside_typewell", "sum"),
            truth_edge_rows=("truth_edge", "sum"),
            pred_edge_rows=("pred_edge", "sum"),
            posterior_std_sum=("posterior_std", "sum"),
        )
        for well, row in grouped.iterrows():
            item = well_acc.setdefault(
                str(well),
                {
                    "rows": 0.0,
                    "sum_error": 0.0,
                    "sum_abs_error": 0.0,
                    "sum_sq_error": 0.0,
                    "outside_rows": 0.0,
                    "truth_outside_typewell_rows": 0.0,
                    "pred_outside_typewell_rows": 0.0,
                    "truth_edge_rows": 0.0,
                    "pred_edge_rows": 0.0,
                    "posterior_std_sum": 0.0,
                },
            )
            for key in item:
                item[key] += float(row[key])
        total_rows += len(chunk)

    if total_rows != 3_783_989:
        raise ValueError(f"expected 3,783,989 candidate rows, got {total_rows}")

    well_rows = []
    for well, item in sorted(well_acc.items()):
        n = int(item["rows"])
        well_rows.append(
            {
                "well": well,
                "rows": n,
                "rmse_ft": math.sqrt(item["sum_sq_error"] / n),
                "mae_ft": item["sum_abs_error"] / n,
                "bias_ft": item["sum_error"] / n,
                "outside_grid_fraction": item["outside_rows"] / n,
                "truth_outside_typewell_fraction": (
                    item["truth_outside_typewell_rows"] / n
                ),
                "prediction_outside_typewell_fraction": (
                    item["pred_outside_typewell_rows"] / n
                ),
                "truth_edge_fraction": item["truth_edge_rows"] / n,
                "prediction_edge_fraction": item["pred_edge_rows"] / n,
                "posterior_std_mean": item["posterior_std_sum"] / n,
                "sse": item["sum_sq_error"],
            }
        )
    by_well = pd.DataFrame(well_rows).merge(ledger, on="well", validate="one_to_one")
    if len(by_well) != 773:
        raise ValueError(f"expected 773 by-well rows, got {len(by_well)}")

    scopes = finalize_sums(scope_acc)
    overall_sse = float(scopes.loc[scopes["scope"] == "all", "sse"].iloc[0])
    outside_sse = float(
        scopes.loc[
            scopes["scope"].isin(["truth_below_grid", "truth_above_grid"]), "sse"
        ].sum()
    )
    outside_rows = int(
        scopes.loc[
            scopes["scope"].isin(["truth_below_grid", "truth_above_grid"]), "rows"
        ].sum()
    )
    def optional_scope_row(scope: str) -> dict[str, float]:
        selected = scopes.loc[scopes["scope"] == scope]
        if selected.empty:
            return {"rows": 0.0, "sse": 0.0}
        row = selected.iloc[0]
        return {"rows": float(row["rows"]), "sse": float(row["sse"])}

    truth_outside_typewell_row = optional_scope_row("truth_outside_typewell")
    prediction_outside_typewell_row = optional_scope_row(
        "prediction_outside_typewell"
    )
    summary = {
        "rows": total_rows,
        "wells": int(len(by_well)),
        "step_ft": STEP_FT,
        "band_pad_ft": BAND_PAD_FT,
        "typewell_pad_ft": TYPEWELL_PAD_FT,
        "edge_width_ft": EDGE_WIDTH_FT,
        "last_tvt_max_abs_diff": last_tvt_max_abs_diff,
        "outside_grid_rows": outside_rows,
        "outside_grid_fraction": outside_rows / total_rows,
        "wells_with_any_outside_grid": int((by_well["outside_grid_fraction"] > 0).sum()),
        "outside_grid_sse_fraction": outside_sse / overall_sse,
        "truth_outside_typewell_rows": int(truth_outside_typewell_row["rows"]),
        "truth_outside_typewell_fraction": float(
            truth_outside_typewell_row["rows"] / total_rows
        ),
        "truth_outside_typewell_sse_fraction": float(
            truth_outside_typewell_row["sse"] / overall_sse
        ),
        "prediction_outside_typewell_rows": int(
            prediction_outside_typewell_row["rows"]
        ),
        "prediction_outside_typewell_fraction": float(
            prediction_outside_typewell_row["rows"] / total_rows
        ),
        "prediction_outside_typewell_sse_fraction": float(
            prediction_outside_typewell_row["sse"] / overall_sse
        ),
        "wells_with_prediction_edge_fraction_ge_0p10": int(
            (by_well["prediction_edge_fraction"] >= 0.10).sum()
        ),
        "spearman_outside_fraction_vs_rmse": spearman(
            by_well["outside_grid_fraction"], by_well["rmse_ft"]
        ),
        "spearman_prediction_edge_fraction_vs_rmse": spearman(
            by_well["prediction_edge_fraction"], by_well["rmse_ft"]
        ),
        "spearman_prediction_edge_fraction_vs_abs_bias": spearman(
            by_well["prediction_edge_fraction"], by_well["bias_ft"].abs()
        ),
        "wells_with_true_rate_outside_fraction_ge_0p01": int(
            (by_well["true_rate_outside_fraction"] >= 0.01).sum()
        ),
        "true_rate_outside_fraction_median": float(
            by_well["true_rate_outside_fraction"].median()
        ),
        "true_rate_outside_fraction_p95": float(
            by_well["true_rate_outside_fraction"].quantile(0.95)
        ),
        "spearman_true_rate_outside_fraction_vs_rmse": spearman(
            by_well["true_rate_outside_fraction"], by_well["rmse_ft"]
        ),
        "wells_with_any_raw_md_step_below_one": int(
            (by_well["raw_md_step_below_one_fraction"] > 0.0).sum()
        ),
        "wells_with_any_raw_md_step_nonpositive": int(
            (by_well["raw_md_step_nonpositive_fraction"] > 0.0).sum()
        ),
        "raw_md_step_median_of_well_medians": float(
            by_well["raw_md_step_median"].median()
        ),
        "rate_lower_edge_lost_mass_mean_median": float(
            by_well["rate_lower_edge_lost_mass_mean"].median()
        ),
        "rate_upper_edge_lost_mass_mean_median": float(
            by_well["rate_upper_edge_lost_mass_mean"].median()
        ),
        "grid_low_source_counts": {
            str(k): int(v) for k, v in ledger["grid_low_source"].value_counts().items()
        },
        "grid_high_source_counts": {
            str(k): int(v) for k, v in ledger["grid_high_source"].value_counts().items()
        },
    }

    ledger.to_csv(output / "grid_ledger.csv", index=False)
    scopes.to_csv(output / "row_scope_metrics.csv", index=False)
    by_well.sort_values("rmse_ft", ascending=False).to_csv(
        output / "by_well_metrics.csv", index=False
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
