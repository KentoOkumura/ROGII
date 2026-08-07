from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "experiments/exp300_exp264_vs_exp274_well_selector_readout"
EXP264_BY_WELL = ROOT / (
    "experiments/exp264_exp263_candidate_confidence_dual_selector/"
    "kaggle/output/stage_d_v3_corrected/artifacts/stage_d_by_well.csv"
)
EXP274_BY_WELL = Path(
    EXPERIMENT_DIR / "artifacts/source_inputs/"
    "exp274_catboost_final_regressor_swap_on_exp238_by_well.csv"
)
RAW_TRAIN = ROOT / "data/raw/train"
OUT_DIR = EXPERIMENT_DIR / "artifacts"
HORIZONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def safe_stat(values: np.ndarray, fn) -> float:
    finite = values[np.isfinite(values)]
    return float(fn(finite)) if finite.size else float("nan")


def linear_slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    if ok.sum() < 2:
        return float("nan")
    x = np.linspace(0.0, 1.0, len(values), dtype=float)[ok]
    return float(np.polyfit(x, values[ok], 1)[0])


def lag1_corr(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return float("nan")
    a, b = values[:-1], values[1:]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def summarize_typewell(path: Path) -> dict[str, float]:
    tw = pd.read_csv(path)
    tvt = pd.to_numeric(tw["TVT"], errors="coerce").to_numpy(float)
    gr = pd.to_numeric(tw["GR"], errors="coerce").to_numpy(float)
    geology = tw["Geology"].astype("string").str.strip()
    geology_valid = geology.notna() & geology.ne("")
    gr_diff = np.diff(gr)
    return {
        "typewell_rows": float(len(tw)),
        "typewell_tvt_span": safe_stat(tvt, np.max) - safe_stat(tvt, np.min),
        "typewell_gr_mean": safe_stat(gr, np.mean),
        "typewell_gr_std": safe_stat(gr, np.std),
        "typewell_gr_iqr": safe_stat(gr, lambda x: np.quantile(x, 0.75) - np.quantile(x, 0.25)),
        "typewell_gr_absdiff_mean": safe_stat(np.abs(gr_diff), np.mean),
        "typewell_gr_missing_frac": float(np.mean(~np.isfinite(gr))),
        "typewell_geology_labeled_frac": float(geology_valid.mean()),
        "typewell_geology_nunique": float(geology[geology_valid].nunique()),
    }


def summarize_well(well: str, expected_tail_rows: int) -> dict[str, float | str]:
    path = RAW_TRAIN / f"{well}__horizontal_well.csv"
    df = pd.read_csv(path)
    df = df.sort_values("MD", kind="stable").reset_index(drop=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    known = df["TVT_input"].notna().to_numpy()
    if known.any():
        last_known_idx = int(np.flatnonzero(known)[-1])
        tail_start = last_known_idx + 1
    else:
        last_known_idx = -1
        tail_start = 0
    tail = df.iloc[tail_start:].copy()
    prefix = df.iloc[:tail_start].copy()
    if len(tail) != expected_tail_rows:
        raise ValueError(
            f"{well}: raw tail rows {len(tail)} != OOF rows {expected_tail_rows}"
        )

    md = df["MD"].to_numpy(float)
    tx = tail["X"].to_numpy(float)
    ty = tail["Y"].to_numpy(float)
    tz = tail["Z"].to_numpy(float)
    tmd = tail["MD"].to_numpy(float)
    tgr = tail["GR"].to_numpy(float)
    pgr = prefix["GR"].to_numpy(float)
    ttvt = tail["TVT"].to_numpy(float)

    dx_step = np.diff(tx)
    dy_step = np.diff(ty)
    dz_step = np.diff(tz)
    dmd_step = np.diff(tmd)
    horizontal_step = np.hypot(dx_step, dy_step)
    path3d_step = np.sqrt(dx_step**2 + dy_step**2 + dz_step**2)
    gr_diff = np.diff(tgr)
    tvt_diff = np.diff(ttvt)

    tail_md_span = float(tmd[-1] - tmd[0]) if len(tmd) > 1 else 0.0
    dx = float(tx[-1] - tx[0]) if len(tx) > 1 else 0.0
    dy = float(ty[-1] - ty[0]) if len(ty) > 1 else 0.0
    dz = float(tz[-1] - tz[0]) if len(tz) > 1 else 0.0
    horizontal_disp = float(np.hypot(dx, dy))
    path_horizontal = safe_stat(horizontal_step, np.sum)
    path3d = safe_stat(path3d_step, np.sum)

    # Heading change is a target-free trajectory complexity proxy.
    headings = np.unwrap(np.arctan2(dy_step, dx_step))
    heading_change = np.abs(np.diff(headings))

    # Formation geometry is available on hidden test. Distances are in the same
    # signed vertical coordinate system as Z and the six formation surfaces.
    form = tail[HORIZONS].to_numpy(float)
    zmat = tz[:, None]
    form_dist = zmat - form
    nearest_abs = np.nanmin(np.abs(form_dist), axis=1)
    sign_changes = 0
    for j in range(form_dist.shape[1]):
        vals = form_dist[:, j]
        ok = np.isfinite(vals[:-1]) & np.isfinite(vals[1:])
        sign_changes += int(np.sum((vals[:-1][ok] * vals[1:][ok]) <= 0))

    if last_known_idx >= 0:
        last_known_tvt = float(df.loc[last_known_idx, "TVT_input"])
        prefix_tvt = prefix["TVT"].to_numpy(float)
        recent = prefix_tvt[-min(100, len(prefix_tvt)) :]
        prefix_slope_per_row = safe_stat(np.diff(recent), np.median)
    else:
        last_known_tvt = float("nan")
        prefix_slope_per_row = float("nan")

    if np.isfinite(last_known_tvt) and np.isfinite(prefix_slope_per_row):
        extrap = last_known_tvt + prefix_slope_per_row * np.arange(1, len(ttvt) + 1)
        linear_extrap_rmse = float(np.sqrt(np.mean((ttvt - extrap) ** 2)))
    else:
        linear_extrap_rmse = float("nan")

    row: dict[str, float | str] = {
        "well": well,
        "total_rows": float(len(df)),
        "prefix_rows": float(len(prefix)),
        "tail_rows_raw": float(len(tail)),
        "tail_fraction": float(len(tail) / len(df)),
        "prefix_to_tail_ratio": float(len(prefix) / len(tail)),
        "md_span_total": float(md[-1] - md[0]) if len(md) > 1 else 0.0,
        "tail_md_span": tail_md_span,
        "tail_start_x": float(tx[0]),
        "tail_start_y": float(ty[0]),
        "tail_start_z": float(tz[0]),
        "tail_end_x": float(tx[-1]),
        "tail_end_y": float(ty[-1]),
        "tail_end_z": float(tz[-1]),
        "tail_dx": dx,
        "tail_dy": dy,
        "tail_dz": dz,
        "tail_horizontal_disp": horizontal_disp,
        "tail_path_horizontal": path_horizontal,
        "tail_path_3d": path3d,
        "tail_horizontal_straightness": horizontal_disp / path_horizontal if path_horizontal > 0 else np.nan,
        "tail_vertical_fraction": abs(dz) / path3d if path3d > 0 else np.nan,
        "tail_abs_dx_per_md": abs(dx) / tail_md_span if tail_md_span > 0 else np.nan,
        "tail_abs_dy_per_md": abs(dy) / tail_md_span if tail_md_span > 0 else np.nan,
        "tail_abs_dz_per_md": abs(dz) / tail_md_span if tail_md_span > 0 else np.nan,
        "tail_z_range": safe_stat(tz, np.max) - safe_stat(tz, np.min),
        "tail_z_step_std": safe_stat(dz_step, np.std),
        "tail_heading_change_mean": safe_stat(heading_change, np.mean),
        "tail_heading_change_p90": safe_stat(heading_change, lambda x: np.quantile(x, 0.90)),
        "tail_gr_mean": safe_stat(tgr, np.mean),
        "tail_gr_std": safe_stat(tgr, np.std),
        "tail_gr_iqr": safe_stat(tgr, lambda x: np.quantile(x, 0.75) - np.quantile(x, 0.25)),
        "tail_gr_p90_p10": safe_stat(tgr, lambda x: np.quantile(x, 0.90) - np.quantile(x, 0.10)),
        "tail_gr_absdiff_mean": safe_stat(np.abs(gr_diff), np.mean),
        "tail_gr_absdiff_p90": safe_stat(np.abs(gr_diff), lambda x: np.quantile(x, 0.90)),
        "tail_gr_missing_frac": float(np.mean(~np.isfinite(tgr))),
        "tail_gr_lag1_corr": lag1_corr(tgr),
        "tail_gr_linear_change": linear_slope(tgr),
        "prefix_gr_mean": safe_stat(pgr, np.mean),
        "prefix_gr_std": safe_stat(pgr, np.std),
        "gr_tail_minus_prefix_mean": safe_stat(tgr, np.mean) - safe_stat(pgr, np.mean),
        "nearest_formation_absdist_start": float(nearest_abs[0]),
        "nearest_formation_absdist_end": float(nearest_abs[-1]),
        "nearest_formation_absdist_mean": safe_stat(nearest_abs, np.mean),
        "nearest_formation_absdist_min": safe_stat(nearest_abs, np.min),
        "formation_crossing_count": float(sign_changes),
        # Target-derived descriptors below are diagnostic/oracle only.
        "oracle_last_known_tvt": last_known_tvt,
        "oracle_tail_tvt_delta_end": float(ttvt[-1] - last_known_tvt),
        "oracle_tail_tvt_abs_delta_end": abs(float(ttvt[-1] - last_known_tvt)),
        "oracle_tail_tvt_range": safe_stat(ttvt, np.max) - safe_stat(ttvt, np.min),
        "oracle_tail_tvt_step_std": safe_stat(tvt_diff, np.std),
        "oracle_tail_tvt_abs_step_mean": safe_stat(np.abs(tvt_diff), np.mean),
        "oracle_tail_tvt_curvature_mean": safe_stat(np.abs(np.diff(ttvt, n=2)), np.mean),
        "oracle_prefix_linear_extrap_rmse": linear_extrap_rmse,
    }
    row.update(summarize_typewell(RAW_TRAIN / f"{well}__typewell.csv"))
    return row


def bh_qvalues(pvalues: pd.Series) -> pd.Series:
    p = pvalues.to_numpy(float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adjusted, 0, 1)
    return pd.Series(out, index=pvalues.index)


def feature_readout(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    group = df["worse_vs_exp274"].astype(bool)
    delta = df["delta_exp264_vs_exp274"]
    for col in feature_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        a = values[group].dropna().to_numpy(float)
        b = values[~group].dropna().to_numpy(float)
        if len(a) < 5 or len(b) < 5 or np.nanstd(values) == 0:
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        cliff = 2 * float(u) / (len(a) * len(b)) - 1
        pooled = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))
        smd = (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else np.nan
        ok = values.notna() & delta.notna()
        rho, rho_p = spearmanr(values[ok], delta[ok])
        rows.append(
            {
                "feature": col,
                "n_worse": len(a),
                "n_better": len(b),
                "worse_median": float(np.median(a)),
                "better_median": float(np.median(b)),
                "worse_mean": float(np.mean(a)),
                "better_mean": float(np.mean(b)),
                "cliffs_delta": cliff,
                "standardized_mean_diff": smd,
                "mannwhitney_p": float(p),
                "spearman_delta": float(rho),
                "spearman_p": float(rho_p),
            }
        )
    out = pd.DataFrame(rows)
    out["mannwhitney_q"] = bh_qvalues(out["mannwhitney_p"])
    out["spearman_q"] = bh_qvalues(out["spearman_p"])
    return out.sort_values(["spearman_q", "spearman_delta"], key=lambda s: np.abs(s) if s.name == "spearman_delta" else s)


def quartile_readout(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    work = df[[feature, "worse_vs_exp274", "delta_exp264_vs_exp274", "rows"]].dropna().copy()
    work["quartile"] = pd.qcut(work[feature], 4, duplicates="drop")
    return (
        work.groupby("quartile", observed=True)
        .agg(
            wells=("worse_vs_exp274", "size"),
            worsened_rate=("worse_vs_exp274", "mean"),
            median_delta=("delta_exp264_vs_exp274", "median"),
            mean_delta=("delta_exp264_vs_exp274", "mean"),
            rows=("rows", "sum"),
        )
        .reset_index()
    )


def weighted_pooled_rmse(frame: pd.DataFrame, col: str) -> float:
    return float(np.sqrt(np.average(frame[col] ** 2, weights=frame["rows"])))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(EXP264_BY_WELL).rename(
        columns={
            "matched_control_rmse": "exp264_clean_control_rmse",
            "selector_compact_addonly_rmse": "exp264_rmse",
            "delta_rmse_addonly_minus_control": "delta_exp264_vs_clean_control",
        }
    )
    b = pd.read_csv(EXP274_BY_WELL).rename(
        columns={
            "parent_rmse": "exp238_parent_rmse",
            "catboost_rmse": "exp274_rmse",
            "blend_rmse": "exp274_blend_rmse",
        }
    )
    merged = a.merge(b, on="well", suffixes=("", "_exp274"), validate="one_to_one")
    if len(merged) != 773:
        raise ValueError(f"Expected 773 wells, got {len(merged)}")
    if not np.array_equal(merged["rows"].to_numpy(), merged["rows_exp274"].to_numpy()):
        raise ValueError("OOF row counts differ between exp264 and exp274")
    merged = merged.drop(columns=["rows_exp274"])
    merged["delta_exp264_vs_exp274"] = merged["exp264_rmse"] - merged["exp274_rmse"]
    merged["delta_exp264_vs_exp238"] = merged["exp264_rmse"] - merged["exp238_parent_rmse"]
    merged["worse_vs_exp274"] = merged["delta_exp264_vs_exp274"] > 0
    merged["worse_vs_exp238"] = merged["delta_exp264_vs_exp238"] > 0

    raw_rows = []
    for i, rec in enumerate(merged[["well", "rows"]].itertuples(index=False), start=1):
        raw_rows.append(summarize_well(str(rec.well), int(rec.rows)))
        if i % 100 == 0:
            print(f"summarized {i}/773", flush=True)
    raw = pd.DataFrame(raw_rows)
    full = merged.merge(raw, on="well", validate="one_to_one")

    metric_cols = {
        "rows",
        "exp264_clean_control_rmse",
        "exp264_rmse",
        "delta_exp264_vs_clean_control",
        "exp238_parent_rmse",
        "exp274_rmse",
        "exp274_blend_rmse",
        "catboost_delta_rmse",
        "blend_delta_rmse",
        "delta_exp264_vs_exp274",
        "delta_exp264_vs_exp238",
        "worse_vs_exp274",
        "worse_vs_exp238",
    }
    feature_cols = [
        c for c in full.columns if c not in metric_cols | {"well"} and pd.api.types.is_numeric_dtype(full[c])
    ]
    deployable_cols = [c for c in feature_cols if not c.startswith("oracle_")]
    oracle_cols = [c for c in feature_cols if c.startswith("oracle_")]
    deploy_readout = feature_readout(full, deployable_cols)
    oracle_readout = feature_readout(full, oracle_cols)

    full.sort_values("delta_exp264_vs_exp274", ascending=False).to_csv(
        OUT_DIR / "well_comparison_and_features.csv", index=False
    )
    ordered = full.sort_values("delta_exp264_vs_exp274", ascending=False).copy()
    selected_columns = [
        "well",
        "rows",
        "exp238_parent_rmse",
        "exp274_rmse",
        "exp264_rmse",
        "delta_exp264_vs_exp274",
        "delta_exp264_vs_exp238",
        "tail_rows_raw",
        "tail_fraction",
        "tail_dx",
        "tail_dy",
        "tail_dz",
        "tail_gr_mean",
        "tail_gr_std",
        "tail_gr_absdiff_mean",
        "tail_gr_missing_frac",
        "nearest_formation_absdist_mean",
        "typewell_gr_std",
        "oracle_tail_tvt_range",
        "oracle_tail_tvt_abs_delta_end",
    ]
    ordered.loc[ordered["delta_exp264_vs_exp274"] > 0, selected_columns].to_csv(
        OUT_DIR / "worsened_vs_exp274_wells.csv", index=False
    )
    ordered.loc[ordered["delta_exp264_vs_exp274"] > 3, selected_columns].to_csv(
        OUT_DIR / "material_worsened_gt3_vs_exp274_wells.csv", index=False
    )
    ordered["mse_delta_contribution"] = (
        ordered["rows"]
        * (ordered["exp264_rmse"] ** 2 - ordered["exp274_rmse"] ** 2)
        / ordered["rows"].sum()
    )
    ordered["positive_sse_contribution"] = ordered["mse_delta_contribution"].clip(
        lower=0
    )
    positive_total = ordered["positive_sse_contribution"].sum()
    ordered["positive_sse_share"] = (
        ordered["positive_sse_contribution"] / positive_total
    )
    ordered["positive_sse_cumulative_share"] = ordered["positive_sse_share"].cumsum()
    ordered.to_csv(OUT_DIR / "well_sse_contributions.csv", index=False)
    deploy_readout.to_csv(OUT_DIR / "deployable_feature_readout.csv", index=False)
    oracle_readout.to_csv(OUT_DIR / "oracle_feature_readout.csv", index=False)

    key_quartiles = [
        "tail_rows_raw",
        "tail_fraction",
        "tail_horizontal_disp",
        "tail_abs_dz_per_md",
        "tail_gr_std",
        "tail_gr_absdiff_mean",
        "tail_gr_missing_frac",
        "nearest_formation_absdist_mean",
        "typewell_gr_std",
        "oracle_tail_tvt_abs_delta_end",
        "oracle_prefix_linear_extrap_rmse",
        "exp274_rmse",
    ]
    quartiles = pd.concat(
        [quartile_readout(full, c).assign(feature=c) for c in key_quartiles],
        ignore_index=True,
    )
    quartiles.to_csv(OUT_DIR / "quartile_readout.csv", index=False)

    summary_lines = []
    for label, col in [
        ("exp238 parent", "exp238_parent_rmse"),
        ("exp274", "exp274_rmse"),
        ("exp264 clean control", "exp264_clean_control_rmse"),
        ("exp264", "exp264_rmse"),
    ]:
        summary_lines.append(f"pooled {label}: {weighted_pooled_rmse(full, col):.9f}")
    d = full["delta_exp264_vs_exp274"]
    summary_lines.extend(
        [
            f"worse vs exp274: {(d > 0).sum()}/773",
            f"worse >0.25: {(d > 0.25).sum()}",
            f"worse >1: {(d > 1).sum()}",
            f"worse >3: {(d > 3).sum()}",
            f"worse >5: {(d > 5).sum()}",
            f"improved: {(d < 0).sum()}/773",
            f"median well delta: {d.median():.6f}",
            f"row-weighted delta of pooled RMSE: {weighted_pooled_rmse(full, 'exp264_rmse') - weighted_pooled_rmse(full, 'exp274_rmse'):.6f}",
        ]
    )
    (OUT_DIR / "summary.txt").write_text("\n".join(summary_lines) + "\n")

    print("\n".join(summary_lines))
    print("\nTOP WORSE VS EXP274")
    print(
        full.sort_values("delta_exp264_vs_exp274", ascending=False)[
            [
                "well",
                "rows",
                "exp274_rmse",
                "exp264_rmse",
                "delta_exp264_vs_exp274",
                "exp238_parent_rmse",
                "delta_exp264_vs_exp238",
            ]
        ].head(25).to_string(index=False)
    )
    print("\nDEPLOYABLE FEATURES: strongest continuous associations")
    print(
        deploy_readout.assign(abs_rho=deploy_readout["spearman_delta"].abs())
        .sort_values(["spearman_q", "abs_rho"], ascending=[True, False])
        .head(30)
        .drop(columns="abs_rho")
        .to_string(index=False)
    )
    print("\nORACLE FEATURES")
    print(
        oracle_readout.assign(abs_rho=oracle_readout["spearman_delta"].abs())
        .sort_values(["spearman_q", "abs_rho"], ascending=[True, False])
        .head(20)
        .drop(columns="abs_rho")
        .to_string(index=False)
    )
    print("\nQUARTILES")
    print(quartiles.to_string(index=False))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        main()
