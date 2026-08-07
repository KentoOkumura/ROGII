from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "experiments/exp300_exp264_vs_exp274_well_selector_readout"
IN = EXPERIMENT_DIR / "artifacts/well_comparison_and_features.csv"
OUT = EXPERIMENT_DIR / "artifacts"


METRIC_COLS = {
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


def bh(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    ranked = p[order]
    qrank = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    q = np.empty_like(qrank)
    q[order] = np.clip(qrank, 0, 1)
    return q


def readout(df: pd.DataFrame, features: list[str], threshold: float) -> pd.DataFrame:
    positive = df["delta_exp264_vs_exp274"] > threshold
    rows = []
    for col in features:
        x = pd.to_numeric(df[col], errors="coerce")
        a = x[positive].dropna().to_numpy(float)
        b = x[~positive].dropna().to_numpy(float)
        if len(a) < 5 or len(b) < 5 or np.nanstd(x) == 0:
            continue
        u, p = mannwhitneyu(a, b)
        pooled = np.sqrt(((len(a)-1)*np.var(a, ddof=1)+(len(b)-1)*np.var(b, ddof=1))/(len(a)+len(b)-2))
        ok = x.notna()
        rho, rp = spearmanr(x[ok], df.loc[ok, "delta_exp264_vs_exp274"])
        rows.append({
            "feature": col,
            "threshold": threshold,
            "n_severe": len(a),
            "n_other": len(b),
            "severe_median": np.median(a),
            "other_median": np.median(b),
            "median_ratio": np.median(a) / np.median(b) if np.median(b) != 0 else np.nan,
            "cliffs_delta": 2*u/(len(a)*len(b))-1,
            "standardized_mean_diff": (np.mean(a)-np.mean(b))/pooled if pooled > 0 else np.nan,
            "p": p,
            "spearman_delta": rho,
            "spearman_p": rp,
        })
    out = pd.DataFrame(rows)
    out["q"] = bh(out["p"].to_numpy())
    out["spearman_q"] = bh(out["spearman_p"].to_numpy())
    return out.sort_values(["q", "cliffs_delta"], key=lambda s: s.abs() if s.name == "cliffs_delta" else s)


def cross_validated_auc(df: pd.DataFrame, features: list[str], threshold: float) -> tuple[float, float]:
    X = df[features]
    y = (df["delta_exp264_vs_exp274"] > threshold).astype(int)
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        RobustScaler(),
        LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced"),
    )
    aucs = []
    for seed in range(5):
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed)
        pred = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, pred))
    return float(np.mean(aucs)), float(np.std(aucs))


def main() -> None:
    df = pd.read_csv(IN)
    numeric = [c for c in df.columns if c not in METRIC_COLS | {"well"} and pd.api.types.is_numeric_dtype(df[c])]
    deployable = [c for c in numeric if not c.startswith("oracle_")]
    oracle = [c for c in numeric if c.startswith("oracle_")]

    all_tables = []
    for threshold in [0.0, 1.0, 3.0, 5.0]:
        dep = readout(df, deployable, threshold).assign(kind="deployable")
        ora = readout(df, oracle, threshold).assign(kind="oracle")
        all_tables.extend([dep, ora])
        print(f"\n=== delta > {threshold:g} ft: {(df.delta_exp264_vs_exp274 > threshold).sum()} wells ===")
        print("deployable")
        print(dep.head(18).to_string(index=False))
        print("oracle")
        print(ora.head(10).to_string(index=False))

    combined = pd.concat(all_tables, ignore_index=True)
    combined.to_csv(OUT / "threshold_feature_readout.csv", index=False)

    # Spatial clusters are diagnostic summaries, not fold-safe router proposals.
    coords = df[["tail_start_x", "tail_start_y"]].copy()
    scaled = RobustScaler().fit_transform(coords)
    df["spatial_cluster_8"] = KMeans(n_clusters=8, random_state=42, n_init=50).fit_predict(scaled)
    cluster = (
        df.groupby("spatial_cluster_8")
        .agg(
            wells=("well", "size"),
            x_median=("tail_start_x", "median"),
            y_median=("tail_start_y", "median"),
            worse_rate=("worse_vs_exp274", "mean"),
            severe_1_rate=("delta_exp264_vs_exp274", lambda x: np.mean(x > 1)),
            severe_3_rate=("delta_exp264_vs_exp274", lambda x: np.mean(x > 3)),
            median_delta=("delta_exp264_vs_exp274", "median"),
            mean_delta=("delta_exp264_vs_exp274", "mean"),
        )
        .sort_values("mean_delta", ascending=False)
    )
    cluster.to_csv(OUT / "spatial_cluster_readout.csv")
    print("\nSPATIAL CLUSTERS")
    print(cluster.to_string())

    compact_features = [
        "tail_rows_raw",
        "tail_fraction",
        "tail_start_x",
        "tail_start_y",
        "tail_start_z",
        "tail_dx",
        "tail_dy",
        "tail_dz",
        "tail_horizontal_straightness",
        "tail_vertical_fraction",
        "tail_z_step_std",
        "tail_heading_change_mean",
        "tail_gr_mean",
        "tail_gr_std",
        "tail_gr_absdiff_mean",
        "tail_gr_missing_frac",
        "tail_gr_lag1_corr",
        "gr_tail_minus_prefix_mean",
        "nearest_formation_absdist_start",
        "nearest_formation_absdist_end",
        "nearest_formation_absdist_min",
        "formation_crossing_count",
        "typewell_rows",
        "typewell_gr_mean",
        "typewell_gr_std",
        "typewell_gr_absdiff_mean",
        "typewell_geology_labeled_frac",
    ]
    print("\nCROSS-VALIDATED TARGET-FREE LOGISTIC AUC")
    auc_rows = []
    for threshold in [0.0, 1.0, 3.0, 5.0]:
        mean_auc, std_auc = cross_validated_auc(df, compact_features, threshold)
        auc_rows.append({"threshold": threshold, "positive_wells": int((df.delta_exp264_vs_exp274 > threshold).sum()), "mean_auc": mean_auc, "std_auc": std_auc})
        print(threshold, mean_auc, std_auc)
    pd.DataFrame(auc_rows).to_csv(OUT / "target_free_logistic_auc.csv", index=False)

    # Top severe wells with the most interpretable target-free and oracle descriptors.
    cols = [
        "well", "rows", "exp238_parent_rmse", "exp274_rmse", "exp264_rmse", "delta_exp264_vs_exp274",
        "tail_fraction", "tail_dx", "tail_dy", "tail_dz", "tail_abs_dz_per_md",
        "tail_gr_mean", "tail_gr_std", "tail_gr_absdiff_mean", "tail_gr_missing_frac",
        "nearest_formation_absdist_mean", "nearest_formation_absdist_min",
        "typewell_gr_std", "oracle_tail_tvt_abs_delta_end", "oracle_prefix_linear_extrap_rmse",
    ]
    top = df.sort_values("delta_exp264_vs_exp274", ascending=False)[cols].head(40)
    top.to_csv(OUT / "top40_worse_characteristics.csv", index=False)
    print("\nTOP 40 CHARACTERISTICS")
    print(top.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
