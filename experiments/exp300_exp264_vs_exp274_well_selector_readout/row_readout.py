from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "experiments/exp300_exp264_vs_exp274_well_selector_readout"
OOF274 = EXPERIMENT_DIR / "artifacts/source_inputs/exp274_catboost_final_regressor_swap_on_exp238_oof_predictions.csv.gz"
OOF264 = ROOT / "experiments/exp264_exp263_candidate_confidence_dual_selector/artifacts/exp264_exp263_candidate_confidence_dual_selector_stage_d_v3_oof_viewer.csv"
WELL_FEATURES = EXPERIMENT_DIR / "artifacts/well_comparison_and_features.csv"
RAW = ROOT / "data/raw/train"
OUT = EXPERIMENT_DIR / "artifacts"
CHUNK = 200_000


class MetricStore:
    def __init__(self) -> None:
        self.data: dict[str, dict[int, dict[str, object]]] = defaultdict(dict)

    def update(
        self,
        name: str,
        categories: np.ndarray,
        well_codes: np.ndarray,
        err274: np.ndarray,
        err264: np.ndarray,
        pred_diff: np.ndarray,
    ) -> None:
        for category in np.unique(categories):
            mask = categories == category
            cat = int(category)
            if cat not in self.data[name]:
                self.data[name][cat] = {
                    "rows": 0,
                    "wells": set(),
                    "sse274": 0.0,
                    "sse264": 0.0,
                    "pred_diff_sum": 0.0,
                    "pred_diff_sq_sum": 0.0,
                }
            rec = self.data[name][cat]
            rec["rows"] += int(mask.sum())
            rec["wells"].update(np.unique(well_codes[mask]).tolist())
            rec["sse274"] += float(np.sum(err274[mask].astype(np.float64) ** 2))
            rec["sse264"] += float(np.sum(err264[mask].astype(np.float64) ** 2))
            rec["pred_diff_sum"] += float(np.sum(pred_diff[mask], dtype=np.float64))
            rec["pred_diff_sq_sum"] += float(np.sum(pred_diff[mask].astype(np.float64) ** 2))

    def frame(self, name: str, labels: dict[int, str] | None = None) -> pd.DataFrame:
        rows = []
        for category, rec in sorted(self.data[name].items()):
            n = int(rec["rows"])
            r274 = np.sqrt(float(rec["sse274"]) / n)
            r264 = np.sqrt(float(rec["sse264"]) / n)
            rows.append(
                {
                    "category": labels.get(category, str(category)) if labels else category,
                    "rows": n,
                    "wells": len(rec["wells"]),
                    "exp274_rmse": r274,
                    "exp264_rmse": r264,
                    "delta_rmse": r264 - r274,
                    "mse_delta": r264**2 - r274**2,
                    "mean_pred_shift_264_minus_274": float(rec["pred_diff_sum"]) / n,
                    "pair_rmse_264_274": np.sqrt(float(rec["pred_diff_sq_sum"]) / n),
                }
            )
        return pd.DataFrame(rows)


def auc(df: pd.DataFrame, features: list[str], threshold: float) -> float:
    y = (df["delta_exp264_vs_exp274"] > threshold).astype(int)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RobustScaler(),
        LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced"),
    )
    pred = cross_val_predict(
        model,
        df[features],
        y,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        method="predict_proba",
    )[:, 1]
    return float(roc_auc_score(y, pred))


def main() -> None:
    well = pd.read_csv(WELL_FEATURES)
    well_names = well["well"].astype(str).tolist()
    well_to_code = {name: i for i, name in enumerate(well_names)}
    delta_map = dict(zip(well_names, well["delta_exp264_vs_exp274"], strict=True))
    n_wells = len(well)

    print("loading raw GR arrays", flush=True)
    tail_start: dict[str, int] = {}
    gr_arrays: dict[str, np.ndarray] = {}
    for i, name in enumerate(well_names, start=1):
        raw = pd.read_csv(
            RAW / f"{name}__horizontal_well.csv",
            usecols=["GR", "TVT_input"],
            dtype={"GR": "float32", "TVT_input": "float32"},
        )
        missing_input = raw["TVT_input"].isna().to_numpy()
        start = int(np.flatnonzero(missing_input)[0])
        tail_start[name] = start
        gr_arrays[name] = raw["GR"].to_numpy(np.float32)
        if i % 200 == 0:
            print(f"raw {i}/773", flush=True)

    stats = MetricStore()
    # Per-well prediction disagreement sufficient statistics.
    n = np.zeros(n_wells, np.int64)
    sum_d274 = np.zeros(n_wells, np.float64)
    sumsq_d274 = np.zeros(n_wells, np.float64)
    sumabs_d274 = np.zeros(n_wells, np.float64)
    maxabs_d274 = np.zeros(n_wells, np.float64)
    last_d274 = np.full(n_wells, np.nan, np.float64)
    sum_t = np.zeros(n_wells, np.float64)
    sum_t2 = np.zeros(n_wells, np.float64)
    sum_td274 = np.zeros(n_wells, np.float64)
    sumsq_d238 = np.zeros(n_wells, np.float64)
    sumabs_d238 = np.zeros(n_wells, np.float64)
    maxabs_d238 = np.zeros(n_wells, np.float64)
    min_p264 = np.full(n_wells, np.inf, np.float64)
    max_p264 = np.full(n_wells, -np.inf, np.float64)
    min_p274 = np.full(n_wells, np.inf, np.float64)
    max_p274 = np.full(n_wells, -np.inf, np.float64)

    iter274 = pd.read_csv(
        OOF274,
        usecols=["id", "well", "outer_fold", "last_known_tvt", "target_tvt", "exp238_lgb_mean_tvt", "catboost_public_cb0_tvt"],
        dtype={"id": "string", "well": "string", "outer_fold": "int8", "last_known_tvt": "float32", "target_tvt": "float32", "exp238_lgb_mean_tvt": "float32", "catboost_public_cb0_tvt": "float32"},
        chunksize=CHUNK,
    )
    iter264 = pd.read_csv(OOF264, dtype={"id": "string", "tvt": "float32"}, chunksize=CHUNK)
    total = 0
    for chunk_no, (a, b) in enumerate(zip(iter274, iter264, strict=True), start=1):
        if len(a) != len(b) or not a["id"].reset_index(drop=True).equals(b["id"].reset_index(drop=True)):
            raise ValueError(f"OOF ID mismatch in chunk {chunk_no}")
        names = a["well"].astype(str)
        codes = names.map(well_to_code).to_numpy(np.int16)
        suffix = a["id"].str.slice(9).astype("int32").to_numpy()
        starts = names.map(tail_start).to_numpy(np.int32)
        distance = suffix - starts
        rows_for_well = codes.copy().astype(np.int32)
        rows_for_well[:] = well["rows"].to_numpy(np.int32)[codes]
        relative = distance / np.maximum(rows_for_well - 1, 1)

        gr = np.empty(len(a), np.float32)
        for name in names.unique():
            mask = names.to_numpy() == name
            gr[mask] = gr_arrays[name][suffix[mask]]

        target = a["target_tvt"].to_numpy(np.float32)
        p274 = a["catboost_public_cb0_tvt"].to_numpy(np.float32)
        p238 = a["exp238_lgb_mean_tvt"].to_numpy(np.float32)
        p264 = b["tvt"].to_numpy(np.float32)
        d274 = p264 - p274
        d238 = p264 - p238
        e274 = p274 - target
        e264 = p264 - target
        residual_abs = np.abs(target - a["last_known_tvt"].to_numpy(np.float32))
        deltas = names.map(delta_map).to_numpy(np.float32)

        distance_cat = np.digitize(distance, [50, 100, 250, 500, 1000], right=False)
        relative_cat = np.minimum((relative * 10).astype(np.int8), 9)
        residual_cat = np.digitize(residual_abs, [2, 5, 10, 20, 50], right=True)
        well_group = np.select([deltas <= 0, deltas <= 1, deltas <= 3, deltas <= 5], [0, 1, 2, 3], default=4).astype(np.int8)

        stats.update("overall", np.zeros(len(a), np.int8), codes, e274, e264, d274)
        stats.update("outer_fold", a["outer_fold"].to_numpy(np.int8), codes, e274, e264, d274)
        stats.update("distance_bucket", distance_cat, codes, e274, e264, d274)
        stats.update("relative_tail_decile", relative_cat, codes, e274, e264, d274)
        stats.update("abs_target_residual_bucket", residual_cat, codes, e274, e264, d274)
        stats.update("gr_missing", np.isnan(gr).astype(np.int8), codes, e274, e264, d274)
        stats.update("well_group", well_group, codes, e274, e264, d274)

        np.add.at(n, codes, 1)
        np.add.at(sum_d274, codes, d274)
        np.add.at(sumsq_d274, codes, d274.astype(np.float64) ** 2)
        np.add.at(sumabs_d274, codes, np.abs(d274))
        np.maximum.at(maxabs_d274, codes, np.abs(d274))
        np.add.at(sum_t, codes, distance)
        np.add.at(sum_t2, codes, distance.astype(np.float64) ** 2)
        np.add.at(sum_td274, codes, distance.astype(np.float64) * d274)
        np.add.at(sumsq_d238, codes, d238.astype(np.float64) ** 2)
        np.add.at(sumabs_d238, codes, np.abs(d238))
        np.maximum.at(maxabs_d238, codes, np.abs(d238))
        np.minimum.at(min_p264, codes, p264)
        np.maximum.at(max_p264, codes, p264)
        np.minimum.at(min_p274, codes, p274)
        np.maximum.at(max_p274, codes, p274)
        for code in np.unique(codes):
            last_d274[code] = float(d274[np.flatnonzero(codes == code)[-1]])

        total += len(a)
        print(f"chunk {chunk_no}: {total:,}", flush=True)

    if total != 3_783_989 or not np.array_equal(n, well["rows"].to_numpy(np.int64)):
        raise ValueError("streamed row counts do not match well contract")

    labels = {
        "outer_fold": None,
        "distance_bucket": dict(enumerate(["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"])),
        "relative_tail_decile": {i: f"q{i+1}" for i in range(10)},
        "abs_target_residual_bucket": dict(enumerate(["000_002", "002_005", "005_010", "010_020", "020_050", "050_plus"])),
        "gr_missing": {0: "GR_observed", 1: "GR_missing"},
        "well_group": {0: "improved", 1: "worse_0_1", 2: "worse_1_3", 3: "worse_3_5", 4: "worse_5_plus"},
        "overall": {0: "all"},
    }
    tables = {}
    for name, label_map in labels.items():
        table = stats.frame(name, label_map)
        tables[name] = table
        table.to_csv(OUT / f"row_metric_{name}.csv", index=False)

    denom = n * sum_t2 - sum_t**2
    slope = np.divide(n * sum_td274 - sum_t * sum_d274, denom, out=np.full(n_wells, np.nan), where=denom != 0)
    disagreement = pd.DataFrame(
        {
            "well": well_names,
            "pair_rmse_264_274": np.sqrt(sumsq_d274 / n),
            "pair_absmean_264_274": sumabs_d274 / n,
            "pair_maxabs_264_274": maxabs_d274,
            "pair_mean_264_274": sum_d274 / n,
            "pair_std_264_274": np.sqrt(np.maximum(sumsq_d274 / n - (sum_d274 / n) ** 2, 0)),
            "pair_end_264_274": last_d274,
            "pair_slope_per_row_264_274": slope,
            "pair_rmse_264_238": np.sqrt(sumsq_d238 / n),
            "pair_absmean_264_238": sumabs_d238 / n,
            "pair_maxabs_264_238": maxabs_d238,
            "exp264_pred_range": max_p264 - min_p264,
            "exp274_pred_range": max_p274 - min_p274,
        }
    )
    combined = well.merge(disagreement, on="well", validate="one_to_one")
    combined.to_csv(OUT / "well_prediction_disagreement.csv", index=False)

    pair_features = [c for c in disagreement if c != "well"]
    effects = []
    for threshold in [0.0, 1.0, 3.0, 5.0]:
        severe = combined["delta_exp264_vs_exp274"] > threshold
        for feature in pair_features:
            a = combined.loc[severe, feature].to_numpy(float)
            b = combined.loc[~severe, feature].to_numpy(float)
            u, p = mannwhitneyu(a, b)
            rho, rp = spearmanr(combined[feature], combined["delta_exp264_vs_exp274"])
            effects.append({
                "threshold": threshold,
                "feature": feature,
                "severe_median": np.median(a),
                "other_median": np.median(b),
                "cliffs_delta": 2*u/(len(a)*len(b))-1,
                "p": p,
                "spearman_delta": rho,
                "spearman_p": rp,
            })
    effects_df = pd.DataFrame(effects).sort_values(["threshold", "p"])
    effects_df.to_csv(OUT / "prediction_disagreement_effects.csv", index=False)

    auc_rows = []
    features_all = ["pair_rmse_264_274", "pair_mean_264_274", "pair_std_264_274", "pair_end_264_274", "pair_slope_per_row_264_274", "pair_maxabs_264_274"]
    for threshold in [0.0, 1.0, 3.0, 5.0]:
        auc_rows.append({
            "threshold": threshold,
            "positive_wells": int((combined["delta_exp264_vs_exp274"] > threshold).sum()),
            "auc_pair_rmse_only": auc(combined, ["pair_rmse_264_274"], threshold),
            "auc_pair_summary": auc(combined, features_all, threshold),
        })
    auc_df = pd.DataFrame(auc_rows)
    auc_df.to_csv(OUT / "prediction_disagreement_auc.csv", index=False)

    for name, table in tables.items():
        print(f"\n=== {name} ===")
        print(table.to_string(index=False))
    print("\n=== pair effects ===")
    for threshold in [0.0, 1.0, 3.0, 5.0]:
        print(f"\nthreshold > {threshold}")
        print(effects_df[effects_df.threshold.eq(threshold)].head(10).to_string(index=False))
    print("\n=== pair AUC ===")
    print(auc_df.to_string(index=False))
    print("\n=== top pair readout ===")
    cols = ["well", "exp274_rmse", "exp264_rmse", "delta_exp264_vs_exp274", *features_all]
    print(combined.sort_values("delta_exp264_vs_exp274", ascending=False)[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
