from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import mannwhitneyu, spearmanr


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "experiments/exp300_exp264_vs_exp274_well_selector_readout"
ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts"
SOURCE_DIR = ARTIFACT_DIR / "source_inputs"
WELL_PATH = ARTIFACT_DIR / "well_comparison_and_features.csv"
EXP274_OOF_PATH = SOURCE_DIR / (
    "exp274_catboost_final_regressor_swap_on_exp238_oof_predictions.csv.gz"
)
EXP264_STAGE_B_COMPACT_PATH = ROOT / (
    "experiments/exp264_exp263_candidate_confidence_dual_selector/"
    "kaggle/output/stage_b_v5/artifacts/compact_meta_oof.parquet"
)
EXP264_SELECTOR_ROOT = ROOT / (
    "experiments/exp264_exp263_candidate_confidence_dual_selector/"
    "kaggle/output/oof_selector_confidence_probe_v3/artifacts"
)
SELECTOR_MANIFEST_PATH = EXP264_SELECTOR_ROOT / (
    "exp264_exp263_candidate_confidence_dual_selector_"
    "oof_selector_confidence_probe_plot_manifest.csv"
)
SELECTOR_SUMMARY_PATH = EXP264_SELECTOR_ROOT / (
    "exp264_exp263_candidate_confidence_dual_selector_"
    "oof_selector_confidence_probe_summary.json"
)
SEVERITY_THRESHOLDS = [0.0, 1.0, 3.0, 5.0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_content(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_inputs() -> tuple[pd.DataFrame, dict[str, Any]]:
    well = pd.read_csv(WELL_PATH)
    manifest = pd.read_csv(SELECTOR_MANIFEST_PATH)
    summary = json.loads(SELECTOR_SUMMARY_PATH.read_text())
    if len(well) != 773 or well["well"].nunique() != 773:
        raise ValueError("well comparison must contain 773 unique wells")
    if len(manifest) != 773 or manifest["well"].nunique() != 773:
        raise ValueError("selector manifest must contain 773 unique wells")
    if summary["selector_contract"]["surface"] != (
        "corrected_stage_c_v6_strict_nested_outer_valid"
    ):
        raise ValueError("selector summary is not the corrected Stage C v6 surface")
    if summary["stage_d_contract"]["surface"] != (
        "corrected_stage_d_v3_selector_compact_addonly_lgb_mean"
    ):
        raise ValueError("selector summary is not aligned to corrected Stage D v3")
    joined = well.merge(manifest, on="well", suffixes=("", "_selector"), validate="one_to_one")
    if not np.array_equal(
        joined["rows"].to_numpy(np.int64),
        joined["rows_selector"].to_numpy(np.int64),
    ):
        raise ValueError("well row counts differ between OOF and selector manifest")
    joined["switches_per_1000"] = (
        1000.0 * joined["primary_candidate_switches"] / joined["rows"]
    )
    joined["hard_primary_minus_final_rmse"] = (
        joined["primary_error_top1_rmse"] - joined["final_oof_rmse"]
    )
    joined["well_group"] = pd.cut(
        joined["delta_exp264_vs_exp274"],
        [-np.inf, 0, 1, 3, 5, np.inf],
        labels=["improved", "worse_0_1", "worse_1_3", "worse_3_5", "worse_5_plus"],
    )
    return joined, summary


def build_fold_assignment_audit() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    exp274_map: dict[str, int] = {}
    for chunk in pd.read_csv(
        EXP274_OOF_PATH,
        usecols=["well", "outer_fold"],
        dtype={"well": "string", "outer_fold": "int8"},
        chunksize=300_000,
    ):
        for well, fold in chunk.drop_duplicates().itertuples(index=False):
            fold = int(fold)
            if str(well) in exp274_map and exp274_map[str(well)] != fold:
                raise ValueError(f"exp274 well occurs in multiple folds: {well}")
            exp274_map[str(well)] = fold

    exp264_map: dict[str, int] = {}
    parquet = pq.ParquetFile(EXP264_STAGE_B_COMPACT_PATH)
    for row_group in range(parquet.num_row_groups):
        frame = parquet.read_row_group(
            row_group, columns=["well", "outer_fold"]
        ).to_pandas().drop_duplicates()
        for well, fold in frame.itertuples(index=False):
            fold = int(fold)
            if str(well) in exp264_map and exp264_map[str(well)] != fold:
                raise ValueError(f"exp264 well occurs in multiple folds: {well}")
            exp264_map[str(well)] = fold
    if set(exp274_map) != set(exp264_map) or len(exp274_map) != 773:
        raise ValueError("exp264/exp274 fold maps do not cover the same 773 wells")
    rows = [
        {
            "well": well,
            "exp274_fold": exp274_map[well],
            "exp264_fold": exp264_map[well],
            "fold_match": exp274_map[well] == exp264_map[well],
        }
        for well in sorted(exp274_map)
    ]
    audit = pd.DataFrame(rows)
    logical_sha = sha256_json(
        audit[["well", "exp274_fold", "exp264_fold"]].to_dict(orient="records")
    )
    well = pd.read_csv(WELL_PATH)
    joined = well.merge(audit, on="well", validate="one_to_one")

    summary_rows = []
    for fold_match, group in joined.groupby("fold_match"):
        exp274_rmse = float(
            np.sqrt(np.average(group["exp274_rmse"] ** 2, weights=group["rows"]))
        )
        exp264_rmse = float(
            np.sqrt(np.average(group["exp264_rmse"] ** 2, weights=group["rows"]))
        )
        summary_rows.append(
            {
                "fold_match": bool(fold_match),
                "wells": len(group),
                "rows": int(group["rows"].sum()),
                "worsened_wells": int((group["delta_exp264_vs_exp274"] > 0).sum()),
                "worsened_gt1_wells": int((group["delta_exp264_vs_exp274"] > 1).sum()),
                "worsened_gt3_wells": int((group["delta_exp264_vs_exp274"] > 3).sum()),
                "exp274_rmse": exp274_rmse,
                "exp264_rmse": exp264_rmse,
                "delta_rmse": exp264_rmse - exp274_rmse,
            }
        )
    return audit, pd.DataFrame(summary_rows), logical_sha


def build_global_distribution(summary: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(summary["primary_candidate_distribution"])
    total = int(frame["rows"].sum())
    if total != 3_783_989 or not np.isclose(frame["share"].sum(), 1.0):
        raise ValueError("selector global candidate distribution contract failed")
    return frame.sort_values("rows", ascending=False).reset_index(drop=True)


def build_dominant_candidate_summary(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, group in joined.groupby("dominant_primary_candidate"):
        n_rows = int(group["rows"].sum())
        final_rmse = float(np.sqrt(group["final_squared_error_sum"].sum() / n_rows))
        hard_rmse = float(np.sqrt(group["primary_squared_error_sum"].sum() / n_rows))
        fixed_rmse = float(np.sqrt(group["fixed_squared_error_sum"].sum() / n_rows))
        exp274_rmse = float(
            np.sqrt(np.average(group["exp274_rmse"] ** 2, weights=group["rows"]))
        )
        rows.append(
            {
                "dominant_primary_candidate": candidate,
                "wells": len(group),
                "rows": n_rows,
                "worse_rate": float((group["delta_exp264_vs_exp274"] > 0).mean()),
                "worse_gt1_rate": float((group["delta_exp264_vs_exp274"] > 1).mean()),
                "worse_gt3_rate": float((group["delta_exp264_vs_exp274"] > 3).mean()),
                "worse_gt5_rate": float((group["delta_exp264_vs_exp274"] > 5).mean()),
                "median_delta_exp264_vs_exp274": float(
                    group["delta_exp264_vs_exp274"].median()
                ),
                "mean_delta_exp264_vs_exp274": float(
                    group["delta_exp264_vs_exp274"].mean()
                ),
                "median_dominant_share": float(
                    group["dominant_primary_candidate_share"].median()
                ),
                "median_switches_per_1000": float(group["switches_per_1000"].median()),
                "exp274_rmse": exp274_rmse,
                "final_exp264_rmse": final_rmse,
                "hard_primary_top1_rmse": hard_rmse,
                "fixed_top1_rmse": fixed_rmse,
                "final_minus_exp274_rmse": final_rmse - exp274_rmse,
                "hard_minus_final_rmse": hard_rmse - final_rmse,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "mean_delta_exp264_vs_exp274", ascending=False
    )


def build_selector_metric_effects(joined: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "primary_error_margin_mean",
        "primary_error_margin_p50",
        "primary_error_margin_p90",
        "primary_probability_margin_mean",
        "dominant_primary_candidate_share",
        "switches_per_1000",
        "hard_primary_minus_final_rmse",
    ]
    rows = []
    for threshold in SEVERITY_THRESHOLDS:
        severe = joined["delta_exp264_vs_exp274"] > threshold
        for metric in metrics:
            a = joined.loc[severe, metric].dropna().to_numpy(float)
            b = joined.loc[~severe, metric].dropna().to_numpy(float)
            u, p = mannwhitneyu(a, b, alternative="two-sided")
            rho, rho_p = spearmanr(
                joined[metric], joined["delta_exp264_vs_exp274"], nan_policy="omit"
            )
            rows.append(
                {
                    "threshold": threshold,
                    "metric": metric,
                    "severe_wells": int(severe.sum()),
                    "other_wells": int((~severe).sum()),
                    "severe_median": float(np.median(a)),
                    "other_median": float(np.median(b)),
                    "cliffs_delta": float(2 * u / (len(a) * len(b)) - 1),
                    "mannwhitney_p": float(p),
                    "spearman_delta": float(rho),
                    "spearman_p": float(rho_p),
                }
            )
    return pd.DataFrame(rows).sort_values(["threshold", "mannwhitney_p"])


def build_candidate_lift(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    candidates = sorted(joined["dominant_primary_candidate"].unique())
    for threshold in SEVERITY_THRESHOLDS:
        severe = joined["delta_exp264_vs_exp274"] > threshold
        for candidate in candidates:
            severe_count = int(
                joined.loc[severe, "dominant_primary_candidate"].eq(candidate).sum()
            )
            other_count = int(
                joined.loc[~severe, "dominant_primary_candidate"].eq(candidate).sum()
            )
            severe_share = severe_count / int(severe.sum())
            other_share = other_count / int((~severe).sum())
            rows.append(
                {
                    "threshold": threshold,
                    "candidate": candidate,
                    "severe_wells": severe_count,
                    "other_wells": other_count,
                    "severe_share": severe_share,
                    "other_share": other_share,
                    "lift": severe_share / other_share if other_share > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["threshold", "lift"], ascending=[True, False])


def build_well_group_performance(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name, group in joined.groupby("well_group", observed=True):
        n_rows = int(group["rows"].sum())
        exp274_rmse = float(
            np.sqrt(np.average(group["exp274_rmse"] ** 2, weights=group["rows"]))
        )
        final_rmse = float(np.sqrt(group["final_squared_error_sum"].sum() / n_rows))
        hard_rmse = float(np.sqrt(group["primary_squared_error_sum"].sum() / n_rows))
        fixed_rmse = float(np.sqrt(group["fixed_squared_error_sum"].sum() / n_rows))
        rows.append(
            {
                "well_group": str(group_name),
                "wells": len(group),
                "rows": n_rows,
                "exp274_rmse": exp274_rmse,
                "final_exp264_rmse": final_rmse,
                "hard_primary_top1_rmse": hard_rmse,
                "fixed_top1_rmse": fixed_rmse,
                "final_minus_exp274_rmse": final_rmse - exp274_rmse,
                "hard_minus_final_rmse": hard_rmse - final_rmse,
            }
        )
    return pd.DataFrame(rows)


def save_plots(
    joined: pd.DataFrame,
    dominant: pd.DataFrame,
    lift: pd.DataFrame,
) -> list[Path]:
    """Write dependency-free SVG diagnostics for the local readout."""

    def write_svg(path: Path, body: str, width: int, height: int) -> Path:
        payload = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">\n'
            '<rect width="100%" height="100%" fill="white"/>\n'
            '<style>text{font-family:system-ui,sans-serif;fill:#111827}'
            '.small{font-size:11px}.axis{stroke:#374151;stroke-width:1}'
            '.grid{stroke:#d1d5db;stroke-width:1}</style>\n'
            f"{body}\n</svg>\n"
        )
        path.write_text(payload)
        return path

    palette = [
        "#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2",
        "#be185d", "#4f46e5", "#65a30d", "#9333ea", "#475569",
    ]
    candidates = sorted(joined["dominant_primary_candidate"].unique())
    colors = {candidate: palette[i % len(palette)] for i, candidate in enumerate(candidates)}

    width, height = 1000, 650
    left, right, top, bottom = 85, 270, 55, 65
    plot_w, plot_h = width - left - right, height - top - bottom
    x = joined["primary_error_margin_mean"].to_numpy(float)
    y = joined["delta_exp264_vs_exp274"].to_numpy(float)
    x_min, x_max = 0.0, max(1.0, float(np.nanmax(x)))
    y_min, y_max = min(-12.0, float(np.nanmin(y))), max(18.0, float(np.nanmax(y)))
    x_pos = left + plot_w * (x - x_min) / (x_max - x_min)
    y_pos = top + plot_h * (y_max - y) / (y_max - y_min)
    body = [
        '<text x="500" y="28" text-anchor="middle" font-size="18">Selector confidence versus exp264 regression</text>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
    ]
    for value, stroke, dash in [(0.0, "#111827", ""), (3.0, "#dc2626", ' stroke-dasharray="5 4"')]:
        py = top + plot_h * (y_max - value) / (y_max - y_min)
        body.append(f'<line x1="{left}" y1="{py:.2f}" x2="{left + plot_w}" y2="{py:.2f}" stroke="{stroke}"{dash}/>')
    for i, row in enumerate(joined.itertuples(index=False)):
        candidate = row.dominant_primary_candidate
        body.append(
            f'<circle cx="{x_pos[i]:.2f}" cy="{y_pos[i]:.2f}" r="2.4" '
            f'fill="{colors[candidate]}" fill-opacity="0.55"><title>{escape(str(row.well))}: '
            f'{row.delta_exp264_vs_exp274:.3f}</title></circle>'
        )
    body.extend([
        f'<text x="{left + plot_w / 2:.0f}" y="625" text-anchor="middle" font-size="13">Primary pred_abs_error top1 margin mean</text>',
        '<text x="18" y="320" transform="rotate(-90 18 320)" text-anchor="middle" font-size="13">exp264 RMSE - exp274 RMSE by well</text>',
    ])
    for i, candidate in enumerate(candidates):
        ly = 70 + i * 28
        body.append(f'<rect x="750" y="{ly - 10}" width="12" height="12" fill="{colors[candidate]}"/>')
        body.append(f'<text class="small" x="770" y="{ly}">{escape(candidate)}</text>')
    plot_paths = [write_svg(ARTIFACT_DIR / "selector_margin_vs_regression.svg", "\n".join(body), width, height)]

    def horizontal_bar_svg(
        frame: pd.DataFrame,
        label_col: str,
        value_col: str,
        title: str,
        x_label: str,
        path: Path,
        reference: float = 0.0,
    ) -> Path:
        frame = frame.reset_index(drop=True)
        width, row_h = 1000, 34
        left, right, top, bottom = 300, 45, 55, 55
        height = top + bottom + row_h * len(frame)
        values = frame[value_col].to_numpy(float)
        x_min = min(0.0, reference, float(np.nanmin(values)))
        x_max = max(reference, float(np.nanmax(values)))
        pad = max((x_max - x_min) * 0.08, 0.05)
        x_min, x_max = x_min - pad, x_max + pad
        plot_w = width - left - right
        scale = lambda value: left + plot_w * (value - x_min) / (x_max - x_min)
        baseline = scale(0.0)
        body = [
            f'<text x="500" y="28" text-anchor="middle" font-size="18">{escape(title)}</text>',
            f'<line class="grid" x1="{scale(reference):.2f}" y1="{top - 10}" x2="{scale(reference):.2f}" y2="{height - bottom}"/>',
        ]
        for i, row in frame.iterrows():
            cy = top + i * row_h + row_h / 2
            value = float(row[value_col])
            x_value = scale(value)
            x0, bar_w = min(baseline, x_value), abs(x_value - baseline)
            color = "#b91c1c" if value > 0 and reference == 0.0 else "#0f766e"
            if value <= 0 and reference == 0.0:
                color = "#2563eb"
            body.append(f'<text class="small" x="290" y="{cy + 4:.2f}" text-anchor="end">{escape(str(row[label_col]))}</text>')
            body.append(f'<rect x="{x0:.2f}" y="{cy - 8:.2f}" width="{max(bar_w, 1):.2f}" height="16" fill="{color}"/>')
            body.append(f'<text class="small" x="{x_value + (5 if value >= 0 else -5):.2f}" y="{cy + 4:.2f}" text-anchor="{"start" if value >= 0 else "end"}">{value:.3f}</text>')
        body.append(f'<text x="{left + plot_w / 2:.0f}" y="{height - 14}" text-anchor="middle" font-size="13">{escape(x_label)}</text>')
        return write_svg(path, "\n".join(body), width, height)

    severe3 = lift[lift["threshold"].eq(3.0)].sort_values("lift")
    plot_paths.append(horizontal_bar_svg(
        severe3, "candidate", "lift", "Selector dominant-candidate lift",
        "Dominant-candidate prevalence lift in wells regressed >3 ft",
        ARTIFACT_DIR / "selector_dominant_candidate_gt3_lift.svg", reference=1.0,
    ))
    ordered = dominant.sort_values("mean_delta_exp264_vs_exp274")
    plot_paths.append(horizontal_bar_svg(
        ordered, "dominant_primary_candidate", "mean_delta_exp264_vs_exp274",
        "Regression by dominant primary selector candidate",
        "Mean well RMSE delta: exp264 - exp274",
        ARTIFACT_DIR / "selector_dominant_candidate_mean_delta.svg",
    ))
    return plot_paths


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joined, selector_summary = load_inputs()
    fold_audit, fold_summary, fold_map_sha = build_fold_assignment_audit()
    global_distribution = build_global_distribution(selector_summary)
    dominant = build_dominant_candidate_summary(joined)
    effects = build_selector_metric_effects(joined)
    lift = build_candidate_lift(joined)
    performance = build_well_group_performance(joined)
    top_worse = joined.sort_values("delta_exp264_vs_exp274", ascending=False).head(100)

    output_frames = {
        "selector_by_well.csv": joined,
        "selector_global_candidate_distribution.csv": global_distribution,
        "selector_dominant_candidate_summary.csv": dominant,
        "selector_metric_effects.csv": effects,
        "selector_candidate_lift_by_threshold.csv": lift,
        "selector_performance_by_well_group.csv": performance,
        "selector_top100_worse_wells.csv": top_worse,
        "fold_assignment_comparison.csv": fold_audit,
        "fold_assignment_summary.csv": fold_summary,
    }
    for filename, frame in output_frames.items():
        frame.to_csv(ARTIFACT_DIR / filename, index=False)
    plot_paths = save_plots(joined, dominant, lift)

    severe3_effects = effects[effects["threshold"].eq(3.0)].set_index("metric")
    summary = {
        "status": "complete_diagnostic_only",
        "rows": int(joined["rows"].sum()),
        "wells": len(joined),
        "source_contract": {
            "selector_surface": selector_summary["selector_contract"]["surface"],
            "stage_d_surface": selector_summary["stage_d_contract"]["surface"],
            "hard_primary_is_final_prediction": selector_summary["selector_contract"][
                "hard_primary_top1_is_final_prediction"
            ],
            "stage_c_outer_valid_score_sha256": selector_summary["inputs"]["sha256"][
                "stage_c_outer_valid_score"
            ],
            "stage_d_oof_sha256": selector_summary["inputs"]["sha256"][
                "stage_d_oof"
            ],
        },
        "global_metrics": selector_summary["global_metrics"],
        "fold_assignment": {
            "matched_wells": int(fold_audit["fold_match"].sum()),
            "mismatched_wells": int((~fold_audit["fold_match"]).sum()),
            "logical_mapping_sha256": fold_map_sha,
        },
        "selector_findings": {
            "beam_dominant_mean_delta": float(
                dominant.set_index("dominant_primary_candidate").loc[
                    "beam_mean", "mean_delta_exp264_vs_exp274"
                ]
            ),
            "likpf_dominant_mean_delta": float(
                dominant.set_index("dominant_primary_candidate").loc[
                    "likpf_mean", "mean_delta_exp264_vs_exp274"
                ]
            ),
            "selfgr_dominant_mean_delta": float(
                dominant.set_index("dominant_primary_candidate").loc[
                    "selfgr_hmm_a070", "mean_delta_exp264_vs_exp274"
                ]
            ),
            "gt3_margin_median": float(
                severe3_effects.loc["primary_error_margin_mean", "severe_median"]
            ),
            "other_margin_median": float(
                severe3_effects.loc["primary_error_margin_mean", "other_median"]
            ),
            "gt3_probability_margin_median": float(
                severe3_effects.loc[
                    "primary_probability_margin_mean", "severe_median"
                ]
            ),
            "other_probability_margin_median": float(
                severe3_effects.loc[
                    "primary_probability_margin_mean", "other_median"
                ]
            ),
            "gt3_dominant_share_median": float(
                severe3_effects.loc[
                    "dominant_primary_candidate_share", "severe_median"
                ]
            ),
            "other_dominant_share_median": float(
                severe3_effects.loc[
                    "dominant_primary_candidate_share", "other_median"
                ]
            ),
            "gt3_switches_per_1000_median": float(
                severe3_effects.loc["switches_per_1000", "severe_median"]
            ),
            "other_switches_per_1000_median": float(
                severe3_effects.loc["switches_per_1000", "other_median"]
            ),
        },
        "input_sha256": {
            "well_comparison": sha256_file(WELL_PATH),
            "selector_plot_manifest": sha256_file(SELECTOR_MANIFEST_PATH),
            "selector_summary": sha256_file(SELECTOR_SUMMARY_PATH),
            "exp274_oof_raw_gzip": sha256_file(EXP274_OOF_PATH),
            "exp274_oof_decompressed": sha256_gzip_content(EXP274_OOF_PATH),
        },
        "output_sha256": {
            filename: sha256_file(ARTIFACT_DIR / filename)
            for filename in output_frames
        }
        | {path.name: sha256_file(path) for path in plot_paths},
        "non_use_contract": [
            "diagnostic only",
            "do not update route anchors",
            "do not approve hard selector or router from this posthoc readout",
            "candidate dominance is a per-well summary, not row-level severe-group selection share",
        ],
    }
    summary_path = ARTIFACT_DIR / "selector_readout_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nDominant candidate summary")
    print(dominant.to_string(index=False))
    print("\nSelector metric effects for >3 ft regression")
    print(effects[effects["threshold"].eq(3.0)].to_string(index=False))


if __name__ == "__main__":
    main()
