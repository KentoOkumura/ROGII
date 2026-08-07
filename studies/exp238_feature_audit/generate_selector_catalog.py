# ruff: noqa: E501
from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP237_IMPORTANCE = ROOT / (
    "studies/exp238_feature_audit/inputs/"
    "exp237_selector_feature_importance_mean.csv"
)
EXP251_IMPORTANCE = ROOT / (
    "experiments/exp251_raw_test_safe_dual_objective_candidate_ranker/"
    "kaggle/output/train_v4/artifacts/"
    "exp251_raw_test_safe_dual_objective_candidate_ranker_feature_importance_mean.csv"
)
EXP251_SCHEMA = ROOT / (
    "experiments/exp251_raw_test_safe_dual_objective_candidate_ranker/"
    "kaggle/output/train_v4/artifacts/"
    "exp251_raw_test_safe_dual_objective_candidate_ranker_selected_feature_schema.csv"
)
OUTPUT_DIR = ROOT / "studies/exp238_feature_audit/outputs"
EXP237_CATALOG = OUTPUT_DIR / "exp237_selector_feature_catalog_importance.csv"
EXP251_CATALOG = OUTPUT_DIR / "exp251_v4_selector_feature_catalog_importance.csv"
EXP238_TVT_CATALOG = OUTPUT_DIR / "exp238_feature_catalog_importance.csv"
REPORT = ROOT / "docs/surveys/selector_feature_catalog_20260716.md"


CANDIDATES = [
    "pf_ancc",
    "beam_mean",
    "likpf_mean",
    "sc_ens",
    "hyb",
    "tvt_dense",
    "tvt_densew",
    "tvt_dense50",
    "blend_likpf_hmm_w500",
    "hmm_selfgr_boost_only_a070_c100",
    "v6_k16_geometry_gr_u_projection",
]

CANDIDATE_LABELS = {
    "pf_ancc": "ANCC PF",
    "beam_mean": "複数Beam path平均",
    "likpf_mean": "likelihood-weighted PF平均",
    "sc_ens": "multi-scale NCC ensemble",
    "hyb": "Beam/NCC hybrid",
    "tvt_dense": "dense spatial ANCC（full-prefix bias）",
    "tvt_densew": "dense spatial ANCC（prefix加重bias）",
    "tvt_dense50": "dense spatial ANCC（prefix末尾50 bias）",
    "blend_likpf_hmm_w500": "likPF / exact-HMM 50:50 blend",
    "hmm_selfgr_boost_only_a070_c100": "Self-GR likelihood HMM",
    "v6_k16_geometry_gr_u_projection": "exp226 K16 geometry/GR/U-projection",
}

PRIOR_LABELS = {
    "typewell_native_overlap_0p999": "native-overlap閾値0.999のtypewell cluster prior",
    "typewell_native_overlap_1": "native-overlap閾値1.0のtypewell cluster prior",
    "spatial_xy_only_k8": "XY距離だけを使うK=8 spatial prior",
    "spatial_xy_plus_trajectory_shape_k8": "XY距離とtrajectory shapeを使うK=8 spatial prior",
}

GATE_LABELS = {
    "any_outlier_signal_k8": "K=8近傍のいずれかのoutlier signal",
    "nearby_majority_diff_k8": "K=8近傍の多数派cluster不一致",
    "nearest_other_closer": "別clusterの方が近い",
    "own_z_gt2p0": "所属cluster距離z-score>2",
}


BASE_DESCRIPTIONS = {
    "last_known_tvt": "既知prefix末尾のTVT。全候補deltaのanchor。",
    "candidate_tvt": "このcandidate-long行が表す候補パスの絶対TVT。",
    "candidate_minus_last": "この候補TVT − last_known_tvt。候補の外挿量。",
    "candidate_index": "固定11候補bank内の候補index。候補identityを木に伝える数値code。",
    "candidate_name_code": "candidate_indexと同じ候補identity codeを別契約名で保持した列。",
    "candidate_mean": "同じbase rowにある候補TVT群の平均。",
    "candidate_std": "同じbase rowにある候補TVT群の標準偏差。候補間disagreement。",
    "candidate_range": "同じbase rowにある候補TVT群の最大−最小。候補間disagreement。",
    "candidate_is_dense_family": "対象候補がdense 3候補なら1。",
    "candidate_is_hmm_family": "対象候補がHMMまたはHMM blendなら1。",
    "candidate_is_default_likpf": "対象候補が既定fallbackのlikpf_meanなら1。",
    "candidate_is_geometry_family": "対象候補がexp226 geometry候補なら1。",
    "candidate_is_pfbeam_family": "対象候補がPF/Beam系なら1。",
    "candidate_multiobs_score": "対象候補自身のmulti-observation GR一致score。",
    "candidate_multiobs_mae": "対象候補自身のmulti-observation GR照合MAE。",
    "candidate_multiobs_ncc": "対象候補自身のmulti-observation GR照合NCC。",
    "eval_len": "そのwellの予測tail行数。",
    "md_since": "既知prefix末尾から予測行までのMD距離。",
    "pf_ancc_std": "ANCC PF粒子の行別TVT標準偏差。粒子分布の幅。",
    "dense_std": "dense spatial KNN近傍ANCCの重み付き標準偏差。",
    "dense_dist": "dense spatial KNNで使う最短正規化XY距離。",
    "pf_vs_dense": "ANCC PF候補 − dense ANCC候補。",
    "hmm_exact_std": "exact HMM posteriorのTVT標準偏差。",
    "hmm_exact_loglik": "exact HMMの行別観測log-likelihood。",
    "hmm_selfgr_std": "Self-GR likelihood HMM posteriorのTVT標準偏差。",
    "hmm_selfgr_loglik": "Self-GR likelihood HMMの行別観測log-likelihood。",
    "hmm_exact_minus_likpf_mean": "exact HMM候補 − likelihood-weighted PF平均候補。",
    "self_gr_quality": "同一horizontal内Self-GR motif照合の品質score。",
    "self_gr_peak_gap": "Self-GR照合のbest peakとsecond peakのscore差。",
    "self_gr_valid": "Self-GR照合が入力条件を満たして有効なら1。",
    "self_gr_typewell_agreement": "Self-GR観測とtypewell観測の整合度。",
    "exp226_gr_delta": "exp226候補位置でのGR整合差。exp251 v4ではraw-test契約外として除外。",
    "exp226_geop_tvt": "exp226 geometry projectionが返す絶対TVT。exp251 v4ではraw-test契約外として除外。",
    "exp226_geop_minus_pred": "exp226 geometry projection TVT − exp226最終候補TVT。",
    "exp226_geop_minus_pred_abs": "|exp226 geometry projection TVT − exp226最終候補TVT|。",
    "beam_mean_d": "複数Beam path平均候補 − last_known_tvt。",
    "likpf_mean_d": "likelihood-weighted PF平均候補 − last_known_tvt。",
    "sc_ens_d": "multi-scale NCC ensemble候補 − last_known_tvt。",
    "hyb_d": "Beam/NCC hybrid候補 − last_known_tvt。",
    "tvt_dense_d": "dense full-prefix候補 − last_known_tvt。",
    "tvt_densew_d": "dense prefix加重候補 − last_known_tvt。",
    "tvt_dense50_d": "dense prefix末尾50候補 − last_known_tvt。",
    "hmm_selfgr_boost_only_a070_c100_mean_tvt": "Self-GR likelihood HMMの絶対TVT候補。",
    "blend_likpf_hmm_w500": "likPF / exact-HMM 50:50 blendの絶対TVT候補。",
    "exp226_v6_k16_geometry_gr_u_projection": "exp226 K16 geometry/GR/U-projectionの絶対TVT候補。",
    "view_candidate_count": "raw-test viewで利用可能な候補数。",
    "view_candidate_mean": "raw-test viewで利用可能な候補TVTの平均。",
    "view_candidate_std": "raw-test viewで利用可能な候補TVTの標準偏差。",
    "view_candidate_std_safe": "candidate z-score用に下限を置いたview内候補標準偏差。",
    "view_candidate_range": "raw-test viewで利用可能な候補TVTの最大−最小。",
    "view_score_best": "raw-test viewで利用可能な候補のmulti-observation score最大値。",
    "view_multiobs_available_count": "multi-observation scoreが利用可能な候補数。",
    "view_hmm_available_count": "HMM診断が利用可能な候補数。",
    "view_exp226_available_count": "exp226診断が利用可能な候補数。",
    "view_dense_available_count": "dense family候補が利用可能な候補数。",
    "view_geometry_available_count": "geometry family候補が利用可能な候補数。",
    "view_pfbeam_available_count": "PF/Beam family候補が利用可能な候補数。",
    "candidate_abs_minus_view_mean": "|対象候補TVT − view内候補平均|。",
    "candidate_z_within_view": "(対象候補TVT − view内候補平均) / view内候補標準偏差。",
    "candidate_score_gap_from_view_best": "view内best multi-observation score − 対象候補score。",
    "multiobs_score_mean": "候補bankのmulti-observation GR一致score平均。",
    "multiobs_score_max": "候補bankのmulti-observation GR一致score最大値。",
    "multiobs_score_gap": "multi-observation GR一致scoreの1位−2位差。",
    "multiobs_top1_mae": "multi-observation score最上位候補のGR照合MAE。",
    "multiobs_top1_ncc": "multi-observation score最上位候補のGR照合NCC。",
    "multiobs_top1_source_id": "multi-observation score最上位候補のsource code。",
}

COPCF_BASE_DESCRIPTIONS = {
    "copcf_own_cluster_dist": "test wellと割当先cluster中心の距離。",
    "copcf_own_cluster_dist_z": "割当先cluster距離のcluster内z-score。",
    "copcf_nearest_other_cluster_dist": "最も近い別cluster中心までの距離。",
    "copcf_nearest_other_closer": "最も近い別clusterが割当先clusterより近ければ1。",
    "copcf_nearby_majority_count_k5": "近傍5 well中、多数派clusterに属するwell数。",
    "copcf_nearby_majority_count_k8": "近傍8 well中、多数派clusterに属するwell数。",
    "copcf_nearby_majority_count_k12": "近傍12 well中、多数派clusterに属するwell数。",
    "copcf_nearby_majority_share_k5": "近傍5 wellの多数派cluster比率。",
    "copcf_nearby_majority_share_k8": "近傍8 wellの多数派cluster比率。",
    "copcf_nearby_majority_share_k12": "近傍12 wellの多数派cluster比率。",
    "copcf_nearby_majority_diff_k5": "近傍5 wellの多数派clusterが自身の割当clusterと異なれば1。",
    "copcf_nearby_majority_diff_k8": "近傍8 wellの多数派clusterが自身の割当clusterと異なれば1。",
    "copcf_nearby_majority_diff_k12": "近傍12 wellの多数派clusterが自身の割当clusterと異なれば1。",
    "copcf_cluster_feature_valid": "cluster/outlier診断が計算可能なら1。",
    "copcf_gate_any_outlier_signal_k8": "K=8 outlier signalの総合gate flag。",
    "copcf_gate_nearby_majority_diff_k8": "K=8近傍多数派不一致のgate flag。",
    "copcf_gate_nearest_other_closer": "別clusterの方が近いgate flag。",
    "copcf_gate_own_z_gt2p0": "所属cluster距離z-score>2のgate flag。",
    "copcf_any_configured_gate": "設定済みcluster/outlier gateのいずれかが有効なら1。",
    "copcf_well_gate_ratio_any_outlier_signal_k8": "well内でany-outlier gateが有効な行の比率。",
    "copcf_well_gate_ratio_nearby_majority_diff_k8": "well内で近傍多数派不一致gateが有効な行の比率。",
    "copcf_well_gate_ratio_nearest_other_closer": "well内で別cluster近接gateが有効な行の比率。",
    "copcf_well_gate_ratio_own_z_gt2p0": "well内で所属cluster距離z>2 gateが有効な行の比率。",
    "copcf_typewell_spatial_prior_delta": "typewell prior − spatial prior。2種類の補正方向の差。",
    "copcf_typewell_spatial_prior_abs_delta": "|typewell prior − spatial prior|。",
}

CRFE_DESCRIPTIONS = {
    "crfe_tvt_dense_abs_delta_from_last": "|dense full-prefix候補 − anchor|。",
    "crfe_tvt_densew_abs_delta_from_last": "|dense prefix加重候補 − anchor|。",
    "crfe_tvt_dense50_abs_delta_from_last": "|dense末尾50候補 − anchor|。",
    "crfe_dense_candidate_mean": "dense 3候補TVTの平均。",
    "crfe_dense_candidate_std": "dense 3候補TVTの標準偏差。",
    "crfe_dense_candidate_range": "dense 3候補TVTの最大−最小。",
    "crfe_pf_ancc_minus_tvt_densew": "ANCC PF候補 − dense prefix加重候補。",
    "crfe_likpf_mean_minus_tvt_densew": "likPF平均候補 − dense prefix加重候補。",
    "crfe_beam_mean_minus_tvt_densew": "Beam平均候補 − dense prefix加重候補。",
    "crfe_pf_ancc_minus_tvt_densew_abs_norm": "|ANCC PF − dense加重| / max(dense_std, 10)。",
    "crfe_likpf_mean_minus_tvt_densew_abs_norm": "|likPF − dense加重| / max(dense_std, 10)。",
    "crfe_beam_mean_minus_tvt_densew_abs_norm": "|Beam平均 − dense加重| / max(dense_std, 10)。",
    "crfe_tail_rank_norm": "min(予測tail内row index / 1000, 5)。",
    "crfe_longtail_1000_flag": "anchorから1000行以上のlong-tailなら1。",
    "crfe_near_md_50_flag": "anchorからMD 50以内なら1。",
    "crfe_tvt_dense_drift_per_md": "(dense full-prefix候補 − anchor) / max(md_since, 1)。",
    "crfe_tvt_densew_drift_per_md": "(dense加重候補 − anchor) / max(md_since, 1)。",
    "crfe_tvt_dense50_drift_per_md": "(dense末尾50候補 − anchor) / max(md_since, 1)。",
    "crfe_pf_vs_dense_abs_norm": "|ANCC PF − dense ANCC| / max(dense_std, 10)。",
    "crfe_dense_std_norm": "dense_std / 10を上限付きで正規化した不確実性proxy。",
    "crfe_dense_dist_norm": "dense_distを上限付きで正規化した近傍距離proxy。",
    "crfe_high_disagreement_proxy": "PF-vs-dense差、dense std、dense距離を0.45/0.35/0.20で合成したdisagreement。",
    "crfe_high_disagreement_x_longtail": "high_disagreement_proxy × longtail_1000_flag。",
}


def family(feature: str) -> str:
    if feature.startswith("copcf_"):
        return "cluster_prior_confidence"
    if feature.startswith("crfe_"):
        return "disagreement_enrichment"
    if feature.startswith("multiobs_") or feature.startswith("candidate_multiobs_"):
        return "multi_observation"
    if feature.startswith("self_gr_"):
        return "self_gr_confidence"
    if feature.startswith("hmm_"):
        return "hmm_confidence"
    if feature.startswith("exp226_"):
        return "exp226_geometry"
    if feature.startswith("view_") or feature in {
        "candidate_abs_minus_view_mean",
        "candidate_z_within_view",
        "candidate_score_gap_from_view_best",
    }:
        return "raw_test_view_context"
    if feature.startswith("candidate_"):
        return "candidate_identity_or_context"
    if "_vs_" in feature or feature in {"pf_vs_dense", "hmm_exact_minus_likpf_mean"}:
        return "candidate_disagreement"
    if feature in {"eval_len", "md_since", "last_known_tvt"}:
        return "distance_or_anchor_context"
    if feature.startswith("dense_") or feature == "pf_ancc_std":
        return "pf_dense_confidence"
    return "candidate_path_value"


def _candidate_label(name: str) -> str:
    return CANDIDATE_LABELS[name]


def _describe_pairwise(feature: str) -> str | None:
    for left in CANDIDATES:
        for right in CANDIDATES:
            if feature == f"{left}_vs_{right}_abs":
                return f"|{_candidate_label(left)} TVT − {_candidate_label(right)} TVT|。"
    return None


def _describe_copcf(feature: str) -> str | None:
    if feature in COPCF_BASE_DESCRIPTIONS:
        return COPCF_BASE_DESCRIPTIONS[feature]
    if not feature.startswith("copcf_"):
        return None
    body = feature[len("copcf_") :]
    source = next((name for name in PRIOR_LABELS if body.startswith(name + "_")), None)
    if source is None:
        return None
    prior = PRIOR_LABELS[source]
    suffix = body[len(source) + 1 :]
    simple = {
        "minus_candidate": f"{prior} TVT − 対象候補TVT。",
        "minus_candidate_abs": f"|{prior} TVT − 対象候補TVT|。",
        "minus_candidate_abs_norm": f"|{prior} TVT − 対象候補TVT|をprior標準偏差で正規化。",
        "prior_std": f"{prior}を作るsource TVTの標準偏差。",
        "prior_count": f"{prior}を作る有効source数。",
        "neighbor_wells": f"{prior}を作る近傍well数。",
        "valid_prior": f"{prior}が計算可能なら1。",
        "std_x_candidate": f"{prior}の標準偏差をcandidate-long各行へ複製した列。積ではない。",
        "count_x_candidate": f"{prior}のsource数をcandidate-long各行へ複製した列。積ではない。",
        "neighbor_wells_x_candidate": f"{prior}の近傍well数をcandidate-long各行へ複製した列。積ではない。",
        "valid_x_candidate": f"{prior}のvalid flagをcandidate-long各行へ複製した列。積ではない。",
    }
    if suffix in simple:
        return simple[suffix]
    for gate, gate_label in GATE_LABELS.items():
        prefix = gate + "_"
        if not suffix.startswith(prefix):
            continue
        operation = suffix[len(prefix) :]
        if operation == "gate_x_candidate":
            return f"{prior}が有効かつ{gate_label}なら1。candidate-long行へ複製するgate。"
        if operation == "gate_x_dense_family":
            return f"{prior}が有効、{gate_label}、かつ対象候補がdense familyなら1。"
        match = re.fullmatch(r"corr_abs_c(20|40)", operation)
        if match:
            cap = match.group(1)
            return f"{gate_label}時に{prior}へ寄せる仮想補正量の絶対値（clip {cap} ft）。"
        match = re.fullmatch(r"clip_hit_c(20|40)", operation)
        if match:
            cap = match.group(1)
            return f"{gate_label}時のprior−候補差がclip {cap} ftを超えれば1。"
    return None


def describe(feature: str) -> str:
    if feature in BASE_DESCRIPTIONS:
        return BASE_DESCRIPTIONS[feature]
    if feature in CRFE_DESCRIPTIONS:
        return CRFE_DESCRIPTIONS[feature]
    description = _describe_copcf(feature)
    if description is not None:
        return description
    description = _describe_pairwise(feature)
    if description is not None:
        return description
    match = re.fullmatch(r"multiobs_(score|mae|ncc)_(pf_ancc|beam_mean|likpf_mean|sc_ens|hyb)", feature)
    if match:
        metric, candidate = match.groups()
        labels = {"score": "GR一致score", "mae": "GR照合MAE", "ncc": "GR照合NCC"}
        return f"{_candidate_label(candidate)}のmulti-observation {labels[metric]}。"
    match = re.fullmatch(
        r"(pf_ancc|beam_mean|likpf_mean|sc_ens|hyb|tvt_dense|tvt_densew|tvt_dense50|blend_likpf_hmm_w500|hmm_selfgr_boost_only_a070_c100|v6_k16_geometry_gr_u_projection)_minus_last",
        feature,
    )
    if match:
        return f"{_candidate_label(match.group(1))} TVT − last_known_tvt。"
    raise ValueError(f"No selector feature description rule for: {feature}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rank_map(values: dict[str, float]) -> dict[str, int]:
    ordered = sorted(values, key=lambda feature: (-values[feature], feature))
    return {feature: rank for rank, feature in enumerate(ordered, start=1)}


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def family_share(rows: list[dict[str, object]], share_field: str) -> list[tuple[str, int, float]]:
    counts: dict[str, int] = defaultdict(int)
    shares: dict[str, float] = defaultdict(float)
    for row in rows:
        key = str(row["family"])
        counts[key] += 1
        shares[key] += float(row[share_field])
    return sorted(
        ((key, counts[key], shares[key]) for key in counts),
        key=lambda item: (-item[2], item[0]),
    )


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("correlation inputs have different sizes")
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    numerator = sum(
        (x - mean_left) * (y - mean_right)
        for x, y in zip(left, right, strict=True)
    )
    denom_left = math.sqrt(sum((x - mean_left) ** 2 for x in left))
    denom_right = math.sqrt(sum((y - mean_right) ** 2 for y in right))
    return numerator / (denom_left * denom_right)


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    def escape(cell: str) -> str:
        return cell.replace("|", r"\|").replace("\n", " ")

    return [
        "| " + " | ".join(escape(header) for header in headers) + " |",
        "| " + " | ".join("---" if index else "---" for index in range(len(headers))) + " |",
        *("| " + " | ".join(escape(cell) for cell in row) + " |" for row in rows),
    ]


def main() -> None:
    exp238_tvt_rows = [
        row
        for row in read_csv(EXP238_TVT_CATALOG)
        if row["family"] == "nested_selector"
    ]
    if len(exp238_tvt_rows) != 35:
        raise ValueError(
            "exp238 TVT catalog must contain 35 nested_selector rows; "
            "run generate_report.py first"
        )

    exp237_raw = read_csv(EXP237_IMPORTANCE)
    exp237_importance = {row["feature"]: float(row["mean_importance"]) for row in exp237_raw}
    exp237_std = {row["feature"]: float(row["std_importance"]) for row in exp237_raw}
    exp237_folds = {row["feature"]: int(row["folds"]) for row in exp237_raw}
    exp237_ranks = rank_map(exp237_importance)
    exp237_total = sum(exp237_importance.values())
    exp237_rows: list[dict[str, object]] = []
    for feature in sorted(exp237_importance, key=exp237_ranks.get):
        exp237_rows.append(
            {
                "rank": exp237_ranks[feature],
                "feature": feature,
                "family": family(feature),
                "mean_split_importance": exp237_importance[feature],
                "std_split_importance": exp237_std[feature],
                "importance_share_pct": 100.0 * exp237_importance[feature] / exp237_total,
                "folds": exp237_folds[feature],
                "description": describe(feature),
            }
        )

    exp251_raw = read_csv(EXP251_IMPORTANCE)
    exp251_schema = [row["feature"] for row in read_csv(EXP251_SCHEMA)]
    objectives = ["expected_error_regressor", "within10_classifier"]
    by_objective: dict[str, dict[str, float]] = {objective: {} for objective in objectives}
    for row in exp251_raw:
        by_objective[row["objective"]][row["feature"]] = float(row["importance"])
    schema_set = set(exp251_schema)
    for objective in objectives:
        if set(by_objective[objective]) != schema_set:
            raise ValueError(f"exp251 schema mismatch for {objective}")
    objective_ranks = {objective: rank_map(by_objective[objective]) for objective in objectives}
    objective_totals = {objective: sum(by_objective[objective].values()) for objective in objectives}
    combined_share = {
        feature: sum(
            100.0 * by_objective[objective][feature] / objective_totals[objective]
            for objective in objectives
        )
        / len(objectives)
        for feature in exp251_schema
    }
    combined_ranks = rank_map(combined_share)
    exp251_rows: list[dict[str, object]] = []
    for feature in sorted(exp251_schema, key=combined_ranks.get):
        exp251_rows.append(
            {
                "combined_rank": combined_ranks[feature],
                "feature": feature,
                "family": family(feature),
                "combined_importance_share_pct": combined_share[feature],
                "expected_error_rank": objective_ranks["expected_error_regressor"][feature],
                "expected_error_split_importance": by_objective["expected_error_regressor"][feature],
                "expected_error_share_pct": 100.0
                * by_objective["expected_error_regressor"][feature]
                / objective_totals["expected_error_regressor"],
                "within10_rank": objective_ranks["within10_classifier"][feature],
                "within10_split_importance": by_objective["within10_classifier"][feature],
                "within10_share_pct": 100.0
                * by_objective["within10_classifier"][feature]
                / objective_totals["within10_classifier"],
                "description": describe(feature),
            }
        )

    write_csv(
        EXP237_CATALOG,
        exp237_rows,
        [
            "rank",
            "feature",
            "family",
            "mean_split_importance",
            "std_split_importance",
            "importance_share_pct",
            "folds",
            "description",
        ],
    )
    write_csv(
        EXP251_CATALOG,
        exp251_rows,
        [
            "combined_rank",
            "feature",
            "family",
            "combined_importance_share_pct",
            "expected_error_rank",
            "expected_error_split_importance",
            "expected_error_share_pct",
            "within10_rank",
            "within10_split_importance",
            "within10_share_pct",
            "description",
        ],
    )

    common = sorted(set(exp237_importance) & schema_set)
    rank_correlation = pearson(
        [float(exp237_ranks[feature]) for feature in common],
        [float(objective_ranks["expected_error_regressor"][feature]) for feature in common],
    )
    objective_rank_correlation = pearson(
        [float(objective_ranks["expected_error_regressor"][feature]) for feature in exp251_schema],
        [float(objective_ranks["within10_classifier"][feature]) for feature in exp251_schema],
    )

    lines = [
        "# Selector入力特徴量カタログと重要度（exp237 / exp251 v4）",
        "",
        "## 結論",
        "",
        "- exp238のnested selector入力は、直接の親であるexp237のcandidate-long **320列**である。exp238の`nsel_*` 35列はselectorの入力ではなく、selectorが出した候補別予測誤差をTVTモデルへ渡すadapterである。",
        "- exp251 v4はraw-test再生成契約を満たす **295列**を使い、`within10_classifier`と`expected_error_regressor`の2目的を学習した。",
        "- 重要度はLightGBMの平均split回数であり、特徴を削除したときの因果効果ではない。候補identity、候補値、距離、HMM σ、cluster priorなど相関した列の間でsplitが分散する。",
        f"- exp237とexp251の共通列は{len(common)}列。共通列におけるexp237誤差rankerとexp251 expected-errorの重要度順位相関は{rank_correlation:.3f}。exp251内のexpected-errorとwithin10の順位相関は{objective_rank_correlation:.3f}で、2目的は同一ではない。",
        "",
        "## 3種類の特徴を混同しない",
        "",
        "| 層 | 列数 | 内容 |",
        "| --- | ---: | --- |",
        "| exp237 selector入力 | 320 | 候補値・候補間差・HMM/PF/dense/Self-GR診断・multiobs・cluster priorなど |",
        "| exp251 v4 selector入力 | 295 | exp237系からraw-test再生成可能な契約へ整理し、view contextを追加 |",
        "| exp238 TVTモデルへのselector出力adapter | 35 | top1/top2候補値、予測誤差、margin、候補別予測誤差など。selector入力ではない |",
        "",
        "## exp238 downstream rank-slot adapter 35列",
        "",
        "exp238最終TVT LightGBMの415列中、`nsel_*`だけを抽出した。`share`は各LightGBM内の総split数で正規化して3 config平均した比率、`TVT rank`は415列全体での順位。35列合計shareは9.573%。",
        "",
    ]
    lines.extend(
        markdown_table(
            ["TVT rank", "feature", "share", "lgb0/lgb1/lgb2 split", "説明"],
            [
                [
                    row["rank"],
                    f"`{row['feature']}`",
                    f"{float(row['importance_share_pct']):.4f}%",
                    f"{float(row['lgb0_split']):.1f}/{float(row['lgb1_split']):.1f}/{float(row['lgb2_split']):.1f}",
                    row["description"],
                ]
                for row in exp238_tvt_rows
            ],
        )
    )
    lines.extend(
        [
        "",
        "## exp237 selector入力 320列",
        "",
        "重要度は5 outer foldの`lgb_candidate_error_ranker`平均split回数。",
        "",
        ]
    )
    lines.extend(
        markdown_table(
            ["rank", "feature", "family", "mean±std split", "share", "説明"],
            [
                [
                    str(row["rank"]),
                    f"`{row['feature']}`",
                    f"`{row['family']}`",
                    f"{float(row['mean_split_importance']):.1f}±{float(row['std_split_importance']):.1f}",
                    f"{float(row['importance_share_pct']):.3f}%",
                    str(row["description"]),
                ]
                for row in exp237_rows
            ],
        )
    )
    lines.extend(["", "### exp237 importance family集計", ""])
    lines.extend(
        markdown_table(
            ["family", "features", "split share"],
            [
                [f"`{name}`", str(count), f"{share:.3f}%"]
                for name, count, share in family_share(exp237_rows, "importance_share_pct")
            ],
        )
    )
    lines.extend(
        [
            "",
            "## exp251 v4 selector入力 295列 × 2目的",
            "",
            "combined順位はexpected-errorとwithin10それぞれのsplit shareを平均したもの。片方だけの強さを隠さないよう、両目的の順位・split回数を併記する。",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            [
                "comb rank",
                "feature",
                "family",
                "comb share",
                "error rank/split",
                "within10 rank/split",
                "説明",
            ],
            [
                [
                    str(row["combined_rank"]),
                    f"`{row['feature']}`",
                    f"`{row['family']}`",
                    f"{float(row['combined_importance_share_pct']):.3f}%",
                    f"{row['expected_error_rank']} / {float(row['expected_error_split_importance']):.1f}",
                    f"{row['within10_rank']} / {float(row['within10_split_importance']):.1f}",
                    str(row["description"]),
                ]
                for row in exp251_rows
            ],
        )
    )
    lines.extend(["", "### exp251 combined importance family集計", ""])
    lines.extend(
        markdown_table(
            ["family", "features", "combined split share"],
            [
                [f"`{name}`", str(count), f"{share:.3f}%"]
                for name, count, share in family_share(
                    exp251_rows, "combined_importance_share_pct"
                )
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 解釈上の注意",
            "",
            "- `candidate_index` / `candidate_name_code`は順序尺度ではなくcategory codeに近い。木では分割可能だが、将来candidate bankの順序を変えると意味が変わる。",
            "- `*_std`、log-likelihood、multiobs scoreはraw confidence proxyである。小さいσや大きいscoreを単独でhard gateにせず、候補identity・距離・disagreementと一緒にOOF学習する。",
            "- `copcf_*_x_candidate`の一部は掛け算ではなくcandidate-long行へ複製したcontext列である。`gate_x_dense_family`だけはgateとdense-family flagの論理積。",
            "- exp251 v4は`exp226_gr_delta`と`exp226_geop_tvt`をraw-test契約から除外した。exp226候補そのものを除外したわけではない。",
            "- selector重要度の重複・相関は、候補identity、候補値/delta、全pair差、dense 3本、prior 4系統×gate展開に構造的に多い。削減するならimportance下位から機械的に落とさず、family単位のOOF ablationで判定する。",
            "",
            "## 出典",
            "",
            "- `studies/exp238_feature_audit/inputs/exp237_selector_feature_importance_mean.csv`",
            "- `experiments/exp251_raw_test_safe_dual_objective_candidate_ranker/kaggle/output/train_v4/artifacts/exp251_raw_test_safe_dual_objective_candidate_ranker_feature_importance_mean.csv`",
            "- `experiments/exp251_raw_test_safe_dual_objective_candidate_ranker/kaggle/output/train_v4/artifacts/exp251_raw_test_safe_dual_objective_candidate_ranker_selected_feature_schema.csv`",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT.relative_to(ROOT)}")
    print(f"wrote {EXP237_CATALOG.relative_to(ROOT)} ({len(exp237_rows)} rows)")
    print(f"wrote {EXP251_CATALOG.relative_to(ROOT)} ({len(exp251_rows)} rows)")


if __name__ == "__main__":
    main()
