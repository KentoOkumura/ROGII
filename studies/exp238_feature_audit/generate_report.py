from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "studies/exp238_feature_audit/inputs"
DEFAULT_IMPORTANCE = INPUT_DIR / "exp238_feature_importance_mean.csv"
DEFAULT_CANDIDATES = INPUT_DIR / "exp237_candidate_readout.csv"
DEFAULT_RESIDUAL_CORR = INPUT_DIR / "exp237_candidate_residual_correlation.csv"
DEFAULT_SELECTOR_IMPORTANCE = INPUT_DIR / "exp237_selector_feature_importance_mean.csv"
DEFAULT_SELECTOR_SCHEMA = INPUT_DIR / "exp237_selector_feature_schema.csv"
DROP_AUDIT = ROOT / (
    "studies/feature_replacement_audit/outputs/"
    "corr_prune_sanity_readout_on_exp148/"
    "corr_prune_sanity_readout_on_exp148_drop_candidates.csv"
)
OUTPUT_DIR = ROOT / "studies/exp238_feature_audit/outputs"
REPORT_PATH = ROOT / "docs/surveys/exp238_selector_tvt_feature_audit_20260716.md"
SELECTION_DISTRIBUTION_INPUT = INPUT_DIR / "exp237_selection_distribution.csv"
BEAM_MARGINAL_INPUT = INPUT_DIR / "exp237_beam_marginal_readout.csv"


CANDIDATE_LABELS = {
    "pf_ancc": "ANCC粒子フィルタ",
    "beam_mean": "複数Beam path平均",
    "likpf_mean": "likelihood-weighted PF平均",
    "sc_ens": "multi-scale NCC ensemble",
    "hyb": "Beam/NCC hybrid",
    "tvt_dense": "dense spatial ANCC（full-prefix bias）",
    "tvt_densew": "dense spatial ANCC（prefix加重bias）",
    "tvt_dense50": "dense spatial ANCC（prefix末尾50 bias）",
    "blend_likpf_hmm_w500": "likPFとexact HMMの50/50平均",
    "hmm_selfgr_boost_only_a070_c100": "self-GR HMM",
    "hmm_selfgr_boost_only_a070_c100_mean_tvt": "self-GR HMM",
    "v6_k16_geometry_gr_u_projection": "exp226 K16 geometry/GR/U-projection",
    "exp226_v6_k16_geometry_gr_u_projection": "exp226 K16 geometry/GR/U-projection",
}

BEAM_LABELS = {
    "cons": "conservative Beam",
    "loose": "loose Beam",
    "vcons": "very-conservative Beam",
    "sm5": "smoothed Beam (r=5)",
    "vloose": "very-loose Beam",
    "mid": "middle Beam",
    "stiff": "stiff Beam",
}

FORMATION_LABELS = {
    "ANCC": "ANCC formation surface",
    "ASTNU": "ASTNU formation surface",
    "ASTNL": "ASTNL formation surface",
    "EGFDU": "EGFDU formation surface",
    "EGFDL": "EGFDL formation surface",
    "BUDA": "BUDA formation surface",
}


def feature_family(feature: str) -> str:
    if feature.startswith("nsel_"):
        return "nested_selector"
    if feature.startswith("grwr_"):
        return "gr_wavelet_rotation"
    if feature.startswith("uproj_"):
        return "u_projection"
    if feature.startswith("ll_"):
        return "learned_likelihood"
    return "base_replay"


BASE_DESCRIPTIONS = {
    "last_known_tvt": "既知prefix末尾のTVT。全残差予測のanchor。",
    "pf_ancc": "ANCC particle filterの絶対TVT候補。",
    "pf_ancc_std": "ANCC particle filter粒子の行別TVT標準偏差。",
    "pf_ancc_delta": "ANCC PF候補 − last_known_tvt。",
    "pf_z": "Z-aware particle filterの絶対TVT候補。",
    "pf_z_delta": "Z-aware PF候補 − last_known_tvt。",
    "pf_vs_z": "ANCC PF候補 − Z-aware PF候補。",
    "beam_mean_d": "7種類のBeam候補deltaの行別平均。",
    "beam_std_d": "7種類のBeam候補deltaの行別標準偏差。",
    "beam_med_d": "7種類のBeam候補deltaの行別中央値。",
    "sc8_d": "half-window 8のmulti-scale NCC候補 − anchor。",
    "sc15_d": "half-window 15のmulti-scale NCC候補 − anchor。",
    "sc25_d": "half-window 25のmulti-scale NCC候補 − anchor。",
    "sc8_sc": "half-window 8 NCC matching score。",
    "sc15_sc": "half-window 15 NCC matching score。",
    "sc25_sc": "half-window 25 NCC matching score。",
    "sc_cons_d": "sc8/sc15/sc25候補の平均 − anchor。",
    "sc_ens_d": "multi-scale NCC ensemble候補 − anchor。",
    "sc_trust": "既知prefix長から作るNCC trust。exp238 train面では定数。",
    "hyb_d": "Beam/NCC hybrid候補 − anchor。",
    "sig_std": "PF・Beam・NCC・formation・dense候補群の行別標準偏差。",
    "sig_mean_d": "PF・Beam・NCC・formation・dense候補群の平均 − anchor。",
    "form_mean_d": "6 formation由来TVT候補の平均 − anchor。",
    "form_std_d": "6 formation由来TVT候補の標準偏差。",
    "form_rng_d": "6 formation由来TVT候補の最大−最小。",
    "spatial_ancc_d": "spatial KNN ANCC surface − anchor位置のtypewell GR値。",
    "spatial_knn_dist": "formation spatial KNNで使う最短正規化XY距離。",
    "dense_ancc": "dense spatial KNNで補完したANCC surface値。",
    "dense_std": "dense spatial KNN近傍ANCCの重み付き標準偏差。",
    "dense_dist": "dense spatial KNNの最短正規化XY距離。",
    "tvt_dense_d": "dense ANCC + full-prefix bias TVT候補 − anchor。",
    "tvt_densew_d": "dense ANCC + prefix指数加重bias TVT候補 − anchor。",
    "tvt_dense50_d": "dense ANCC + prefix末尾50行bias TVT候補 − anchor。",
    "dense_rmse": "既知prefixでのdense ANCC TVT式のRMSE（well定数）。",
    "dense_bias": "既知prefixでのdense ANCC TVT式の平均bias（well定数）。",
    "dense_nb_std": "既知prefixのdense KNN近傍標準偏差平均（well定数）。",
    "pf_vs_spatial": "ANCC PF候補 − ANCC formation-spatial候補。",
    "pf_vs_dense": "ANCC PF候補 − dense ANCC候補。",
    "spatial_vs_dense": "ANCC formation-spatial候補 − dense ANCC候補。",
    "beam_vs_spatial": "conservative Beam候補 − ANCC formation-spatial候補。",
    "sc_vs_beam": "NCC ensemble候補 − conservative Beam候補。",
    "cal_a": "既知prefix GRをtypewell GRへaffine fitした傾き。",
    "cal_b": "既知prefix GRをtypewell GRへaffine fitした切片。",
    "pfx_rmse": "既知prefix GRとTVT対応typewell GRのRMSE。",
    "known_len": "既知TVT_input prefixの行数。",
    "eval_len": "予測tailの行数。",
    "slp_all": "既知prefix全体のTVT/MD robust slope。",
    "slp_50": "既知prefix末尾50行のTVT/MD robust slope。",
    "slp_z": "既知prefixのTVT/Z robust slope。",
    "slp_b_d_all": "全prefix slope外挿TVT − anchor。",
    "slp_b_d_50": "末尾50行 slope外挿TVT − anchor。",
    "ktvt_range": "既知prefix TVT_inputのrange。",
    "ktvt_std": "既知prefix TVT_inputの標準偏差。",
    "md_since": "anchor行からのMD距離。",
    "frac": "予測tail内の0〜1正規化行位置。",
    "frac2": "fracの二乗。",
    "sqrt_frac": "fracの平方根。",
    "z": "予測行のZ座標。",
    "dx": "予測行X − anchor行X。",
    "dy": "予測行Y − anchor行Y。",
    "dz": "予測行Z − anchor行Z。",
    "dxy": "anchorからのXY平面距離。",
    "dzdmd": "行差分 dZ/dMD。",
    "dxdmd": "行差分 dX/dMD。",
    "dydmd": "行差分 dY/dMD。",
    "gr": "水平井の補間済みraw GR。",
    "gr_d1": "raw GRの1階行差分。",
    "gr_d2": "raw GRの2階行差分。",
    "gr_env": "raw GRのcentered rolling-21最大値。",
    "gr_nrg": "raw GR二乗のrolling-21平均平方根。",
    "gr_vs_tw_anc": "raw GR − anchor TVTでのtypewell GR。",
    "gr_vs_slp_all": "raw GR − 全prefix slope外挿TVTでのtypewell GR。",
    "tw_range": "typewell TVT軸のrange（well定数）。",
    "tw_gr_mean": "typewell GR平均（well定数）。",
    "likpf_mean_d": "likelihood-weighted PF平均候補 − anchor。",
}


def _offset_from_name(feature: str, prefix: str) -> int:
    return int(feature[len(prefix) :])


def describe_base(feature: str) -> str:
    if feature in BASE_DESCRIPTIONS:
        return BASE_DESCRIPTIONS[feature]
    match = re.fullmatch(r"beam_(cons|loose|vcons|sm5|vloose|mid|stiff)_d", feature)
    if match:
        return f"{BEAM_LABELS[match.group(1)]}候補 − last_known_tvt。"
    match = re.fullmatch(r"(tvtFw|tvtF50|tvtF|bw_early|bw_mid|bww|bw50|bw)_(.+)", feature)
    if match and match.group(2) in FORMATION_LABELS:
        kind, formation = match.groups()
        label = FORMATION_LABELS[formation]
        meanings = {
            "tvtF": "spatial KNN surfaceとfull-prefix median biasから作るTVT候補",
            "tvtFw": "spatial KNN surfaceとprefix指数加重biasから作るTVT候補",
            "tvtF50": "spatial KNN surfaceとprefix末尾50行biasから作るTVT候補",
            "bw": "full-prefixで推定したTVT+Z−formationのmedian bias",
            "bww": "prefix後半を重くしたTVT+Z−formationの指数加重bias",
            "bw50": "prefix末尾50行のTVT+Z−formation median bias",
            "bw_early": "prefix前1/3のTVT+Z−formation median bias",
            "bw_mid": "prefix中1/3のTVT+Z−formation median bias",
        }
        return f"{label}: {meanings[kind]}。"
    match = re.fullmatch(r"frm_rmse_(.+)", feature)
    if match and match.group(1) in FORMATION_LABELS:
        return f"{FORMATION_LABELS[match.group(1)]} TVT式の既知prefix RMSE。"
    match = re.fullmatch(r"grm(5|21|51|101)", feature)
    if match:
        return f"raw GRのcentered rolling-{match.group(1)}平均。"
    match = re.fullmatch(r"grs(5|21|51|101)", feature)
    if match:
        return f"raw GRのcentered rolling-{match.group(1)}標準偏差。"
    match = re.fullmatch(r"glag(1|5|15|30)", feature)
    if match:
        return f"raw GRを{match.group(1)}行lagした値。"
    match = re.fullmatch(r"glead(1|5|15|30)", feature)
    if match:
        return f"raw GRを{match.group(1)}行leadした値。"
    if re.fullmatch(r"tda-?\d+", feature):
        offset = _offset_from_name(feature, "tda")
        return f"raw GR − typewell GR(anchor TVT {offset:+d} ft)。"
    if re.fullmatch(r"tdbc-?\d+", feature):
        offset = _offset_from_name(feature, "tdbc")
        return f"raw GR − typewell GR(Beam reference TVT {offset:+d} ft)。"
    if re.fullmatch(r"tdsc-?\d+", feature):
        offset = _offset_from_name(feature, "tdsc")
        return f"raw GR − typewell GR(NCC ensemble TVT {offset:+d} ft)。"
    if re.fullmatch(r"tdpf-?\d+", feature):
        offset = _offset_from_name(feature, "tdpf")
        return f"raw GR − typewell GR(ANCC PF TVT {offset:+d} ft)。"
    return "exp072/exp063 full-replay由来のanchor-conditioned特徴。"


def describe_uproj(feature: str) -> str:
    sources = "pf_ancc|pf_z|beam_mean|beam_med|likpf_mean"
    match = re.fullmatch(rf"uproj_({sources})_(corr|resid|abs_resid|resid_mad)", feature)
    if match:
        source, kind = match.groups()
        meanings = {
            "corr": "well内robust polynomial U-trend − source U",
            "resid": "source U − well内robust polynomial U-trend",
            "abs_resid": "上記residualの絶対値",
            "resid_mad": "上記|residual|のwell内中央値",
        }
        return f"{source} U-projection: {meanings[kind]}。"
    match = re.fullmatch(rf"uproj_diff_({sources})_minus_({sources})", feature)
    if match:
        return f"U-spaceの{match.group(1)} − {match.group(2)}。Z/anchor共通項は相殺。"
    match = re.fullmatch(rf"uproj_absdiff_({sources})_({sources})", feature)
    if match:
        return f"U-spaceの|{match.group(1)} − {match.group(2)}|。"
    if feature == "uproj_source_u_std":
        return "5候補のU値の行別標準偏差。"
    if feature == "uproj_source_u_range":
        return "5候補のU値の行別range。"
    if feature == "uproj_corr_std":
        return "5候補のpolynomial correction値の行別標準偏差。"
    if feature == "uproj_corr_range":
        return "5候補のpolynomial correction値の行別range。"
    return "U=TVT+Z−anchor_U上のcandidate correction/disagreement特徴。"


def _candidate_label(name: str) -> str:
    return CANDIDATE_LABELS.get(name, name)


def describe_ll(feature: str) -> str:
    name = feature.removeprefix("ll_")
    match = re.fullmatch(r"learned_prob_(pf_ancc|beam_mean|likpf_mean|sc_ens|hyb)", name)
    if match:
        return f"exp111 classifierが推定した{_candidate_label(match.group(1))}のP(|error|≤10ft)。"
    match = re.fullmatch(r"learned_pred_abs_error_(pf_ancc|beam_mean|likpf_mean|sc_ens|hyb)", name)
    if match:
        return f"exp111 L1 modelが推定した{_candidate_label(match.group(1))}の絶対誤差。"
    match = re.fullmatch(r"multiobs_(score|mae|ncc)_(pf_ancc|beam_mean|likpf_mean|sc_ens|hyb)", name)
    if match:
        metric, candidate = match.groups()
        meanings = {"score": "multi-observation一致score", "mae": "multi-observation GR MAE", "ncc": "multi-observation NCC"}
        return f"{_candidate_label(candidate)}の{meanings[metric]}。"
    match = re.fullmatch(r"candidate_tvt_(pf_ancc|beam_mean|likpf_mean|sc_ens|hyb)_minus_(last_known_tvt|likpf_mean_tvt)", name)
    if match:
        candidate, reference = match.groups()
        return f"元の{_candidate_label(candidate)} TVT − {reference}。learned予測値ではない。"
    if name == "candidate_tvt_std":
        return "exp111の5候補TVTの行別標準偏差。"
    if name == "candidate_tvt_range":
        return "exp111の5候補TVTの行別range。"
    if name.startswith("learned_prob_"):
        return "exp111 within10確率面のtop-k、rank、margin、entropyまたは確率加重TVT要約。"
    if name.startswith("learned_error_"):
        return "exp111 predicted-absolute-error面のtop-k、rank、marginまたは誤差逆重みTVT要約。"
    return "exp111/exp145 learned candidate-likelihood confidence特徴。"


def describe_grwr(feature: str) -> str:
    name = feature.removeprefix("grwr_")
    if name == "gr_missing_rate":
        return "水平井GRのwell内欠損率。"
    if name == "typewell_gr_missing_rate":
        return "typewell GRのwell内欠損率。"
    if name == "known_prefix_rows_log1p":
        return "既知TVT_input prefix行数のlog1p。"
    if name == "known_prefix_fraction":
        return "全horizontal行に占める既知prefix比率。"
    if name == "candidate_tvt_std":
        return "8候補TVTの行別標準偏差。"
    if name == "candidate_tvt_range":
        return "8候補TVTの行別range。"
    fft = {
        "fft_dominant_energy_ratio": "detrend後GR FFTの最大周波数bin energy比率。",
        "fft_dominant_frequency_norm": "detrend後GR FFTのdominant frequency（正規化）。",
        "fft_high_frequency_ratio": "正規化周波数0.35超のGR FFT energy比率。",
        "fft_notch_residual_energy_ratio": "上位3周波数を除いたGR FFT residual energy比率。",
        "fft_rotation_energy_ratio": "正規化周波数0.06〜0.35のrotation-band energy比率。",
    }
    if name in fft:
        return fft[name]
    match = re.fullmatch(r"raw_std_w(033|065|129)", name)
    if match:
        return f"raw GRのlocal rolling標準偏差（window {int(match.group(1))}）。"
    match = re.fullmatch(r"dwt_detail_(energy|absmean|energy_ratio)_w(033|065|129)", name)
    if match:
        metric, window = match.groups()
        meanings = {"energy": "detail二乗平均", "absmean": "detail絶対値平均", "energy_ratio": "detail/(raw-local+detail) energy比"}
        return f"db4 level-3 DWT detailの{meanings[metric]}（window {int(window)}）。"
    match = re.fullmatch(r"raw_minus_(rolling|savgol|dwt)_absmean_w(033|065|129)", name)
    if match:
        filt, window = match.groups()
        return f"|raw GR − {filt} denoised GR|のlocal平均（window {int(window)}）。"
    match = re.fullmatch(r"raw_(rolling|savgol|dwt)_corr_w(033|065|129)", name)
    if match:
        filt, window = match.groups()
        return f"raw GRと{filt} denoised GRのlocal相関（window {int(window)}）。"
    filters = "raw|rolling_median_11|savgol_31_p2|dwt_approx"
    match = re.fullmatch(rf"({filters})_(default_candidate_cost|default_candidate_ncc|candidate_cost_entropy|best_minus_default_cost|best_is_default_candidate|candidate_cost_std|zero_rank_norm|zero_minus_best_cost)", name)
    if match:
        filt, metric = match.groups()
        meanings = {
            "default_candidate_cost": "default likPFのlocal GR observation cost",
            "default_candidate_ncc": "default likPFのlocal GR NCC",
            "candidate_cost_entropy": "候補cost分布のentropy",
            "best_minus_default_cost": "最良候補cost − default候補cost",
            "best_is_default_candidate": "最良cost候補がdefault likPFかのflag",
            "candidate_cost_std": "候補costの標準偏差",
            "zero_rank_norm": "anchor固定候補のcost順位（正規化）",
            "zero_minus_best_cost": "anchor固定候補cost − 全候補最良cost",
        }
        return f"{filt} GR面: {meanings[metric]}。"
    match = re.fullmatch(r"(rolling_median_11|savgol_31_p2|dwt_approx)_minus_raw_default_candidate_(cost|ncc)", name)
    if match:
        return f"{match.group(1)}面とraw面のdefault likPF {match.group(2)}差。"
    interactions = {
        "dwt_energy_ratio_w065_x_candidate_std": "DWT detail energy比(w65) × 8候補TVT標準偏差。",
        "raw_std_w065_x_log1p_md_since": "raw GR local std(w65) × log1p(md_since)。",
        "fft_rotation_ratio_x_log1p_md_since": "FFT rotation-band energy比 × log1p(md_since)。",
        "fft_rotation_ratio_x_candidate_range": "FFT rotation-band energy比 × 8候補TVT range。",
        "dwt_minus_raw_ncc_gap_x_candidate_range": "DWT-vs-raw default NCC差 × 候補TVT range。",
        "dwt_minus_raw_ncc_gap_x_dwt_energy_ratio_w065": "DWT-vs-raw default NCC差 × DWT detail energy比(w65)。",
        "ll_entropy_x_dwt_energy_ratio_w065": "learned-likelihood entropy × DWT detail energy比(w65)。",
    }
    if name in interactions:
        return interactions[name]
    return "target-free GR denoise・wavelet・FFT・candidate observation consistency特徴。"


def describe_nsel(feature: str) -> str:
    name = feature.removeprefix("nsel_")
    direct = {
        "top1_code": "予測絶対誤差が最小の候補index（数値code）。",
        "top2_code": "予測絶対誤差が2番目に小さい候補index（数値code）。",
        "top1_minus_anchor": "selector top1候補TVT − last_known_tvt。",
        "top2_minus_anchor": "selector top2候補TVT − last_known_tvt。",
        "top2_minus_top1": "selector top2候補TVT − top1候補TVT。",
        "error_top1": "selectorが予測したtop1候補の絶対誤差。",
        "error_top2": "selectorが予測したtop2候補の絶対誤差。",
        "error_margin": "predicted error top2 − top1。",
        "error_ratio": "predicted error top1 / max(top2, 1e-3)。",
        "score_mean": "11候補predicted errorの行別平均。",
        "score_std": "11候補predicted errorの行別標準偏差。",
        "candidate_std": "11候補TVTの行別標準偏差。",
        "candidate_range": "11候補TVTの行別range。",
    }
    if name in direct:
        return direct[name]
    if name.startswith("top1_is_"):
        candidate = name.removeprefix("top1_is_")
        return f"selector top1が{_candidate_label(candidate)}であるone-hot flag。"
    if name.startswith("pred_error_"):
        candidate = name.removeprefix("pred_error_")
        return f"nested selectorが予測した{_candidate_label(candidate)}の絶対誤差。"
    return "strict outer/inner nested selectorのrank-slot confidence特徴。"


def describe_feature(feature: str) -> str:
    family = feature_family(feature)
    if family == "base_replay":
        return describe_base(feature)
    if family == "u_projection":
        return describe_uproj(feature)
    if family == "learned_likelihood":
        return describe_ll(feature)
    if family == "gr_wavelet_rotation":
        return describe_grwr(feature)
    return describe_nsel(feature)


def read_importance(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1245:
        raise ValueError(f"expected 3 x 415 importance rows, got {len(rows)}")
    return rows


def importance_catalog(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    models = sorted({row["model"] for row in rows})
    by_model: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_model[row["model"]][row["feature"]] = float(row["importance"])
    features = sorted({row["feature"] for row in rows})
    if len(features) != 415 or models != ["lgb0", "lgb1", "lgb2"]:
        raise ValueError({"features": len(features), "models": models})
    totals = {model: sum(by_model[model].values()) for model in models}
    model_ranks: dict[str, dict[str, int]] = {}
    for model in models:
        ordered = sorted(by_model[model], key=lambda f: (-by_model[model][f], f))
        model_ranks[model] = {feature: index + 1 for index, feature in enumerate(ordered)}
    catalog: list[dict[str, object]] = []
    for feature in features:
        share = sum(by_model[model][feature] / totals[model] for model in models) / len(models)
        mean_rank = sum(model_ranks[model][feature] for model in models) / len(models)
        catalog.append(
            {
                "feature": feature,
                "family": feature_family(feature),
                "importance_share_pct": 100.0 * share,
                "mean_model_rank": mean_rank,
                **{f"{model}_split": by_model[model][feature] for model in models},
                "all_models_zero": all(by_model[model][feature] == 0.0 for model in models),
                "description": describe_feature(feature),
            }
        )
    catalog.sort(
        key=lambda row: (
            -float(row["importance_share_pct"]),
            float(row["mean_model_rank"]),
            str(row["feature"]),
        )
    )
    for rank, row in enumerate(catalog, 1):
        row["rank"] = rank
    return catalog


def duplicate_evidence() -> dict[str, dict[str, str]]:
    with DROP_AUDIT.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        output[row["feature"]] = {
            "bucket": row["bucket"],
            "relation": row["relation"],
            "keep_feature": row["keep_feature"],
            "pair_corr": row["pair_corr"],
            "recommended_action": row["recommended_action"],
        }
    return output


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def format_candidate_table(
    rows: list[dict[str, str]], selection_rows: list[dict[str, str]]
) -> list[str]:
    recommendation = {
        "v6_k16_geometry_gr_u_projection": "core keep",
        "hmm_selfgr_boost_only_a070_c100": "core keep",
        "likpf_mean": "core fallback",
        "pf_ancc": "core diversity",
        "beam_mean": "reserve keep; do not add Beam variants",
        "tvt_dense": "keep one broad dense path",
        "tvt_dense50": "preferred late-bias dense path",
        "tvt_densew": "family-exclusion audit; corr 0.99935 with dense50",
        "blend_likpf_hmm_w500": "cheap bridge; audit because it is a deterministic blend",
        "sc_ens": "exclude first; weak and redundant with hyb",
        "hyb": "exclude with sc_ens unless an outer-safe subgroup proves value",
    }
    lines = [
        "| candidate | 内容 | RMSE | unique-best | exp237 OOF選択率 | 推奨 |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    rates: dict[str, float] = {}
    for row in selection_rows:
        if row["mode"] == "oof":
            rates[row["selected_candidate"]] = float(row["rate"])
    for row in sorted(rows, key=lambda item: float(item["rmse_tvt"])):
        name = row["candidate"]
        rate = rates.get(name)
        rate_text = "-" if rate is None else f"{100.0 * rate:.2f}%"
        lines.append(
            f"| `{name}` | {_candidate_label(name)} | {float(row['rmse_tvt']):.3f} | "
            f"{100.0 * float(row['unique_best_rate']):.2f}% | {rate_text} | "
            f"{recommendation[name]} |"
        )
    return lines


def write_catalog_csv(catalog: list[dict[str, object]], duplicates: dict[str, dict[str, str]]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "exp238_feature_catalog_importance.csv"
    fieldnames = [
        "rank",
        "feature",
        "family",
        "importance_share_pct",
        "mean_model_rank",
        "lgb0_split",
        "lgb1_split",
        "lgb2_split",
        "all_models_zero",
        "description",
        "duplicate_bucket",
        "duplicate_relation",
        "keep_feature",
        "pair_corr",
        "recommended_action",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in catalog:
            evidence = duplicates.get(str(row["feature"]), {})
            writer.writerow(
                {
                    **row,
                    "duplicate_bucket": evidence.get("bucket", ""),
                    "duplicate_relation": evidence.get("relation", ""),
                    "keep_feature": evidence.get("keep_feature", ""),
                    "pair_corr": evidence.get("pair_corr", ""),
                    "recommended_action": evidence.get("recommended_action", ""),
                }
            )
    return path


def write_report(
    catalog: list[dict[str, object]],
    duplicates: dict[str, dict[str, str]],
    candidates: list[dict[str, str]],
    residual_corr: list[dict[str, str]],
    selector_importance: list[dict[str, str]],
    selector_schema: list[dict[str, str]],
    selection_distribution: list[dict[str, str]],
    beam_marginal: list[dict[str, str]],
) -> None:
    family_counts = Counter(str(row["family"]) for row in catalog)
    family_share: dict[str, float] = defaultdict(float)
    for row in catalog:
        family_share[str(row["family"])] += float(row["importance_share_pct"])
    all_zero = [str(row["feature"]) for row in catalog if bool(row["all_models_zero"])]
    high_candidate_corr = sorted(
        residual_corr,
        key=lambda row: -abs(float(row["residual_correlation"])),
    )[:10]
    selector_top = sorted(
        selector_importance,
        key=lambda row: (-float(row["mean_importance"]), row["feature"]),
    )[:12]
    selector_ranked = sorted(
        selector_importance,
        key=lambda row: (-float(row["mean_importance"]), row["feature"]),
    )
    selector_by_feature = {
        row["feature"]: (rank, float(row["mean_importance"]))
        for rank, row in enumerate(selector_ranked, 1)
    }
    beam_by_surface = {row["surface"]: row for row in beam_marginal}

    lines: list[str] = [
        "# exp238 selector / TVT feature audit",
        "",
        "作成日: 2026-07-16",
        "",
        "## 結論",
        "",
        "- selector候補bankは11本をそのまま増やさず、`exp226 K16`、`self-GR HMM`、`likpf_mean`、`pf_ancc`をcoreにする。denseは`dense50`を優先し、`densew`はfamily-exclusion audit対象。`sc_ens`と`hyb`は最初の除外候補。",
        "- `beam_mean`は単体RMSE 15.774で直線的なpathが多いが、exp237 selectorが選んだ行ではlikPFより良い。削除ではなく1本だけreserveに残し、Beam variant追加はしない。",
        "- `likpf_mean`は単純算術平均ではなくseed likelihood-weighted mean。非平均化したexp243 K8 medoidはdirect replacementには弱いが、base8 unionのwhole-well oracleを6.5924から5.4996へ改善するため、優先する実験候補へ引き上げる。",
        "- self-GRは既に検討済み。exp091/093のraw self-GR 5本はoracle headroomがあるがdirect pathと旧scorerは失敗。exp223の弱いSelf-GR likelihoodを加えたHMMは改善し、exp237の現行11候補bankに入っている。",
        "- 過去候補の再監査では、`last_anchor_tvt`、exp221 HMM+LGB、exp103 `xy_likpf_scale_12`に未回収の可能性がある。`recent_linear`はnear専用、raw self-GRとheatmap path生成はdirect候補としてclosed/rejectedのままとする。",
        "- selectorはhard TVTを主出力にせず、outer-fold別の候補誤差分布とrank-slot confidenceを出す。direct pathはdiagnosticに限定する。",
        "- TVT LightGBMは`TVT-last_known_tvt`残差を維持し、selectorはadd-onlyにする。exp257のreplacement-onlyはsame-fold exp238よりRMSE +0.164641で悪化した。",
        "- exp238の415列には、exp148 lineageから継承したhigh-confidence重複17列が残る。さらにformation末尾重視12列は相関0.999990〜0.999993。最初はexact 17だけのdrop ablation、その後にformation 12を別ablationにする。",
        "- 415列全体のtrain-row相関matrixは未保存。既存の数値相関は前半294列の600,000-row auditと、11 candidate pathの全3,783,989-row residual相関。GRWR 86列とnsel 35列は生成式上の従属関係まで監査したが、全組合せの実測相関は別のno-training readoutが必要。",
        "",
        "## Evidence boundary",
        "",
        "- exp238 final model: 380 base + 35 nested selector = 415 features、3 LightGBM configs × 5 folds。OOF `lgb_mean` 7.936690、Public LB 7.775。",
        "- 現行11候補rankerの参照はexp237。raw-test-safe化した最新exp251の295列 expected-error fixed Viterbiは8.502212で、overall、1000+、worst-well guardが不通過。採用済みhard selectorはまだない。",
        "- importanceはLightGBM `feature_importances_`のsplit回数でありgainではない。各configの総split数で正規化してから3 config平均した値で順位付けした。",
        "- exp238のhistorical exp218との差はouter-fold assignmentが一致しないため、selector特徴だけの因果差ではない。",
        "",
        "## Candidate paths",
        "",
        *format_candidate_table(candidates, selection_distribution),
        "",
        "主な候補残差相関:",
        "",
        "| left | right | residual correlation |",
        "| --- | --- | ---: |",
    ]
    for row in high_candidate_corr:
        lines.append(
            f"| `{row['left_candidate']}` | `{row['right_candidate']}` | "
            f"{float(row['residual_correlation']):.6f} |"
        )
    lines.extend(
        [
            "",
            "解釈:",
            "",
            "- `tvt_densew` / `tvt_dense50`は0.999350で、同時にselectableにする価値が小さい。",
            "- `sc_ens` / `hyb`は0.997180、Viterbi選択率もほぼ0なので最初のcandidate exclusion候補。",
            "- `blend_likpf_hmm_w500`はlikPFとHMMの決定的50/50平均で、残差相関も各親と0.86台。新しいgeneratorではなくcheap bridgeとして扱う。",
            "",
            "### `beam_mean`は候補に残すか",
            "",
            "exp237 OOF 3,783,989行を再集計した。見た目の直線性だけでは除外せず、oracle contributionと実際にselectorが選んだ領域のcounterfactualを優先する。",
            "",
            "| surface | Beam rows | rate | Beam RMSE on rows | likPF RMSE on same rows | Beam beats likPF | all-row RMSE | BeamをlikPFへ強制置換 | delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *[
                f"| {surface} | {int(beam_by_surface[surface]['beam_rows']):,} | "
                f"{100 * float(beam_by_surface[surface]['beam_rate']):.2f}% | "
                f"{float(beam_by_surface[surface]['beam_rmse_on_rows']):.4f} | "
                f"{float(beam_by_surface[surface]['likpf_rmse_on_rows']):.4f} | "
                f"{100 * float(beam_by_surface[surface]['beam_beats_likpf_rate']):.2f}% | "
                f"{float(beam_by_surface[surface]['full_rmse']):.6f} | "
                f"{float(beam_by_surface[surface]['forced_likpf_full_rmse']):.5f} | "
                f"+{float(beam_by_surface[surface]['forced_likpf_delta']):.5f} |"
                for surface in ["oracle", "rowwise", "viterbi"]
            ],
            "",
            "`rowwise`のoracle-Beam recallは4.43%、Viterbiは3.95%に留まる一方、Beamを選んだ行のoracle precisionは17.62% / 18.87%。つまりBeam自体が不要なのではなく、Beamが勝つ領域の識別が弱い。強制likPF置換は、selector再学習や次善候補を使わないため削除損失の上限寄りの診断である。判断は**1本だけreserveに残す、Beam top-K/posterior variantは増やさない**。exp173/177のBeam top-K posterior・gap/entropy gateはnegativeでclosedのままとする。",
            "Beam残差相関はself-GR HMM 0.4134、blend 0.4728、likPF 0.4876、exp226 0.5481、pfANCC 0.5672で、current coreに対して一定の多様性もある。",
            "",
            "### PFを単純平均しない候補",
            "",
            "`likpf_mean`は500 particles × 128 seedsの各seed予測をseed likelihoodで重み付けした平均で、元から単純算術平均ではない。exp243はさらに平均前の実在seed trajectoryをcluster medoidとして保持した。",
            "",
            "- 最良direct medoidはRMSE 12.296667で`likpf_mean` 11.594898より+0.701770。直接置換は不採用。",
            "- base8 + K8 medoid oracleはrow 4.564605→3.216218、block128 4.805040→3.399936、whole-well 6.592426→5.499587。K8 unique-bestは43.88%、374/773 wellsを改善。",
            "- K3/K5/K8全部を入れてもK8単独からwhole-well -0.006406だけなので、候補化するならK8だけ。",
            "- exp252ではK8内の`cluster_likelihood_mass` / likelihood rank / gapがwhole-well AUC 0.6752 / 0.6551 / 0.6542。ただしbank gateは最良でも0.5606、固定top1はbest base8比+3.1949 ft。",
            "- raw生成はexp243で約10時間18分/773 wells、hidden約200 wellsの単純比例は約2時間40分。したがってbase8 fallback付きのfold-safe selectorへ**高優先の実験候補**として追加するが、current coreやdirect pathにはまだしない。",
            "",
            "### 過去実験から再検討する候補",
            "",
            "| 候補 | 根拠 | 判断 |",
            "| --- | --- | --- |",
            "| `last_anchor_tvt` | RMSE 15.910だがexp093 oracle best 370,631行（9.79%）。raw-test-safeで非常に安い | **高**。まず現行pruned bankへのrow/block/well oracle追加量とnear選択を監査 |",
            "| `recent_linear` / prefix末尾slope | exp019の0–49行で0.7966、50–249行で3.6155。exp238の`slp_b_d_50`は重要度2位 | **中**のnear専用expert。exp001 full OOF 41.022のため、global候補ではなく250 ft以内・fallback必須 |",
            "| exp221 HMM+LGB | OOF 8.327737、exp148比-0.173554、全距離/hidden-like改善。Public LB 7.953で現行ML anchor未満 | **高**のno-training oracle/残差相関audit。exp148 lineageとの高相関とfold alignmentを確認してから候補化 |",
            "| exp082 fle3n final ensemble | Public LB 7.601でroute anchorだが、public/pretrained branchを含むfinal blend | aligned outer-fold OOFと候補固有confidence契約がないため、selector候補にはしない。LB anchorとしてのみ保持 |",
            "| exp243 K8 PF medoids | row/block/wellすべて大きいoracle headroom、K8内confidence signalあり | **高**の実験候補。高コストかつbank gate未成立なのでbase8 fallback必須 |",
            "| exp091/093 raw self-GR (`self_gr_sc8/15/25`, `best`, `ens`) | baseline oracle 7.4340→6.9589、within10 0.9065→0.9225。`sc25`はoracle best 175,030行 | **direct再追加はしない**。best単体250.162、ens 191.216、旧rankerはsc25/best/sc15を0行選択。Self-GRのscore/gap/qualityだけをconfidenceへ使う |",
            "| exp223 Self-GR likelihood HMM | likPF 11.59490→11.34995。全距離/hidden-like改善だがworst well +46.95 | **現行core**。raw self-GR pathとは別物で、exp237の`hmm_selfgr_boost_only_a070_c100`として既に候補化済み |",
            "| exp225 state-known TVT Self-GR HMM | RMSE 14.21295、likPF比+2.618、long-tail/hidden/worst-wellが悪化 | **再候補化しない**。state-known emission設計はnegative |",
            "| exp103 `xy_likpf_scale_12` | 単体13.916。`likPF+pf_z` oracle 9.1152へ追加すると7.8084（-1.3068） | **中**。seed std、既存PFとの差、smoothnessを同伴し、K8との重複を先に確認 |",
            "| exp106 `pf_z_ms_scale_3` | 単体16.146、3候補oracle 8.1237。exp103より弱いがstrict parity | **低-中**。exp103またはK8とfamily-exclusionし、同時追加しない |",
            "| exp142 trajectory-aware PF | global 23.132で不採用だが0–50/50–100/100–250 ftは0.551/1.267/2.630でlikPFより良い | **低**のnear-only。`last_anchor`/recent slopeの方が安く安全なので後順位 |",
            "| exp202–215 heatmap topK/path | union oracleは大きいが生成path単体32–50 ft級、selector/feature follow-upも親を更新せず | ユーザー判断の**closed/rejectedを維持**。path候補として再開しない |",
            "",
            "その他、exp128/134のSelf-GR hard switch/gate、Beam top-K（exp173/177）、MAP/dominant HMM（exp236）、adaptive/robust PF（exp232/233/241/242）、quantile/DTW/atlas HMM（exp229–231）、alt typewell path（exp187）はnegativeまたはclosed。exp129 spatial priorとexp176 typewell late-rangeはpath置換ではなくconfidence/context featureとしてのみpositiveで、current candidate pathにはしない。新規案の`exp218_centered_residual_diverse_hmm`は過去候補ではないため、保存OOFで小Kのoracle/diversity auditを先に行う。",
            "",
            "## Selector features",
            "",
            "推奨入力はcandidate-long形式にする。row contextとcandidate-specific featureを分離し、candidate indexを連続量として扱わない。",
            "",
            f"exp237は{len(selector_schema)}列のcontext schemaからcandidate-long展開後{len(selector_importance)}列を学習に使用した。split importance上位は次のとおり。`candidate_index`が2位なのは候補固有biasを拾う一方、任意の候補順序に閾値構造を入れるため、次版ではcategorical/one-hotへ置き換える。",
            "",
            "| rank | exp237 selector feature | mean split importance |",
            "| ---: | --- | ---: |",
            *[
                f"| {rank} | `{row['feature']}` | {float(row['mean_importance']):.1f} |"
                for rank, row in enumerate(selector_top, 1)
            ],
            "",
            "全320列の説明・順位・mean±std split importanceと、exp251 v4の全295列×2目的の重要度は[`selector_feature_catalog_20260716.md`](selector_feature_catalog_20260716.md)に分離した。exp237/251の共通282列におけるexpected-error重要度順位相関は0.952、exp251内のexpected-error/within10順位相関は0.981。高相関でも2目的のrowwise RMSE優劣はversion間で反転しており、同じ出力とは扱わない。",
            "",
            "### 候補パスの信頼度を入力できるか",
            "",
            "できる。HMMのσに相当する`hmm_exact_std`と`hmm_selfgr_std`はexp237/251ですでに入力済みで、split importanceも上位だった。",
            "現状はrow contextとして全candidate行へ反復される。次版では、対応するHMM candidate行の`candidate_sigma_tvt`へ写し、`candidate_has_sigma`を付ける方がcandidate固有confidenceとして明確になる。元のglobal HMM stdも、他familyからHMMへ切り替える判断用contextとして残す。",
            "",
            "| signal | exp237 importance rank | mean split importance | 証拠と扱い |",
            "| --- | ---: | ---: | --- |",
            f"| `hmm_exact_std` | {selector_by_feature['hmm_exact_std'][0]} | {selector_by_feature['hmm_exact_std'][1]:.1f} | exp205でHMM absolute errorとの相関0.3995。粗いrisk signal |",
            f"| `hmm_selfgr_std` | {selector_by_feature['hmm_selfgr_std'][0]} | {selector_by_feature['hmm_selfgr_std'][1]:.1f} | self-GR HMMのposterior TVT幅 |",
            f"| `hmm_exact_loglik` | {selector_by_feature['hmm_exact_loglik'][0]} | {selector_by_feature['hmm_exact_loglik'][1]:.1f} | 行数で正規化して使う |",
            f"| `hmm_selfgr_loglik` | {selector_by_feature['hmm_selfgr_loglik'][0]} | {selector_by_feature['hmm_selfgr_loglik'][1]:.1f} | stdと別のfit quality |",
            f"| `pf_ancc_std` | {selector_by_feature['pf_ancc_std'][0]} | {selector_by_feature['pf_ancc_std'][1]:.1f} | PF粒子の行別spread。単独では弱い |",
            f"| `crfe_dense_candidate_std` | {selector_by_feature['crfe_dense_candidate_std'][0]} | {selector_by_feature['crfe_dense_candidate_std'][1]:.1f} | dense候補集合のspread |",
            "",
            "ただしσをそのまま「小さいほど正しい」とは扱わない。exp221ではposterior std最低binのRMSEが8.986、中央binは7.66–7.80、最高binは9.997で非単調だった。exp223もlow-std bin RMSE 9.365を記録している。",
            "",
            "#### 「候補別・outer-fold内で予測誤差へ校正」の正確な意味",
            "",
            "ユーザーの理解どおり、**現行exp237/251ではσはselectorの入力特徴量の1つ**である。別のσ専用calibratorを直列に置いた、という意味ではない。selector LightGBMが`candidate identity + σ + loglik + 距離 + 候補間差 + その他context`から`|candidate_tvt-true_tvt|`または`P(error≤10)`を学ぶため、これは多変量モデル内での暗黙のcalibrationである。前の「校正する」という表現は別モデルを示すように読めて曖昧だった。",
            "",
            "outer-fold内とは、outer-valid wellの正解をσ→error対応の学習に使わないというleakage guardを指す。outer fold fについて、outer-train側はinner OOF selector score、outer-valid側はouter-train wellsだけで学習したselector scoreを作る。exp238はouter 5 × inner 4 = 20モデルでこの契約を実装した。単変量の`σ→expected error`曲線を別途fitする案はdiagnosticにはなるが、σと誤差が非単調なので現時点の主案ではない。",
            "",
            "candidate-longの共通confidence schemaは次を推奨する。family固有のraw値を保持し、該当しないfamilyは0埋めせず`confidence_valid`とmissing indicatorを付ける。",
            "",
            "- 共通: `candidate_family` one-hot、`confidence_valid`、`confidence_source`、`sigma_tvt`、`loglik_per_row`、`score_margin`、`entropy`、`support_count`、`candidate_tvt-anchor`、candidate間distance/disagreement。",
            "- HMM: posterior TVT std、GR observationの`hmm_prefix_sigma`、loglik/row、posterior entropy、top1/top2 mass gap、grid-edge mass、bimodal/mean-in-valley flag。固定hyperparameterのLGB emission σはconfidenceにしない。",
            "- PF/K8: particle std、ESS fraction、resampling/collapse rate、seed prediction std、seed likelihood dispersion、cluster mass、likelihood mass/rank/gap、assignment distance、cluster entropy。",
            "- Beam: retained top-K posterior/gap gateはexp173/177でnegativeなので再利用しない。保持する`beam_mean`にはlikPF/HMM/exp226とのdisagreement、local slope/curvature、直線度、anchorからのdriftだけを使う。",
            "- dense/spatial/exp226: neighbor weighted std、distance、neighbor count、coverage/fallback、geometry gap、GR delta、donor agreement。",
            "- 変換: `log1p(sigma)`、well内percentile、local rolling p50/p90/max、σの変化量、候補間σ比、`sigma × disagreement`。すべてouter-train fit、outer-valid applyにする。",
            "",
            "候補ごとに同じ意味のσが必ず存在するわけではない。そのため、raw confidence proxyはfamily別に計算し、最終的な共通尺度をselectorの`pred_abs_error` / `p_within10`にする。",
            "",
            "| candidate family | target-free raw confidence | 現状 |",
            "| --- | --- | --- |",
            "| `pf_ancc` | particle TVT std、ESS、resampling/collapse率、likelihood entropy/margin、multiobs score | `pf_ancc_std`とmultiobsは入力済み。ESS等は追加可能 |",
            "| `likpf_mean` | seed予測のlikelihood加重std、seed weight entropy/max、effective seed数、PF ESS、multiobs | std/entropyの完全なcandidate固有化は未実装。再生成可能 |",
            "| `beam_mean` | Beam間spread、直線度、local slope/curvature、boundary/clip率、他core候補とのdisagreement | disagreementは入力済み。exp173/177のtop-K posterior gateはnegativeなので主信頼度にしない |",
            "| `sc_ens` / `hyb` | scale別NCC、1位−2位gap、scale間TVT spread、coverage/trust、Beamとのagreement | multiobs/NCCとSelf-GR系scoreを転用可。ただしpath自体は除外優先 |",
            "| dense 3本 | neighbor weighted std、距離、count/coverage、prefix bias RMSE、3本のspread | `dense_std/dist`、dense pair差、CRFEは入力済み |",
            "| exact / Self-GR HMM | posterior std、loglik/row、entropy、top-mass gap、grid-edge mass、prefix σ、Self-GR quality/valid | std/loglik/Self-GR qualityは入力済み。entropy/edge massは追加可能 |",
            "| likPF-HMM blend | 両親のconfidence、両親のTVT差、blend位置、親ごとの予測誤差 | 固有σはない。親confidenceとdisagreementから作る |",
            "| exp226 geometry | donor/neighbor support、KNN距離、geometry projection gap、GR delta、condition、family disagreement | gapは入力済み。`exp226_gr_delta/geop_tvt`はexp251 v4 raw-test契約外なのでparity実装後のみ復帰 |",
            "| K8 PF medoids | cluster likelihood mass/rank/gap、assignment distance、cluster entropy、seed std、ESS | exp252でmass/rank/gapのwhole-well AUC 0.675/0.655/0.654。候補bank全体のgateは未成立 |",
            "| `last_anchor` / recent slope | prefix slope fit残差、窓間slope分散、外挿距離、曲率/step、anchor quality | 安価に算出可能。near専用expertとして監査前 |",
            "| xy-likPF / trajectory PF | seed std、XY slope fit残差/condition、PF stats、core候補との差 | 過去artifactから追加可能。K8とのfamily重複を先に監査 |",
            "| raw self-GR path | NCC peak gap、scale間spread、match coverage、Self-GR/typewell agreement | exp091/093から算出可能。ただしraw pathは弱いのでconfidence featureだけを再利用 |",
            "| exp221 HMM+LGB | HMM posterior診断、LGB fold/ensemble spread、quantile幅、HMM centerとの差 | OOF artifactで作成可能。candidate化前にfold alignmentが必要 |",
            "",
            "- Row context: `md_since`、tail位置、`eval_len`、anchor geometry、GR missing/prefix coverage、DWT/FFT、candidate spread。",
            "- Candidate-specific: `candidate_tvt-anchor`、family、上記confidence、各observation likelihood、exp226 geometry gap、candidate-vs-family disagreement。",
            "- Safety: outer-train wellsだけで作るwell/segment risk、near-row flag、fallback候補、raw-test parity flag。",
            "- 避けるもの: target由来gate、same-fold true error、ordinal `candidate_index` / `top1_code`の連続値扱い、全欠損列の黙示0補完。",
            "",
            "## Selector output format",
            "",
            "### exp238の35 rank-slotとexp251のdual outputはどちらがよいか",
            "",
            "両者は同じ層ではないため、排他的に選ばない。**canonical selector出力はexp251形式**、**現行TVTモデルへ渡すadapterはexp238形式**がよい。",
            "",
            "| 形式 | 情報 | 長所 | 短所 / evidence | 判断 |",
            "| --- | --- | --- | --- | --- |",
            "| exp238 35 rank-slot | 11候補の予測誤差、top1/top2候補値・誤差・margin、one-hot、spread | exp238 add-only OOF 7.936690、Public LB 7.775。TVTモデルがsoftに再利用できる | `p_within10`を持たず、一部summaryは決定的重複。historical exp218とはfold不一致 | **現行downstream adapterとして維持** |",
            "| exp251 dual raw output | 各候補の`p_within10`と`pred_abs_error` | 情報を落とさず、rowwise/Viterbi/TVT adapterへ派生可能。確率校正も監査できる | v4 fixed Viterbi 8.502212でguard不通過。v2/v4でどちらの目的が良いか反転 | **canonical出力契約として採用、hard selectorは不採用** |",
            "| exp257 replacement-only | selector出力を既存29 slotへ圧縮し、`nsel_*`を追加しない | schemaを380列に維持 | same-fold exp238比+0.164641悪化 | **不採用** |",
            "",
            "数値も一方的ではない。exp251 v2ではprobability rowwise 8.682860、expected-error rowwise 8.464866、expected-error Viterbi 8.402086。v4では8.479603 / 8.548425 / 8.502212となり、probabilityとexpected-errorの優劣が反転した。よって片方を捨てず、各候補2 scoreを保存する。ただし現時点でexp251 scoreをexp238 TVTモデルへadd-onlyしたsame-fold比較はなく、exp238の35列を直ちに置換する根拠はない。",
            "",
            "正規出力は2層にする。",
            "",
            "1. canonical long artifact: `id, well, row, outer_fold, role, candidate, candidate_family, candidate_tvt, confidence_valid, pred_abs_error, p_within10`。`pred_sq_error`や`p_abs_gt25`は将来objectiveとして検証できた場合だけ追加する。raw confidence列は同じschema manifestに持つ。",
            "2. downstream wide artifact: `pred_error__<candidate>`、`p_within10__<candidate>`と、top1/top2 delta、margin、ratio、score mean/std、candidate spread、top1 one-hotからなるfold別rank-slot features。現行exp238は予測誤差側35列だけなので、確率側はsame-fold add-only ablation後に採否を決める。",
            "",
            "推論ではouter foldごとに4 inner selectorを平均し、そのouter fold用35列を同じouter foldのTVT LightGBMへ渡す。model/schema/SHA manifest、candidate order、missing/nonfinite countsを必須にし、public-test row artifactは入力にしない。hard top1 TVTは監査列にだけ残す。exp255ではassertive bounded correctionがglobal -0.058700でもworst-well +3.151245でguard不通過だった。",
            "",
            "## TVT prediction model features",
            "",
            "推奨順序:",
            "",
            "1. exp238のanchor residual targetとadd-only selector構成をbaselineに固定。",
            "2. high-confidence exact duplicate 17列だけをdropしたsame-fold ablation。exp198ではexp148 CVを-0.043358改善したがPublic LB 7.930で、exp238への転移は未検証。",
            "3. formation `bw50/tvtF50` 12列を別ablationでdrop。weighted側を残す。",
            "4. nselはreplacementしない。まずordinal codeとdeterministic summariesを相関監査し、slim化は別ablation。",
            "5. GRWRはall-zero列とfilter/window間高相関をno-training auditしてからgroup単位で削る。importanceだけで一括削除しない。",
            "",
            "## Importance family summary",
            "",
            "| family | features | normalized split share |",
            "| --- | ---: | ---: |",
        ]
    )
    for family, share in sorted(family_share.items(), key=lambda item: -item[1]):
        lines.append(f"| `{family}` | {family_counts[family]} | {share:.3f}% |")
    lines.extend(
        [
            "",
            "### Top 40",
            "",
            "| rank | feature | family | share | mean rank | lgb0/lgb1/lgb2 split | 説明 |",
            "| ---: | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in catalog[:40]:
        lines.append(
            f"| {row['rank']} | `{row['feature']}` | `{row['family']}` | "
            f"{float(row['importance_share_pct']):.4f}% | {float(row['mean_model_rank']):.1f} | "
            f"{float(row['lgb0_split']):.1f}/{float(row['lgb1_split']):.1f}/{float(row['lgb2_split']):.1f} | "
            f"{row['description']} |"
        )
    lines.extend(
        [
            "",
            "14列は3 configすべてsplit importance 0: " + ", ".join(f"`{name}`" for name in sorted(all_zero)) + "。",
            "",
            "## Duplicate and correlation audit",
            "",
            "### High-confidence exact / functional duplicates inherited into exp238",
            "",
            "| drop candidate | keep | relation | correlation |",
            "| --- | --- | --- | ---: |",
        ]
    )
    exact_rows = [
        (feature, evidence)
        for feature, evidence in duplicates.items()
        if evidence["bucket"] == "exact_prune_17"
    ]
    for feature, evidence in exact_rows:
        corr = evidence["pair_corr"] or ("constant" if "constant" in evidence["relation"] else "-")
        keep = f"`{evidence['keep_feature']}`" if evidence["keep_feature"] else "-"
        lines.append(f"| `{feature}` | {keep} | `{evidence['relation']}` | {corr} |")
    lines.extend(
        [
            "",
            "### Near duplicates / high correlations from the 600k-row exp148-lineage audit",
            "",
            "- formation weighted vs last50 12 pairs: |r| 0.999990〜0.999993。",
            "- `md_since` / `dxy`: 0.999997950。意味は異なるため即dropではない。",
            "- `gr_nrg` / `grm21`: 0.999921773。",
            "- `tvt_dense50_d` / `tvt_densew_d`: 0.999539184。",
            "- `form_rng_d` / `form_std_d`: 0.996395546。",
            "- `ll_candidate_tvt_std` / `ll_candidate_tvt_range`: 0.999463033。",
            "- `ll_candidate_tvt_sc_ens_minus_likpf_mean_tvt` / `ll_candidate_tvt_hyb_minus_likpf_mean_tvt`: 0.997646402。",
            "- `uproj_beam_mean_resid_mad` / `uproj_beam_med_resid_mad`: 0.979712902。",
            "- `uproj_source_u_std` / `uproj_source_u_range`: 0.986839609。",
            "- `uproj_corr_std` / `uproj_corr_range`: 0.985400052。",
            "",
            "nselの生成式上の従属関係:",
            "",
            "- `top2_minus_top1 = top2_minus_anchor - top1_minus_anchor`。",
            "- `error_margin = error_top2 - error_top1`、`error_ratio = error_top1 / max(error_top2,1e-3)`。",
            "- 11個の`top1_is_*`の和は1、`top1_code`はone-hotの線形結合。",
            "- `score_mean/std`は11個の`pred_error_*`から、`candidate_std/range`は11 candidate TVTから決定される。",
            "",
            "これらは木モデルで即バグではないが、importanceを分散し、ordinal codeに不自然な閾値順序を与える。slim化は`top1/top2 delta + per-candidate error + one-hot`をcoreとし、code/派生summaryを別ablationで落とすのが安全。",
            "",
            "## All 415 features in normalized importance order",
            "",
            "| rank | feature | family | share | mean rank | lgb0/lgb1/lgb2 split | duplicate note | 説明 |",
            "| ---: | --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in catalog:
        evidence = duplicates.get(str(row["feature"]), {})
        duplicate_note = ""
        if evidence:
            partner = evidence.get("keep_feature", "")
            duplicate_note = evidence["relation"]
            if partner:
                duplicate_note += f" → keep `{partner}`"
            if evidence.get("pair_corr"):
                duplicate_note += f" (r={evidence['pair_corr']})"
        if bool(row["all_models_zero"]):
            duplicate_note = (duplicate_note + "; " if duplicate_note else "") + "all models zero split"
        lines.append(
            f"| {row['rank']} | `{row['feature']}` | `{row['family']}` | "
            f"{float(row['importance_share_pct']):.4f}% | {float(row['mean_model_rank']):.1f} | "
            f"{float(row['lgb0_split']):.1f}/{float(row['lgb1_split']):.1f}/{float(row['lgb2_split']):.1f} | "
            f"{duplicate_note} | {row['description']} |"
        )
    lines.extend(
        [
            "",
            "## Sources",
            "",
            "- `experiments/exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218/{config.yaml,result.md,metrics.json}`",
            "- Kaggle output `kentookumura/exp238-nested-rank-slot-exp218-train` v5 feature importance artifact。",
            "- Kaggle output `kentookumura/exp237-hmm-exp226-candidate-selector-exp183-train` v1 candidate readout / residual correlation。",
            "- `studies/feature_replacement_audit/README.md`とそのcorr-prune outputs。",
            "- `experiments/exp198_exact_replacement_prune_on_exp148/result.md`。",
            "- `experiments/exp252_pf_seed_medoid_selectability_audit/result.md`。",
            "- `experiments/exp019_pf_beam_candidate_quality_audit/result.md`、`exp091_self_gr_likelihood_pf_beam_probe/result.md`、`exp093_pf_candidate_coverage_then_ranker_audit/result.md`。",
            "- `experiments/exp128_trajectory_local_typewell_self_gr_switch_audit/result.md`、`exp134_self_gr_multiscale_longtail_gate/result.md`。",
            "- `experiments/exp103_pf_z_xy_likpf_ensemble_parity/result.md`、`exp106_strict_exp072_pf_z_multiseed_scale_cache/result.md`。",
            "- `experiments/exp142_trajectory_aware_pf_transition_prior/result.md`、`exp173_beam_topk_path_posterior_audit/result.md`、`exp177_beam_topk_bimodal_gate_posthoc_audit/result.md`。",
            "- `experiments/exp205_exact_hmm_smoother_exp072_compatible_cache_audit/result.md`、`exp221_lgb_oof_gaussian_emission_hmm_on_exp148/result.md`、`exp223_joint_typewell_self_gr_hmm_likelihood_probe/result.md`、`exp225_state_known_tvt_self_gr_hmm_emission/result.md`。",
            "- `experiments/exp243_pf_seed_medoids/result.md`、`experiments/exp251_raw_test_safe_dual_objective_candidate_ranker/result.md`。",
            "- `experiment_summary.md`のexp202–215 heatmap route close記録。",
            "- exp237 OOF chunked Beam marginal readout。source gzip SHA256 `c5d94361c2582f3f2e419ff70e8f87c1e4d3613b4cc21981e11f009f956d66c9`。",
            "- `experiments/exp255_nested_selector_gated_bounded_direct_readout_on_exp238/result.md`。",
            "- `experiments/exp257_nested_selector_output_replacement_only_on_exp218/result.md`。",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--importance", type=Path, default=DEFAULT_IMPORTANCE)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--residual-corr", type=Path, default=DEFAULT_RESIDUAL_CORR)
    parser.add_argument("--selector-importance", type=Path, default=DEFAULT_SELECTOR_IMPORTANCE)
    parser.add_argument("--selector-schema", type=Path, default=DEFAULT_SELECTOR_SCHEMA)
    parser.add_argument(
        "--selection-distribution",
        type=Path,
        default=SELECTION_DISTRIBUTION_INPUT,
    )
    args = parser.parse_args()
    for path in [
        args.importance,
        args.candidates,
        args.residual_corr,
        args.selector_importance,
        args.selector_schema,
        args.selection_distribution,
        DROP_AUDIT,
        BEAM_MARGINAL_INPUT,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    importance_rows = read_importance(args.importance)
    catalog = importance_catalog(importance_rows)
    duplicates = duplicate_evidence()
    candidates = read_csv(args.candidates)
    residual_corr = read_csv(args.residual_corr)
    selector_importance = read_csv(args.selector_importance)
    selector_schema = read_csv(args.selector_schema)
    selection_distribution = read_csv(args.selection_distribution)
    beam_marginal = read_csv(BEAM_MARGINAL_INPUT)

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved_inputs = [
        (args.importance, INPUT_DIR / "exp238_feature_importance_mean.csv"),
        (args.candidates, INPUT_DIR / "exp237_candidate_readout.csv"),
        (args.residual_corr, INPUT_DIR / "exp237_candidate_residual_correlation.csv"),
        (
            args.selector_importance,
            INPUT_DIR / "exp237_selector_feature_importance_mean.csv",
        ),
        (args.selector_schema, INPUT_DIR / "exp237_selector_feature_schema.csv"),
        (args.selection_distribution, SELECTION_DISTRIBUTION_INPUT),
    ]
    for source, destination in saved_inputs:
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
    write_catalog_csv(catalog, duplicates)
    write_report(
        catalog,
        duplicates,
        candidates,
        residual_corr,
        selector_importance,
        selector_schema,
        selection_distribution,
        beam_marginal,
    )
    print(
        {
            "report": str(REPORT_PATH),
            "catalog": str(OUTPUT_DIR / "exp238_feature_catalog_importance.csv"),
            "features": len(catalog),
            "all_models_zero": sum(bool(row["all_models_zero"]) for row in catalog),
        }
    )


if __name__ == "__main__":
    main()
