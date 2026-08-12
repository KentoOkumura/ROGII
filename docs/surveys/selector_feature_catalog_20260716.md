---
title: Selector入力特徴量カタログと重要度
date: 2026-07-16
types:
  - model_explanation
  - feature_analysis
  - comparison
experiments:
  - exp237
  - exp238
  - exp251
topics:
  - selector
  - feature_catalog
  - feature_importance
  - confidence
status: final
summary: "exp237・exp251のselector入力とexp238 downstream adapterを区別し、全特徴の意味と重要度をカタログ化した。"
---

# Selector入力特徴量カタログと重要度（exp237 / exp251 v4）

- 対応する上位仮説: なし

## 結論

- exp238のnested selector入力は、直接の親であるexp237のcandidate-long **320列**である。exp238の`nsel_*` 35列はselectorの入力ではなく、selectorが出した候補別予測誤差をTVTモデルへ渡すadapterである。
- exp251 v4はraw-test再生成契約を満たす **295列**を使い、`within10_classifier`と`expected_error_regressor`の2目的を学習した。
- 重要度はLightGBMの平均split回数であり、特徴を削除したときの因果効果ではない。候補identity、候補値、距離、HMM σ、cluster priorなど相関した列の間でsplitが分散する。
- exp237とexp251の共通列は282列。共通列におけるexp237誤差rankerとexp251 expected-errorの重要度順位相関は0.952。exp251内のexpected-errorとwithin10の順位相関は0.981で、2目的は同一ではない。

## 3種類の特徴を混同しない

| 層 | 列数 | 内容 |
| --- | ---: | --- |
| exp237 selector入力 | 320 | 候補値・候補間差・HMM/PF/dense/Self-GR診断・multiobs・cluster priorなど |
| exp251 v4 selector入力 | 295 | exp237系からraw-test再生成可能な契約へ整理し、view contextを追加 |
| exp238 TVTモデルへのselector出力adapter | 35 | top1/top2候補値、予測誤差、margin、候補別予測誤差など。selector入力ではない |

## exp238 downstream rank-slot adapter 35列

exp238最終TVT LightGBMの415列中、`nsel_*`だけを抽出した。`share`は各LightGBM内の総split数で正規化して3 config平均した比率、`TVT rank`は415列全体での順位。35列合計shareは9.573%。

| TVT rank | feature | share | lgb0/lgb1/lgb2 split | 説明 |
| --- | --- | --- | --- | --- |
| 1 | `nsel_top1_minus_anchor` | 1.6228% | 3547.6/2209.6/2358.0 | selector top1候補TVT − last_known_tvt。 |
| 9 | `nsel_top2_minus_anchor` | 1.2707% | 2823.0/1768.6/1775.0 | selector top2候補TVT − last_known_tvt。 |
| 30 | `nsel_pred_error_hmm_selfgr_boost_only_a070_c100_mean_tvt` | 0.8549% | 1513.0/1396.0/1288.6 | nested selectorが予測したself-GR HMMの絶対誤差。 |
| 56 | `nsel_pred_error_exp226_v6_k16_geometry_gr_u_projection` | 0.6931% | 1408.6/1048.4/987.0 | nested selectorが予測したexp226 K16 geometry/GR/U-projectionの絶対誤差。 |
| 60 | `nsel_pred_error_beam_mean` | 0.6572% | 1417.0/896.6/968.4 | nested selectorが予測した複数Beam path平均の絶対誤差。 |
| 68 | `nsel_pred_error_blend_likpf_hmm_w500` | 0.5927% | 968.4/962.0/960.6 | nested selectorが予測したlikPFとexact HMMの50/50平均の絶対誤差。 |
| 83 | `nsel_top2_minus_top1` | 0.4608% | 1824.2/325.2/340.4 | selector top2候補TVT − top1候補TVT。 |
| 90 | `nsel_pred_error_tvt_dense` | 0.4111% | 878.8/576.2/596.6 | nested selectorが予測したdense spatial ANCC（full-prefix bias）の絶対誤差。 |
| 94 | `nsel_pred_error_likpf_mean` | 0.3990% | 750.2/620.8/597.8 | nested selectorが予測したlikelihood-weighted PF平均の絶対誤差。 |
| 104 | `nsel_pred_error_tvt_densew` | 0.3648% | 656.6/590.2/547.0 | nested selectorが予測したdense spatial ANCC（prefix加重bias）の絶対誤差。 |
| 106 | `nsel_pred_error_tvt_dense50` | 0.3634% | 664.4/554.0/569.6 | nested selectorが予測したdense spatial ANCC（prefix末尾50 bias）の絶対誤差。 |
| 107 | `nsel_error_top1` | 0.3622% | 604.6/606.2/559.4 | selectorが予測したtop1候補の絶対誤差。 |
| 111 | `nsel_error_top2` | 0.3569% | 534.4/598.4/596.8 | selectorが予測したtop2候補の絶対誤差。 |
| 117 | `nsel_pred_error_pf_ancc` | 0.3225% | 656.6/470.8/475.0 | nested selectorが予測したANCC粒子フィルタの絶対誤差。 |
| 143 | `nsel_top1_code` | 0.2222% | 591.6/262.8/280.6 | 予測絶対誤差が最小の候補index（数値code）。 |
| 164 | `nsel_top2_code` | 0.1582% | 524.6/147.2/159.8 | 予測絶対誤差が2番目に小さい候補index（数値code）。 |
| 198 | `nsel_error_margin` | 0.0811% | 262.0/79.2/83.8 | predicted error top2 − top1。 |
| 220 | `nsel_error_ratio` | 0.0575% | 188.4/47.6/65.6 | predicted error top1 / max(top2, 1e-3)。 |
| 232 | `nsel_top1_is_blend_likpf_hmm_w500` | 0.0459% | 127.6/43.8/64.0 | selector top1がlikPFとexact HMMの50/50平均であるone-hot flag。 |
| 233 | `nsel_top1_is_likpf_mean` | 0.0437% | 113.0/50.6/58.6 | selector top1がlikelihood-weighted PF平均であるone-hot flag。 |
| 257 | `nsel_top1_is_hmm_selfgr_boost_only_a070_c100_mean_tvt` | 0.0289% | 85.8/22.8/40.8 | selector top1がself-GR HMMであるone-hot flag。 |
| 262 | `nsel_score_std` | 0.0255% | 95.6/18.8/22.2 | 11候補predicted errorの行別標準偏差。 |
| 264 | `nsel_top1_is_hyb` | 0.0243% | 125.8/4.4/7.6 | selector top1がBeam/NCC hybridであるone-hot flag。 |
| 265 | `nsel_top1_is_exp226_v6_k16_geometry_gr_u_projection` | 0.0239% | 27.8/42.8/43.6 | selector top1がexp226 K16 geometry/GR/U-projectionであるone-hot flag。 |
| 274 | `nsel_top1_is_beam_mean` | 0.0204% | 81.4/13.0/16.0 | selector top1が複数Beam path平均であるone-hot flag。 |
| 275 | `nsel_top1_is_tvt_dense` | 0.0204% | 86.4/13.2/12.0 | selector top1がdense spatial ANCC（full-prefix bias）であるone-hot flag。 |
| 289 | `nsel_pred_error_hyb` | 0.0135% | 41.6/14.2/14.6 | nested selectorが予測したBeam/NCC hybridの絶対誤差。 |
| 290 | `nsel_top1_is_tvt_densew` | 0.0135% | 46.8/11.2/13.4 | selector top1がdense spatial ANCC（prefix加重bias）であるone-hot flag。 |
| 298 | `nsel_pred_error_sc_ens` | 0.0111% | 38.6/9.2/11.0 | nested selectorが予測したmulti-scale NCC ensembleの絶対誤差。 |
| 299 | `nsel_top1_is_pf_ancc` | 0.0110% | 35.4/10.2/12.0 | selector top1がANCC粒子フィルタであるone-hot flag。 |
| 302 | `nsel_top1_is_sc_ens` | 0.0102% | 58.8/0.4/0.2 | selector top1がmulti-scale NCC ensembleであるone-hot flag。 |
| 303 | `nsel_score_mean` | 0.0097% | 24.8/10.8/13.8 | 11候補predicted errorの行別平均。 |
| 309 | `nsel_candidate_range` | 0.0077% | 23.4/6.4/10.2 | 11候補TVTの行別range。 |
| 316 | `nsel_top1_is_tvt_dense50` | 0.0064% | 30.8/2.2/2.8 | selector top1がdense spatial ANCC（prefix末尾50 bias）であるone-hot flag。 |
| 327 | `nsel_candidate_std` | 0.0050% | 15.4/5.6/5.2 | 11候補TVTの行別標準偏差。 |

## exp237 selector入力 320列

重要度は5 outer foldの`lgb_candidate_error_ranker`平均split回数。

| rank | feature | family | mean±std split | share | 説明 |
| --- | --- | --- | --- | --- | --- |
| 1 | `candidate_minus_last` | `candidate_identity_or_context` | 1903.2±53.7 | 5.598% | この候補TVT − last_known_tvt。候補の外挿量。 |
| 2 | `candidate_index` | `candidate_identity_or_context` | 1155.4±50.1 | 3.398% | 固定11候補bank内の候補index。候補identityを木に伝える数値code。 |
| 3 | `v6_k16_geometry_gr_u_projection_minus_last` | `candidate_path_value` | 979.6±68.8 | 2.881% | exp226 K16 geometry/GR/U-projection TVT − last_known_tvt。 |
| 4 | `candidate_tvt` | `candidate_identity_or_context` | 838.8±47.3 | 2.467% | このcandidate-long行が表す候補パスの絶対TVT。 |
| 5 | `hmm_selfgr_boost_only_a070_c100_minus_last` | `hmm_confidence` | 774.6±63.7 | 2.278% | Self-GR likelihood HMM TVT − last_known_tvt。 |
| 6 | `copcf_spatial_xy_plus_trajectory_shape_k8_minus_candidate` | `cluster_prior_confidence` | 704.2±69.7 | 2.071% | XY距離とtrajectory shapeを使うK=8 spatial prior TVT − 対象候補TVT。 |
| 7 | `tvt_densew_vs_tvt_dense50_abs` | `candidate_disagreement` | 703.8±25.5 | 2.070% | \|dense spatial ANCC（prefix加重bias） TVT − dense spatial ANCC（prefix末尾50 bias） TVT\|。 |
| 8 | `copcf_nearest_other_cluster_dist` | `cluster_prior_confidence` | 691.4±62.5 | 2.033% | 最も近い別cluster中心までの距離。 |
| 9 | `hmm_exact_std` | `hmm_confidence` | 648.2±77.6 | 1.906% | exact HMM posteriorのTVT標準偏差。 |
| 10 | `eval_len` | `distance_or_anchor_context` | 566.4±35.3 | 1.666% | そのwellの予測tail行数。 |
| 11 | `copcf_own_cluster_dist` | `cluster_prior_confidence` | 562.2±34.6 | 1.653% | test wellと割当先cluster中心の距離。 |
| 12 | `hmm_selfgr_boost_only_a070_c100_vs_v6_k16_geometry_gr_u_projection_abs` | `hmm_confidence` | 558.8±46.2 | 1.643% | \|Self-GR likelihood HMM TVT − exp226 K16 geometry/GR/U-projection TVT\|。 |
| 13 | `copcf_spatial_xy_only_k8_minus_candidate` | `cluster_prior_confidence` | 538.6±33.7 | 1.584% | XY距離だけを使うK=8 spatial prior TVT − 対象候補TVT。 |
| 14 | `hmm_selfgr_std` | `hmm_confidence` | 536.4±39.2 | 1.578% | Self-GR likelihood HMM posteriorのTVT標準偏差。 |
| 15 | `blend_likpf_hmm_w500_minus_last` | `candidate_path_value` | 536.0±33.9 | 1.576% | likPF / exact-HMM 50:50 blend TVT − last_known_tvt。 |
| 16 | `copcf_own_cluster_dist_z` | `cluster_prior_confidence` | 522.6±44.3 | 1.537% | 割当先cluster距離のcluster内z-score。 |
| 17 | `hmm_exact_loglik` | `hmm_confidence` | 487.2±24.4 | 1.433% | exact HMMの行別観測log-likelihood。 |
| 18 | `copcf_typewell_native_overlap_1_minus_candidate` | `cluster_prior_confidence` | 470.6±15.2 | 1.384% | native-overlap閾値1.0のtypewell cluster prior TVT − 対象候補TVT。 |
| 19 | `dense_dist` | `pf_dense_confidence` | 469.6±38.6 | 1.381% | dense spatial KNNで使う最短正規化XY距離。 |
| 20 | `beam_mean_d` | `candidate_path_value` | 458.2±19.1 | 1.348% | 複数Beam path平均候補 − last_known_tvt。 |
| 21 | `copcf_spatial_xy_plus_trajectory_shape_k8_minus_candidate_abs` | `cluster_prior_confidence` | 431.6±33.3 | 1.269% | \|XY距離とtrajectory shapeを使うK=8 spatial prior TVT − 対象候補TVT\|。 |
| 22 | `beam_mean_vs_hmm_selfgr_boost_only_a070_c100_abs` | `candidate_disagreement` | 415.2±36.6 | 1.221% | \|複数Beam path平均 TVT − Self-GR likelihood HMM TVT\|。 |
| 23 | `tvt_dense_vs_tvt_densew_abs` | `candidate_disagreement` | 407.0±26.2 | 1.197% | \|dense spatial ANCC（full-prefix bias） TVT − dense spatial ANCC（prefix加重bias） TVT\|。 |
| 24 | `pf_ancc_minus_last` | `candidate_path_value` | 399.4±37.5 | 1.175% | ANCC PF TVT − last_known_tvt。 |
| 25 | `copcf_spatial_xy_plus_trajectory_shape_k8_minus_candidate_abs_norm` | `cluster_prior_confidence` | 395.8±38.8 | 1.164% | \|XY距離とtrajectory shapeを使うK=8 spatial prior TVT − 対象候補TVT\|をprior標準偏差で正規化。 |
| 26 | `beam_mean_vs_v6_k16_geometry_gr_u_projection_abs` | `candidate_disagreement` | 371.0±10.3 | 1.091% | \|複数Beam path平均 TVT − exp226 K16 geometry/GR/U-projection TVT\|。 |
| 27 | `hmm_selfgr_loglik` | `hmm_confidence` | 370.2±15.5 | 1.089% | Self-GR likelihood HMMの行別観測log-likelihood。 |
| 28 | `likpf_mean_d` | `candidate_path_value` | 369.2±33.1 | 1.086% | likelihood-weighted PF平均候補 − last_known_tvt。 |
| 29 | `md_since` | `distance_or_anchor_context` | 349.4±25.3 | 1.028% | 既知prefix末尾から予測行までのMD距離。 |
| 30 | `pf_ancc_vs_v6_k16_geometry_gr_u_projection_abs` | `candidate_disagreement` | 326.8±8.7 | 0.961% | \|ANCC PF TVT − exp226 K16 geometry/GR/U-projection TVT\|。 |
| 31 | `pf_ancc_vs_hmm_selfgr_boost_only_a070_c100_abs` | `candidate_disagreement` | 320.8±28.8 | 0.944% | \|ANCC PF TVT − Self-GR likelihood HMM TVT\|。 |
| 32 | `copcf_typewell_native_overlap_0p999_neighbor_wells` | `cluster_prior_confidence` | 312.2±26.1 | 0.918% | native-overlap閾値0.999のtypewell cluster priorを作る近傍well数。 |
| 33 | `tvt_dense_vs_tvt_dense50_abs` | `candidate_disagreement` | 311.4±15.6 | 0.916% | \|dense spatial ANCC（full-prefix bias） TVT − dense spatial ANCC（prefix末尾50 bias） TVT\|。 |
| 34 | `likpf_mean_vs_v6_k16_geometry_gr_u_projection_abs` | `candidate_disagreement` | 309.2±23.0 | 0.909% | \|likelihood-weighted PF平均 TVT − exp226 K16 geometry/GR/U-projection TVT\|。 |
| 35 | `crfe_dense_dist_norm` | `disagreement_enrichment` | 294.4±5.6 | 0.866% | dense_distを上限付きで正規化した近傍距離proxy。 |
| 36 | `hmm_exact_minus_likpf_mean` | `hmm_confidence` | 290.4±36.0 | 0.854% | exact HMM候補 − likelihood-weighted PF平均候補。 |
| 37 | `copcf_spatial_xy_only_k8_minus_candidate_abs_norm` | `cluster_prior_confidence` | 279.8±75.6 | 0.823% | \|XY距離だけを使うK=8 spatial prior TVT − 対象候補TVT\|をprior標準偏差で正規化。 |
| 38 | `dense_std` | `pf_dense_confidence` | 278.2±19.1 | 0.818% | dense spatial KNN近傍ANCCの重み付き標準偏差。 |
| 39 | `last_known_tvt` | `distance_or_anchor_context` | 276.6±40.2 | 0.814% | 既知prefix末尾のTVT。全候補deltaのanchor。 |
| 40 | `copcf_typewell_spatial_prior_delta` | `cluster_prior_confidence` | 269.6±22.2 | 0.793% | typewell prior − spatial prior。2種類の補正方向の差。 |
| 41 | `blend_likpf_hmm_w500_vs_v6_k16_geometry_gr_u_projection_abs` | `candidate_disagreement` | 266.8±22.5 | 0.785% | \|likPF / exact-HMM 50:50 blend TVT − exp226 K16 geometry/GR/U-projection TVT\|。 |
| 42 | `copcf_spatial_xy_only_k8_minus_candidate_abs` | `cluster_prior_confidence` | 261.2±26.5 | 0.768% | \|XY距離だけを使うK=8 spatial prior TVT − 対象候補TVT\|。 |
| 43 | `beam_mean_vs_blend_likpf_hmm_w500_abs` | `candidate_disagreement` | 257.2±36.5 | 0.756% | \|複数Beam path平均 TVT − likPF / exact-HMM 50:50 blend TVT\|。 |
| 44 | `exp226_gr_delta` | `exp226_geometry` | 245.2±25.8 | 0.721% | exp226候補位置でのGR整合差。exp251 v4ではraw-test契約外として除外。 |
| 45 | `copcf_typewell_native_overlap_1_minus_candidate_abs` | `cluster_prior_confidence` | 244.6±16.9 | 0.719% | \|native-overlap閾値1.0のtypewell cluster prior TVT − 対象候補TVT\|。 |
| 46 | `tvt_dense_vs_hmm_selfgr_boost_only_a070_c100_abs` | `candidate_disagreement` | 241.6±14.5 | 0.711% | \|dense spatial ANCC（full-prefix bias） TVT − Self-GR likelihood HMM TVT\|。 |
| 47 | `copcf_typewell_native_overlap_1_minus_candidate_abs_norm` | `cluster_prior_confidence` | 241.0±24.7 | 0.709% | \|native-overlap閾値1.0のtypewell cluster prior TVT − 対象候補TVT\|をprior標準偏差で正規化。 |
| 48 | `exp226_geop_minus_pred` | `exp226_geometry` | 235.8±44.5 | 0.694% | exp226 geometry projection TVT − exp226最終候補TVT。 |
| 49 | `likpf_mean_vs_blend_likpf_hmm_w500_abs` | `candidate_disagreement` | 234.6±21.4 | 0.690% | \|likelihood-weighted PF平均 TVT − likPF / exact-HMM 50:50 blend TVT\|。 |
| 50 | `pf_ancc_vs_likpf_mean_abs` | `candidate_disagreement` | 232.0±20.5 | 0.682% | \|ANCC PF TVT − likelihood-weighted PF平均 TVT\|。 |
| 51 | `pf_ancc_vs_beam_mean_abs` | `candidate_disagreement` | 230.6±29.9 | 0.678% | \|ANCC PF TVT − 複数Beam path平均 TVT\|。 |
| 52 | `blend_likpf_hmm_w500_vs_hmm_selfgr_boost_only_a070_c100_abs` | `candidate_disagreement` | 230.2±9.0 | 0.677% | \|likPF / exact-HMM 50:50 blend TVT − Self-GR likelihood HMM TVT\|。 |
| 53 | `crfe_tvt_dense_drift_per_md` | `disagreement_enrichment` | 227.0±18.7 | 0.668% | (dense full-prefix候補 − anchor) / max(md_since, 1)。 |
| 54 | `pf_ancc_vs_blend_likpf_hmm_w500_abs` | `candidate_disagreement` | 227.0±17.9 | 0.668% | \|ANCC PF TVT − likPF / exact-HMM 50:50 blend TVT\|。 |
| 55 | `copcf_spatial_xy_plus_trajectory_shape_k8_prior_std` | `cluster_prior_confidence` | 222.8±23.7 | 0.655% | XY距離とtrajectory shapeを使うK=8 spatial priorを作るsource TVTの標準偏差。 |
| 56 | `copcf_typewell_native_overlap_0p999_prior_std` | `cluster_prior_confidence` | 220.0±28.8 | 0.647% | native-overlap閾値0.999のtypewell cluster priorを作るsource TVTの標準偏差。 |
| 57 | `copcf_typewell_native_overlap_1_prior_count` | `cluster_prior_confidence` | 214.8±37.3 | 0.632% | native-overlap閾値1.0のtypewell cluster priorを作る有効source数。 |
| 58 | `tvt_dense_vs_v6_k16_geometry_gr_u_projection_abs` | `candidate_disagreement` | 211.4±18.5 | 0.622% | \|dense spatial ANCC（full-prefix bias） TVT − exp226 K16 geometry/GR/U-projection TVT\|。 |
| 59 | `copcf_spatial_xy_only_k8_prior_std` | `cluster_prior_confidence` | 207.2±20.0 | 0.609% | XY距離だけを使うK=8 spatial priorを作るsource TVTの標準偏差。 |
| 60 | `crfe_tvt_dense50_drift_per_md` | `disagreement_enrichment` | 203.4±24.7 | 0.598% | (dense末尾50候補 − anchor) / max(md_since, 1)。 |
| 61 | `crfe_dense_candidate_mean` | `disagreement_enrichment` | 200.8±16.1 | 0.591% | dense 3候補TVTの平均。 |
| 62 | `tvt_densew_vs_hmm_selfgr_boost_only_a070_c100_abs` | `candidate_disagreement` | 199.8±19.3 | 0.588% | \|dense spatial ANCC（prefix加重bias） TVT − Self-GR likelihood HMM TVT\|。 |
| 63 | `tvt_dense50_vs_hmm_selfgr_boost_only_a070_c100_abs` | `candidate_disagreement` | 199.0±18.6 | 0.585% | \|dense spatial ANCC（prefix末尾50 bias） TVT − Self-GR likelihood HMM TVT\|。 |
| 64 | `likpf_mean_vs_hmm_selfgr_boost_only_a070_c100_abs` | `candidate_disagreement` | 189.2±31.7 | 0.556% | \|likelihood-weighted PF平均 TVT − Self-GR likelihood HMM TVT\|。 |
| 65 | `crfe_dense_candidate_std` | `disagreement_enrichment` | 177.4±11.5 | 0.522% | dense 3候補TVTの標準偏差。 |
| 66 | `beam_mean_vs_likpf_mean_abs` | `candidate_disagreement` | 176.6±27.5 | 0.519% | \|複数Beam path平均 TVT − likelihood-weighted PF平均 TVT\|。 |
| 67 | `hmm_selfgr_boost_only_a070_c100_mean_tvt` | `hmm_confidence` | 176.6±16.6 | 0.519% | Self-GR likelihood HMMの絶対TVT候補。 |
| 68 | `candidate_name_code` | `candidate_identity_or_context` | 175.6±50.9 | 0.516% | candidate_indexと同じ候補identity codeを別契約名で保持した列。 |
| 69 | `blend_likpf_hmm_w500` | `candidate_path_value` | 170.4±6.0 | 0.501% | likPF / exact-HMM 50:50 blendの絶対TVT候補。 |
| 70 | `tvt_dense_d` | `candidate_path_value` | 166.6±8.2 | 0.490% | dense full-prefix候補 − last_known_tvt。 |
| 71 | `crfe_tvt_densew_drift_per_md` | `disagreement_enrichment` | 164.0±29.0 | 0.482% | (dense加重候補 − anchor) / max(md_since, 1)。 |
| 72 | `candidate_is_dense_family` | `candidate_identity_or_context` | 159.6±16.5 | 0.469% | 対象候補がdense 3候補なら1。 |
| 73 | `crfe_tail_rank_norm` | `disagreement_enrichment` | 155.2±17.2 | 0.456% | min(予測tail内row index / 1000, 5)。 |
| 74 | `pf_vs_dense` | `candidate_disagreement` | 147.8±13.3 | 0.435% | ANCC PF候補 − dense ANCC候補。 |
| 75 | `candidate_is_hmm_family` | `candidate_identity_or_context` | 147.4±10.6 | 0.434% | 対象候補がHMMまたはHMM blendなら1。 |
| 76 | `exp226_geop_tvt` | `exp226_geometry` | 144.2±11.6 | 0.424% | exp226 geometry projectionが返す絶対TVT。exp251 v4ではraw-test契約外として除外。 |
| 77 | `copcf_nearby_majority_count_k12` | `cluster_prior_confidence` | 142.2±18.2 | 0.418% | 近傍12 well中、多数派clusterに属するwell数。 |
| 78 | `candidate_is_default_likpf` | `candidate_identity_or_context` | 137.6±15.7 | 0.405% | 対象候補が既定fallbackのlikpf_meanなら1。 |
| 79 | `crfe_dense_candidate_range` | `disagreement_enrichment` | 137.0±16.1 | 0.403% | dense 3候補TVTの最大−最小。 |
| 80 | `crfe_pf_ancc_minus_tvt_densew` | `disagreement_enrichment` | 135.2±13.1 | 0.398% | ANCC PF候補 − dense prefix加重候補。 |
| 81 | `exp226_v6_k16_geometry_gr_u_projection` | `exp226_geometry` | 134.8±12.6 | 0.396% | exp226 K16 geometry/GR/U-projectionの絶対TVT候補。 |
| 82 | `crfe_likpf_mean_minus_tvt_densew` | `disagreement_enrichment` | 132.2±19.0 | 0.389% | likPF平均候補 − dense prefix加重候補。 |
| 83 | `tvt_densew_vs_v6_k16_geometry_gr_u_projection_abs` | `candidate_disagreement` | 128.4±11.3 | 0.378% | \|dense spatial ANCC（prefix加重bias） TVT − exp226 K16 geometry/GR/U-projection TVT\|。 |
| 84 | `beam_mean_minus_last` | `candidate_path_value` | 126.0±19.2 | 0.371% | 複数Beam path平均 TVT − last_known_tvt。 |
| 85 | `candidate_multiobs_mae` | `multi_observation` | 124.2±12.6 | 0.365% | 対象候補自身のmulti-observation GR照合MAE。 |
| 86 | `crfe_beam_mean_minus_tvt_densew_abs_norm` | `disagreement_enrichment` | 117.0±16.5 | 0.344% | \|Beam平均 − dense加重\| / max(dense_std, 10)。 |
| 87 | `crfe_high_disagreement_proxy` | `disagreement_enrichment` | 116.2±2.9 | 0.342% | PF-vs-dense差、dense std、dense距離を0.45/0.35/0.20で合成したdisagreement。 |
| 88 | `tvt_dense50_vs_v6_k16_geometry_gr_u_projection_abs` | `candidate_disagreement` | 109.4±5.5 | 0.322% | \|dense spatial ANCC（prefix末尾50 bias） TVT − exp226 K16 geometry/GR/U-projection TVT\|。 |
| 89 | `tvt_dense_vs_blend_likpf_hmm_w500_abs` | `candidate_disagreement` | 106.4±21.9 | 0.313% | \|dense spatial ANCC（full-prefix bias） TVT − likPF / exact-HMM 50:50 blend TVT\|。 |
| 90 | `beam_mean_vs_tvt_dense_abs` | `candidate_disagreement` | 103.8±11.4 | 0.305% | \|複数Beam path平均 TVT − dense spatial ANCC（full-prefix bias） TVT\|。 |
| 91 | `likpf_mean_vs_tvt_dense_abs` | `candidate_disagreement` | 103.6±11.2 | 0.305% | \|likelihood-weighted PF平均 TVT − dense spatial ANCC（full-prefix bias） TVT\|。 |
| 92 | `crfe_tvt_dense_abs_delta_from_last` | `disagreement_enrichment` | 101.2±10.6 | 0.298% | \|dense full-prefix候補 − anchor\|。 |
| 93 | `pf_ancc_vs_tvt_dense_abs` | `candidate_disagreement` | 99.8±10.8 | 0.294% | \|ANCC PF TVT − dense spatial ANCC（full-prefix bias） TVT\|。 |
| 94 | `copcf_nearby_majority_count_k8` | `cluster_prior_confidence` | 96.6±8.8 | 0.284% | 近傍8 well中、多数派clusterに属するwell数。 |
| 95 | `tvt_dense50_d` | `candidate_path_value` | 96.4±12.7 | 0.284% | dense prefix末尾50候補 − last_known_tvt。 |
| 96 | `crfe_beam_mean_minus_tvt_densew` | `disagreement_enrichment` | 93.4±17.0 | 0.275% | Beam平均候補 − dense prefix加重候補。 |
| 97 | `copcf_spatial_xy_only_k8_prior_count` | `cluster_prior_confidence` | 92.8±9.5 | 0.273% | XY距離だけを使うK=8 spatial priorを作る有効source数。 |
| 98 | `crfe_pf_vs_dense_abs_norm` | `disagreement_enrichment` | 91.0±13.1 | 0.268% | \|ANCC PF − dense ANCC\| / max(dense_std, 10)。 |
| 99 | `pf_ancc_std` | `pf_dense_confidence` | 91.0±15.8 | 0.268% | ANCC PF粒子の行別TVT標準偏差。粒子分布の幅。 |
| 100 | `crfe_likpf_mean_minus_tvt_densew_abs_norm` | `disagreement_enrichment` | 85.4±12.3 | 0.251% | \|likPF − dense加重\| / max(dense_std, 10)。 |
| 101 | `crfe_pf_ancc_minus_tvt_densew_abs_norm` | `disagreement_enrichment` | 83.8±15.3 | 0.246% | \|ANCC PF − dense加重\| / max(dense_std, 10)。 |
| 102 | `copcf_typewell_native_overlap_0p999_minus_candidate` | `cluster_prior_confidence` | 81.4±9.4 | 0.239% | native-overlap閾値0.999のtypewell cluster prior TVT − 対象候補TVT。 |
| 103 | `copcf_typewell_spatial_prior_abs_delta` | `cluster_prior_confidence` | 81.0±9.5 | 0.238% | \|typewell prior − spatial prior\|。 |
| 104 | `pf_ancc_vs_tvt_densew_abs` | `candidate_disagreement` | 80.6±9.9 | 0.237% | \|ANCC PF TVT − dense spatial ANCC（prefix加重bias） TVT\|。 |
| 105 | `pf_ancc_vs_tvt_dense50_abs` | `candidate_disagreement` | 79.0±14.0 | 0.232% | \|ANCC PF TVT − dense spatial ANCC（prefix末尾50 bias） TVT\|。 |
| 106 | `tvt_densew_d` | `candidate_path_value` | 76.8±10.7 | 0.226% | dense prefix加重候補 − last_known_tvt。 |
| 107 | `exp226_geop_minus_pred_abs` | `exp226_geometry` | 71.6±13.0 | 0.211% | \|exp226 geometry projection TVT − exp226最終候補TVT\|。 |
| 108 | `candidate_multiobs_score` | `multi_observation` | 70.4±15.4 | 0.207% | 対象候補自身のmulti-observation GR一致score。 |
| 109 | `likpf_mean_vs_tvt_dense50_abs` | `candidate_disagreement` | 68.8±11.4 | 0.202% | \|likelihood-weighted PF平均 TVT − dense spatial ANCC（prefix末尾50 bias） TVT\|。 |
| 110 | `tvt_densew_vs_blend_likpf_hmm_w500_abs` | `candidate_disagreement` | 68.6±8.7 | 0.202% | \|dense spatial ANCC（prefix加重bias） TVT − likPF / exact-HMM 50:50 blend TVT\|。 |
| 111 | `crfe_tvt_densew_abs_delta_from_last` | `disagreement_enrichment` | 62.4±7.6 | 0.184% | \|dense prefix加重候補 − anchor\|。 |
| 112 | `crfe_tvt_dense50_abs_delta_from_last` | `disagreement_enrichment` | 62.0±8.5 | 0.182% | \|dense末尾50候補 − anchor\|。 |
| 113 | `tvt_dense50_vs_blend_likpf_hmm_w500_abs` | `candidate_disagreement` | 60.6±9.0 | 0.178% | \|dense spatial ANCC（prefix末尾50 bias） TVT − likPF / exact-HMM 50:50 blend TVT\|。 |
| 114 | `likpf_mean_vs_tvt_densew_abs` | `candidate_disagreement` | 56.6±2.7 | 0.166% | \|likelihood-weighted PF平均 TVT − dense spatial ANCC（prefix加重bias） TVT\|。 |
| 115 | `copcf_nearby_majority_count_k5` | `cluster_prior_confidence` | 53.8±10.4 | 0.158% | 近傍5 well中、多数派clusterに属するwell数。 |
| 116 | `likpf_mean_minus_last` | `candidate_path_value` | 53.6±10.9 | 0.158% | likelihood-weighted PF平均 TVT − last_known_tvt。 |
| 117 | `sc_ens_vs_v6_k16_geometry_gr_u_projection_abs` | `candidate_disagreement` | 52.8±9.2 | 0.155% | \|multi-scale NCC ensemble TVT − exp226 K16 geometry/GR/U-projection TVT\|。 |
| 118 | `hyb_vs_v6_k16_geometry_gr_u_projection_abs` | `candidate_disagreement` | 46.6±10.0 | 0.137% | \|Beam/NCC hybrid TVT − exp226 K16 geometry/GR/U-projection TVT\|。 |
| 119 | `beam_mean_vs_tvt_densew_abs` | `candidate_disagreement` | 45.0±6.9 | 0.132% | \|複数Beam path平均 TVT − dense spatial ANCC（prefix加重bias） TVT\|。 |
| 120 | `sc_ens_vs_hmm_selfgr_boost_only_a070_c100_abs` | `candidate_disagreement` | 42.8±11.3 | 0.126% | \|multi-scale NCC ensemble TVT − Self-GR likelihood HMM TVT\|。 |
| 121 | `copcf_typewell_native_overlap_1_neighbor_wells` | `cluster_prior_confidence` | 42.0±8.0 | 0.124% | native-overlap閾値1.0のtypewell cluster priorを作る近傍well数。 |
| 122 | `beam_mean_vs_tvt_dense50_abs` | `candidate_disagreement` | 41.8±6.9 | 0.123% | \|複数Beam path平均 TVT − dense spatial ANCC（prefix末尾50 bias） TVT\|。 |
| 123 | `sc_ens_vs_blend_likpf_hmm_w500_abs` | `candidate_disagreement` | 40.0±10.7 | 0.118% | \|multi-scale NCC ensemble TVT − likPF / exact-HMM 50:50 blend TVT\|。 |
| 124 | `copcf_typewell_native_overlap_0p999_minus_candidate_abs_norm` | `cluster_prior_confidence` | 36.2±5.1 | 0.106% | \|native-overlap閾値0.999のtypewell cluster prior TVT − 対象候補TVT\|をprior標準偏差で正規化。 |
| 125 | `hyb_vs_blend_likpf_hmm_w500_abs` | `candidate_disagreement` | 36.2±5.7 | 0.106% | \|Beam/NCC hybrid TVT − likPF / exact-HMM 50:50 blend TVT\|。 |
| 126 | `copcf_spatial_xy_only_k8_nearby_majority_diff_k8_corr_abs_c20` | `cluster_prior_confidence` | 36.0±4.9 | 0.106% | K=8近傍の多数派cluster不一致時にXY距離だけを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 127 | `hyb_vs_hmm_selfgr_boost_only_a070_c100_abs` | `candidate_disagreement` | 35.4±6.1 | 0.104% | \|Beam/NCC hybrid TVT − Self-GR likelihood HMM TVT\|。 |
| 128 | `copcf_spatial_xy_only_k8_any_outlier_signal_k8_corr_abs_c40` | `cluster_prior_confidence` | 34.8±8.4 | 0.102% | K=8近傍のいずれかのoutlier signal時にXY距離だけを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 129 | `copcf_spatial_xy_plus_trajectory_shape_k8_std_x_candidate` | `cluster_prior_confidence` | 33.8±4.8 | 0.099% | XY距離とtrajectory shapeを使うK=8 spatial priorの標準偏差をcandidate-long各行へ複製した列。積ではない。 |
| 130 | `crfe_dense_std_norm` | `disagreement_enrichment` | 33.8±6.1 | 0.099% | dense_std / 10を上限付きで正規化した不確実性proxy。 |
| 131 | `copcf_typewell_native_overlap_0p999_minus_candidate_abs` | `cluster_prior_confidence` | 33.6±7.0 | 0.099% | \|native-overlap閾値0.999のtypewell cluster prior TVT − 対象候補TVT\|。 |
| 132 | `sc_ens_d` | `candidate_path_value` | 33.4±5.9 | 0.098% | multi-scale NCC ensemble候補 − last_known_tvt。 |
| 133 | `copcf_spatial_xy_only_k8_nearby_majority_diff_k8_corr_abs_c40` | `cluster_prior_confidence` | 33.2±7.7 | 0.098% | K=8近傍の多数派cluster不一致時にXY距離だけを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 134 | `copcf_spatial_xy_only_k8_nearest_other_closer_corr_abs_c40` | `cluster_prior_confidence` | 32.0±10.3 | 0.094% | 別clusterの方が近い時にXY距離だけを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 135 | `copcf_spatial_xy_only_k8_nearest_other_closer_corr_abs_c20` | `cluster_prior_confidence` | 31.8±10.4 | 0.094% | 別clusterの方が近い時にXY距離だけを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 136 | `copcf_spatial_xy_only_k8_std_x_candidate` | `cluster_prior_confidence` | 31.0±6.7 | 0.091% | XY距離だけを使うK=8 spatial priorの標準偏差をcandidate-long各行へ複製した列。積ではない。 |
| 137 | `copcf_spatial_xy_plus_trajectory_shape_k8_nearest_other_closer_corr_abs_c20` | `cluster_prior_confidence` | 30.8±4.2 | 0.091% | 別clusterの方が近い時にXY距離とtrajectory shapeを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 138 | `copcf_typewell_native_overlap_0p999_prior_count` | `cluster_prior_confidence` | 29.8±10.5 | 0.088% | native-overlap閾値0.999のtypewell cluster priorを作る有効source数。 |
| 139 | `copcf_spatial_xy_plus_trajectory_shape_k8_nearby_majority_diff_k8_corr_abs_c40` | `cluster_prior_confidence` | 28.4±9.3 | 0.084% | K=8近傍の多数派cluster不一致時にXY距離とtrajectory shapeを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 140 | `copcf_spatial_xy_plus_trajectory_shape_k8_nearest_other_closer_corr_abs_c40` | `cluster_prior_confidence` | 28.0±7.8 | 0.082% | 別clusterの方が近い時にXY距離とtrajectory shapeを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 141 | `copcf_spatial_xy_only_k8_any_outlier_signal_k8_corr_abs_c20` | `cluster_prior_confidence` | 27.4±4.0 | 0.081% | K=8近傍のいずれかのoutlier signal時にXY距離だけを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 142 | `copcf_spatial_xy_plus_trajectory_shape_k8_nearby_majority_diff_k8_corr_abs_c20` | `cluster_prior_confidence` | 27.4±4.4 | 0.081% | K=8近傍の多数派cluster不一致時にXY距離とtrajectory shapeを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 143 | `copcf_typewell_native_overlap_1_prior_std` | `cluster_prior_confidence` | 27.4±6.1 | 0.081% | native-overlap閾値1.0のtypewell cluster priorを作るsource TVTの標準偏差。 |
| 144 | `candidate_std` | `candidate_identity_or_context` | 26.4±7.0 | 0.078% | 同じbase rowにある候補TVT群の標準偏差。候補間disagreement。 |
| 145 | `copcf_typewell_native_overlap_1_nearest_other_closer_corr_abs_c20` | `cluster_prior_confidence` | 26.4±7.5 | 0.078% | 別clusterの方が近い時にnative-overlap閾値1.0のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 146 | `copcf_spatial_xy_plus_trajectory_shape_k8_any_outlier_signal_k8_corr_abs_c40` | `cluster_prior_confidence` | 26.2±6.6 | 0.077% | K=8近傍のいずれかのoutlier signal時にXY距離とtrajectory shapeを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 147 | `copcf_spatial_xy_plus_trajectory_shape_k8_prior_count` | `cluster_prior_confidence` | 26.2±4.9 | 0.077% | XY距離とtrajectory shapeを使うK=8 spatial priorを作る有効source数。 |
| 148 | `copcf_typewell_native_overlap_1_nearby_majority_diff_k8_corr_abs_c40` | `cluster_prior_confidence` | 25.6±6.1 | 0.075% | K=8近傍の多数派cluster不一致時にnative-overlap閾値1.0のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 149 | `copcf_nearby_majority_share_k12` | `cluster_prior_confidence` | 25.0±5.5 | 0.074% | 近傍12 wellの多数派cluster比率。 |
| 150 | `tvt_dense_minus_last` | `candidate_path_value` | 23.2±2.4 | 0.068% | dense spatial ANCC（full-prefix bias） TVT − last_known_tvt。 |
| 151 | `likpf_mean_vs_sc_ens_abs` | `candidate_disagreement` | 23.0±6.0 | 0.068% | \|likelihood-weighted PF平均 TVT − multi-scale NCC ensemble TVT\|。 |
| 152 | `copcf_typewell_native_overlap_1_nearest_other_closer_corr_abs_c40` | `cluster_prior_confidence` | 22.4±8.0 | 0.066% | 別clusterの方が近い時にnative-overlap閾値1.0のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 153 | `crfe_high_disagreement_x_longtail` | `disagreement_enrichment` | 22.2±4.1 | 0.065% | high_disagreement_proxy × longtail_1000_flag。 |
| 154 | `copcf_spatial_xy_only_k8_own_z_gt2p0_corr_abs_c20` | `cluster_prior_confidence` | 22.0±9.7 | 0.065% | 所属cluster距離z-score>2時にXY距離だけを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 155 | `copcf_typewell_native_overlap_1_any_outlier_signal_k8_corr_abs_c40` | `cluster_prior_confidence` | 21.6±4.3 | 0.064% | K=8近傍のいずれかのoutlier signal時にnative-overlap閾値1.0のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 156 | `copcf_typewell_native_overlap_1_nearby_majority_diff_k8_corr_abs_c20` | `cluster_prior_confidence` | 21.6±4.6 | 0.064% | K=8近傍の多数派cluster不一致時にnative-overlap閾値1.0のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 157 | `self_gr_peak_gap` | `self_gr_confidence` | 21.6±9.6 | 0.064% | Self-GR照合のbest peakとsecond peakのscore差。 |
| 158 | `copcf_nearby_majority_diff_k5` | `cluster_prior_confidence` | 21.4±3.4 | 0.063% | 近傍5 wellの多数派clusterが自身の割当clusterと異なれば1。 |
| 159 | `copcf_spatial_xy_plus_trajectory_shape_k8_any_outlier_signal_k8_corr_abs_c20` | `cluster_prior_confidence` | 19.6±3.0 | 0.058% | K=8近傍のいずれかのoutlier signal時にXY距離とtrajectory shapeを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 160 | `candidate_mean` | `candidate_identity_or_context` | 18.8±5.1 | 0.055% | 同じbase rowにある候補TVT群の平均。 |
| 161 | `copcf_nearby_majority_share_k8` | `cluster_prior_confidence` | 17.6±3.8 | 0.052% | 近傍8 wellの多数派cluster比率。 |
| 162 | `hyb_d` | `candidate_path_value` | 17.6±9.8 | 0.052% | Beam/NCC hybrid候補 − last_known_tvt。 |
| 163 | `copcf_typewell_native_overlap_1_any_outlier_signal_k8_corr_abs_c20` | `cluster_prior_confidence` | 17.4±3.3 | 0.051% | K=8近傍のいずれかのoutlier signal時にnative-overlap閾値1.0のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 164 | `likpf_mean_vs_hyb_abs` | `candidate_disagreement` | 17.4±6.8 | 0.051% | \|likelihood-weighted PF平均 TVT − Beam/NCC hybrid TVT\|。 |
| 165 | `copcf_spatial_xy_only_k8_own_z_gt2p0_corr_abs_c40` | `cluster_prior_confidence` | 17.2±5.0 | 0.051% | 所属cluster距離z-score>2時にXY距離だけを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 166 | `copcf_spatial_xy_plus_trajectory_shape_k8_own_z_gt2p0_corr_abs_c40` | `cluster_prior_confidence` | 16.6±8.2 | 0.049% | 所属cluster距離z-score>2時にXY距離とtrajectory shapeを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 167 | `multiobs_mae_beam_mean` | `multi_observation` | 15.4±4.7 | 0.045% | 複数Beam path平均のmulti-observation GR照合MAE。 |
| 168 | `tvt_dense50_minus_last` | `candidate_path_value` | 14.6±4.7 | 0.043% | dense spatial ANCC（prefix末尾50 bias） TVT − last_known_tvt。 |
| 169 | `copcf_spatial_xy_plus_trajectory_shape_k8_nearest_other_closer_gate_x_dense_family` | `cluster_prior_confidence` | 14.0±6.6 | 0.041% | XY距離とtrajectory shapeを使うK=8 spatial priorが有効、別clusterの方が近い、かつ対象候補がdense familyなら1。 |
| 170 | `copcf_spatial_xy_plus_trajectory_shape_k8_own_z_gt2p0_corr_abs_c20` | `cluster_prior_confidence` | 13.8±5.9 | 0.041% | 所属cluster距離z-score>2時にXY距離とtrajectory shapeを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 171 | `copcf_spatial_xy_only_k8_count_x_candidate` | `cluster_prior_confidence` | 13.6±4.5 | 0.040% | XY距離だけを使うK=8 spatial priorのsource数をcandidate-long各行へ複製した列。積ではない。 |
| 172 | `copcf_typewell_native_overlap_1_nearest_other_closer_gate_x_dense_family` | `cluster_prior_confidence` | 13.4±2.4 | 0.039% | native-overlap閾値1.0のtypewell cluster priorが有効、別clusterの方が近い、かつ対象候補がdense familyなら1。 |
| 173 | `copcf_spatial_xy_plus_trajectory_shape_k8_nearby_majority_diff_k8_gate_x_dense_family` | `cluster_prior_confidence` | 13.2±6.8 | 0.039% | XY距離とtrajectory shapeを使うK=8 spatial priorが有効、K=8近傍の多数派cluster不一致、かつ対象候補がdense familyなら1。 |
| 174 | `copcf_spatial_xy_only_k8_nearby_majority_diff_k8_gate_x_dense_family` | `cluster_prior_confidence` | 12.8±6.1 | 0.038% | XY距離だけを使うK=8 spatial priorが有効、K=8近傍の多数派cluster不一致、かつ対象候補がdense familyなら1。 |
| 175 | `copcf_nearby_majority_diff_k12` | `cluster_prior_confidence` | 12.6±3.9 | 0.037% | 近傍12 wellの多数派clusterが自身の割当clusterと異なれば1。 |
| 176 | `pf_ancc_vs_sc_ens_abs` | `candidate_disagreement` | 12.2±3.8 | 0.036% | \|ANCC PF TVT − multi-scale NCC ensemble TVT\|。 |
| 177 | `sc_ens_vs_hyb_abs` | `candidate_disagreement` | 11.6±1.8 | 0.034% | \|multi-scale NCC ensemble TVT − Beam/NCC hybrid TVT\|。 |
| 178 | `copcf_typewell_native_overlap_1_own_z_gt2p0_corr_abs_c40` | `cluster_prior_confidence` | 11.2±7.6 | 0.033% | 所属cluster距離z-score>2時にnative-overlap閾値1.0のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 179 | `copcf_typewell_native_overlap_1_any_outlier_signal_k8_gate_x_dense_family` | `cluster_prior_confidence` | 10.8±6.9 | 0.032% | native-overlap閾値1.0のtypewell cluster priorが有効、K=8近傍のいずれかのoutlier signal、かつ対象候補がdense familyなら1。 |
| 180 | `tvt_densew_minus_last` | `candidate_path_value` | 10.8±4.9 | 0.032% | dense spatial ANCC（prefix加重bias） TVT − last_known_tvt。 |
| 181 | `candidate_range` | `candidate_identity_or_context` | 10.6±4.7 | 0.031% | 同じbase rowにある候補TVT群の最大−最小。候補間disagreement。 |
| 182 | `copcf_spatial_xy_plus_trajectory_shape_k8_valid_prior` | `cluster_prior_confidence` | 10.6±7.1 | 0.031% | XY距離とtrajectory shapeを使うK=8 spatial priorが計算可能なら1。 |
| 183 | `copcf_typewell_native_overlap_1_own_z_gt2p0_corr_abs_c20` | `cluster_prior_confidence` | 9.0±5.5 | 0.026% | 所属cluster距離z-score>2時にnative-overlap閾値1.0のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 184 | `multiobs_mae_pf_ancc` | `multi_observation` | 8.4±4.4 | 0.025% | ANCC PFのmulti-observation GR照合MAE。 |
| 185 | `copcf_spatial_xy_only_k8_nearest_other_closer_gate_x_dense_family` | `cluster_prior_confidence` | 7.6±5.3 | 0.022% | XY距離だけを使うK=8 spatial priorが有効、別clusterの方が近い、かつ対象候補がdense familyなら1。 |
| 186 | `copcf_typewell_native_overlap_1_neighbor_wells_x_candidate` | `cluster_prior_confidence` | 7.6±4.9 | 0.022% | native-overlap閾値1.0のtypewell cluster priorの近傍well数をcandidate-long各行へ複製した列。積ではない。 |
| 187 | `copcf_nearby_majority_diff_k8` | `cluster_prior_confidence` | 7.4±3.2 | 0.022% | 近傍8 wellの多数派clusterが自身の割当clusterと異なれば1。 |
| 188 | `multiobs_mae_likpf_mean` | `multi_observation` | 7.2±1.8 | 0.021% | likelihood-weighted PF平均のmulti-observation GR照合MAE。 |
| 189 | `copcf_spatial_xy_plus_trajectory_shape_k8_any_outlier_signal_k8_gate_x_dense_family` | `cluster_prior_confidence` | 7.0±2.5 | 0.021% | XY距離とtrajectory shapeを使うK=8 spatial priorが有効、K=8近傍のいずれかのoutlier signal、かつ対象候補がdense familyなら1。 |
| 190 | `copcf_nearby_majority_share_k5` | `cluster_prior_confidence` | 6.8±3.1 | 0.020% | 近傍5 wellの多数派cluster比率。 |
| 191 | `copcf_spatial_xy_plus_trajectory_shape_k8_own_z_gt2p0_gate_x_dense_family` | `cluster_prior_confidence` | 6.8±9.1 | 0.020% | XY距離とtrajectory shapeを使うK=8 spatial priorが有効、所属cluster距離z-score>2、かつ対象候補がdense familyなら1。 |
| 192 | `pf_ancc_vs_hyb_abs` | `candidate_disagreement` | 6.4±1.3 | 0.019% | \|ANCC PF TVT − Beam/NCC hybrid TVT\|。 |
| 193 | `beam_mean_vs_sc_ens_abs` | `candidate_disagreement` | 5.8±2.2 | 0.017% | \|複数Beam path平均 TVT − multi-scale NCC ensemble TVT\|。 |
| 194 | `copcf_typewell_native_overlap_1_count_x_candidate` | `cluster_prior_confidence` | 5.8±1.8 | 0.017% | native-overlap閾値1.0のtypewell cluster priorのsource数をcandidate-long各行へ複製した列。積ではない。 |
| 195 | `sc_ens_minus_last` | `candidate_path_value` | 5.8±7.8 | 0.017% | multi-scale NCC ensemble TVT − last_known_tvt。 |
| 196 | `copcf_typewell_native_overlap_1_nearby_majority_diff_k8_gate_x_dense_family` | `cluster_prior_confidence` | 5.4±3.8 | 0.016% | native-overlap閾値1.0のtypewell cluster priorが有効、K=8近傍の多数派cluster不一致、かつ対象候補がdense familyなら1。 |
| 197 | `self_gr_quality` | `self_gr_confidence` | 5.4±2.2 | 0.016% | 同一horizontal内Self-GR motif照合の品質score。 |
| 198 | `beam_mean_vs_hyb_abs` | `candidate_disagreement` | 5.0±2.1 | 0.015% | \|複数Beam path平均 TVT − Beam/NCC hybrid TVT\|。 |
| 199 | `hyb_vs_tvt_dense50_abs` | `candidate_disagreement` | 5.0±2.0 | 0.015% | \|Beam/NCC hybrid TVT − dense spatial ANCC（prefix末尾50 bias） TVT\|。 |
| 200 | `hyb_vs_tvt_dense_abs` | `candidate_disagreement` | 4.8±4.3 | 0.014% | \|Beam/NCC hybrid TVT − dense spatial ANCC（full-prefix bias） TVT\|。 |
| 201 | `copcf_gate_nearby_majority_diff_k8` | `cluster_prior_confidence` | 4.6±2.5 | 0.014% | K=8近傍多数派不一致のgate flag。 |
| 202 | `copcf_typewell_native_overlap_1_own_z_gt2p0_gate_x_dense_family` | `cluster_prior_confidence` | 4.4±1.5 | 0.013% | native-overlap閾値1.0のtypewell cluster priorが有効、所属cluster距離z-score>2、かつ対象候補がdense familyなら1。 |
| 203 | `sc_ens_vs_tvt_dense50_abs` | `candidate_disagreement` | 4.4±1.7 | 0.013% | \|multi-scale NCC ensemble TVT − dense spatial ANCC（prefix末尾50 bias） TVT\|。 |
| 204 | `copcf_spatial_xy_only_k8_valid_prior` | `cluster_prior_confidence` | 4.2±3.0 | 0.012% | XY距離だけを使うK=8 spatial priorが計算可能なら1。 |
| 205 | `copcf_spatial_xy_plus_trajectory_shape_k8_count_x_candidate` | `cluster_prior_confidence` | 4.2±1.6 | 0.012% | XY距離とtrajectory shapeを使うK=8 spatial priorのsource数をcandidate-long各行へ複製した列。積ではない。 |
| 206 | `copcf_typewell_native_overlap_1_any_outlier_signal_k8_gate_x_candidate` | `cluster_prior_confidence` | 4.2±1.9 | 0.012% | native-overlap閾値1.0のtypewell cluster priorが有効かつK=8近傍のいずれかのoutlier signalなら1。candidate-long行へ複製するgate。 |
| 207 | `copcf_spatial_xy_only_k8_nearest_other_closer_clip_hit_c40` | `cluster_prior_confidence` | 4.0±3.5 | 0.012% | 別clusterの方が近い時のprior−候補差がclip 40 ftを超えれば1。 |
| 208 | `copcf_spatial_xy_only_k8_own_z_gt2p0_gate_x_dense_family` | `cluster_prior_confidence` | 4.0±3.2 | 0.012% | XY距離だけを使うK=8 spatial priorが有効、所属cluster距離z-score>2、かつ対象候補がdense familyなら1。 |
| 209 | `copcf_typewell_native_overlap_1_std_x_candidate` | `cluster_prior_confidence` | 4.0±2.5 | 0.012% | native-overlap閾値1.0のtypewell cluster priorの標準偏差をcandidate-long各行へ複製した列。積ではない。 |
| 210 | `copcf_spatial_xy_plus_trajectory_shape_k8_nearby_majority_diff_k8_gate_x_candidate` | `cluster_prior_confidence` | 3.8±1.9 | 0.011% | XY距離とtrajectory shapeを使うK=8 spatial priorが有効かつK=8近傍の多数派cluster不一致なら1。candidate-long行へ複製するgate。 |
| 211 | `copcf_typewell_native_overlap_1_nearby_majority_diff_k8_gate_x_candidate` | `cluster_prior_confidence` | 3.8±2.4 | 0.011% | native-overlap閾値1.0のtypewell cluster priorが有効かつK=8近傍の多数派cluster不一致なら1。candidate-long行へ複製するgate。 |
| 212 | `copcf_gate_any_outlier_signal_k8` | `cluster_prior_confidence` | 3.6±2.5 | 0.011% | K=8 outlier signalの総合gate flag。 |
| 213 | `copcf_spatial_xy_only_k8_any_outlier_signal_k8_gate_x_dense_family` | `cluster_prior_confidence` | 3.6±1.1 | 0.011% | XY距離だけを使うK=8 spatial priorが有効、K=8近傍のいずれかのoutlier signal、かつ対象候補がdense familyなら1。 |
| 214 | `copcf_spatial_xy_only_k8_nearby_majority_diff_k8_gate_x_candidate` | `cluster_prior_confidence` | 3.6±2.6 | 0.011% | XY距離だけを使うK=8 spatial priorが有効かつK=8近傍の多数派cluster不一致なら1。candidate-long行へ複製するgate。 |
| 215 | `copcf_typewell_native_overlap_0p999_nearest_other_closer_corr_abs_c40` | `cluster_prior_confidence` | 3.6±1.7 | 0.011% | 別clusterの方が近い時にnative-overlap閾値0.999のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 216 | `copcf_nearest_other_closer` | `cluster_prior_confidence` | 3.4±2.3 | 0.010% | 最も近い別clusterが割当先clusterより近ければ1。 |
| 217 | `hyb_vs_tvt_densew_abs` | `candidate_disagreement` | 3.2±1.1 | 0.009% | \|Beam/NCC hybrid TVT − dense spatial ANCC（prefix加重bias） TVT\|。 |
| 218 | `copcf_typewell_native_overlap_0p999_any_outlier_signal_k8_corr_abs_c40` | `cluster_prior_confidence` | 3.0±2.3 | 0.009% | K=8近傍のいずれかのoutlier signal時にnative-overlap閾値0.999のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 219 | `copcf_spatial_xy_only_k8_nearest_other_closer_gate_x_candidate` | `cluster_prior_confidence` | 2.8±1.6 | 0.008% | XY距離だけを使うK=8 spatial priorが有効かつ別clusterの方が近いなら1。candidate-long行へ複製するgate。 |
| 220 | `copcf_typewell_native_overlap_1_nearest_other_closer_gate_x_candidate` | `cluster_prior_confidence` | 2.8±1.3 | 0.008% | native-overlap閾値1.0のtypewell cluster priorが有効かつ別clusterの方が近いなら1。candidate-long行へ複製するgate。 |
| 221 | `multiobs_mae_sc_ens` | `multi_observation` | 2.6±2.6 | 0.008% | multi-scale NCC ensembleのmulti-observation GR照合MAE。 |
| 222 | `copcf_typewell_native_overlap_0p999_nearby_majority_diff_k8_corr_abs_c40` | `cluster_prior_confidence` | 2.4±1.5 | 0.007% | K=8近傍の多数派cluster不一致時にnative-overlap閾値0.999のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 223 | `crfe_near_md_50_flag` | `disagreement_enrichment` | 2.4±3.6 | 0.007% | anchorからMD 50以内なら1。 |
| 224 | `sc_ens_vs_tvt_dense_abs` | `candidate_disagreement` | 2.4±1.3 | 0.007% | \|multi-scale NCC ensemble TVT − dense spatial ANCC（full-prefix bias） TVT\|。 |
| 225 | `copcf_spatial_xy_plus_trajectory_shape_k8_nearest_other_closer_gate_x_candidate` | `cluster_prior_confidence` | 2.2±1.8 | 0.006% | XY距離とtrajectory shapeを使うK=8 spatial priorが有効かつ別clusterの方が近いなら1。candidate-long行へ複製するgate。 |
| 226 | `hyb_minus_last` | `candidate_path_value` | 2.2±0.8 | 0.006% | Beam/NCC hybrid TVT − last_known_tvt。 |
| 227 | `copcf_typewell_native_overlap_0p999_any_outlier_signal_k8_corr_abs_c20` | `cluster_prior_confidence` | 2.0±1.6 | 0.006% | K=8近傍のいずれかのoutlier signal時にnative-overlap閾値0.999のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 228 | `copcf_typewell_native_overlap_0p999_nearby_majority_diff_k8_corr_abs_c20` | `cluster_prior_confidence` | 2.0±1.2 | 0.006% | K=8近傍の多数派cluster不一致時にnative-overlap閾値0.999のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 229 | `copcf_typewell_native_overlap_0p999_valid_prior` | `cluster_prior_confidence` | 2.0±1.2 | 0.006% | native-overlap閾値0.999のtypewell cluster priorが計算可能なら1。 |
| 230 | `multiobs_mae_hyb` | `multi_observation` | 2.0±2.5 | 0.006% | Beam/NCC hybridのmulti-observation GR照合MAE。 |
| 231 | `sc_ens_vs_tvt_densew_abs` | `candidate_disagreement` | 2.0±2.3 | 0.006% | \|multi-scale NCC ensemble TVT − dense spatial ANCC（prefix加重bias） TVT\|。 |
| 232 | `copcf_spatial_xy_only_k8_any_outlier_signal_k8_gate_x_candidate` | `cluster_prior_confidence` | 1.8±1.3 | 0.005% | XY距離だけを使うK=8 spatial priorが有効かつK=8近傍のいずれかのoutlier signalなら1。candidate-long行へ複製するgate。 |
| 233 | `copcf_typewell_native_overlap_0p999_nearest_other_closer_corr_abs_c20` | `cluster_prior_confidence` | 1.8±0.8 | 0.005% | 別clusterの方が近い時にnative-overlap閾値0.999のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 234 | `candidate_is_geometry_family` | `candidate_identity_or_context` | 1.6±1.3 | 0.005% | 対象候補がexp226 geometry候補なら1。 |
| 235 | `copcf_cluster_feature_valid` | `cluster_prior_confidence` | 1.6±1.1 | 0.005% | cluster/outlier診断が計算可能なら1。 |
| 236 | `copcf_spatial_xy_plus_trajectory_shape_k8_own_z_gt2p0_gate_x_candidate` | `cluster_prior_confidence` | 1.6±1.3 | 0.005% | XY距離とtrajectory shapeを使うK=8 spatial priorが有効かつ所属cluster距離z-score>2なら1。candidate-long行へ複製するgate。 |
| 237 | `copcf_spatial_xy_plus_trajectory_shape_k8_any_outlier_signal_k8_gate_x_candidate` | `cluster_prior_confidence` | 1.4±1.1 | 0.004% | XY距離とtrajectory shapeを使うK=8 spatial priorが有効かつK=8近傍のいずれかのoutlier signalなら1。candidate-long行へ複製するgate。 |
| 238 | `candidate_is_pfbeam_family` | `candidate_identity_or_context` | 1.2±2.2 | 0.004% | 対象候補がPF/Beam系なら1。 |
| 239 | `copcf_typewell_native_overlap_0p999_any_outlier_signal_k8_gate_x_dense_family` | `cluster_prior_confidence` | 1.2±1.3 | 0.004% | native-overlap閾値0.999のtypewell cluster priorが有効、K=8近傍のいずれかのoutlier signal、かつ対象候補がdense familyなら1。 |
| 240 | `copcf_typewell_native_overlap_0p999_own_z_gt2p0_corr_abs_c20` | `cluster_prior_confidence` | 1.2±0.8 | 0.004% | 所属cluster距離z-score>2時にnative-overlap閾値0.999のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 241 | `copcf_typewell_native_overlap_0p999_std_x_candidate` | `cluster_prior_confidence` | 1.2±1.8 | 0.004% | native-overlap閾値0.999のtypewell cluster priorの標準偏差をcandidate-long各行へ複製した列。積ではない。 |
| 242 | `copcf_typewell_native_overlap_1_own_z_gt2p0_gate_x_candidate` | `cluster_prior_confidence` | 1.2±0.8 | 0.004% | native-overlap閾値1.0のtypewell cluster priorが有効かつ所属cluster距離z-score>2なら1。candidate-long行へ複製するgate。 |
| 243 | `copcf_gate_nearest_other_closer` | `cluster_prior_confidence` | 1.0±0.7 | 0.003% | 別clusterの方が近いgate flag。 |
| 244 | `copcf_spatial_xy_only_k8_own_z_gt2p0_gate_x_candidate` | `cluster_prior_confidence` | 1.0±0.7 | 0.003% | XY距離だけを使うK=8 spatial priorが有効かつ所属cluster距離z-score>2なら1。candidate-long行へ複製するgate。 |
| 245 | `copcf_typewell_native_overlap_0p999_neighbor_wells_x_candidate` | `cluster_prior_confidence` | 1.0±1.0 | 0.003% | native-overlap閾値0.999のtypewell cluster priorの近傍well数をcandidate-long各行へ複製した列。積ではない。 |
| 246 | `copcf_well_gate_ratio_nearby_majority_diff_k8` | `cluster_prior_confidence` | 1.0±1.2 | 0.003% | well内で近傍多数派不一致gateが有効な行の比率。 |
| 247 | `multiobs_score_beam_mean` | `multi_observation` | 1.0±1.2 | 0.003% | 複数Beam path平均のmulti-observation GR一致score。 |
| 248 | `multiobs_score_sc_ens` | `multi_observation` | 1.0±1.4 | 0.003% | multi-scale NCC ensembleのmulti-observation GR一致score。 |
| 249 | `candidate_multiobs_ncc` | `multi_observation` | 0.8±1.3 | 0.002% | 対象候補自身のmulti-observation GR照合NCC。 |
| 250 | `copcf_spatial_xy_plus_trajectory_shape_k8_any_outlier_signal_k8_clip_hit_c40` | `cluster_prior_confidence` | 0.8±0.4 | 0.002% | K=8近傍のいずれかのoutlier signal時のprior−候補差がclip 40 ftを超えれば1。 |
| 251 | `copcf_spatial_xy_plus_trajectory_shape_k8_nearest_other_closer_clip_hit_c40` | `cluster_prior_confidence` | 0.8±1.3 | 0.002% | 別clusterの方が近い時のprior−候補差がclip 40 ftを超えれば1。 |
| 252 | `multiobs_score_likpf_mean` | `multi_observation` | 0.8±0.8 | 0.002% | likelihood-weighted PF平均のmulti-observation GR一致score。 |
| 253 | `copcf_typewell_native_overlap_0p999_nearby_majority_diff_k8_gate_x_dense_family` | `cluster_prior_confidence` | 0.6±0.5 | 0.002% | native-overlap閾値0.999のtypewell cluster priorが有効、K=8近傍の多数派cluster不一致、かつ対象候補がdense familyなら1。 |
| 254 | `copcf_typewell_native_overlap_0p999_nearest_other_closer_gate_x_dense_family` | `cluster_prior_confidence` | 0.6±0.9 | 0.002% | native-overlap閾値0.999のtypewell cluster priorが有効、別clusterの方が近い、かつ対象候補がdense familyなら1。 |
| 255 | `copcf_typewell_native_overlap_0p999_own_z_gt2p0_corr_abs_c40` | `cluster_prior_confidence` | 0.6±0.9 | 0.002% | 所属cluster距離z-score>2時にnative-overlap閾値0.999のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 256 | `copcf_typewell_native_overlap_0p999_own_z_gt2p0_gate_x_dense_family` | `cluster_prior_confidence` | 0.6±0.5 | 0.002% | native-overlap閾値0.999のtypewell cluster priorが有効、所属cluster距離z-score>2、かつ対象候補がdense familyなら1。 |
| 257 | `copcf_typewell_native_overlap_1_nearest_other_closer_clip_hit_c40` | `cluster_prior_confidence` | 0.6±0.9 | 0.002% | 別clusterの方が近い時のprior−候補差がclip 40 ftを超えれば1。 |
| 258 | `copcf_gate_own_z_gt2p0` | `cluster_prior_confidence` | 0.4±0.5 | 0.001% | 所属cluster距離z-score>2のgate flag。 |
| 259 | `copcf_typewell_native_overlap_0p999_any_outlier_signal_k8_gate_x_candidate` | `cluster_prior_confidence` | 0.4±0.5 | 0.001% | native-overlap閾値0.999のtypewell cluster priorが有効かつK=8近傍のいずれかのoutlier signalなら1。candidate-long行へ複製するgate。 |
| 260 | `copcf_typewell_native_overlap_0p999_count_x_candidate` | `cluster_prior_confidence` | 0.4±0.9 | 0.001% | native-overlap閾値0.999のtypewell cluster priorのsource数をcandidate-long各行へ複製した列。積ではない。 |
| 261 | `copcf_typewell_native_overlap_1_any_outlier_signal_k8_clip_hit_c40` | `cluster_prior_confidence` | 0.4±0.5 | 0.001% | K=8近傍のいずれかのoutlier signal時のprior−候補差がclip 40 ftを超えれば1。 |
| 262 | `multiobs_score_mean` | `multi_observation` | 0.4±0.5 | 0.001% | 候補bankのmulti-observation GR一致score平均。 |
| 263 | `multiobs_score_pf_ancc` | `multi_observation` | 0.4±0.5 | 0.001% | ANCC PFのmulti-observation GR一致score。 |
| 264 | `multiobs_top1_mae` | `multi_observation` | 0.4±0.9 | 0.001% | multi-observation score最上位候補のGR照合MAE。 |
| 265 | `copcf_any_configured_gate` | `cluster_prior_confidence` | 0.2±0.4 | 0.001% | 設定済みcluster/outlier gateのいずれかが有効なら1。 |
| 266 | `copcf_spatial_xy_only_k8_nearby_majority_diff_k8_clip_hit_c40` | `cluster_prior_confidence` | 0.2±0.4 | 0.001% | K=8近傍の多数派cluster不一致時のprior−候補差がclip 40 ftを超えれば1。 |
| 267 | `copcf_spatial_xy_only_k8_nearest_other_closer_clip_hit_c20` | `cluster_prior_confidence` | 0.2±0.4 | 0.001% | 別clusterの方が近い時のprior−候補差がclip 20 ftを超えれば1。 |
| 268 | `copcf_spatial_xy_only_k8_own_z_gt2p0_clip_hit_c40` | `cluster_prior_confidence` | 0.2±0.4 | 0.001% | 所属cluster距離z-score>2時のprior−候補差がclip 40 ftを超えれば1。 |
| 269 | `copcf_spatial_xy_plus_trajectory_shape_k8_any_outlier_signal_k8_clip_hit_c20` | `cluster_prior_confidence` | 0.2±0.4 | 0.001% | K=8近傍のいずれかのoutlier signal時のprior−候補差がclip 20 ftを超えれば1。 |
| 270 | `copcf_spatial_xy_plus_trajectory_shape_k8_nearby_majority_diff_k8_clip_hit_c20` | `cluster_prior_confidence` | 0.2±0.4 | 0.001% | K=8近傍の多数派cluster不一致時のprior−候補差がclip 20 ftを超えれば1。 |
| 271 | `copcf_typewell_native_overlap_1_nearby_majority_diff_k8_clip_hit_c20` | `cluster_prior_confidence` | 0.2±0.4 | 0.001% | K=8近傍の多数派cluster不一致時のprior−候補差がclip 20 ftを超えれば1。 |
| 272 | `copcf_typewell_native_overlap_1_valid_prior` | `cluster_prior_confidence` | 0.2±0.4 | 0.001% | native-overlap閾値1.0のtypewell cluster priorが計算可能なら1。 |
| 273 | `copcf_well_gate_ratio_any_outlier_signal_k8` | `cluster_prior_confidence` | 0.2±0.4 | 0.001% | well内でany-outlier gateが有効な行の比率。 |
| 274 | `multiobs_ncc_beam_mean` | `multi_observation` | 0.2±0.4 | 0.001% | 複数Beam path平均のmulti-observation GR照合NCC。 |
| 275 | `multiobs_score_hyb` | `multi_observation` | 0.2±0.4 | 0.001% | Beam/NCC hybridのmulti-observation GR一致score。 |
| 276 | `multiobs_top1_source_id` | `multi_observation` | 0.2±0.4 | 0.001% | multi-observation score最上位候補のsource code。 |
| 277 | `self_gr_valid` | `self_gr_confidence` | 0.2±0.4 | 0.001% | Self-GR照合が入力条件を満たして有効なら1。 |
| 278 | `copcf_spatial_xy_only_k8_any_outlier_signal_k8_clip_hit_c20` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | K=8近傍のいずれかのoutlier signal時のprior−候補差がclip 20 ftを超えれば1。 |
| 279 | `copcf_spatial_xy_only_k8_any_outlier_signal_k8_clip_hit_c40` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | K=8近傍のいずれかのoutlier signal時のprior−候補差がclip 40 ftを超えれば1。 |
| 280 | `copcf_spatial_xy_only_k8_nearby_majority_diff_k8_clip_hit_c20` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | K=8近傍の多数派cluster不一致時のprior−候補差がclip 20 ftを超えれば1。 |
| 281 | `copcf_spatial_xy_only_k8_neighbor_wells` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | XY距離だけを使うK=8 spatial priorを作る近傍well数。 |
| 282 | `copcf_spatial_xy_only_k8_neighbor_wells_x_candidate` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | XY距離だけを使うK=8 spatial priorの近傍well数をcandidate-long各行へ複製した列。積ではない。 |
| 283 | `copcf_spatial_xy_only_k8_own_z_gt2p0_clip_hit_c20` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | 所属cluster距離z-score>2時のprior−候補差がclip 20 ftを超えれば1。 |
| 284 | `copcf_spatial_xy_only_k8_valid_x_candidate` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | XY距離だけを使うK=8 spatial priorのvalid flagをcandidate-long各行へ複製した列。積ではない。 |
| 285 | `copcf_spatial_xy_plus_trajectory_shape_k8_nearby_majority_diff_k8_clip_hit_c40` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | K=8近傍の多数派cluster不一致時のprior−候補差がclip 40 ftを超えれば1。 |
| 286 | `copcf_spatial_xy_plus_trajectory_shape_k8_nearest_other_closer_clip_hit_c20` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | 別clusterの方が近い時のprior−候補差がclip 20 ftを超えれば1。 |
| 287 | `copcf_spatial_xy_plus_trajectory_shape_k8_neighbor_wells` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | XY距離とtrajectory shapeを使うK=8 spatial priorを作る近傍well数。 |
| 288 | `copcf_spatial_xy_plus_trajectory_shape_k8_neighbor_wells_x_candidate` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | XY距離とtrajectory shapeを使うK=8 spatial priorの近傍well数をcandidate-long各行へ複製した列。積ではない。 |
| 289 | `copcf_spatial_xy_plus_trajectory_shape_k8_own_z_gt2p0_clip_hit_c20` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | 所属cluster距離z-score>2時のprior−候補差がclip 20 ftを超えれば1。 |
| 290 | `copcf_spatial_xy_plus_trajectory_shape_k8_own_z_gt2p0_clip_hit_c40` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | 所属cluster距離z-score>2時のprior−候補差がclip 40 ftを超えれば1。 |
| 291 | `copcf_spatial_xy_plus_trajectory_shape_k8_valid_x_candidate` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | XY距離とtrajectory shapeを使うK=8 spatial priorのvalid flagをcandidate-long各行へ複製した列。積ではない。 |
| 292 | `copcf_typewell_native_overlap_0p999_any_outlier_signal_k8_clip_hit_c20` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | K=8近傍のいずれかのoutlier signal時のprior−候補差がclip 20 ftを超えれば1。 |
| 293 | `copcf_typewell_native_overlap_0p999_any_outlier_signal_k8_clip_hit_c40` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | K=8近傍のいずれかのoutlier signal時のprior−候補差がclip 40 ftを超えれば1。 |
| 294 | `copcf_typewell_native_overlap_0p999_nearby_majority_diff_k8_clip_hit_c20` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | K=8近傍の多数派cluster不一致時のprior−候補差がclip 20 ftを超えれば1。 |
| 295 | `copcf_typewell_native_overlap_0p999_nearby_majority_diff_k8_clip_hit_c40` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | K=8近傍の多数派cluster不一致時のprior−候補差がclip 40 ftを超えれば1。 |
| 296 | `copcf_typewell_native_overlap_0p999_nearby_majority_diff_k8_gate_x_candidate` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | native-overlap閾値0.999のtypewell cluster priorが有効かつK=8近傍の多数派cluster不一致なら1。candidate-long行へ複製するgate。 |
| 297 | `copcf_typewell_native_overlap_0p999_nearest_other_closer_clip_hit_c20` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | 別clusterの方が近い時のprior−候補差がclip 20 ftを超えれば1。 |
| 298 | `copcf_typewell_native_overlap_0p999_nearest_other_closer_clip_hit_c40` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | 別clusterの方が近い時のprior−候補差がclip 40 ftを超えれば1。 |
| 299 | `copcf_typewell_native_overlap_0p999_nearest_other_closer_gate_x_candidate` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | native-overlap閾値0.999のtypewell cluster priorが有効かつ別clusterの方が近いなら1。candidate-long行へ複製するgate。 |
| 300 | `copcf_typewell_native_overlap_0p999_own_z_gt2p0_clip_hit_c20` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | 所属cluster距離z-score>2時のprior−候補差がclip 20 ftを超えれば1。 |
| 301 | `copcf_typewell_native_overlap_0p999_own_z_gt2p0_clip_hit_c40` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | 所属cluster距離z-score>2時のprior−候補差がclip 40 ftを超えれば1。 |
| 302 | `copcf_typewell_native_overlap_0p999_own_z_gt2p0_gate_x_candidate` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | native-overlap閾値0.999のtypewell cluster priorが有効かつ所属cluster距離z-score>2なら1。candidate-long行へ複製するgate。 |
| 303 | `copcf_typewell_native_overlap_0p999_valid_x_candidate` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | native-overlap閾値0.999のtypewell cluster priorのvalid flagをcandidate-long各行へ複製した列。積ではない。 |
| 304 | `copcf_typewell_native_overlap_1_any_outlier_signal_k8_clip_hit_c20` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | K=8近傍のいずれかのoutlier signal時のprior−候補差がclip 20 ftを超えれば1。 |
| 305 | `copcf_typewell_native_overlap_1_nearby_majority_diff_k8_clip_hit_c40` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | K=8近傍の多数派cluster不一致時のprior−候補差がclip 40 ftを超えれば1。 |
| 306 | `copcf_typewell_native_overlap_1_nearest_other_closer_clip_hit_c20` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | 別clusterの方が近い時のprior−候補差がclip 20 ftを超えれば1。 |
| 307 | `copcf_typewell_native_overlap_1_own_z_gt2p0_clip_hit_c20` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | 所属cluster距離z-score>2時のprior−候補差がclip 20 ftを超えれば1。 |
| 308 | `copcf_typewell_native_overlap_1_own_z_gt2p0_clip_hit_c40` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | 所属cluster距離z-score>2時のprior−候補差がclip 40 ftを超えれば1。 |
| 309 | `copcf_typewell_native_overlap_1_valid_x_candidate` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | native-overlap閾値1.0のtypewell cluster priorのvalid flagをcandidate-long各行へ複製した列。積ではない。 |
| 310 | `copcf_well_gate_ratio_nearest_other_closer` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | well内で別cluster近接gateが有効な行の比率。 |
| 311 | `copcf_well_gate_ratio_own_z_gt2p0` | `cluster_prior_confidence` | 0.0±0.0 | 0.000% | well内で所属cluster距離z>2 gateが有効な行の比率。 |
| 312 | `crfe_longtail_1000_flag` | `disagreement_enrichment` | 0.0±0.0 | 0.000% | anchorから1000行以上のlong-tailなら1。 |
| 313 | `multiobs_ncc_hyb` | `multi_observation` | 0.0±0.0 | 0.000% | Beam/NCC hybridのmulti-observation GR照合NCC。 |
| 314 | `multiobs_ncc_likpf_mean` | `multi_observation` | 0.0±0.0 | 0.000% | likelihood-weighted PF平均のmulti-observation GR照合NCC。 |
| 315 | `multiobs_ncc_pf_ancc` | `multi_observation` | 0.0±0.0 | 0.000% | ANCC PFのmulti-observation GR照合NCC。 |
| 316 | `multiobs_ncc_sc_ens` | `multi_observation` | 0.0±0.0 | 0.000% | multi-scale NCC ensembleのmulti-observation GR照合NCC。 |
| 317 | `multiobs_score_gap` | `multi_observation` | 0.0±0.0 | 0.000% | multi-observation GR一致scoreの1位−2位差。 |
| 318 | `multiobs_score_max` | `multi_observation` | 0.0±0.0 | 0.000% | 候補bankのmulti-observation GR一致score最大値。 |
| 319 | `multiobs_top1_ncc` | `multi_observation` | 0.0±0.0 | 0.000% | multi-observation score最上位候補のGR照合NCC。 |
| 320 | `self_gr_typewell_agreement` | `self_gr_confidence` | 0.0±0.0 | 0.000% | Self-GR観測とtypewell観測の整合度。 |

### exp237 importance family集計

| family | features | split share |
| --- | --- | --- |
| `cluster_prior_confidence` | 165 | 25.002% |
| `candidate_disagreement` | 55 | 22.701% |
| `candidate_identity_or_context` | 12 | 13.459% |
| `hmm_confidence` | 8 | 11.301% |
| `candidate_path_value` | 18 | 10.411% |
| `disagreement_enrichment` | 23 | 7.933% |
| `distance_or_anchor_context` | 3 | 3.507% |
| `pf_dense_confidence` | 3 | 2.467% |
| `exp226_geometry` | 5 | 2.446% |
| `multi_observation` | 24 | 0.693% |
| `self_gr_confidence` | 4 | 0.080% |

## exp251 v4 selector入力 295列 × 2目的

combined順位はexpected-errorとwithin10それぞれのsplit shareを平均したもの。片方だけの強さを隠さないよう、両目的の順位・split回数を併記する。

| comb rank | feature | family | comb share | error rank/split | within10 rank/split | 説明 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `candidate_z_within_view` | `raw_test_view_context` | 4.666% | 1 / 1486.0 | 1 / 742.8 | (対象候補TVT − view内候補平均) / view内候補標準偏差。 |
| 2 | `candidate_minus_last` | `candidate_identity_or_context` | 3.061% | 4 / 837.4 | 2 / 565.0 | この候補TVT − last_known_tvt。候補の外挿量。 |
| 3 | `copcf_nearest_other_cluster_dist` | `cluster_prior_confidence` | 3.060% | 3 / 840.8 | 3 / 562.6 | 最も近い別cluster中心までの距離。 |
| 4 | `candidate_index` | `candidate_identity_or_context` | 2.940% | 2 / 880.2 | 4 / 499.8 | 固定11候補bank内の候補index。候補identityを木に伝える数値code。 |
| 5 | `v6_k16_geometry_gr_u_projection_minus_last` | `candidate_path_value` | 2.780% | 5 / 829.8 | 5 / 474.0 | exp226 K16 geometry/GR/U-projection TVT − last_known_tvt。 |
| 6 | `tvt_densew_vs_tvt_dense50_abs` | `candidate_disagreement` | 2.509% | 6 / 711.4 | 6 / 449.0 | \|dense spatial ANCC（prefix加重bias） TVT − dense spatial ANCC（prefix末尾50 bias） TVT\|。 |
| 7 | `copcf_own_cluster_dist` | `cluster_prior_confidence` | 2.346% | 8 / 645.6 | 7 / 430.8 | test wellと割当先cluster中心の距離。 |
| 8 | `eval_len` | `distance_or_anchor_context` | 2.288% | 9 / 643.4 | 9 / 412.4 | そのwellの予測tail行数。 |
| 9 | `candidate_tvt` | `candidate_identity_or_context` | 2.254% | 11 / 605.0 | 8 / 422.6 | このcandidate-long行が表す候補パスの絶対TVT。 |
| 10 | `hmm_selfgr_boost_only_a070_c100_minus_last` | `hmm_confidence` | 2.243% | 7 / 678.6 | 11 / 377.4 | Self-GR likelihood HMM TVT − last_known_tvt。 |
| 11 | `copcf_own_cluster_dist_z` | `cluster_prior_confidence` | 2.146% | 12 / 589.2 | 10 / 394.8 | 割当先cluster距離のcluster内z-score。 |
| 12 | `hmm_exact_std` | `hmm_confidence` | 2.054% | 10 / 621.4 | 12 / 345.6 | exact HMM posteriorのTVT標準偏差。 |
| 13 | `hmm_exact_loglik` | `hmm_confidence` | 1.842% | 15 / 516.4 | 13 / 332.8 | exact HMMの行別観測log-likelihood。 |
| 14 | `hmm_selfgr_boost_only_a070_c100_vs_v6_k16_geometry_gr_u_projection_abs` | `hmm_confidence` | 1.791% | 13 / 586.4 | 18 / 276.0 | \|Self-GR likelihood HMM TVT − exp226 K16 geometry/GR/U-projection TVT\|。 |
| 15 | `tvt_dense_vs_tvt_densew_abs` | `candidate_disagreement` | 1.699% | 17 / 498.8 | 14 / 294.4 | \|dense spatial ANCC（full-prefix bias） TVT − dense spatial ANCC（prefix加重bias） TVT\|。 |
| 16 | `tvt_dense_vs_tvt_dense50_abs` | `candidate_disagreement` | 1.679% | 18 / 495.8 | 16 / 289.2 | \|dense spatial ANCC（full-prefix bias） TVT − dense spatial ANCC（prefix末尾50 bias） TVT\|。 |
| 17 | `candidate_abs_minus_view_mean` | `raw_test_view_context` | 1.652% | 14 / 547.6 | 21 / 250.8 | \|対象候補TVT − view内候補平均\|。 |
| 18 | `beam_mean_minus_last` | `candidate_path_value` | 1.621% | 19 / 492.4 | 19 / 271.6 | 複数Beam path平均 TVT − last_known_tvt。 |
| 19 | `md_since` | `distance_or_anchor_context` | 1.583% | 20 / 445.4 | 17 / 285.0 | 既知prefix末尾から予測行までのMD距離。 |
| 20 | `hmm_selfgr_std` | `hmm_confidence` | 1.577% | 16 / 508.2 | 22 / 247.8 | Self-GR likelihood HMM posteriorのTVT標準偏差。 |
| 21 | `hmm_selfgr_loglik` | `hmm_confidence` | 1.541% | 21 / 407.6 | 15 / 292.2 | Self-GR likelihood HMMの行別観測log-likelihood。 |
| 22 | `beam_mean_vs_hmm_selfgr_boost_only_a070_c100_abs` | `candidate_disagreement` | 1.335% | 22 / 375.4 | 23 / 240.6 | \|複数Beam path平均 TVT − Self-GR likelihood HMM TVT\|。 |
| 23 | `last_known_tvt` | `distance_or_anchor_context` | 1.314% | 25 / 365.2 | 24 / 239.2 | 既知prefix末尾のTVT。全候補deltaのanchor。 |
| 24 | `beam_mean_vs_v6_k16_geometry_gr_u_projection_abs` | `candidate_disagreement` | 1.276% | 31 / 309.0 | 20 / 258.0 | \|複数Beam path平均 TVT − exp226 K16 geometry/GR/U-projection TVT\|。 |
| 25 | `pf_ancc_minus_last` | `candidate_path_value` | 1.205% | 23 / 371.6 | 25 / 198.8 | ANCC PF TVT − last_known_tvt。 |
| 26 | `copcf_spatial_xy_plus_trajectory_shape_k8_minus_candidate` | `cluster_prior_confidence` | 1.132% | 26 / 353.0 | 27 / 184.6 | XY距離とtrajectory shapeを使うK=8 spatial prior TVT − 対象候補TVT。 |
| 27 | `blend_likpf_hmm_w500_minus_last` | `candidate_path_value` | 1.125% | 24 / 367.6 | 30 / 173.8 | likPF / exact-HMM 50:50 blend TVT − last_known_tvt。 |
| 28 | `copcf_typewell_native_overlap_0p999_neighbor_wells` | `cluster_prior_confidence` | 1.113% | 29 / 323.6 | 26 / 194.6 | native-overlap閾値0.999のtypewell cluster priorを作る近傍well数。 |
| 29 | `copcf_spatial_xy_only_k8_minus_candidate` | `cluster_prior_confidence` | 1.077% | 28 / 325.2 | 28 / 181.6 | XY距離だけを使うK=8 spatial prior TVT − 対象候補TVT。 |
| 30 | `pf_ancc_vs_v6_k16_geometry_gr_u_projection_abs` | `candidate_disagreement` | 1.031% | 30 / 322.0 | 31 / 167.6 | \|ANCC PF TVT − exp226 K16 geometry/GR/U-projection TVT\|。 |
| 31 | `pf_ancc_vs_hmm_selfgr_boost_only_a070_c100_abs` | `candidate_disagreement` | 1.005% | 27 / 340.4 | 35 / 148.6 | \|ANCC PF TVT − Self-GR likelihood HMM TVT\|。 |
| 32 | `copcf_spatial_xy_plus_trajectory_shape_k8_minus_candidate_abs` | `cluster_prior_confidence` | 0.990% | 36 / 283.0 | 29 / 175.8 | \|XY距離とtrajectory shapeを使うK=8 spatial prior TVT − 対象候補TVT\|。 |
| 33 | `likpf_mean_minus_last` | `candidate_path_value` | 0.962% | 33 / 295.4 | 33 / 159.2 | likelihood-weighted PF平均 TVT − last_known_tvt。 |
| 34 | `likpf_mean_vs_v6_k16_geometry_gr_u_projection_abs` | `candidate_disagreement` | 0.925% | 32 / 307.2 | 40 / 140.2 | \|likelihood-weighted PF平均 TVT − exp226 K16 geometry/GR/U-projection TVT\|。 |
| 35 | `likpf_mean_vs_blend_likpf_hmm_w500_abs` | `candidate_disagreement` | 0.876% | 37 / 277.8 | 39 / 140.2 | \|likelihood-weighted PF平均 TVT − likPF / exact-HMM 50:50 blend TVT\|。 |
| 36 | `tvt_dense_vs_v6_k16_geometry_gr_u_projection_abs` | `candidate_disagreement` | 0.855% | 41 / 245.8 | 34 / 151.0 | \|dense spatial ANCC（full-prefix bias） TVT − exp226 K16 geometry/GR/U-projection TVT\|。 |
| 37 | `pf_ancc_vs_beam_mean_abs` | `candidate_disagreement` | 0.835% | 40 / 247.2 | 37 / 143.6 | \|ANCC PF TVT − 複数Beam path平均 TVT\|。 |
| 38 | `blend_likpf_hmm_w500_vs_v6_k16_geometry_gr_u_projection_abs` | `candidate_disagreement` | 0.829% | 35 / 284.8 | 53 / 120.4 | \|likPF / exact-HMM 50:50 blend TVT − exp226 K16 geometry/GR/U-projection TVT\|。 |
| 39 | `copcf_spatial_xy_plus_trajectory_shape_k8_minus_candidate_abs_norm` | `cluster_prior_confidence` | 0.822% | 56 / 206.0 | 32 / 162.2 | \|XY距離とtrajectory shapeを使うK=8 spatial prior TVT − 対象候補TVT\|をprior標準偏差で正規化。 |
| 40 | `tvt_dense_vs_hmm_selfgr_boost_only_a070_c100_abs` | `candidate_disagreement` | 0.821% | 46 / 236.0 | 36 / 145.0 | \|dense spatial ANCC（full-prefix bias） TVT − Self-GR likelihood HMM TVT\|。 |
| 41 | `tvt_dense_minus_last` | `candidate_path_value` | 0.807% | 34 / 285.4 | 57 / 112.4 | dense spatial ANCC（full-prefix bias） TVT − last_known_tvt。 |
| 42 | `pf_ancc_vs_likpf_mean_abs` | `candidate_disagreement` | 0.804% | 43 / 243.4 | 44 / 135.2 | \|ANCC PF TVT − likelihood-weighted PF平均 TVT\|。 |
| 43 | `copcf_typewell_native_overlap_0p999_prior_std` | `cluster_prior_confidence` | 0.803% | 44 / 241.2 | 42 / 136.2 | native-overlap閾値0.999のtypewell cluster priorを作るsource TVTの標準偏差。 |
| 44 | `beam_mean_vs_blend_likpf_hmm_w500_abs` | `candidate_disagreement` | 0.792% | 39 / 250.4 | 48 / 127.0 | \|複数Beam path平均 TVT − likPF / exact-HMM 50:50 blend TVT\|。 |
| 45 | `exp226_v6_k16_geometry_gr_u_projection` | `exp226_geometry` | 0.772% | 49 / 223.2 | 43 / 135.8 | exp226 K16 geometry/GR/U-projectionの絶対TVT候補。 |
| 46 | `copcf_typewell_native_overlap_1_minus_candidate` | `cluster_prior_confidence` | 0.760% | 42 / 245.2 | 54 / 119.2 | native-overlap閾値1.0のtypewell cluster prior TVT − 対象候補TVT。 |
| 47 | `copcf_spatial_xy_plus_trajectory_shape_k8_prior_std` | `cluster_prior_confidence` | 0.755% | 45 / 236.4 | 50 / 122.6 | XY距離とtrajectory shapeを使うK=8 spatial priorを作るsource TVTの標準偏差。 |
| 48 | `blend_likpf_hmm_w500_vs_hmm_selfgr_boost_only_a070_c100_abs` | `candidate_disagreement` | 0.752% | 38 / 257.4 | 58 / 109.6 | \|likPF / exact-HMM 50:50 blend TVT − Self-GR likelihood HMM TVT\|。 |
| 49 | `blend_likpf_hmm_w500` | `candidate_path_value` | 0.749% | 51 / 218.0 | 46 / 131.0 | likPF / exact-HMM 50:50 blendの絶対TVT候補。 |
| 50 | `copcf_typewell_native_overlap_1_prior_count` | `cluster_prior_confidence` | 0.739% | 53 / 209.0 | 45 / 132.4 | native-overlap閾値1.0のtypewell cluster priorを作る有効source数。 |
| 51 | `pf_ancc_vs_blend_likpf_hmm_w500_abs` | `candidate_disagreement` | 0.728% | 47 / 235.4 | 56 / 113.8 | \|ANCC PF TVT − likPF / exact-HMM 50:50 blend TVT\|。 |
| 52 | `hmm_selfgr_boost_only_a070_c100_mean_tvt` | `hmm_confidence` | 0.707% | 55 / 207.6 | 51 / 122.4 | Self-GR likelihood HMMの絶対TVT候補。 |
| 53 | `copcf_spatial_xy_only_k8_prior_std` | `cluster_prior_confidence` | 0.704% | 57 / 199.2 | 49 / 126.2 | XY距離だけを使うK=8 spatial priorを作るsource TVTの標準偏差。 |
| 54 | `beam_mean_vs_likpf_mean_abs` | `candidate_disagreement` | 0.693% | 58 / 190.0 | 47 / 127.8 | \|複数Beam path平均 TVT − likelihood-weighted PF平均 TVT\|。 |
| 55 | `copcf_spatial_xy_only_k8_minus_candidate_abs_norm` | `cluster_prior_confidence` | 0.692% | 63 / 162.2 | 38 / 143.0 | \|XY距離だけを使うK=8 spatial prior TVT − 対象候補TVT\|をprior標準偏差で正規化。 |
| 56 | `copcf_typewell_native_overlap_1_minus_candidate_abs` | `cluster_prior_confidence` | 0.674% | 66 / 159.0 | 41 / 138.8 | \|native-overlap閾値1.0のtypewell cluster prior TVT − 対象候補TVT\|。 |
| 57 | `copcf_typewell_spatial_prior_delta` | `cluster_prior_confidence` | 0.670% | 48 / 230.0 | 61 / 97.2 | typewell prior − spatial prior。2種類の補正方向の差。 |
| 58 | `likpf_mean_vs_hmm_selfgr_boost_only_a070_c100_abs` | `candidate_disagreement` | 0.657% | 52 / 209.8 | 59 / 104.2 | \|likelihood-weighted PF平均 TVT − Self-GR likelihood HMM TVT\|。 |
| 59 | `copcf_spatial_xy_only_k8_minus_candidate_abs` | `cluster_prior_confidence` | 0.637% | 60 / 176.8 | 55 / 116.2 | \|XY距離だけを使うK=8 spatial prior TVT − 対象候補TVT\|。 |
| 60 | `tvt_dense50_vs_hmm_selfgr_boost_only_a070_c100_abs` | `candidate_disagreement` | 0.630% | 50 / 218.8 | 65 / 90.2 | \|dense spatial ANCC（prefix末尾50 bias） TVT − Self-GR likelihood HMM TVT\|。 |
| 61 | `copcf_typewell_native_overlap_1_minus_candidate_abs_norm` | `cluster_prior_confidence` | 0.627% | 64 / 161.8 | 52 / 121.2 | \|native-overlap閾値1.0のtypewell cluster prior TVT − 対象候補TVT\|をprior標準偏差で正規化。 |
| 62 | `pf_ancc_vs_tvt_dense_abs` | `candidate_disagreement` | 0.582% | 61 / 170.8 | 60 / 101.0 | \|ANCC PF TVT − dense spatial ANCC（full-prefix bias） TVT\|。 |
| 63 | `tvt_densew_vs_hmm_selfgr_boost_only_a070_c100_abs` | `candidate_disagreement` | 0.580% | 54 / 208.0 | 68 / 79.2 | \|dense spatial ANCC（prefix加重bias） TVT − Self-GR likelihood HMM TVT\|。 |
| 64 | `tvt_densew_vs_v6_k16_geometry_gr_u_projection_abs` | `candidate_disagreement` | 0.554% | 65 / 161.6 | 62 / 96.6 | \|dense spatial ANCC（prefix加重bias） TVT − exp226 K16 geometry/GR/U-projection TVT\|。 |
| 65 | `copcf_nearby_majority_count_k12` | `cluster_prior_confidence` | 0.521% | 68 / 146.0 | 64 / 94.2 | 近傍12 well中、多数派clusterに属するwell数。 |
| 66 | `candidate_name_code` | `candidate_identity_or_context` | 0.510% | 71 / 139.0 | 63 / 94.4 | candidate_indexと同じ候補identity codeを別契約名で保持した列。 |
| 67 | `beam_mean_vs_tvt_dense_abs` | `candidate_disagreement` | 0.490% | 67 / 149.8 | 67 / 81.6 | \|複数Beam path平均 TVT − dense spatial ANCC（full-prefix bias） TVT\|。 |
| 68 | `likpf_mean_vs_tvt_dense_abs` | `candidate_disagreement` | 0.479% | 72 / 133.4 | 66 / 87.0 | \|likelihood-weighted PF平均 TVT − dense spatial ANCC（full-prefix bias） TVT\|。 |
| 69 | `tvt_densew_minus_last` | `candidate_path_value` | 0.469% | 59 / 180.2 | 75 / 57.4 | dense spatial ANCC（prefix加重bias） TVT − last_known_tvt。 |
| 70 | `hyb_vs_v6_k16_geometry_gr_u_projection_abs` | `candidate_disagreement` | 0.442% | 69 / 144.2 | 71 / 68.4 | \|Beam/NCC hybrid TVT − exp226 K16 geometry/GR/U-projection TVT\|。 |
| 71 | `candidate_is_dense_family` | `candidate_identity_or_context` | 0.416% | 73 / 128.2 | 70 / 68.6 | 対象候補がdense 3候補なら1。 |
| 72 | `sc_ens_vs_v6_k16_geometry_gr_u_projection_abs` | `candidate_disagreement` | 0.403% | 62 / 165.0 | 85 / 43.6 | \|multi-scale NCC ensemble TVT − exp226 K16 geometry/GR/U-projection TVT\|。 |
| 73 | `tvt_dense_vs_blend_likpf_hmm_w500_abs` | `candidate_disagreement` | 0.397% | 74 / 128.0 | 72 / 62.4 | \|dense spatial ANCC（full-prefix bias） TVT − likPF / exact-HMM 50:50 blend TVT\|。 |
| 74 | `tvt_dense50_minus_last` | `candidate_path_value` | 0.394% | 70 / 142.2 | 81 / 53.4 | dense spatial ANCC（prefix末尾50 bias） TVT − last_known_tvt。 |
| 75 | `candidate_is_default_likpf` | `candidate_identity_or_context` | 0.378% | 80 / 105.4 | 69 / 68.6 | 対象候補が既定fallbackのlikpf_meanなら1。 |
| 76 | `tvt_dense50_vs_v6_k16_geometry_gr_u_projection_abs` | `candidate_disagreement` | 0.377% | 75 / 125.6 | 76 / 57.0 | \|dense spatial ANCC（prefix末尾50 bias） TVT − exp226 K16 geometry/GR/U-projection TVT\|。 |
| 77 | `copcf_nearby_majority_count_k8` | `cluster_prior_confidence` | 0.364% | 78 / 112.0 | 73 / 60.2 | 近傍8 well中、多数派clusterに属するwell数。 |
| 78 | `candidate_is_hmm_family` | `candidate_identity_or_context` | 0.355% | 79 / 111.8 | 74 / 57.4 | 対象候補がHMMまたはHMM blendなら1。 |
| 79 | `pf_ancc_vs_tvt_densew_abs` | `candidate_disagreement` | 0.335% | 77 / 117.4 | 84 / 47.4 | \|ANCC PF TVT − dense spatial ANCC（prefix加重bias） TVT\|。 |
| 80 | `hyb_vs_hmm_selfgr_boost_only_a070_c100_abs` | `candidate_disagreement` | 0.334% | 82 / 104.0 | 78 / 54.4 | \|Beam/NCC hybrid TVT − Self-GR likelihood HMM TVT\|。 |
| 81 | `pf_ancc_vs_tvt_dense50_abs` | `candidate_disagreement` | 0.333% | 81 / 104.8 | 79 / 53.8 | \|ANCC PF TVT − dense spatial ANCC（prefix末尾50 bias） TVT\|。 |
| 82 | `likpf_mean_vs_tvt_dense50_abs` | `candidate_disagreement` | 0.325% | 83 / 100.4 | 80 / 53.4 | \|likelihood-weighted PF平均 TVT − dense spatial ANCC（prefix末尾50 bias） TVT\|。 |
| 83 | `likpf_mean_vs_tvt_densew_abs` | `candidate_disagreement` | 0.313% | 89 / 89.4 | 77 / 55.6 | \|likelihood-weighted PF平均 TVT − dense spatial ANCC（prefix加重bias） TVT\|。 |
| 84 | `beam_mean_vs_tvt_densew_abs` | `candidate_disagreement` | 0.306% | 87 / 93.6 | 82 / 51.0 | \|複数Beam path平均 TVT − dense spatial ANCC（prefix加重bias） TVT\|。 |
| 85 | `sc_ens_minus_last` | `candidate_path_value` | 0.302% | 76 / 120.4 | 91 / 34.4 | multi-scale NCC ensemble TVT − last_known_tvt。 |
| 86 | `copcf_spatial_xy_only_k8_prior_count` | `cluster_prior_confidence` | 0.300% | 86 / 94.0 | 83 / 48.8 | XY距離だけを使うK=8 spatial priorを作る有効source数。 |
| 87 | `beam_mean_vs_tvt_dense50_abs` | `candidate_disagreement` | 0.274% | 88 / 92.6 | 86 / 40.6 | \|複数Beam path平均 TVT − dense spatial ANCC（prefix末尾50 bias） TVT\|。 |
| 88 | `pf_ancc_std` | `pf_dense_confidence` | 0.268% | 85 / 97.6 | 89 / 35.8 | ANCC PF粒子の行別TVT標準偏差。粒子分布の幅。 |
| 89 | `tvt_densew_vs_blend_likpf_hmm_w500_abs` | `candidate_disagreement` | 0.264% | 84 / 97.8 | 92 / 34.4 | \|dense spatial ANCC（prefix加重bias） TVT − likPF / exact-HMM 50:50 blend TVT\|。 |
| 90 | `copcf_typewell_spatial_prior_abs_delta` | `cluster_prior_confidence` | 0.260% | 91 / 85.8 | 87 / 39.6 | \|typewell prior − spatial prior\|。 |
| 91 | `tvt_dense50_vs_blend_likpf_hmm_w500_abs` | `candidate_disagreement` | 0.258% | 90 / 89.2 | 88 / 37.2 | \|dense spatial ANCC（prefix末尾50 bias） TVT − likPF / exact-HMM 50:50 blend TVT\|。 |
| 92 | `candidate_std` | `candidate_identity_or_context` | 0.221% | 92 / 73.6 | 93 / 33.4 | 同じbase rowにある候補TVT群の標準偏差。候補間disagreement。 |
| 93 | `candidate_multiobs_mae` | `multi_observation` | 0.198% | 95 / 62.6 | 94 / 31.8 | 対象候補自身のmulti-observation GR照合MAE。 |
| 94 | `copcf_nearby_majority_count_k5` | `cluster_prior_confidence` | 0.197% | 98 / 56.6 | 90 / 34.8 | 近傍5 well中、多数派clusterに属するwell数。 |
| 95 | `candidate_mean` | `candidate_identity_or_context` | 0.188% | 96 / 61.2 | 95 / 29.2 | 同じbase rowにある候補TVT群の平均。 |
| 96 | `hyb_vs_blend_likpf_hmm_w500_abs` | `candidate_disagreement` | 0.163% | 94 / 64.8 | 104 / 18.6 | \|Beam/NCC hybrid TVT − likPF / exact-HMM 50:50 blend TVT\|。 |
| 97 | `hyb_minus_last` | `candidate_path_value` | 0.160% | 93 / 69.2 | 115 / 15.2 | Beam/NCC hybrid TVT − last_known_tvt。 |
| 98 | `sc_ens_vs_hmm_selfgr_boost_only_a070_c100_abs` | `candidate_disagreement` | 0.156% | 97 / 60.6 | 105 / 18.6 | \|multi-scale NCC ensemble TVT − Self-GR likelihood HMM TVT\|。 |
| 99 | `copcf_typewell_native_overlap_1_neighbor_wells` | `cluster_prior_confidence` | 0.156% | 100 / 47.4 | 97 / 26.0 | native-overlap閾値1.0のtypewell cluster priorを作る近傍well数。 |
| 100 | `likpf_mean_vs_hyb_abs` | `candidate_disagreement` | 0.140% | 101 / 45.6 | 100 / 21.8 | \|likelihood-weighted PF平均 TVT − Beam/NCC hybrid TVT\|。 |
| 101 | `copcf_typewell_native_overlap_0p999_minus_candidate_abs` | `cluster_prior_confidence` | 0.136% | 108 / 29.8 | 96 / 29.2 | \|native-overlap閾値0.999のtypewell cluster prior TVT − 対象候補TVT\|。 |
| 102 | `copcf_spatial_xy_plus_trajectory_shape_k8_std_x_candidate` | `cluster_prior_confidence` | 0.129% | 103 / 34.8 | 98 / 24.0 | XY距離とtrajectory shapeを使うK=8 spatial priorの標準偏差をcandidate-long各行へ複製した列。積ではない。 |
| 103 | `copcf_typewell_native_overlap_1_prior_std` | `cluster_prior_confidence` | 0.118% | 102 / 35.2 | 101 / 20.2 | native-overlap閾値1.0のtypewell cluster priorを作るsource TVTの標準偏差。 |
| 104 | `sc_ens_vs_blend_likpf_hmm_w500_abs` | `candidate_disagreement` | 0.113% | 99 / 55.4 | 147 / 7.0 | \|multi-scale NCC ensemble TVT − likPF / exact-HMM 50:50 blend TVT\|。 |
| 105 | `copcf_typewell_native_overlap_0p999_minus_candidate_abs_norm` | `cluster_prior_confidence` | 0.106% | 116 / 25.2 | 99 / 21.8 | \|native-overlap閾値0.999のtypewell cluster prior TVT − 対象候補TVT\|をprior標準偏差で正規化。 |
| 106 | `copcf_typewell_native_overlap_0p999_prior_count` | `cluster_prior_confidence` | 0.104% | 107 / 30.4 | 106 / 18.2 | native-overlap閾値0.999のtypewell cluster priorを作る有効source数。 |
| 107 | `copcf_spatial_xy_only_k8_std_x_candidate` | `cluster_prior_confidence` | 0.100% | 110 / 29.2 | 107 / 17.4 | XY距離だけを使うK=8 spatial priorの標準偏差をcandidate-long各行へ複製した列。積ではない。 |
| 108 | `copcf_typewell_native_overlap_0p999_minus_candidate` | `cluster_prior_confidence` | 0.097% | 106 / 31.2 | 114 / 15.2 | native-overlap閾値0.999のtypewell cluster prior TVT − 対象候補TVT。 |
| 109 | `copcf_spatial_xy_only_k8_nearby_majority_diff_k8_corr_abs_c20` | `cluster_prior_confidence` | 0.095% | 121 / 24.0 | 103 / 18.6 | K=8近傍の多数派cluster不一致時にXY距離だけを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 110 | `copcf_spatial_xy_plus_trajectory_shape_k8_nearby_majority_diff_k8_corr_abs_c40` | `cluster_prior_confidence` | 0.094% | 125 / 22.6 | 102 / 19.2 | K=8近傍の多数派cluster不一致時にXY距離とtrajectory shapeを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 111 | `copcf_spatial_xy_only_k8_any_outlier_signal_k8_corr_abs_c40` | `cluster_prior_confidence` | 0.093% | 105 / 32.2 | 119 / 13.2 | K=8近傍のいずれかのoutlier signal時にXY距離だけを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 112 | `copcf_spatial_xy_only_k8_nearest_other_closer_corr_abs_c20` | `cluster_prior_confidence` | 0.089% | 109 / 29.2 | 116 / 13.8 | 別clusterの方が近い時にXY距離だけを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 113 | `copcf_spatial_xy_plus_trajectory_shape_k8_nearby_majority_diff_k8_corr_abs_c20` | `cluster_prior_confidence` | 0.087% | 123 / 23.4 | 109 / 16.2 | K=8近傍の多数派cluster不一致時にXY距離とtrajectory shapeを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 114 | `copcf_nearby_majority_share_k12` | `cluster_prior_confidence` | 0.086% | 131 / 21.4 | 108 / 17.0 | 近傍12 wellの多数派cluster比率。 |
| 115 | `copcf_spatial_xy_only_k8_own_z_gt2p0_corr_abs_c20` | `cluster_prior_confidence` | 0.085% | 124 / 23.2 | 111 / 15.8 | 所属cluster距離z-score>2時にXY距離だけを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 116 | `copcf_typewell_native_overlap_1_nearest_other_closer_corr_abs_c20` | `cluster_prior_confidence` | 0.084% | 130 / 21.6 | 110 / 16.2 | 別clusterの方が近い時にnative-overlap閾値1.0のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 117 | `copcf_spatial_xy_only_k8_any_outlier_signal_k8_corr_abs_c20` | `cluster_prior_confidence` | 0.083% | 113 / 27.2 | 121 / 12.8 | K=8近傍のいずれかのoutlier signal時にXY距離だけを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 118 | `self_gr_peak_gap` | `self_gr_confidence` | 0.083% | 114 / 26.4 | 120 / 13.2 | Self-GR照合のbest peakとsecond peakのscore差。 |
| 119 | `copcf_nearby_majority_diff_k5` | `cluster_prior_confidence` | 0.082% | 126 / 22.2 | 112 / 15.2 | 近傍5 wellの多数派clusterが自身の割当clusterと異なれば1。 |
| 120 | `copcf_spatial_xy_plus_trajectory_shape_k8_prior_count` | `cluster_prior_confidence` | 0.082% | 117 / 25.0 | 118 / 13.6 | XY距離とtrajectory shapeを使うK=8 spatial priorを作る有効source数。 |
| 121 | `copcf_spatial_xy_plus_trajectory_shape_k8_nearest_other_closer_corr_abs_c20` | `cluster_prior_confidence` | 0.082% | 120 / 24.6 | 117 / 13.8 | 別clusterの方が近い時にXY距離とtrajectory shapeを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 122 | `hyb_vs_tvt_dense_abs` | `candidate_disagreement` | 0.082% | 104 / 33.8 | 140 / 8.6 | \|Beam/NCC hybrid TVT − dense spatial ANCC（full-prefix bias） TVT\|。 |
| 123 | `candidate_range` | `candidate_identity_or_context` | 0.081% | 111 / 28.8 | 129 / 11.2 | 同じbase rowにある候補TVT群の最大−最小。候補間disagreement。 |
| 124 | `copcf_spatial_xy_only_k8_nearby_majority_diff_k8_corr_abs_c40` | `cluster_prior_confidence` | 0.079% | 134 / 20.6 | 113 / 15.2 | K=8近傍の多数派cluster不一致時にXY距離だけを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 125 | `copcf_spatial_xy_plus_trajectory_shape_k8_any_outlier_signal_k8_corr_abs_c40` | `cluster_prior_confidence` | 0.079% | 118 / 24.8 | 122 / 12.8 | K=8近傍のいずれかのoutlier signal時にXY距離とtrajectory shapeを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 126 | `multiobs_mae_beam_mean` | `multi_observation` | 0.079% | 115 / 26.2 | 125 / 12.0 | 複数Beam path平均のmulti-observation GR照合MAE。 |
| 127 | `multiobs_mae_likpf_mean` | `multi_observation` | 0.075% | 112 / 28.0 | 136 / 9.6 | likelihood-weighted PF平均のmulti-observation GR照合MAE。 |
| 128 | `copcf_spatial_xy_plus_trajectory_shape_k8_nearest_other_closer_corr_abs_c40` | `cluster_prior_confidence` | 0.068% | 135 / 19.4 | 123 / 12.0 | 別clusterの方が近い時にXY距離とtrajectory shapeを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 129 | `hyb_vs_tvt_dense50_abs` | `candidate_disagreement` | 0.068% | 127 / 22.2 | 134 / 10.4 | \|Beam/NCC hybrid TVT − dense spatial ANCC（prefix末尾50 bias） TVT\|。 |
| 130 | `copcf_typewell_native_overlap_1_any_outlier_signal_k8_corr_abs_c40` | `cluster_prior_confidence` | 0.067% | 136 / 19.0 | 124 / 12.0 | K=8近傍のいずれかのoutlier signal時にnative-overlap閾値1.0のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 131 | `copcf_spatial_xy_only_k8_nearest_other_closer_corr_abs_c40` | `cluster_prior_confidence` | 0.067% | 128 / 21.8 | 132 / 10.4 | 別clusterの方が近い時にXY距離だけを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 132 | `hyb_vs_tvt_densew_abs` | `candidate_disagreement` | 0.066% | 119 / 24.8 | 142 / 8.4 | \|Beam/NCC hybrid TVT − dense spatial ANCC（prefix加重bias） TVT\|。 |
| 133 | `likpf_mean_vs_sc_ens_abs` | `candidate_disagreement` | 0.065% | 122 / 23.6 | 141 / 8.6 | \|likelihood-weighted PF平均 TVT − multi-scale NCC ensemble TVT\|。 |
| 134 | `copcf_typewell_native_overlap_1_nearest_other_closer_corr_abs_c40` | `cluster_prior_confidence` | 0.063% | 140 / 17.6 | 128 / 11.4 | 別clusterの方が近い時にnative-overlap閾値1.0のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 135 | `pf_ancc_vs_hyb_abs` | `candidate_disagreement` | 0.062% | 142 / 16.2 | 126 / 12.0 | \|ANCC PF TVT − Beam/NCC hybrid TVT\|。 |
| 136 | `copcf_typewell_native_overlap_1_any_outlier_signal_k8_corr_abs_c20` | `cluster_prior_confidence` | 0.062% | 132 / 21.4 | 139 / 9.0 | K=8近傍のいずれかのoutlier signal時にnative-overlap閾値1.0のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 137 | `copcf_spatial_xy_plus_trajectory_shape_k8_any_outlier_signal_k8_corr_abs_c20` | `cluster_prior_confidence` | 0.061% | 137 / 18.6 | 135 / 10.2 | K=8近傍のいずれかのoutlier signal時にXY距離とtrajectory shapeを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 138 | `copcf_nearby_majority_share_k8` | `cluster_prior_confidence` | 0.060% | 146 / 15.2 | 127 / 11.6 | 近傍8 wellの多数派cluster比率。 |
| 139 | `pf_ancc_vs_sc_ens_abs` | `candidate_disagreement` | 0.059% | 143 / 16.2 | 131 / 10.8 | \|ANCC PF TVT − multi-scale NCC ensemble TVT\|。 |
| 140 | `copcf_typewell_native_overlap_1_nearby_majority_diff_k8_corr_abs_c20` | `cluster_prior_confidence` | 0.058% | 138 / 18.4 | 137 / 9.4 | K=8近傍の多数派cluster不一致時にnative-overlap閾値1.0のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 141 | `multiobs_mae_pf_ancc` | `multi_observation` | 0.057% | 139 / 18.2 | 138 / 9.2 | ANCC PFのmulti-observation GR照合MAE。 |
| 142 | `copcf_typewell_native_overlap_1_nearby_majority_diff_k8_corr_abs_c40` | `cluster_prior_confidence` | 0.057% | 145 / 16.0 | 133 / 10.4 | K=8近傍の多数派cluster不一致時にnative-overlap閾値1.0のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 143 | `copcf_spatial_xy_only_k8_own_z_gt2p0_corr_abs_c40` | `cluster_prior_confidence` | 0.056% | 149 / 14.2 | 130 / 11.0 | 所属cluster距離z-score>2時にXY距離だけを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 144 | `sc_ens_vs_tvt_dense_abs` | `candidate_disagreement` | 0.053% | 129 / 21.8 | 151 / 5.8 | \|multi-scale NCC ensemble TVT − dense spatial ANCC（full-prefix bias） TVT\|。 |
| 145 | `sc_ens_vs_hyb_abs` | `candidate_disagreement` | 0.053% | 133 / 21.0 | 148 / 6.2 | \|multi-scale NCC ensemble TVT − Beam/NCC hybrid TVT\|。 |
| 146 | `copcf_nearby_majority_diff_k12` | `cluster_prior_confidence` | 0.049% | 144 / 16.0 | 145 / 7.6 | 近傍12 wellの多数派clusterが自身の割当clusterと異なれば1。 |
| 147 | `copcf_spatial_xy_only_k8_count_x_candidate` | `cluster_prior_confidence` | 0.045% | 150 / 13.4 | 144 / 7.8 | XY距離だけを使うK=8 spatial priorのsource数をcandidate-long各行へ複製した列。積ではない。 |
| 148 | `beam_mean_vs_hyb_abs` | `candidate_disagreement` | 0.042% | 147 / 14.6 | 150 / 6.0 | \|複数Beam path平均 TVT − Beam/NCC hybrid TVT\|。 |
| 149 | `sc_ens_vs_tvt_densew_abs` | `candidate_disagreement` | 0.041% | 141 / 17.2 | 161 / 4.2 | \|multi-scale NCC ensemble TVT − dense spatial ANCC（prefix加重bias） TVT\|。 |
| 150 | `copcf_spatial_xy_plus_trajectory_shape_k8_own_z_gt2p0_corr_abs_c20` | `cluster_prior_confidence` | 0.041% | 156 / 10.0 | 143 / 8.2 | 所属cluster距離z-score>2時にXY距離とtrajectory shapeを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 151 | `copcf_typewell_native_overlap_1_nearest_other_closer_gate_x_dense_family` | `cluster_prior_confidence` | 0.037% | 157 / 9.8 | 146 / 7.0 | native-overlap閾値1.0のtypewell cluster priorが有効、別clusterの方が近い、かつ対象候補がdense familyなら1。 |
| 152 | `sc_ens_vs_tvt_dense50_abs` | `candidate_disagreement` | 0.036% | 148 / 14.4 | 160 / 4.2 | \|multi-scale NCC ensemble TVT − dense spatial ANCC（prefix末尾50 bias） TVT\|。 |
| 153 | `self_gr_quality` | `self_gr_confidence` | 0.034% | 152 / 10.8 | 154 / 5.4 | 同一horizontal内Self-GR motif照合の品質score。 |
| 154 | `view_candidate_std` | `raw_test_view_context` | 0.033% | 153 / 10.8 | 155 / 5.0 | raw-test viewで利用可能な候補TVTの標準偏差。 |
| 155 | `beam_mean_vs_sc_ens_abs` | `candidate_disagreement` | 0.030% | 151 / 11.8 | 163 / 3.6 | \|複数Beam path平均 TVT − multi-scale NCC ensemble TVT\|。 |
| 156 | `copcf_typewell_native_overlap_1_own_z_gt2p0_corr_abs_c20` | `cluster_prior_confidence` | 0.030% | 163 / 8.4 | 153 / 5.4 | 所属cluster距離z-score>2時にnative-overlap閾値1.0のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 157 | `copcf_typewell_native_overlap_1_any_outlier_signal_k8_gate_x_dense_family` | `cluster_prior_confidence` | 0.029% | 159 / 9.2 | 157 / 4.8 | native-overlap閾値1.0のtypewell cluster priorが有効、K=8近傍のいずれかのoutlier signal、かつ対象候補がdense familyなら1。 |
| 158 | `copcf_spatial_xy_plus_trajectory_shape_k8_any_outlier_signal_k8_gate_x_dense_family` | `cluster_prior_confidence` | 0.029% | 164 / 8.0 | 152 / 5.4 | XY距離とtrajectory shapeを使うK=8 spatial priorが有効、K=8近傍のいずれかのoutlier signal、かつ対象候補がdense familyなら1。 |
| 159 | `view_candidate_mean` | `raw_test_view_context` | 0.029% | 168 / 6.2 | 149 / 6.2 | raw-test viewで利用可能な候補TVTの平均。 |
| 160 | `copcf_spatial_xy_plus_trajectory_shape_k8_own_z_gt2p0_corr_abs_c40` | `cluster_prior_confidence` | 0.028% | 154 / 10.6 | 165 / 3.6 | 所属cluster距離z-score>2時にXY距離とtrajectory shapeを使うK=8 spatial priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 161 | `copcf_nearby_majority_share_k5` | `cluster_prior_confidence` | 0.027% | 158 / 9.6 | 164 / 3.6 | 近傍5 wellの多数派cluster比率。 |
| 162 | `copcf_typewell_native_overlap_1_own_z_gt2p0_corr_abs_c40` | `cluster_prior_confidence` | 0.025% | 165 / 7.0 | 158 / 4.6 | 所属cluster距離z-score>2時にnative-overlap閾値1.0のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 163 | `copcf_spatial_xy_plus_trajectory_shape_k8_nearest_other_closer_gate_x_dense_family` | `cluster_prior_confidence` | 0.024% | 167 / 6.6 | 159 / 4.4 | XY距離とtrajectory shapeを使うK=8 spatial priorが有効、別clusterの方が近い、かつ対象候補がdense familyなら1。 |
| 164 | `copcf_typewell_native_overlap_1_std_x_candidate` | `cluster_prior_confidence` | 0.023% | 166 / 7.0 | 162 / 4.0 | native-overlap閾値1.0のtypewell cluster priorの標準偏差をcandidate-long各行へ複製した列。積ではない。 |
| 165 | `copcf_nearby_majority_diff_k8` | `cluster_prior_confidence` | 0.023% | 161 / 8.4 | 167 / 3.2 | 近傍8 wellの多数派clusterが自身の割当clusterと異なれば1。 |
| 166 | `copcf_nearest_other_closer` | `cluster_prior_confidence` | 0.023% | 174 / 5.2 | 156 / 4.8 | 最も近い別clusterが割当先clusterより近ければ1。 |
| 167 | `copcf_typewell_native_overlap_1_neighbor_wells_x_candidate` | `cluster_prior_confidence` | 0.022% | 160 / 8.6 | 171 / 2.6 | native-overlap閾値1.0のtypewell cluster priorの近傍well数をcandidate-long各行へ複製した列。積ではない。 |
| 168 | `candidate_multiobs_score` | `multi_observation` | 0.020% | 155 / 10.2 | 200 / 1.0 | 対象候補自身のmulti-observation GR一致score。 |
| 169 | `copcf_spatial_xy_only_k8_nearby_majority_diff_k8_gate_x_dense_family` | `cluster_prior_confidence` | 0.019% | 162 / 8.4 | 179 / 1.8 | XY距離だけを使うK=8 spatial priorが有効、K=8近傍の多数派cluster不一致、かつ対象候補がdense familyなら1。 |
| 170 | `copcf_typewell_native_overlap_1_count_x_candidate` | `cluster_prior_confidence` | 0.017% | 170 / 5.8 | 170 / 2.6 | native-overlap閾値1.0のtypewell cluster priorのsource数をcandidate-long各行へ複製した列。積ではない。 |
| 171 | `copcf_spatial_xy_only_k8_nearest_other_closer_gate_x_dense_family` | `cluster_prior_confidence` | 0.016% | 172 / 5.6 | 173 / 2.4 | XY距離だけを使うK=8 spatial priorが有効、別clusterの方が近い、かつ対象候補がdense familyなら1。 |
| 172 | `copcf_spatial_xy_only_k8_own_z_gt2p0_gate_x_dense_family` | `cluster_prior_confidence` | 0.016% | 177 / 4.4 | 168 / 3.0 | XY距離だけを使うK=8 spatial priorが有効、所属cluster距離z-score>2、かつ対象候補がdense familyなら1。 |
| 173 | `copcf_typewell_native_overlap_1_own_z_gt2p0_gate_x_dense_family` | `cluster_prior_confidence` | 0.016% | 171 / 5.8 | 175 / 2.2 | native-overlap閾値1.0のtypewell cluster priorが有効、所属cluster距離z-score>2、かつ対象候補がdense familyなら1。 |
| 174 | `copcf_spatial_xy_plus_trajectory_shape_k8_own_z_gt2p0_gate_x_dense_family` | `cluster_prior_confidence` | 0.016% | 169 / 6.0 | 177 / 2.0 | XY距離とtrajectory shapeを使うK=8 spatial priorが有効、所属cluster距離z-score>2、かつ対象候補がdense familyなら1。 |
| 175 | `candidate_is_pfbeam_family` | `candidate_identity_or_context` | 0.015% | 181 / 3.6 | 166 / 3.2 | 対象候補がPF/Beam系なら1。 |
| 176 | `copcf_spatial_xy_plus_trajectory_shape_k8_count_x_candidate` | `cluster_prior_confidence` | 0.014% | 178 / 4.0 | 169 / 2.6 | XY距離とtrajectory shapeを使うK=8 spatial priorのsource数をcandidate-long各行へ複製した列。積ではない。 |
| 177 | `copcf_gate_any_outlier_signal_k8` | `cluster_prior_confidence` | 0.013% | 179 / 3.8 | 172 / 2.4 | K=8 outlier signalの総合gate flag。 |
| 178 | `multiobs_score_mean` | `multi_observation` | 0.013% | 173 / 5.6 | 198 / 1.2 | 候補bankのmulti-observation GR一致score平均。 |
| 179 | `multiobs_top1_mae` | `multi_observation` | 0.013% | 187 / 3.4 | 174 / 2.4 | multi-observation score最上位候補のGR照合MAE。 |
| 180 | `copcf_typewell_native_overlap_1_nearby_majority_diff_k8_gate_x_dense_family` | `cluster_prior_confidence` | 0.012% | 183 / 3.6 | 178 / 2.0 | native-overlap閾値1.0のtypewell cluster priorが有効、K=8近傍の多数派cluster不一致、かつ対象候補がdense familyなら1。 |
| 181 | `multiobs_mae_hyb` | `multi_observation` | 0.011% | 185 / 3.4 | 183 / 1.8 | Beam/NCC hybridのmulti-observation GR照合MAE。 |
| 182 | `copcf_typewell_native_overlap_1_any_outlier_signal_k8_gate_x_candidate` | `cluster_prior_confidence` | 0.011% | 182 / 3.6 | 186 / 1.6 | native-overlap閾値1.0のtypewell cluster priorが有効かつK=8近傍のいずれかのoutlier signalなら1。candidate-long行へ複製するgate。 |
| 183 | `view_candidate_range` | `raw_test_view_context` | 0.011% | 176 / 4.6 | 204 / 1.0 | raw-test viewで利用可能な候補TVTの最大−最小。 |
| 184 | `copcf_gate_nearby_majority_diff_k8` | `cluster_prior_confidence` | 0.011% | 195 / 2.8 | 176 / 2.0 | K=8近傍多数派不一致のgate flag。 |
| 185 | `copcf_spatial_xy_only_k8_nearest_other_closer_gate_x_candidate` | `cluster_prior_confidence` | 0.010% | 180 / 3.8 | 189 / 1.4 | XY距離だけを使うK=8 spatial priorが有効かつ別clusterの方が近いなら1。candidate-long行へ複製するgate。 |
| 186 | `copcf_spatial_xy_only_k8_any_outlier_signal_k8_gate_x_dense_family` | `cluster_prior_confidence` | 0.010% | 175 / 5.2 | 208 / 0.6 | XY距離だけを使うK=8 spatial priorが有効、K=8近傍のいずれかのoutlier signal、かつ対象候補がdense familyなら1。 |
| 187 | `copcf_typewell_native_overlap_0p999_nearest_other_closer_corr_abs_c40` | `cluster_prior_confidence` | 0.010% | 196 / 2.8 | 181 / 1.8 | 別clusterの方が近い時にnative-overlap閾値0.999のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 188 | `multiobs_score_max` | `multi_observation` | 0.010% | 194 / 3.0 | 187 / 1.6 | 候補bankのmulti-observation GR一致score最大値。 |
| 189 | `copcf_spatial_xy_plus_trajectory_shape_k8_nearby_majority_diff_k8_gate_x_dense_family` | `cluster_prior_confidence` | 0.009% | 188 / 3.2 | 191 / 1.4 | XY距離とtrajectory shapeを使うK=8 spatial priorが有効、K=8近傍の多数派cluster不一致、かつ対象候補がdense familyなら1。 |
| 190 | `copcf_spatial_xy_plus_trajectory_shape_k8_nearest_other_closer_gate_x_candidate` | `cluster_prior_confidence` | 0.009% | 197 / 2.4 | 180 / 1.8 | XY距離とtrajectory shapeを使うK=8 spatial priorが有効かつ別clusterの方が近いなら1。candidate-long行へ複製するgate。 |
| 191 | `copcf_spatial_xy_plus_trajectory_shape_k8_nearby_majority_diff_k8_gate_x_candidate` | `cluster_prior_confidence` | 0.009% | 192 / 3.0 | 190 / 1.4 | XY距離とtrajectory shapeを使うK=8 spatial priorが有効かつK=8近傍の多数派cluster不一致なら1。candidate-long行へ複製するgate。 |
| 192 | `copcf_typewell_native_overlap_0p999_any_outlier_signal_k8_corr_abs_c40` | `cluster_prior_confidence` | 0.009% | 193 / 3.0 | 193 / 1.4 | K=8近傍のいずれかのoutlier signal時にnative-overlap閾値0.999のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 193 | `copcf_typewell_native_overlap_1_nearby_majority_diff_k8_gate_x_candidate` | `cluster_prior_confidence` | 0.009% | 200 / 2.2 | 182 / 1.8 | native-overlap閾値1.0のtypewell cluster priorが有効かつK=8近傍の多数派cluster不一致なら1。candidate-long行へ複製するgate。 |
| 194 | `copcf_spatial_xy_only_k8_nearby_majority_diff_k8_gate_x_candidate` | `cluster_prior_confidence` | 0.008% | 198 / 2.2 | 184 / 1.6 | XY距離だけを使うK=8 spatial priorが有効かつK=8近傍の多数派cluster不一致なら1。candidate-long行へ複製するgate。 |
| 195 | `multiobs_score_beam_mean` | `multi_observation` | 0.008% | 186 / 3.4 | 205 / 0.8 | 複数Beam path平均のmulti-observation GR一致score。 |
| 196 | `copcf_typewell_native_overlap_0p999_any_outlier_signal_k8_corr_abs_c20` | `cluster_prior_confidence` | 0.008% | 199 / 2.2 | 192 / 1.4 | K=8近傍のいずれかのoutlier signal時にnative-overlap閾値0.999のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 197 | `multiobs_score_pf_ancc` | `multi_observation` | 0.007% | 184 / 3.6 | 224 / 0.4 | ANCC PFのmulti-observation GR一致score。 |
| 198 | `copcf_cluster_feature_valid` | `cluster_prior_confidence` | 0.007% | 205 / 1.6 | 188 / 1.4 | cluster/outlier診断が計算可能なら1。 |
| 199 | `multiobs_mae_sc_ens` | `multi_observation` | 0.007% | 207 / 1.6 | 194 / 1.4 | multi-scale NCC ensembleのmulti-observation GR照合MAE。 |
| 200 | `copcf_spatial_xy_only_k8_any_outlier_signal_k8_gate_x_candidate` | `cluster_prior_confidence` | 0.007% | 191 / 3.0 | 207 / 0.6 | XY距離だけを使うK=8 spatial priorが有効かつK=8近傍のいずれかのoutlier signalなら1。candidate-long行へ複製するgate。 |
| 201 | `copcf_typewell_native_overlap_0p999_valid_prior` | `cluster_prior_confidence` | 0.007% | 189 / 3.2 | 218 / 0.4 | native-overlap閾値0.999のtypewell cluster priorが計算可能なら1。 |
| 202 | `copcf_typewell_native_overlap_0p999_nearest_other_closer_corr_abs_c20` | `cluster_prior_confidence` | 0.006% | 222 / 0.8 | 185 / 1.6 | 別clusterの方が近い時にnative-overlap閾値0.999のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 203 | `candidate_is_geometry_family` | `candidate_identity_or_context` | 0.006% | 202 / 1.8 | 199 / 1.0 | 対象候補がexp226 geometry候補なら1。 |
| 204 | `copcf_typewell_native_overlap_0p999_nearby_majority_diff_k8_corr_abs_c40` | `cluster_prior_confidence` | 0.006% | 204 / 1.8 | 203 / 1.0 | K=8近傍の多数派cluster不一致時にnative-overlap閾値0.999のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 205 | `copcf_spatial_xy_plus_trajectory_shape_k8_own_z_gt2p0_gate_x_candidate` | `cluster_prior_confidence` | 0.006% | 206 / 1.6 | 201 / 1.0 | XY距離とtrajectory shapeを使うK=8 spatial priorが有効かつ所属cluster距離z-score>2なら1。candidate-long行へ複製するgate。 |
| 206 | `multiobs_ncc_beam_mean` | `multi_observation` | 0.005% | 190 / 3.2 | 289 / 0.0 | 複数Beam path平均のmulti-observation GR照合NCC。 |
| 207 | `copcf_typewell_native_overlap_1_nearest_other_closer_gate_x_candidate` | `cluster_prior_confidence` | 0.005% | 218 / 1.0 | 197 / 1.2 | native-overlap閾値1.0のtypewell cluster priorが有効かつ別clusterの方が近いなら1。candidate-long行へ複製するgate。 |
| 208 | `multiobs_score_sc_ens` | `multi_observation` | 0.005% | 232 / 0.6 | 195 / 1.4 | multi-scale NCC ensembleのmulti-observation GR一致score。 |
| 209 | `multiobs_score_likpf_mean` | `multi_observation` | 0.005% | 201 / 2.2 | 223 / 0.4 | likelihood-weighted PF平均のmulti-observation GR一致score。 |
| 210 | `copcf_spatial_xy_only_k8_own_z_gt2p0_gate_x_candidate` | `cluster_prior_confidence` | 0.005% | 227 / 0.6 | 196 / 1.2 | XY距離だけを使うK=8 spatial priorが有効かつ所属cluster距離z-score>2なら1。candidate-long行へ複製するgate。 |
| 211 | `candidate_score_gap_from_view_best` | `raw_test_view_context` | 0.004% | 203 / 1.8 | 212 / 0.4 | view内best multi-observation score − 対象候補score。 |
| 212 | `copcf_typewell_native_overlap_0p999_nearby_majority_diff_k8_corr_abs_c20` | `cluster_prior_confidence` | 0.004% | 229 / 0.6 | 202 / 1.0 | K=8近傍の多数派cluster不一致時にnative-overlap閾値0.999のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 213 | `copcf_spatial_xy_plus_trajectory_shape_k8_any_outlier_signal_k8_gate_x_candidate` | `cluster_prior_confidence` | 0.003% | 216 / 1.0 | 209 / 0.6 | XY距離とtrajectory shapeを使うK=8 spatial priorが有効かつK=8近傍のいずれかのoutlier signalなら1。candidate-long行へ複製するgate。 |
| 214 | `copcf_typewell_native_overlap_0p999_nearest_other_closer_gate_x_dense_family` | `cluster_prior_confidence` | 0.003% | 209 / 1.2 | 214 / 0.4 | native-overlap閾値0.999のtypewell cluster priorが有効、別clusterの方が近い、かつ対象候補がdense familyなら1。 |
| 215 | `view_candidate_std_safe` | `raw_test_view_context` | 0.003% | 213 / 1.2 | 225 / 0.4 | candidate z-score用に下限を置いたview内候補標準偏差。 |
| 216 | `multiobs_ncc_pf_ancc` | `multi_observation` | 0.003% | 224 / 0.8 | 211 / 0.6 | ANCC PFのmulti-observation GR照合NCC。 |
| 217 | `multiobs_score_hyb` | `multi_observation` | 0.003% | 242 / 0.4 | 206 / 0.8 | Beam/NCC hybridのmulti-observation GR一致score。 |
| 218 | `copcf_typewell_native_overlap_1_own_z_gt2p0_gate_x_candidate` | `cluster_prior_confidence` | 0.003% | 210 / 1.2 | 234 / 0.2 | native-overlap閾値1.0のtypewell cluster priorが有効かつ所属cluster距離z-score>2なら1。candidate-long行へ複製するgate。 |
| 219 | `multiobs_score_gap` | `multi_observation` | 0.003% | 211 / 1.2 | 236 / 0.2 | multi-observation GR一致scoreの1位−2位差。 |
| 220 | `self_gr_typewell_agreement` | `self_gr_confidence` | 0.003% | 212 / 1.2 | 238 / 0.2 | Self-GR観測とtypewell観測の整合度。 |
| 221 | `copcf_typewell_native_overlap_0p999_any_outlier_signal_k8_gate_x_dense_family` | `cluster_prior_confidence` | 0.003% | 220 / 0.8 | 213 / 0.4 | native-overlap閾値0.999のtypewell cluster priorが有効、K=8近傍のいずれかのoutlier signal、かつ対象候補がdense familyなら1。 |
| 222 | `copcf_spatial_xy_plus_trajectory_shape_k8_nearby_majority_diff_k8_clip_hit_c40` | `cluster_prior_confidence` | 0.002% | 237 / 0.4 | 210 / 0.6 | K=8近傍の多数派cluster不一致時のprior−候補差がclip 40 ftを超えれば1。 |
| 223 | `copcf_spatial_xy_plus_trajectory_shape_k8_valid_prior` | `cluster_prior_confidence` | 0.002% | 208 / 1.4 | 261 / 0.0 | XY距離とtrajectory shapeを使うK=8 spatial priorが計算可能なら1。 |
| 224 | `candidate_multiobs_ncc` | `multi_observation` | 0.002% | 214 / 1.0 | 226 / 0.2 | 対象候補自身のmulti-observation GR照合NCC。 |
| 225 | `copcf_gate_own_z_gt2p0` | `cluster_prior_confidence` | 0.002% | 215 / 1.0 | 228 / 0.2 | 所属cluster距離z-score>2のgate flag。 |
| 226 | `copcf_typewell_native_overlap_0p999_std_x_candidate` | `cluster_prior_confidence` | 0.002% | 230 / 0.6 | 217 / 0.4 | native-overlap閾値0.999のtypewell cluster priorの標準偏差をcandidate-long各行へ複製した列。積ではない。 |
| 227 | `multiobs_ncc_likpf_mean` | `multi_observation` | 0.002% | 231 / 0.6 | 221 / 0.4 | likelihood-weighted PF平均のmulti-observation GR照合NCC。 |
| 228 | `copcf_gate_nearest_other_closer` | `cluster_prior_confidence` | 0.002% | 219 / 0.8 | 227 / 0.2 | 別clusterの方が近いgate flag。 |
| 229 | `copcf_typewell_native_overlap_0p999_nearby_majority_diff_k8_gate_x_dense_family` | `cluster_prior_confidence` | 0.002% | 221 / 0.8 | 231 / 0.2 | native-overlap閾値0.999のtypewell cluster priorが有効、K=8近傍の多数派cluster不一致、かつ対象候補がdense familyなら1。 |
| 230 | `copcf_typewell_native_overlap_0p999_own_z_gt2p0_corr_abs_c20` | `cluster_prior_confidence` | 0.002% | 223 / 0.8 | 232 / 0.2 | 所属cluster距離z-score>2時にnative-overlap閾値0.999のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 20 ft）。 |
| 231 | `self_gr_valid` | `self_gr_confidence` | 0.002% | 225 / 0.8 | 239 / 0.2 | Self-GR照合が入力条件を満たして有効なら1。 |
| 232 | `copcf_typewell_native_overlap_0p999_own_z_gt2p0_corr_abs_c40` | `cluster_prior_confidence` | 0.002% | 238 / 0.4 | 216 / 0.4 | 所属cluster距離z-score>2時にnative-overlap閾値0.999のtypewell cluster priorへ寄せる仮想補正量の絶対値（clip 40 ft）。 |
| 233 | `copcf_well_gate_ratio_any_outlier_signal_k8` | `cluster_prior_confidence` | 0.002% | 240 / 0.4 | 219 / 0.4 | well内でany-outlier gateが有効な行の比率。 |
| 234 | `copcf_typewell_native_overlap_0p999_count_x_candidate` | `cluster_prior_confidence` | 0.002% | 217 / 1.0 | 266 / 0.0 | native-overlap閾値0.999のtypewell cluster priorのsource数をcandidate-long各行へ複製した列。積ではない。 |
| 235 | `copcf_typewell_native_overlap_0p999_neighbor_wells_x_candidate` | `cluster_prior_confidence` | 0.002% | 250 / 0.2 | 215 / 0.4 | native-overlap閾値0.999のtypewell cluster priorの近傍well数をcandidate-long各行へ複製した列。積ではない。 |
| 236 | `multiobs_ncc_sc_ens` | `multi_observation` | 0.002% | 257 / 0.2 | 222 / 0.4 | multi-scale NCC ensembleのmulti-observation GR照合NCC。 |
| 237 | `copcf_spatial_xy_only_k8_own_z_gt2p0_clip_hit_c40` | `cluster_prior_confidence` | 0.001% | 236 / 0.4 | 230 / 0.2 | 所属cluster距離z-score>2時のprior−候補差がclip 40 ftを超えれば1。 |
| 238 | `copcf_typewell_native_overlap_0p999_own_z_gt2p0_gate_x_dense_family` | `cluster_prior_confidence` | 0.001% | 239 / 0.4 | 233 / 0.2 | native-overlap閾値0.999のtypewell cluster priorが有効、所属cluster距離z-score>2、かつ対象候補がdense familyなら1。 |
| 239 | `view_score_best` | `raw_test_view_context` | 0.001% | 243 / 0.4 | 240 / 0.2 | raw-test viewで利用可能な候補のmulti-observation score最大値。 |
| 240 | `copcf_well_gate_ratio_nearby_majority_diff_k8` | `cluster_prior_confidence` | 0.001% | 289 / 0.0 | 220 / 0.4 | well内で近傍多数派不一致gateが有効な行の比率。 |
| 241 | `copcf_spatial_xy_only_k8_nearest_other_closer_clip_hit_c40` | `cluster_prior_confidence` | 0.001% | 226 / 0.6 | 246 / 0.0 | 別clusterの方が近い時のprior−候補差がclip 40 ftを超えれば1。 |
| 242 | `copcf_spatial_xy_only_k8_valid_prior` | `cluster_prior_confidence` | 0.001% | 228 / 0.6 | 250 / 0.0 | XY距離だけを使うK=8 spatial priorが計算可能なら1。 |
| 243 | `multiobs_ncc_hyb` | `multi_observation` | 0.001% | 256 / 0.2 | 235 / 0.2 | Beam/NCC hybridのmulti-observation GR照合NCC。 |
| 244 | `copcf_any_configured_gate` | `cluster_prior_confidence` | 0.001% | 233 / 0.4 | 241 / 0.0 | 設定済みcluster/outlier gateのいずれかが有効なら1。 |
| 245 | `copcf_spatial_xy_only_k8_any_outlier_signal_k8_clip_hit_c20` | `cluster_prior_confidence` | 0.001% | 234 / 0.4 | 242 / 0.0 | K=8近傍のいずれかのoutlier signal時のprior−候補差がclip 20 ftを超えれば1。 |
| 246 | `copcf_spatial_xy_only_k8_any_outlier_signal_k8_clip_hit_c40` | `cluster_prior_confidence` | 0.001% | 235 / 0.4 | 243 / 0.0 | K=8近傍のいずれかのoutlier signal時のprior−候補差がclip 40 ftを超えれば1。 |
| 247 | `copcf_well_gate_ratio_own_z_gt2p0` | `cluster_prior_confidence` | 0.001% | 241 / 0.4 | 288 / 0.0 | well内で所属cluster距離z>2 gateが有効な行の比率。 |
| 248 | `copcf_spatial_xy_only_k8_nearby_majority_diff_k8_clip_hit_c20` | `cluster_prior_confidence` | 0.001% | 259 / 0.0 | 229 / 0.2 | K=8近傍の多数派cluster不一致時のprior−候補差がclip 20 ftを超えれば1。 |
| 249 | `multiobs_top1_ncc` | `multi_observation` | 0.001% | 290 / 0.0 | 237 / 0.2 | multi-observation score最上位候補のGR照合NCC。 |
| 250 | `copcf_spatial_xy_only_k8_nearest_other_closer_clip_hit_c20` | `cluster_prior_confidence` | 0.000% | 244 / 0.2 | 245 / 0.0 | 別clusterの方が近い時のprior−候補差がclip 20 ftを超えれば1。 |
| 251 | `copcf_spatial_xy_only_k8_own_z_gt2p0_clip_hit_c20` | `cluster_prior_confidence` | 0.000% | 245 / 0.2 | 249 / 0.0 | 所属cluster距離z-score>2時のprior−候補差がclip 20 ftを超えれば1。 |
| 252 | `copcf_spatial_xy_plus_trajectory_shape_k8_any_outlier_signal_k8_clip_hit_c20` | `cluster_prior_confidence` | 0.000% | 246 / 0.2 | 252 / 0.0 | K=8近傍のいずれかのoutlier signal時のprior−候補差がclip 20 ftを超えれば1。 |
| 253 | `copcf_spatial_xy_plus_trajectory_shape_k8_any_outlier_signal_k8_clip_hit_c40` | `cluster_prior_confidence` | 0.000% | 247 / 0.2 | 253 / 0.0 | K=8近傍のいずれかのoutlier signal時のprior−候補差がclip 40 ftを超えれば1。 |
| 254 | `copcf_spatial_xy_plus_trajectory_shape_k8_nearest_other_closer_clip_hit_c40` | `cluster_prior_confidence` | 0.000% | 248 / 0.2 | 256 / 0.0 | 別clusterの方が近い時のprior−候補差がclip 40 ftを超えれば1。 |
| 255 | `copcf_typewell_native_overlap_0p999_nearest_other_closer_gate_x_candidate` | `cluster_prior_confidence` | 0.000% | 249 / 0.2 | 272 / 0.0 | native-overlap閾値0.999のtypewell cluster priorが有効かつ別clusterの方が近いなら1。candidate-long行へ複製するgate。 |
| 256 | `copcf_typewell_native_overlap_1_nearby_majority_diff_k8_clip_hit_c40` | `cluster_prior_confidence` | 0.000% | 251 / 0.2 | 280 / 0.0 | K=8近傍の多数派cluster不一致時のprior−候補差がclip 40 ftを超えれば1。 |
| 257 | `copcf_typewell_native_overlap_1_nearest_other_closer_clip_hit_c40` | `cluster_prior_confidence` | 0.000% | 252 / 0.2 | 282 / 0.0 | 別clusterの方が近い時のprior−候補差がclip 40 ftを超えれば1。 |
| 258 | `copcf_typewell_native_overlap_1_own_z_gt2p0_clip_hit_c40` | `cluster_prior_confidence` | 0.000% | 253 / 0.2 | 284 / 0.0 | 所属cluster距離z-score>2時のprior−候補差がclip 40 ftを超えれば1。 |
| 259 | `copcf_typewell_native_overlap_1_valid_prior` | `cluster_prior_confidence` | 0.000% | 254 / 0.2 | 285 / 0.0 | native-overlap閾値1.0のtypewell cluster priorが計算可能なら1。 |
| 260 | `copcf_well_gate_ratio_nearest_other_closer` | `cluster_prior_confidence` | 0.000% | 255 / 0.2 | 287 / 0.0 | well内で別cluster近接gateが有効な行の比率。 |
| 261 | `multiobs_top1_source_id` | `multi_observation` | 0.000% | 258 / 0.2 | 290 / 0.0 | multi-observation score最上位候補のsource code。 |
| 262 | `copcf_spatial_xy_only_k8_nearby_majority_diff_k8_clip_hit_c40` | `cluster_prior_confidence` | 0.000% | 260 / 0.0 | 244 / 0.0 | K=8近傍の多数派cluster不一致時のprior−候補差がclip 40 ftを超えれば1。 |
| 263 | `copcf_spatial_xy_only_k8_neighbor_wells` | `cluster_prior_confidence` | 0.000% | 261 / 0.0 | 247 / 0.0 | XY距離だけを使うK=8 spatial priorを作る近傍well数。 |
| 264 | `copcf_spatial_xy_only_k8_neighbor_wells_x_candidate` | `cluster_prior_confidence` | 0.000% | 262 / 0.0 | 248 / 0.0 | XY距離だけを使うK=8 spatial priorの近傍well数をcandidate-long各行へ複製した列。積ではない。 |
| 265 | `copcf_spatial_xy_only_k8_valid_x_candidate` | `cluster_prior_confidence` | 0.000% | 263 / 0.0 | 251 / 0.0 | XY距離だけを使うK=8 spatial priorのvalid flagをcandidate-long各行へ複製した列。積ではない。 |
| 266 | `copcf_spatial_xy_plus_trajectory_shape_k8_nearby_majority_diff_k8_clip_hit_c20` | `cluster_prior_confidence` | 0.000% | 264 / 0.0 | 254 / 0.0 | K=8近傍の多数派cluster不一致時のprior−候補差がclip 20 ftを超えれば1。 |
| 267 | `copcf_spatial_xy_plus_trajectory_shape_k8_nearest_other_closer_clip_hit_c20` | `cluster_prior_confidence` | 0.000% | 265 / 0.0 | 255 / 0.0 | 別clusterの方が近い時のprior−候補差がclip 20 ftを超えれば1。 |
| 268 | `copcf_spatial_xy_plus_trajectory_shape_k8_neighbor_wells` | `cluster_prior_confidence` | 0.000% | 266 / 0.0 | 257 / 0.0 | XY距離とtrajectory shapeを使うK=8 spatial priorを作る近傍well数。 |
| 269 | `copcf_spatial_xy_plus_trajectory_shape_k8_neighbor_wells_x_candidate` | `cluster_prior_confidence` | 0.000% | 267 / 0.0 | 258 / 0.0 | XY距離とtrajectory shapeを使うK=8 spatial priorの近傍well数をcandidate-long各行へ複製した列。積ではない。 |
| 270 | `copcf_spatial_xy_plus_trajectory_shape_k8_own_z_gt2p0_clip_hit_c20` | `cluster_prior_confidence` | 0.000% | 268 / 0.0 | 259 / 0.0 | 所属cluster距離z-score>2時のprior−候補差がclip 20 ftを超えれば1。 |
| 271 | `copcf_spatial_xy_plus_trajectory_shape_k8_own_z_gt2p0_clip_hit_c40` | `cluster_prior_confidence` | 0.000% | 269 / 0.0 | 260 / 0.0 | 所属cluster距離z-score>2時のprior−候補差がclip 40 ftを超えれば1。 |
| 272 | `copcf_spatial_xy_plus_trajectory_shape_k8_valid_x_candidate` | `cluster_prior_confidence` | 0.000% | 270 / 0.0 | 262 / 0.0 | XY距離とtrajectory shapeを使うK=8 spatial priorのvalid flagをcandidate-long各行へ複製した列。積ではない。 |
| 273 | `copcf_typewell_native_overlap_0p999_any_outlier_signal_k8_clip_hit_c20` | `cluster_prior_confidence` | 0.000% | 271 / 0.0 | 263 / 0.0 | K=8近傍のいずれかのoutlier signal時のprior−候補差がclip 20 ftを超えれば1。 |
| 274 | `copcf_typewell_native_overlap_0p999_any_outlier_signal_k8_clip_hit_c40` | `cluster_prior_confidence` | 0.000% | 272 / 0.0 | 264 / 0.0 | K=8近傍のいずれかのoutlier signal時のprior−候補差がclip 40 ftを超えれば1。 |
| 275 | `copcf_typewell_native_overlap_0p999_any_outlier_signal_k8_gate_x_candidate` | `cluster_prior_confidence` | 0.000% | 273 / 0.0 | 265 / 0.0 | native-overlap閾値0.999のtypewell cluster priorが有効かつK=8近傍のいずれかのoutlier signalなら1。candidate-long行へ複製するgate。 |
| 276 | `copcf_typewell_native_overlap_0p999_nearby_majority_diff_k8_clip_hit_c20` | `cluster_prior_confidence` | 0.000% | 274 / 0.0 | 267 / 0.0 | K=8近傍の多数派cluster不一致時のprior−候補差がclip 20 ftを超えれば1。 |
| 277 | `copcf_typewell_native_overlap_0p999_nearby_majority_diff_k8_clip_hit_c40` | `cluster_prior_confidence` | 0.000% | 275 / 0.0 | 268 / 0.0 | K=8近傍の多数派cluster不一致時のprior−候補差がclip 40 ftを超えれば1。 |
| 278 | `copcf_typewell_native_overlap_0p999_nearby_majority_diff_k8_gate_x_candidate` | `cluster_prior_confidence` | 0.000% | 276 / 0.0 | 269 / 0.0 | native-overlap閾値0.999のtypewell cluster priorが有効かつK=8近傍の多数派cluster不一致なら1。candidate-long行へ複製するgate。 |
| 279 | `copcf_typewell_native_overlap_0p999_nearest_other_closer_clip_hit_c20` | `cluster_prior_confidence` | 0.000% | 277 / 0.0 | 270 / 0.0 | 別clusterの方が近い時のprior−候補差がclip 20 ftを超えれば1。 |
| 280 | `copcf_typewell_native_overlap_0p999_nearest_other_closer_clip_hit_c40` | `cluster_prior_confidence` | 0.000% | 278 / 0.0 | 271 / 0.0 | 別clusterの方が近い時のprior−候補差がclip 40 ftを超えれば1。 |
| 281 | `copcf_typewell_native_overlap_0p999_own_z_gt2p0_clip_hit_c20` | `cluster_prior_confidence` | 0.000% | 279 / 0.0 | 273 / 0.0 | 所属cluster距離z-score>2時のprior−候補差がclip 20 ftを超えれば1。 |
| 282 | `copcf_typewell_native_overlap_0p999_own_z_gt2p0_clip_hit_c40` | `cluster_prior_confidence` | 0.000% | 280 / 0.0 | 274 / 0.0 | 所属cluster距離z-score>2時のprior−候補差がclip 40 ftを超えれば1。 |
| 283 | `copcf_typewell_native_overlap_0p999_own_z_gt2p0_gate_x_candidate` | `cluster_prior_confidence` | 0.000% | 281 / 0.0 | 275 / 0.0 | native-overlap閾値0.999のtypewell cluster priorが有効かつ所属cluster距離z-score>2なら1。candidate-long行へ複製するgate。 |
| 284 | `copcf_typewell_native_overlap_0p999_valid_x_candidate` | `cluster_prior_confidence` | 0.000% | 282 / 0.0 | 276 / 0.0 | native-overlap閾値0.999のtypewell cluster priorのvalid flagをcandidate-long各行へ複製した列。積ではない。 |
| 285 | `copcf_typewell_native_overlap_1_any_outlier_signal_k8_clip_hit_c20` | `cluster_prior_confidence` | 0.000% | 283 / 0.0 | 277 / 0.0 | K=8近傍のいずれかのoutlier signal時のprior−候補差がclip 20 ftを超えれば1。 |
| 286 | `copcf_typewell_native_overlap_1_any_outlier_signal_k8_clip_hit_c40` | `cluster_prior_confidence` | 0.000% | 284 / 0.0 | 278 / 0.0 | K=8近傍のいずれかのoutlier signal時のprior−候補差がclip 40 ftを超えれば1。 |
| 287 | `copcf_typewell_native_overlap_1_nearby_majority_diff_k8_clip_hit_c20` | `cluster_prior_confidence` | 0.000% | 285 / 0.0 | 279 / 0.0 | K=8近傍の多数派cluster不一致時のprior−候補差がclip 20 ftを超えれば1。 |
| 288 | `copcf_typewell_native_overlap_1_nearest_other_closer_clip_hit_c20` | `cluster_prior_confidence` | 0.000% | 286 / 0.0 | 281 / 0.0 | 別clusterの方が近い時のprior−候補差がclip 20 ftを超えれば1。 |
| 289 | `copcf_typewell_native_overlap_1_own_z_gt2p0_clip_hit_c20` | `cluster_prior_confidence` | 0.000% | 287 / 0.0 | 283 / 0.0 | 所属cluster距離z-score>2時のprior−候補差がclip 20 ftを超えれば1。 |
| 290 | `copcf_typewell_native_overlap_1_valid_x_candidate` | `cluster_prior_confidence` | 0.000% | 288 / 0.0 | 286 / 0.0 | native-overlap閾値1.0のtypewell cluster priorのvalid flagをcandidate-long各行へ複製した列。積ではない。 |
| 291 | `view_candidate_count` | `raw_test_view_context` | 0.000% | 291 / 0.0 | 291 / 0.0 | raw-test viewで利用可能な候補数。 |
| 292 | `view_dense_available_count` | `raw_test_view_context` | 0.000% | 292 / 0.0 | 292 / 0.0 | dense family候補が利用可能な候補数。 |
| 293 | `view_geometry_available_count` | `raw_test_view_context` | 0.000% | 293 / 0.0 | 293 / 0.0 | geometry family候補が利用可能な候補数。 |
| 294 | `view_hmm_available_count` | `raw_test_view_context` | 0.000% | 294 / 0.0 | 294 / 0.0 | HMM診断が利用可能な候補数。 |
| 295 | `view_pfbeam_available_count` | `raw_test_view_context` | 0.000% | 295 / 0.0 | 295 / 0.0 | PF/Beam family候補が利用可能な候補数。 |

### exp251 combined importance family集計

| family | features | combined split share |
| --- | --- | --- |
| `candidate_disagreement` | 54 | 29.008% |
| `cluster_prior_confidence` | 165 | 24.963% |
| `hmm_confidence` | 7 | 11.755% |
| `candidate_path_value` | 11 | 10.575% |
| `candidate_identity_or_context` | 12 | 10.426% |
| `raw_test_view_context` | 13 | 6.398% |
| `distance_or_anchor_context` | 3 | 5.184% |
| `exp226_geometry` | 1 | 0.772% |
| `multi_observation` | 24 | 0.529% |
| `pf_dense_confidence` | 1 | 0.268% |
| `self_gr_confidence` | 4 | 0.121% |

## 解釈上の注意

- `candidate_index` / `candidate_name_code`は順序尺度ではなくcategory codeに近い。木では分割可能だが、将来candidate bankの順序を変えると意味が変わる。
- `*_std`、log-likelihood、multiobs scoreはraw confidence proxyである。小さいσや大きいscoreを単独でhard gateにせず、候補identity・距離・disagreementと一緒にOOF学習する。
- `copcf_*_x_candidate`の一部は掛け算ではなくcandidate-long行へ複製したcontext列である。`gate_x_dense_family`だけはgateとdense-family flagの論理積。
- exp251 v4は`exp226_gr_delta`と`exp226_geop_tvt`をraw-test契約から除外した。exp226候補そのものを除外したわけではない。
- selector重要度の重複・相関は、候補identity、候補値/delta、全pair差、dense 3本、prior 4系統×gate展開に構造的に多い。削減するならimportance下位から機械的に落とさず、family単位のOOF ablationで判定する。

## 出典

- `studies/exp238_feature_audit/inputs/exp237_selector_feature_importance_mean.csv`
- `experiments/exp251_raw_test_safe_dual_objective_candidate_ranker/kaggle/output/train_v4/artifacts/exp251_raw_test_safe_dual_objective_candidate_ranker_feature_importance_mean.csv`
- `experiments/exp251_raw_test_safe_dual_objective_candidate_ranker/kaggle/output/train_v4/artifacts/exp251_raw_test_safe_dual_objective_candidate_ranker_selected_feature_schema.csv`
