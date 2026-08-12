---
title: GR matching deep research
date: 2026-06-25
types:
  - survey
  - experiment_review
  - comparison
experiments:
  - exp008
  - exp017
  - exp042
  - exp048
  - exp091
  - exp093
  - exp099
  - exp120
  - exp223
topics:
  - gr_matching
  - typewell
  - candidate_path
  - observation_likelihood
status: final
summary: "GR matchingの外部知見と既存実験を統合し、直接TVT決定ではなく候補confidence・likelihoodとして使う方向を残した。"
---

# GR matching deep research: ROGII Wellbore Geology Prediction

- 対応する上位仮説: なし

調査日: 2026-06-25

## 結論

GR matching はまだ使い道がある。ただし、このコンペでは `typewell GR` と `horizontal GR` を直接 NCC/DTW で合わせて TVT を決める形は、既存実験と最新ディスカッションの両方から弱い。実行価値が残るのは、次の 4 方向。

| 優先度 | アイデア | 実装形 | 期待 | 主なリスク |
|---:|---|---|---|---|
| 1 | PF/Beam 候補の learned observation likelihood / ranker | 候補 `pf_ancc`, `beam_mean`, `likpf_mean`, neighbor prior, self-GR/multiobs score を N-way/ranking で選ぶ | `exp093/099` の oracle headroom を使う | target-free scorer は既に失敗。well-level holdout で過学習しやすい |
| 2 | typewell/native-overlap neighbor drift prior + GR confidence | `exp120` の neighbor drift prior を ML feature とし、GR match score は信用度/interaction に限定 | longtail bucket で改善済み | direct correction は worst-well regression が大きい |
| 3 | shape-aware / multi-scale GR local similarity | raw point GR ではなく local descriptor、CWT/DWT energy、shapeDTW 的 descriptor、multi-observation residual を feature 化 | 単点 GR のノイズを減らす | add-only NCC/DTW は既に悪化。候補選択側に限定する |
| 4 | bimodal datum detector | +/-15-25ft 周期候補を明示し、二峰性が強い well は mode commit でなく平均/uncertainty feature | ambiguous wells の大外しを抑える | 正解 mode を当てに行くと RMSE が悪化し得る |

やらない方がよい方向は明確。`NCC/DTW/DWT をそのまま LightGBM へ add-only`、`self-GR path の直接 TVT 候補化`、`same-typewell 他 horizontal GR の直接転写`、`typewell marker likelihood 単体` は、既存結果では採用根拠が弱い。

## コンペ内証拠

### 直接 GR matching は弱い

- `exp008_gr_ncc_matcher`: typewell/horizontal GR の multi-scale NCC add-only は悪化。`control_exp003_no_gr` CV 13.882944 に対し、`gr_ncc_no_gr_multi` 14.641514。
- `exp017_deterministic_dtw_addonly`: LightGBM no-GR CV 13.549257 に対し、DTW/DWT add-only 13.949718、bucket postprocess 13.910963。
- `exp042_ravaghi_ncc_gr_match_features`: Ravaghi 風 NCC/GR match は弱い base geometry には効くが、同一 surface ML control / PF controls には届かない。
- `exp048_ravaghi_single_model_feature_parity_revisit`: NCC/GR match + PF context blend は一部 split で良いが、original-fold / well-hash の両方で strict supported になる候補はなし。

解釈: GR matching は signal 自体が無いのではなく、直接特徴追加や直接 TVT path 化では noise/regime mismatch が勝つ。

### 候補集合には headroom がある

- `exp091_self_gr_likelihood_pf_beam_probe`: best single は `likpf_mean` RMSE 11.594897。self-GR 単体は RMSE 191+ と失敗。ただし oracle best は RMSE 6.873199 / within10 0.925153 で、self-GR が一部 row の当たり候補になる。
- `exp093_pf_candidate_coverage_then_ranker_audit`: baseline候補 oracle RMSE 7.434030、baseline+self-GR oracle RMSE 6.958935。だが target-free rank score top1 は baseline 12.507841、self-GR入り 29.985529 と失敗。`pf_ancc` は oracle best 1,092,069 rows なのに rank score top1 0 rows。
- `exp099_pf_multi_observation_likelihood_probe`: multi-observation likelihood 追加で oracle RMSE 6.897510、within10 0.922941 に改善。ただし direct top1 / softmax blend は弱い。

解釈: 次にやるべきは「候補を増やす」より「候補を選ぶ observation likelihood / ranker」。特に `pf_ancc` を現行 scorer が全く選べない問題を潰す価値が高い。

### neighbor drift prior は GR 直接 matching より強い

- `exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit`: same-typewell 他 horizontal prefix GR transfer は `likpf_mean` より悪化し、random/different-typewell controls より弱い。
- `exp120_typewell_geology_neighbor_prior_pf_likelihood_probe`: `neighbor_drift_prior` が `likpf_mean` RMSE 11.594897884 から 11.207143527 へ改善。特に `1000_plus` longtail bucket で -0.423798 RMSE。ただし 453 wells 改善 / 320 wells 悪化、最大悪化 +6.323216。

解釈: 波形そのものの横流しではなく、native overlap / typewell group の drift prior を作り、GR likelihood はその prior を信用するかどうかの補助に回すのがよい。

## Kaggle discussion readout

- [#697431 besides regression, also dwt](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697431): horizontal GR を typewell TVT-GR に stretch/fold して合わせる発想、local search、reverse index への注意、hidden/shared typewell・neighbor の重要性。
- [#702919 Dynamic Programming for TVT Tracking](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/702919): Viterbi/DP features は OOF を改善したが LB はほぼ動かず、単点 GR observation model が弱いという結論。構造 guide + local matcher の hybrid が重要。
- [#707613 PF baseline got LB 8.863](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/707613): PF は GR likelihood と物理制約で強いが、NN 化するなら candidate coverage、N-way rank/classification、transition/observation model 学習が焦点。
- [#708167 Formation Columns Are Derived from Typewell](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/708167): formation columns は独立 3D surface ではなく、実質 1 base surface + constant offsets。構造 surface prior の重要性を補強。
- [#708367 Problem Breakdown](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/708367): typewell は GR vs TVT lookup、many wells share common typewell subsequence、TVT+Z/layers は piecewise continuous linear。
- [#711878 The +/-15 ft datum](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/711878): rhythmic carbonate-marl cycles により、1 bundle 離れた二つの datum が同程度に合う well がある。二峰性が強い場合は midpoint が RMSE 最適になり得る。
- [#712037 Stop reforking](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/712037): GR は coarse signal として有効だが、fine per-well slope を legal features で pin しにくいという見立て。PF は 16ft から 10ft 程度へ下げるが、それ以上は構造/field generalization が壁。

## Literature and patent readout

### Direct multi-modal inversion of geophysical logs using deep learning

- URL: https://arxiv.org/abs/2201.01871
- 年: 2022
- 要点: gamma-ray logs の stratigraphic inversion は非一意であり、MTP loss + mixture density DNN で複数 trajectory と確率を出す。
- ROGII への転用: single TVT regression ではなく、PF/Beam候補の top-k trajectory と probability/rank を扱う設計に向く。`exp091/093/099` の oracle headroom と整合する。
- 注意: 本格 NN で置き換えるより、まず candidate ranker / uncertainty feature に落とす方が現実的。

### Soft-DTW

- URL: https://arxiv.org/abs/1703.01541
- 年: 2017
- 要点: DTW を soft-min 化し、微分可能な time-series loss にする。
- ROGII への転用: PF/Beam/NN observation model を train するなら使える。ただし exp017 の通常 DTW add-only 失敗から、直接 TVT path loss ではなく local descriptor / candidate scoring loss に限定。
- 注意: O(NM) 系のコスト、長い hidden test、NaN GR、reverse/non-monotone TVT に注意。

### shapeDTW

- URL: https://arxiv.org/abs/1606.01601
- 年: 2016
- 要点: vanilla DTW は局所構造が違う点同士を合わせ得るため、point-wise local structural descriptor を使う。
- ROGII への転用: GR 単点差ではなく、rolling slope、curvature、DWT/CWT energy、local z-score shape を cost に使う。`exp099` multiobs score を descriptor 化する方向。
- 注意: それでも direct path 採用ではなく ranker feature にする。

### Differentiable particle filters

- URL: https://arxiv.org/abs/2302.09639
- 年: 2023
- 要点: PF の dynamics / measurement / proposal / resampling を neural modules として学習する枠組み。
- ROGII への転用: いきなり end-to-end DPF は重い。まず `measurement model = candidate, GR context, neighbor prior -> log likelihood` を supervised に学習し、PF本体は既存候補生成を使う。
- 注意: validation leakage を避けるため well-grouped OOF が必須。

### Seismic horizon tracking / domain-prior contrastive learning

- URLs:
  - https://arxiv.org/abs/1804.06814
  - https://arxiv.org/abs/1812.11092
  - https://arxiv.org/abs/2606.16271
- 要点: horizon tracking は local signal continuity と large-scale geological structure の両方が必要。2026 の contrastive paper は signal-derived correspondences を domain prior として embedding を学ぶ。
- ROGII への転用: local GR matcher だけでなく、構造 surface / neighbor prior と fusion する。高信頼 GR correspondence だけを positive pair として使う contrastive/local embedding も候補。
- 注意: seismic volume と ROGII 1D logs はデータ形状が違うため、主役は構造 prior + candidate ranker。

### ROGII geosteering patents

- URLs:
  - https://patents.google.com/patent/US20190106974A1/en
  - https://patents.google.com/patent/US20230019126A1/en
- 要点: variable formation thickness、formation dip model、type log comparison、nearby wells/geomodel constraints、segment-wise algorithms、Pearson/RMS/self-correlation、confidence factor、geosteering spectrum が出てくる。
- ROGII への転用: patent 的にも「single log matching」ではなく「segment-wise, multi-algorithm, structural constraints, confidence factor」が自然。`exp120` neighbor drift prior + `exp093/099` candidate likelihood/ranker が最も近い。

## 次の実験候補

### A. `pf_candidate_ranker_gr_likelihood_features`

- route: `pf_beam` または `ensemble`
- 親: `exp093`, `exp099`, `exp120`
- 入力候補: `pf_ancc`, `beam_mean`, `likpf_mean`, `sc_ens`, `hyb`, `neighbor_drift_prior`, optional self-GR/multiobs candidates
- feature:
  - candidate absolute TVT / delta from anchor / delta from `likpf_mean`
  - multiobs MAE/NCC/score at offsets `[-24,-12,0,12,24]`
  - shape descriptor residual: GR value, slope, curvature, local z-score, rolling energy
  - candidate continuity: step, second diff, distance bucket, tail rank
  - neighbor prior validity/std/count, native overlap group, prior disagreement
  - ambiguity features: top1-top2 margin, entropy, +/-15ft alternative score gap
- label: row-wise oracle best candidate or soft label by absolute TVT error bucket。
- validation: well-grouped OOF。candidate-generation fold と ranker fold を混ぜない。
- success condition: `likpf_mean` RMSE 11.594897 と `neighbor_drift_prior` 11.207144 を下回る。well-level max regression を明示。

### B. `gr_shape_likelihood_ablation`

- route: `pf_beam`
- 目的: raw GR single-point likelihood、multiobs、shapeDTW-like local descriptor、CWT/DWT descriptor のどれが candidate ranking に効くかを比較。
- 実装: full DTW path ではなく、候補 TVT 近傍の local window descriptor distance を計算して 40-80 個程度の feature cache にする。
- success condition: direct candidate は問わず、ranker feature importance / candidate top-k coverage / NDCG / oracle gap reduction で評価。

### C. `datum_bimodality_detector`

- route: `ensemble` or `ml_model`
- 目的: +/-15-25ft 周期 decoy を検出し、hard mode selection を避ける。
- feature:
  - best GR score と +/-15/20/25ft shifted score の差
  - candidate distribution bimodality
  - PF/Beam/neighbor prior disagreement
  - local cyclicity / spectral energy
- 出力: prediction averaging weight、uncertainty feature、worst-well guard。

## 捨てる方向

- NCC/DTW/DWT add-only を再度そのまま試す。
- self-GR candidate を単体で提出候補にする。
- same-typewell cross-horizontal GR waveform を direct TVT correction に使う。
- typewell marker boundary prior を単体 likelihood として強く使う。
- public LB だけで GR matching parameter を選ぶ。

## 参照したローカル実験/メモ

- `experiments/exp008_gr_ncc_matcher/result.md`
- `experiments/exp017_deterministic_dtw_addonly/result.md`
- `experiments/exp042_ravaghi_ncc_gr_match_features/result.md`
- `experiments/exp048_ravaghi_single_model_feature_parity_revisit/result.md`
- `experiments/exp090_lateral_self_gr_match_pseudotail_probe/result.md`
- `experiments/exp091_self_gr_likelihood_pf_beam_probe/result.md`
- `experiments/exp093_pf_candidate_coverage_then_ranker_audit/result.md`
- `experiments/exp099_pf_multi_observation_likelihood_probe/result.md`
- `experiments/exp119_same_typewell_other_horizontal_prefix_gr_transfer_audit/result.md`
- `experiments/exp120_typewell_geology_neighbor_prior_pf_likelihood_probe/result.md`
- `docs/surveys/maybe_related_research.md`
