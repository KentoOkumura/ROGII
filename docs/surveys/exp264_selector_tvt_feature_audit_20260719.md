---
title: exp264 selector / TVT feature audit
date: 2026-07-19
types:
  - experiment_review
  - model_explanation
  - oof_analysis
  - feature_analysis
experiments:
  - exp263
  - exp264
topics:
  - selector
  - candidate_path
  - feature_importance
  - hidden_safe
  - lightgbm
status: final
summary: "exp264の12候補bank、hidden-safe selector、compact特徴、最終TVTモデルとguard結果を統合して説明する。"
---

# exp264 selector / TVT feature audit

作成日: 2026-07-19

対象: `exp264_exp263_candidate_confidence_dual_selector` の修正版Stage A v4 / Stage B v5 / Stage C v6 / Stage D v3 / inference v4。

## 結論

- 候補bankはexp263 Stage 1でcurrent-test生成できる12 surface。6 primitive、5つの固定50/50 pair、1つのnamed fixedからなる。pair/fixedは既存pathの決定的平均で、新しいpath generatorではない。
- selector入力はhidden-safeな88列。候補IDは12 one-hotで、連続的なordinal indexは使わない。目的は候補ごとの`pred_abs_error`と`p_within10`である。
- selectorのhard top1は不採用。修正版Stage Cではhard RMSE 8.652532でfixed fallback 8.238332より+0.414200、改善1/5 foldsだった。Viterbi、hard path推論、softmax候補平均は行わない。
- selector scoreは74列compactへ決定的に変換し、TVT LightGBMへadd-onlyする。TVTモデルはclean 273 + compact 74 = 347列、3 config×5 folds=15本の等重み平均。
- 347列add-onlyはclean 273 controlを10.476169→8.460811へ改善し5/5 folds改善、Public LBは7.562。ただしworst-well +14.482873でtrain-side総合guardはFAILのまま。
- TVT gain上位4列は2 legal domainのtop1候補TVT−anchorで、合計61.0343%。5位は`beam_mean`の予測誤差score 5.8196%。これはBeamをhard採用した結果ではない。

## Evidence boundary

- selector特徴重要度は修正版Stage B v5の5-fold gainをobjective別に正規化した値。Stage Cも同じ88列schemaを使う。旧100列schemaと旧Stage B/C/Dはtraining-only formation 12列のfeature availability leakageにより無効。
- TVT特徴重要度は修正版Stage D v3 `selector_compact_addonly`の15モデルについて、各モデル内のgain/splitを347列合計で正規化して平均した値。
- final 347列はhidden-safe allowlistであり、旧exp218 380列からformation依存74列、非nested exp111 score依存27列、推移依存GRWR 6列を除外している。
- selectorの選択率は修正版Stage B v5 outer-fold OOFのprimary 11候補domain。Stage C outer-validの選択率artifactはローカル保存していないため、Stage Bの率をStage C実測と誤記しない。

## 1. 候補パスの概要

### 候補bankとlegal domain

- `primitive_pair_bank`: primitive 6本 + pair 5本 = 11本。primary meta-featureのtop1/top2/marginを計算する。
- `primitive_fixed_bank`: primitive 6本 + `exp226_w500_50_50` = 7本。fixed fallback比較用のtop1/top2/marginを計算する。
- 12本すべてを1つのhard-selectable domainへ入れない。pairとnamed fixedを同時に競わせると同じ親pathを重複評価するため。
- `blend_likpf_hmm_w500`は`likpf_mean__exact_hmm`と同値なので独立candidate IDを持たない。

### 全12候補とselector選出率

path RMSE / actual within10はStage B v5 `candidate_score_oof.parquet`の全3,783,989 OOF行から再集計した。top1率はprimary 11候補内で、`pred_abs_error`は最小score、`p_within10`は最大scoreを選んだ率。named fixedはprimary選択率の対象外。

| candidate | kind | family / 内容 | path RMSE | actual within10 | score MAE | score logloss | error top1 | within10 top1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `exp226_k16` | primitive | geometry。exp226 K16 geometry/GR/U-projection候補。 | 9.427110 | 80.773% | 3.720195 | 0.379012 | 7.685% | 12.947% |
| `selfgr_hmm_a070` | primitive | self-GR HMM。exact HMMへself-GR likelihoodをalpha=0.70で加えた候補。 | 11.349943 | 79.484% | 4.253823 | 0.361743 | 19.775% | 9.320% |
| `likpf_mean` | primitive | likelihood PF。PF seed予測をseed likelihoodで重み付けした平均。単純算術平均ではない。 | 11.594898 | 77.281% | 4.251639 | 0.383646 | 7.390% | 5.862% |
| `exact_hmm` | primitive | exact HMM。typewell GR emissionと固定transitionを使うtarget-free exact HMM候補。 | 11.938287 | 78.439% | 4.311015 | 0.358570 | 5.245% | 2.215% |
| `pf_ancc` | primitive | particle filter。ANCC particle filterの平均TVT path。粒子spreadをnative confidenceに持つ。 | 14.493051 | 69.174% | 4.484758 | 0.380708 | 16.507% | 5.163% |
| `beam_mean` | primitive | Beam。複数Beam設定のpath平均。単体は弱いがcore候補と残差多様性を持つ。 | 15.774327 | 59.165% | 4.428738 | 0.413552 | 3.399% | 3.050% |
| `exp226_k16__selfgr_hmm_a070` | pair 50/50 | geometry + self-GR HMM。`exp226_k16`と`selfgr_hmm_a070`の固定50/50平均。 | 8.532715 | 84.350% | 3.127198 | 0.324873 | 6.735% | 15.797% |
| `exp226_k16__exact_hmm` | pair 50/50 | geometry + exact HMM。`exp226_k16`と`exact_hmm`の固定50/50平均。 | 8.635074 | 83.434% | 3.112552 | 0.325779 | 8.933% | 21.682% |
| `exp226_k16__likpf_mean` | pair 50/50 | geometry + likelihood PF。`exp226_k16`と`likpf_mean`の固定50/50平均。 | 8.813822 | 83.566% | 3.295010 | 0.338377 | 11.523% | 15.115% |
| `selfgr_hmm_a070__likpf_mean` | pair 50/50 | self-GR HMM + likelihood PF。`selfgr_hmm_a070`と`likpf_mean`の固定50/50平均。 | 10.123457 | 80.090% | 3.732081 | 0.364509 | 6.159% | 4.267% |
| `likpf_mean__exact_hmm` | pair 50/50 | likelihood PF + exact HMM。`likpf_mean`と`exact_hmm`の固定50/50平均。旧`blend_likpf_hmm_w500`の同値alias。 | 10.269697 | 79.322% | 3.718808 | 0.363235 | 6.649% | 4.582% |
| `exp226_w500_50_50` | named fixed | geometry + w500。0.50×`exp226_k16` + 0.25×`likpf_mean` + 0.25×`exact_hmm`。fixed fallback比較用。 | 8.238332 | 84.590% | 3.113805 | 0.325655 | - | - |

解釈:

- path単体では`exp226_w500_50_50`が8.238332で最良。ただしこれはprimary 11候補domainのhard選択対象ではなく、比較用fixed fallback。
- expected-error top1は`selfgr_hmm_a070` 19.775%、`pf_ancc` 16.507%が多いが、候補単体RMSEの順位とは一致しない。selectorは行ごとのregimeを拾おうとしている。
- probability top1は`exp226_k16__exact_hmm` 21.682%、`exp226_k16__selfgr_hmm_a070` 15.797%、`exp226_k16__likpf_mean` 15.115%が中心。
- `beam_mean`も3%前後選ばれるが、hard top1全体がfixedより悪いため「Beamを直接選ぶべき」という根拠にはしない。

## 2. Selector

### 学習目的とnested構成

| 項目 | 内容 |
| --- | --- |
| 単位 | 1行×1候補のcandidate-long |
| 候補数 | 12 |
| 回帰objective | `actual_abs_error = abs(candidate_tvt - true_tvt)`を教師に`pred_abs_error`を予測 |
| 分類objective | `actual_abs_error <= 10`を教師に`p_within10`を予測 |
| Stage B | outer 5 folds×2 objectives = 10 CPU boosters。selector単体のOOF診断 |
| Stage C | outer 5×inner 4×2 objectives = 40 CPU boosters。outer-trainはinner OOF、outer-validは4 inner model ensemble |
| 後段利用 | 候補別scoreを同じprocess内で74列compactへ変換し、同じouter foldのTVT LightGBMへ渡す |
| 使用しないもの | hard selector推論、Viterbi、candidate TVTのsoftmax平均、score CSVの再読込 |

### 88入力特徴のgroup概要

| group | 列数 | 内容 | pred_abs_error gain | p_within10 gain |
| --- | ---: | --- | ---: | ---: |
| `bank` | 13 | 候補bank内の中央値、spread、順位、現在候補のdisagreement。 | 56.789% | 56.765% |
| `ctx` | 22 | raw MD/X/Y/Z/GR、anchor、evaluation位置、typewell要約。 | 24.728% | 33.684% |
| `formula` | 14 | pair/fixedの親候補spread、weight、親confidence。 | 9.137% | 3.567% |
| `cand` | 13 | 現在候補TVT、anchor差、step、局所slope/curvature/straightness。 | 4.716% | 4.136% |
| `conf` | 12 | source-native sigma/loglik/margin/qualityとvalidity。 | 4.267% | 1.461% |
| `id` | 14 | 12 candidate one-hotとprimitive/pair kind。 | 0.364% | 0.387% |

### Selectorの出力形式

canonical audit出力はcandidate-long Parquetで、TVTの最終予測ではない。

| 区分 | 列 | 用途 |
| --- | --- | --- |
| key | `id`, `well`, `well_row_idx`, `outer_fold`, `candidate_id` | 行・well・fold・候補を一意化 |
| candidate | `candidate_tvt`, `candidate_available`, `confidence_valid` | 候補値とavailability/native confidence有効性 |
| dual score | `pred_abs_error`, `p_within10` | 後段compact化するcanonical selector score |
| label-only audit | `actual_abs_error`, `actual_within10` | OOF評価専用。selector入力やcurrent-testへ入れない |
| provenance | `feature_schema_sha`, `candidate_contract_sha`, `model_fold`, model SHA | schema/model/fold対応の再現性監査 |

運用時はcandidate-long scoreを保存して再読込せず、直ちに74列wide compactへ変換する。

| compact group | 列数 | 内容 |
| --- | ---: | --- |
| 候補別dual score | 24 | 12候補×`pred_abs_error` / `p_within10` |
| primitive+pair domain summary | 13 | 2 objectivesのtop1/top2 value・score・margin・top1-minus-anchorとobjective一致 |
| primitive+fixed domain summary | 13 | 上記と同じ要約をfixed fallback domainで計算 |
| global score/value summary | 9 | score mean/std/entropy、candidate value range/std、available/confidence-valid数 |
| candidate kind flags | 4 | primary top1 primitive/pair、fixed top1 primitive/fixed |
| primary error top1 one-hot | 11 | primary domainのpredicted-error top1候補ID |
| 合計 | 74 | `compact_meta_schema.json`で順序とSHAを固定 |

### Selectorによる選出結果

| stage | expected-error MAE | prior | within10 logloss | prior | Brier | prior | hard RMSE | fixed RMSE | hard改善fold | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Stage B v5 | 3.795801 | 5.788783 | 0.359972 | 0.510131 | 0.112451 | 0.165095 | 8.587004 | 8.238332 | 0/5 | score PASS / hard FAIL |
| Stage C v6 nested | 3.798819 | 5.788783 | 0.359412 | 0.510131 | 0.111830 | 0.165095 | 8.652532 | 8.238332 | 1/5 | score PASS / hard FAIL |

- dual scoreの校正は全3指標でpooledかつ5/5 folds改善し、後段meta-featureとして使う条件を満たした。
- hard top1はStage Bでnear +0.079326、1000+ +0.389208、worst-well +14.684481。hidden-like spatial / typewell-purgedも+0.768585 / +0.721137で悪化した。
- したがって「selectorが候補を直接選び、そのTVTを提出する」構成は棄却。「selector scoreを連続特徴として後段が解釈する」構成だけを採用した。

### 全88 selector入力特徴（pred_abs_error重要度順）

`pred share` / `within share`は修正版Stage B v5の各objective 5-fold mean gain share。missingはcandidate-long全体の構造的欠損率で、0補完せずLightGBMのNaNとして扱う。

| pred rank | within rank | feature | group | 説明 | pred share | within share | missing |
| ---: | ---: | --- | --- | --- | ---: | ---: | ---: |
| 1 | 1 | `bank__candidate_mean_abs_disagreement` | `bank` | 現在候補と12候補それぞれのTVT絶対差の平均。候補bankからの孤立度。 | 45.401% | 44.100% | 0.00% |
| 2 | 17 | `formula__component_std` | `formula` | formula親候補TVTの標準偏差。 | 3.993% | 1.349% | 50.00% |
| 3 | 2 | `bank__candidate_abs_minus_median` | `bank` | 現在候補TVTと12候補中央値の絶対差。 | 3.795% | 4.613% | 0.00% |
| 4 | 33 | `conf__native__sigma_tvt` | `conf` | sourceが推定したTVT uncertainty幅。HMM posterior幅やPF粒子spreadに相当。 | 2.958% | 0.494% | 75.00% |
| 5 | 3 | `ctx__eval_len` | `ctx` | 当該wellのevaluation zone総行数。 | 2.592% | 3.613% | 0.00% |
| 6 | 5 | `ctx__raw__y` | `ctx` | horizontal現在行のraw Y。 | 2.507% | 3.208% | 0.00% |
| 7 | 6 | `cand__minus_last` | `cand` | 候補TVT−last-known TVT anchor。 | 2.254% | 2.683% | 0.00% |
| 8 | 4 | `ctx__typewell__rows` | `ctx` | 対応typewellの総行数。 | 2.008% | 3.215% | 0.00% |
| 9 | 11 | `ctx__typewell__tvt_min` | `ctx` | 対応typewellで有限なTVTの最小値。 | 1.983% | 2.204% | 0.00% |
| 10 | 16 | `bank__candidate_minus_median` | `bank` | 現在候補TVT−12候補中央値。 | 1.969% | 1.502% | 0.00% |
| 11 | 29 | `formula__parent__exact_hmm__sigma_tvt` | `formula` | formulaが親`exact_hmm`を含む場合に写した親のTVT uncertainty幅。非該当行は構造的欠損または0 flag。 | 1.855% | 0.659% | 75.00% |
| 12 | 8 | `ctx__typewell__gr_mean` | `ctx` | 対応typewell GRの平均。 | 1.811% | 2.292% | 0.00% |
| 13 | 7 | `ctx__raw__x` | `ctx` | horizontal現在行のraw X。 | 1.799% | 2.544% | 0.00% |
| 14 | 26 | `bank__fixed_std` | `bank` | primitive 6本+named fixed 1本の7候補TVT標準偏差。 | 1.539% | 0.745% | 0.00% |
| 15 | 12 | `ctx__typewell__gr_std` | `ctx` | 対応typewell GRの標準偏差。 | 1.521% | 2.176% | 0.00% |
| 16 | 41 | `formula__parent__selfgr_hmm_a070__sigma_tvt` | `formula` | formulaが親`selfgr_hmm_a070`を含む場合に写した親のTVT uncertainty幅。非該当行は構造的欠損または0 flag。 | 1.499% | 0.278% | 83.33% |
| 17 | 13 | `ctx__raw_delta_last__x` | `ctx` | horizontal現在行X−既知prefix末尾行X。 | 1.374% | 2.006% | 0.00% |
| 18 | 30 | `formula__component_range` | `formula` | formula親候補TVTの最大−最小。 | 1.337% | 0.622% | 50.00% |
| 19 | 15 | `ctx__raw_delta_last__z` | `ctx` | horizontal現在行Z−既知prefix末尾行Z。 | 1.245% | 1.616% | 0.00% |
| 20 | 10 | `ctx__raw__z` | `ctx` | horizontal現在行のraw Z。 | 1.210% | 2.218% | 0.00% |
| 21 | 9 | `ctx__last_known_tvt` | `ctx` | 既知prefix末尾のTVT。候補差分と後段残差targetのanchor。 | 1.206% | 2.229% | 0.00% |
| 22 | 23 | `ctx__md_since` | `ctx` | 既知prefix末尾から現在行までのMD距離。 | 1.077% | 0.840% | 0.00% |
| 23 | 18 | `bank__primary_std` | `bank` | primitive+pairのprimary 11候補TVT標準偏差。 | 1.040% | 1.254% | 0.00% |
| 24 | 19 | `ctx__typewell__gr_max` | `ctx` | 対応typewell GRの最大値。 | 0.950% | 1.094% | 0.00% |
| 25 | 14 | `bank__std` | `bank` | 12候補TVTの行別標準偏差。 | 0.913% | 1.771% | 0.00% |
| 26 | 20 | `ctx__raw_delta_last__y` | `ctx` | horizontal現在行Y−既知prefix末尾行Y。 | 0.862% | 1.057% | 0.00% |
| 27 | 35 | `cand__slope_512` | `cand` | 現在行とwindow行前の候補TVT差を行spanで割った局所slope（window 512）。 | 0.849% | 0.427% | 0.00% |
| 28 | 25 | `bank__range` | `bank` | 12候補TVTの行別最大−最小。 | 0.744% | 0.768% | 0.00% |
| 29 | 21 | `cand__tvt` | `cand` | 現在score対象となっている候補パスのTVT値。 | 0.736% | 0.898% | 0.00% |
| 30 | 27 | `ctx__typewell__gr_min` | `ctx` | 対応typewell GRの最小値。 | 0.648% | 0.707% | 0.00% |
| 31 | 40 | `conf__native__loglik_per_row` | `conf` | source log-likelihoodを対象行数で割った値。 | 0.519% | 0.279% | 83.33% |
| 32 | 24 | `ctx__raw__md` | `ctx` | horizontal現在行のraw MD。 | 0.519% | 0.773% | 0.00% |
| 33 | 22 | `ctx__evaluation_progress` | `ctx` | evaluation zone内の進捗率（先頭0寄り、末尾1）。 | 0.476% | 0.897% | 0.00% |
| 34 | 38 | `conf__native__source_loglik` | `conf` | source generatorが出した未正規化log-likelihood。 | 0.459% | 0.300% | 83.33% |
| 35 | 28 | `bank__median` | `bank` | 12候補TVTの行別中央値。 | 0.446% | 0.699% | 0.00% |
| 36 | 32 | `ctx__well_row_idx` | `ctx` | horizontal well内の元行index。 | 0.427% | 0.538% | 0.00% |
| 37 | 60 | `cand__slope_128` | `cand` | 現在行とwindow行前の候補TVT差を行spanで割った局所slope（window 128）。 | 0.332% | 0.007% | 0.00% |
| 38 | 34 | `bank__fixed_median` | `bank` | primitive 6本+named fixed 1本の7候補TVT中央値。 | 0.322% | 0.467% | 0.00% |
| 39 | 31 | `bank__primary_median` | `bank` | primitive+pairのprimary 11候補TVT中央値。 | 0.312% | 0.581% | 0.00% |
| 40 | 55 | `cand__straightness_128` | `cand` | 局所net TVT変化の絶対値÷累積絶対step。1に近いほど直線的（window 128）。 | 0.311% | 0.013% | 0.00% |
| 41 | 37 | `ctx__typewell__tvt_max` | `ctx` | 対応typewellで有限なTVTの最大値。 | 0.310% | 0.310% | 0.00% |
| 42 | 42 | `bank__candidate_rank_fraction` | `bank` | 現在候補TVTの12候補内順位を0〜1へ正規化。 | 0.293% | 0.257% | 0.00% |
| 43 | 39 | `conf__native__geometry_gr_delta` | `conf` | exp226 geometry候補のGR整合差。 | 0.183% | 0.287% | 91.67% |
| 44 | 36 | `formula__parent__exp226_k16__confidence_valid` | `formula` | formulaが親`exp226_k16`を含む場合に写した親のnative confidence有効flag。非該当行は構造的欠損または0 flag。 | 0.167% | 0.337% | 0.00% |
| 45 | 43 | `id__candidate__exp226_k16` | `id` | 現在候補IDが`exp226_k16`なら1となるone-hot。ordinal indexは使わない。 | 0.163% | 0.240% | 0.00% |
| 46 | 47 | `conf__native__beam_family_std` | `conf` | Beam family内の候補TVT spread。 | 0.145% | 0.100% | 91.67% |
| 47 | 46 | `cand__straightness_512` | `cand` | 局所net TVT変化の絶対値÷累積絶対step。1に近いほど直線的（window 512）。 | 0.139% | 0.106% | 0.00% |
| 48 | 45 | `ctx__raw_delta_last__gr` | `ctx` | horizontal現在行GR−既知prefix末尾行GR。 | 0.114% | 0.116% | 47.78% |
| 49 | 44 | `formula__parent_valid_count` | `formula` | pair/fixed formulaを構成する親候補のうちnative confidenceが有効な本数。 | 0.100% | 0.125% | 50.00% |
| 50 | 48 | `formula__parent__exact_hmm__loglik_per_row` | `formula` | formulaが親`exact_hmm`を含む場合に写した親の1行当たりlog-likelihood。非該当行は構造的欠損または0 flag。 | 0.076% | 0.099% | 75.00% |
| 51 | 54 | `ctx__typewell__row_gr_z` | `ctx` | horizontal現在行GRをtypewell全体のGR平均・標準偏差で標準化した値。 | 0.065% | 0.015% | 31.82% |
| 52 | 50 | `id__candidate__exp226_w500_50_50` | `id` | 現在候補IDが`exp226_w500_50_50`なら1となるone-hot。ordinal indexは使わない。 | 0.065% | 0.057% | 0.00% |
| 53 | 71 | `cand__slope_32` | `cand` | 現在行とwindow行前の候補TVT差を行spanで割った局所slope（window 32）。 | 0.062% | 0.001% | 0.00% |
| 54 | 49 | `formula__parent__selfgr_hmm_a070__loglik_per_row` | `formula` | formulaが親`selfgr_hmm_a070`を含む場合に写した親の1行当たりlog-likelihood。非該当行は構造的欠損または0 flag。 | 0.060% | 0.064% | 83.33% |
| 55 | 51 | `id__kind__pair_mean_50` | `id` | 現在候補が50/50 pair formulaなら1。 | 0.049% | 0.035% | 0.00% |
| 56 | 52 | `id__candidate__beam_mean` | `id` | 現在候補IDが`beam_mean`なら1となるone-hot。ordinal indexは使わない。 | 0.040% | 0.032% | 0.00% |
| 57 | 57 | `formula__parent_direction_agreement` | `formula` | formula親がanchorに対して全て同じ符号方向なら1。 | 0.035% | 0.012% | 50.00% |
| 58 | 53 | `ctx__raw__gr` | `ctx` | horizontal現在行のraw GR。 | 0.024% | 0.015% | 31.82% |
| 59 | 73 | `cand__straightness_32` | `cand` | 局所net TVT変化の絶対値÷累積絶対step。1に近いほど直線的（window 32）。 | 0.021% | 0.000% | 0.00% |
| 60 | 59 | `id__candidate__likpf_mean__exact_hmm` | `id` | 現在候補IDが`likpf_mean__exact_hmm`なら1となるone-hot。ordinal indexは使わない。 | 0.015% | 0.009% | 0.00% |
| 61 | 66 | `id__candidate__likpf_mean` | `id` | 現在候補IDが`likpf_mean`なら1となるone-hot。ordinal indexは使わない。 | 0.014% | 0.001% | 0.00% |
| 62 | 76 | `cand__step` | `cand` | 候補パスの現在行TVT−前行TVT。 | 0.011% | 0.000% | 0.00% |
| 63 | 56 | `formula__parent__selfgr_hmm_a070__score_margin` | `formula` | formulaが親`selfgr_hmm_a070`を含む場合に写した親のsource score margin。非該当行は構造的欠損または0 flag。 | 0.011% | 0.013% | 83.33% |
| 64 | 65 | `bank__candidate_is_max` | `bank` | 現在候補が12候補中の最大TVTなら1。 | 0.010% | 0.003% | 0.00% |
| 65 | 63 | `id__candidate__exp226_k16__likpf_mean` | `id` | 現在候補IDが`exp226_k16__likpf_mean`なら1となるone-hot。ordinal indexは使わない。 | 0.008% | 0.004% | 0.00% |
| 66 | 61 | `bank__candidate_is_min` | `bank` | 現在候補が12候補中の最小TVTなら1。 | 0.005% | 0.005% | 0.00% |
| 67 | 62 | `id__candidate__pf_ancc` | `id` | 現在候補IDが`pf_ancc`なら1となるone-hot。ordinal indexは使わない。 | 0.004% | 0.004% | 0.00% |
| 68 | 58 | `formula__weight_entropy` | `formula` | formula固定weight分布のentropy。 | 0.004% | 0.009% | 50.00% |
| 69 | 70 | `conf__native_valid` | `conf` | 現在候補でsource-native confidenceが有効なら1。候補値のavailabilityとは別。 | 0.002% | 0.001% | 0.00% |
| 70 | 68 | `cand__curvature_512` | `cand` | 局所slopeの1行差分（window 512）。 | 0.002% | 0.001% | 0.00% |
| 71 | 75 | `id__candidate__exp226_k16__selfgr_hmm_a070` | `id` | 現在候補IDが`exp226_k16__selfgr_hmm_a070`なら1となるone-hot。ordinal indexは使わない。 | 0.002% | 0.000% | 0.00% |
| 72 | 76 | `id__kind__primitive` | `id` | 現在候補がprimitive pathなら1。 | 0.002% | 0.000% | 0.00% |
| 73 | 64 | `id__candidate__selfgr_hmm_a070__likpf_mean` | `id` | 現在候補IDが`selfgr_hmm_a070__likpf_mean`なら1となるone-hot。ordinal indexは使わない。 | 0.002% | 0.004% | 0.00% |
| 74 | 67 | `id__candidate__exact_hmm` | `id` | 現在候補IDが`exact_hmm`なら1となるone-hot。ordinal indexは使わない。 | 0.001% | 0.001% | 0.00% |
| 75 | 76 | `conf__native__score_margin` | `conf` | source confidenceのbest/second score margin。主にself-GR matchの分離度。 | 0.001% | 0.000% | 91.67% |
| 76 | 69 | `conf__native__selfgr_quality` | `conf` | self-GR照合のquality score。 | 0.000% | 0.001% | 91.67% |
| 76 | 72 | `id__candidate__exp226_k16__exact_hmm` | `id` | 現在候補IDが`exp226_k16__exact_hmm`なら1となるone-hot。ordinal indexは使わない。 | 0.000% | 0.001% | 0.00% |
| 76 | 74 | `conf__native__selfgr_peak_tvt` | `conf` | self-GR照合で得たpeak位置のTVT。 | 0.000% | 0.000% | 91.67% |
| 76 | 76 | `cand__curvature` | `cand` | 候補パスstepの1行差分。 | 0.000% | 0.000% | 0.00% |
| 76 | 76 | `cand__curvature_128` | `cand` | 局所slopeの1行差分（window 128）。 | 0.000% | 0.000% | 0.00% |
| 76 | 76 | `cand__curvature_32` | `cand` | 局所slopeの1行差分（window 32）。 | 0.000% | 0.000% | 0.00% |
| 76 | 76 | `conf__native__candidate_finite_source` | `conf` | source生成時の候補値がfiniteだったことを示すflag。 | 0.000% | 0.000% | 91.67% |
| 76 | 76 | `conf__native__selfgr_typewell_agreement` | `conf` | self-GR evidenceとtypewell evidenceの一致度。 | 0.000% | 0.000% | 91.67% |
| 76 | 76 | `conf__native__selfgr_valid` | `conf` | self-GR confidenceが有効なら1。 | 0.000% | 0.000% | 91.67% |
| 76 | 76 | `formula__parent__exact_hmm__confidence_valid` | `formula` | formulaが親`exact_hmm`を含む場合に写した親のnative confidence有効flag。非該当行は構造的欠損または0 flag。 | 0.000% | 0.000% | 0.00% |
| 76 | 76 | `formula__parent__selfgr_hmm_a070__confidence_valid` | `formula` | formulaが親`selfgr_hmm_a070`を含む場合に写した親のnative confidence有効flag。非該当行は構造的欠損または0 flag。 | 0.000% | 0.000% | 0.00% |
| 76 | 76 | `formula__weight_max` | `formula` | formulaに使う固定weightの最大値。 | 0.000% | 0.000% | 50.00% |
| 76 | 76 | `id__candidate__selfgr_hmm_a070` | `id` | 現在候補IDが`selfgr_hmm_a070`なら1となるone-hot。ordinal indexは使わない。 | 0.000% | 0.000% | 0.00% |

### Selector入力の重複・相関

- Stage A v4では150候補列から全欠損41、定数5、完全重複16を除外し、採用88列内の完全重複は0。
- 次の14組は600,000 candidate-long auditでPearsonまたはSpearmanの絶対値が0.999以上だったが、差分・欠損pattern・domain定義が異なるためreport-onlyとして保持した。

| left | right | Pearson | Spearman |
| --- | --- | ---: | ---: |
| `ctx__last_known_tvt` | `cand__tvt` | 0.999771 | 0.998395 |
| `ctx__last_known_tvt` | `bank__median` | 0.999801 | 0.998525 |
| `cand__tvt` | `bank__median` | 0.999944 | 0.999598 |
| `ctx__last_known_tvt` | `bank__primary_median` | 0.999792 | 0.998472 |
| `cand__tvt` | `bank__primary_median` | 0.999941 | 0.999576 |
| `bank__median` | `bank__primary_median` | 0.999998 | 0.999985 |
| `bank__std` | `bank__primary_std` | 0.999928 | 0.999931 |
| `ctx__last_known_tvt` | `bank__fixed_median` | 0.999802 | 0.998540 |
| `cand__tvt` | `bank__fixed_median` | 0.999938 | 0.999559 |
| `bank__median` | `bank__fixed_median` | 0.999995 | 0.999946 |
| `bank__primary_median` | `bank__fixed_median` | 0.999994 | 0.999935 |
| `id__candidate__likpf_mean` | `conf__native_valid` | -1.000000 | -1.000000 |
| `id__candidate__exp226_w500_50_50` | `formula__weight_entropy` | 1.000000 | 1.000000 |
| `id__kind__pair_mean_50` | `formula__weight_entropy` | -1.000000 | -1.000000 |

## 3. TVT予測モデル

### 入力・target・出力

| 項目 | 内容 |
| --- | --- |
| model | LightGBM regression 3 configs×5 outer folds=15 models |
| target | `true_tvt - last_known_tvt`残差 |
| base | exp218 lineageをavailability監査したclean 273列 |
| add-only | Stage C v6 nested selector compact 74列 |
| final schema | 347列。列順はinference v4 schemaと一致 |
| prediction | 各モデルの残差予測へanchorを戻し、15本を等重み平均 |
| hard path | 使用しない。候補値、score、marginをLightGBMが連続特徴として組み合わせる |

### 特徴family概要

| family | 列数 | 概要 | gain share | split share |
| --- | ---: | --- | ---: | ---: |
| `selector_compact` | 74 | 12候補のnested dual score、top1/top2、margin、anchor差、spread、one-hot。 | 76.9258% | 25.2013% |
| `base_replay` | 122 | raw trajectory/GR、anchor、PF/Beam/NCC候補、slope、typewell差などhidden-safe replay。 | 12.6067% | 42.6638% |
| `u_projection` | 44 | PF/Beam/likPF候補をU空間へ投影した補正量・residual・相互差。 | 5.8862% | 18.4805% |
| `gr_wavelet_rotation` | 80 | GRのFFT/DWT/rolling/savgol形状・回転・候補cost診断。 | 2.9499% | 12.0582% |
| `learned_likelihood` | 27 | target-free候補値差とmulti-observation GR一致。非nested exp111 score 27列は含まない。 | 1.6314% | 1.5961% |

重要度の読み方:

- compact 74列がgain 76.9258%を占める一方、splitは25.2013%。少数のtop1-minus-anchor特徴が大きいgainを持つ。
- 最重要4列は2 domain×2 objectivesのtop1-minus-anchor。後段targetもanchor残差なので表現が整合している。
- `selector__pred_abs_error__beam_mean`が5.8196%でも、Beam hard path採用を意味しない。Beamの危険度・regimeを表す連続特徴として使われた。
- compact以外では`likpf_mean_d` 1.7257%、`slp_z` 0.6982%、safe learned candidate delta、U-projection residual、GR FFTが上位。

### Stage Dの結果

| 指標 | clean 273 control | 347 add-only | delta |
| --- | ---: | ---: | ---: |
| pooled RMSE | 10.476169 | 8.460811 | -2.015358 |
| near 0-250 | 2.029054 | 1.583151 | -0.445903 |
| 250-1000 | 4.856472 | 4.099686 | -0.756786 |
| 1000+ | 11.535491 | 9.302283 | -2.233208 |
| hidden-like spatial | 12.493329 | 9.420315 | -3.073014 |
| hidden-like typewell-purged | 12.433031 | 9.341391 | -3.091639 |

5/5 foldsで改善し、773 wells中518改善・255悪化。+1 ft超135、+3 ft超39、+5 ft超14 wellsで、worst `70925e23`は+14.482873。Public LB 7.562は更新したが、train-side総合guardはFAILとして保持する。

### 全347 TVT特徴（15-model normalized gain順）

`gain nonzero`は15 add-only modelsのうちgainが正だった本数。重複注記はexp238 lineage監査で既知の関係だけであり、final 347列全組合せの新しい相関matrixではない。

| rank | feature | family | gain share | split share | gain nonzero | 重複注記 | 説明 |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | `selector__primitive_fixed_bank__p_within10__top1_minus_anchor` | `selector_compact` | 17.3289% | 1.3104% | 15/15 |  | primitive+fixed 7候補domainのwithin10確率最大top1候補TVT−last-known anchor |
| 2 | `selector__primitive_pair_bank__p_within10__top1_minus_anchor` | `selector_compact` | 16.8119% | 1.2408% | 15/15 |  | primitive+pair 11候補domainのwithin10確率最大top1候補TVT−last-known anchor |
| 3 | `selector__primitive_fixed_bank__pred_abs_error__top1_minus_anchor` | `selector_compact` | 16.2704% | 1.1686% | 15/15 |  | primitive+fixed 7候補domainの予測誤差最小top1候補TVT−last-known anchor |
| 4 | `selector__primitive_pair_bank__pred_abs_error__top1_minus_anchor` | `selector_compact` | 10.6231% | 1.2011% | 15/15 |  | primitive+pair 11候補domainの予測誤差最小top1候補TVT−last-known anchor |
| 5 | `selector__pred_abs_error__beam_mean` | `selector_compact` | 5.8196% | 0.9611% | 15/15 |  | `beam_mean`候補のnested selector予測絶対誤差。小さいほど有望 |
| 6 | `likpf_mean_d` | `base_replay` | 1.7257% | 0.5981% | 15/15 |  | likelihood-weighted PF平均候補 − anchor。 |
| 7 | `selector__candidate_value_range` | `selector_compact` | 1.3595% | 0.6562% | 15/15 |  | 12候補TVT値の最大−最小 |
| 8 | `selector__p_within10__beam_mean` | `selector_compact` | 1.1078% | 0.5154% | 15/15 |  | `beam_mean`候補が誤差10以内となるnested校正確率 |
| 9 | `selector__candidate_value_std` | `selector_compact` | 0.8626% | 0.5619% | 15/15 |  | 12候補TVT値の標準偏差 |
| 10 | `slp_z` | `base_replay` | 0.6982% | 1.5871% | 15/15 |  | 既知prefixのTVT/Z robust slope。 |
| 11 | `ll_candidate_tvt_likpf_mean_minus_last_known_tvt` | `learned_likelihood` | 0.6854% | 0.2690% | 15/15 | existing_delta_duplicate → keep `likpf_mean_d` (r=1.0) | 元のlikelihood-weighted PF平均 TVT − last_known_tvt。learned予測値ではない。 |
| 12 | `uproj_diff_beam_med_minus_likpf_mean` | `u_projection` | 0.5790% | 0.3164% | 15/15 |  | U-spaceのbeam_med − likpf_mean。Z/anchor共通項は相殺。 |
| 13 | `ll_candidate_tvt_beam_mean_minus_likpf_mean_tvt` | `learned_likelihood` | 0.5513% | 0.2158% | 15/15 | existing_disagreement_duplicate → keep `uproj_diff_beam_mean_minus_likpf_mean` (r=1.0) | 元の複数Beam path平均 TVT − likpf_mean_tvt。learned予測値ではない。 |
| 14 | `selector__pred_abs_error__exp226_k16` | `selector_compact` | 0.5180% | 1.0627% | 15/15 |  | `exp226_k16`候補のnested selector予測絶対誤差 |
| 15 | `uproj_pf_ancc_resid_mad` | `u_projection` | 0.5076% | 1.4351% | 15/15 | u_projection_family_slim_review | pf_ancc U-projection: 上記\|residual\|のwell内中央値。 |
| 16 | `uproj_beam_mean_resid_mad` | `u_projection` | 0.5056% | 1.1798% | 15/15 | u_projection_family_slim_review | beam_mean U-projection: 上記\|residual\|のwell内中央値。 |
| 17 | `uproj_likpf_mean_resid_mad` | `u_projection` | 0.5053% | 1.4158% | 15/15 | u_projection_family_slim_review | likpf_mean U-projection: 上記\|residual\|のwell内中央値。 |
| 18 | `slp_50` | `base_replay` | 0.4955% | 1.5094% | 15/15 |  | 既知prefix末尾50行のTVT/MD robust slope。 |
| 19 | `dx` | `base_replay` | 0.4873% | 1.7208% | 15/15 |  | 予測行X − anchor行X。 |
| 20 | `pfx_rmse` | `base_replay` | 0.4855% | 1.5526% | 15/15 |  | 既知prefix GRとTVT対応typewell GRのRMSE。 |
| 21 | `tw_gr_mean` | `base_replay` | 0.4532% | 1.2692% | 15/15 |  | typewell GR平均（well定数）。 |
| 22 | `slp_b_d_50` | `base_replay` | 0.4308% | 1.8272% | 15/15 |  | 末尾50行 slope外挿TVT − anchor。 |
| 23 | `grwr_gr_missing_rate` | `gr_wavelet_rotation` | 0.4205% | 1.5293% | 15/15 |  | 水平井GRのwell内欠損率。 |
| 24 | `dz` | `base_replay` | 0.4157% | 1.5975% | 15/15 |  | 予測行Z − anchor行Z。 |
| 25 | `uproj_diff_beam_mean_minus_likpf_mean` | `u_projection` | 0.4004% | 0.4527% | 15/15 |  | U-spaceのbeam_mean − likpf_mean。Z/anchor共通項は相殺。 |
| 26 | `uproj_pf_z_resid_mad` | `u_projection` | 0.3979% | 1.3224% | 15/15 | u_projection_family_slim_review | pf_z U-projection: 上記\|residual\|のwell内中央値。 |
| 27 | `grwr_fft_high_frequency_ratio` | `gr_wavelet_rotation` | 0.3897% | 1.3659% | 15/15 |  | 正規化周波数0.35超のGR FFT energy比率。 |
| 28 | `grwr_fft_dominant_frequency_norm` | `gr_wavelet_rotation` | 0.3896% | 1.3968% | 15/15 |  | detrend後GR FFTのdominant frequency（正規化）。 |
| 29 | `selector__p_within10__exp226_k16` | `selector_compact` | 0.3758% | 1.0237% | 15/15 |  | `exp226_k16`候補が誤差10以内となるnested校正確率 |
| 30 | `grwr_fft_dominant_energy_ratio` | `gr_wavelet_rotation` | 0.3757% | 1.1603% | 15/15 |  | detrend後GR FFTの最大周波数bin energy比率。 |
| 31 | `pf_ancc_delta` | `base_replay` | 0.3715% | 0.6424% | 15/15 |  | ANCC PF候補 − last_known_tvt。 |
| 32 | `uproj_beam_med_resid_mad` | `u_projection` | 0.3691% | 1.0846% | 15/15 | u_projection_family_slim_review | beam_med U-projection: 上記\|residual\|のwell内中央値。 |
| 33 | `known_len` | `base_replay` | 0.3650% | 1.1110% | 15/15 |  | 既知TVT_input prefixの行数。 |
| 34 | `z` | `base_replay` | 0.3573% | 1.7201% | 15/15 |  | 予測行のZ座標。 |
| 35 | `cal_b` | `base_replay` | 0.3414% | 1.2146% | 15/15 |  | 既知prefix GRをtypewell GRへaffine fitした切片。 |
| 36 | `cal_a` | `base_replay` | 0.3331% | 1.4139% | 15/15 |  | 既知prefix GRをtypewell GRへaffine fitした傾き。 |
| 37 | `eval_len` | `base_replay` | 0.3266% | 1.2074% | 15/15 |  | 予測tailの行数。 |
| 38 | `grwr_fft_rotation_ratio_x_log1p_md_since` | `gr_wavelet_rotation` | 0.3185% | 1.6153% | 15/15 |  | FFT rotation-band energy比 × log1p(md_since)。 |
| 39 | `grwr_fft_rotation_energy_ratio` | `gr_wavelet_rotation` | 0.3083% | 1.1771% | 15/15 |  | 正規化周波数0.06〜0.35のrotation-band energy比率。 |
| 40 | `grwr_fft_notch_residual_energy_ratio` | `gr_wavelet_rotation` | 0.3049% | 1.1268% | 15/15 |  | 上位3周波数を除いたGR FFT residual energy比率。 |
| 41 | `dy` | `base_replay` | 0.2938% | 1.3584% | 15/15 |  | 予測行Y − anchor行Y。 |
| 42 | `selector__p_within10__exp226_k16__likpf_mean` | `selector_compact` | 0.2914% | 0.5266% | 15/15 |  | `exp226_k16__likpf_mean`候補が誤差10以内となるnested校正確率 |
| 43 | `last_known_tvt` | `base_replay` | 0.2824% | 0.6839% | 15/15 |  | 既知prefix末尾のTVT。全残差予測のanchor。 |
| 44 | `selector__pred_abs_error__exp226_k16__selfgr_hmm_a070` | `selector_compact` | 0.2796% | 0.5685% | 15/15 |  | `exp226_k16__selfgr_hmm_a070`候補のnested selector予測絶対誤差 |
| 45 | `slp_all` | `base_replay` | 0.2789% | 0.9502% | 15/15 |  | 既知prefix全体のTVT/MD robust slope。 |
| 46 | `beam_vloose_d` | `base_replay` | 0.2713% | 0.9174% | 15/15 |  | very-loose Beam候補 − last_known_tvt。 |
| 47 | `selector__pred_abs_error__exact_hmm` | `selector_compact` | 0.2703% | 0.7153% | 15/15 |  | `exact_hmm`候補のnested selector予測絶対誤差 |
| 48 | `selector__pred_abs_error_std` | `selector_compact` | 0.2611% | 0.4190% | 15/15 |  | 12候補の予測絶対誤差score標準偏差 |
| 49 | `grwr_known_prefix_fraction` | `gr_wavelet_rotation` | 0.2594% | 1.0200% | 15/15 |  | 全horizontal行に占める既知prefix比率。 |
| 50 | `beam_cons_d` | `base_replay` | 0.2542% | 0.8361% | 15/15 |  | conservative Beam候補 − last_known_tvt。 |
| 51 | `beam_stiff_d` | `base_replay` | 0.2475% | 0.9242% | 15/15 |  | stiff Beam候補 − last_known_tvt。 |
| 52 | `pf_z_delta` | `base_replay` | 0.2472% | 0.7165% | 15/15 |  | Z-aware PF候補 − last_known_tvt。 |
| 53 | `selector__pred_abs_error__selfgr_hmm_a070` | `selector_compact` | 0.2409% | 0.7532% | 15/15 |  | `selfgr_hmm_a070`候補のnested selector予測絶対誤差 |
| 54 | `ktvt_range` | `base_replay` | 0.2407% | 0.7527% | 15/15 |  | 既知prefix TVT_inputのrange。 |
| 55 | `dxdmd` | `base_replay` | 0.2391% | 0.7408% | 15/15 |  | 行差分 dX/dMD。 |
| 56 | `selector__pred_abs_error__exp226_k16__exact_hmm` | `selector_compact` | 0.2318% | 0.5091% | 15/15 |  | `exp226_k16__exact_hmm`候補のnested selector予測絶対誤差 |
| 57 | `ktvt_std` | `base_replay` | 0.2271% | 0.7244% | 15/15 |  | 既知prefix TVT_inputの標準偏差。 |
| 58 | `beam_vcons_d` | `base_replay` | 0.2156% | 0.9542% | 15/15 |  | very-conservative Beam候補 − last_known_tvt。 |
| 59 | `tw_range` | `base_replay` | 0.2137% | 0.8082% | 15/15 |  | typewell TVT軸のrange（well定数）。 |
| 60 | `selector__p_within10__exp226_k16__selfgr_hmm_a070` | `selector_compact` | 0.2128% | 0.4415% | 15/15 |  | `exp226_k16__selfgr_hmm_a070`候補が誤差10以内となるnested校正確率 |
| 61 | `selector__p_within10__likpf_mean__exact_hmm` | `selector_compact` | 0.2102% | 0.3660% | 15/15 |  | `likpf_mean__exact_hmm`候補が誤差10以内となるnested校正確率 |
| 62 | `selector__p_within10__likpf_mean` | `selector_compact` | 0.2045% | 0.3838% | 15/15 |  | `likpf_mean`候補が誤差10以内となるnested校正確率 |
| 63 | `selector__pred_abs_error__exp226_k16__likpf_mean` | `selector_compact` | 0.1994% | 0.4229% | 15/15 |  | `exp226_k16__likpf_mean`候補のnested selector予測絶対誤差 |
| 64 | `uproj_diff_pf_z_minus_beam_med` | `u_projection` | 0.1979% | 0.5379% | 15/15 |  | U-spaceのpf_z − beam_med。Z/anchor共通項は相殺。 |
| 65 | `selector__p_within10__exact_hmm` | `selector_compact` | 0.1955% | 0.5686% | 15/15 |  | `exact_hmm`候補が誤差10以内となるnested校正確率 |
| 66 | `uproj_diff_pf_z_minus_likpf_mean` | `u_projection` | 0.1925% | 0.6657% | 15/15 |  | U-spaceのpf_z − likpf_mean。Z/anchor共通項は相殺。 |
| 67 | `uproj_diff_pf_z_minus_beam_mean` | `u_projection` | 0.1860% | 0.5334% | 15/15 |  | U-spaceのpf_z − beam_mean。Z/anchor共通項は相殺。 |
| 68 | `selector__pred_abs_error__likpf_mean` | `selector_compact` | 0.1858% | 0.3861% | 15/15 |  | `likpf_mean`候補のnested selector予測絶対誤差 |
| 69 | `ll_candidate_tvt_pf_ancc_minus_last_known_tvt` | `learned_likelihood` | 0.1799% | 0.3448% | 15/15 | existing_delta_duplicate → keep `pf_ancc_delta` (r=1.0) | 元のANCC粒子フィルタ TVT − last_known_tvt。learned予測値ではない。 |
| 70 | `pf_vs_z` | `base_replay` | 0.1793% | 0.6009% | 15/15 |  | ANCC PF候補 − Z-aware PF候補。 |
| 71 | `selector__pred_abs_error__likpf_mean__exact_hmm` | `selector_compact` | 0.1791% | 0.3143% | 15/15 |  | `likpf_mean__exact_hmm`候補のnested selector予測絶対誤差 |
| 72 | `dydmd` | `base_replay` | 0.1719% | 0.6136% | 15/15 |  | 行差分 dY/dMD。 |
| 73 | `selector__primitive_pair_bank__pred_abs_error__top1_score` | `selector_compact` | 0.1677% | 0.3205% | 15/15 |  | primitive+pair domainの予測誤差最小score |
| 74 | `beam_mean_d` | `base_replay` | 0.1661% | 0.6096% | 15/15 |  | 7種類のBeam候補deltaの行別平均。 |
| 75 | `pf_ancc` | `base_replay` | 0.1568% | 0.5589% | 15/15 |  | ANCC particle filterの絶対TVT候補。 |
| 76 | `selector__p_within10__selfgr_hmm_a070__likpf_mean` | `selector_compact` | 0.1552% | 0.2926% | 15/15 |  | `selfgr_hmm_a070__likpf_mean`候補が誤差10以内となるnested校正確率 |
| 77 | `uproj_diff_pf_ancc_minus_beam_mean` | `u_projection` | 0.1546% | 0.4147% | 15/15 |  | U-spaceのpf_ancc − beam_mean。Z/anchor共通項は相殺。 |
| 78 | `grwr_known_prefix_rows_log1p` | `gr_wavelet_rotation` | 0.1542% | 0.5577% | 15/15 |  | 既知TVT_input prefix行数のlog1p。 |
| 79 | `selector__p_within10__exp226_k16__exact_hmm` | `selector_compact` | 0.1520% | 0.3571% | 15/15 |  | `exp226_k16__exact_hmm`候補が誤差10以内となるnested校正確率 |
| 80 | `selector__primitive_pair_bank__pred_abs_error__top2_score` | `selector_compact` | 0.1493% | 0.3602% | 15/15 |  | primitive+pair domainの予測誤差2位score |
| 81 | `selector__p_within10__selfgr_hmm_a070` | `selector_compact` | 0.1471% | 0.5014% | 15/15 |  | `selfgr_hmm_a070`候補が誤差10以内となるnested校正確率 |
| 82 | `beam_sm5_d` | `base_replay` | 0.1443% | 0.6767% | 15/15 |  | smoothed Beam (r=5)候補 − last_known_tvt。 |
| 83 | `beam_loose_d` | `base_replay` | 0.1424% | 0.7034% | 15/15 |  | loose Beam候補 − last_known_tvt。 |
| 84 | `uproj_absdiff_beam_mean_likpf_mean` | `u_projection` | 0.1410% | 0.2402% | 15/15 |  | U-spaceの\|beam_mean − likpf_mean\|。 |
| 85 | `beam_mid_d` | `base_replay` | 0.1390% | 0.6959% | 15/15 |  | middle Beam候補 − last_known_tvt。 |
| 86 | `ll_candidate_tvt_beam_mean_minus_last_known_tvt` | `learned_likelihood` | 0.1383% | 0.4284% | 15/15 | existing_delta_duplicate → keep `beam_mean_d` (r=0.9999999962819376) | 元の複数Beam path平均 TVT − last_known_tvt。learned予測値ではない。 |
| 87 | `selector__p_within10__exp226_w500_50_50` | `selector_compact` | 0.1371% | 0.3703% | 15/15 |  | fixed `exp226_w500_50_50`候補が誤差10以内となるnested校正確率 |
| 88 | `uproj_absdiff_pf_z_beam_med` | `u_projection` | 0.1346% | 0.4118% | 15/15 |  | U-spaceの\|pf_z − beam_med\|。 |
| 89 | `uproj_diff_pf_ancc_minus_likpf_mean` | `u_projection` | 0.1310% | 0.5831% | 15/15 |  | U-spaceのpf_ancc − likpf_mean。Z/anchor共通項は相殺。 |
| 90 | `pf_z` | `base_replay` | 0.1298% | 0.5802% | 15/15 |  | Z-aware particle filterの絶対TVT候補。 |
| 91 | `beam_med_d` | `base_replay` | 0.1265% | 0.5008% | 15/15 |  | 7種類のBeam候補deltaの行別中央値。 |
| 92 | `selector__pred_abs_error_mean` | `selector_compact` | 0.1262% | 0.2482% | 15/15 |  | 12候補の予測絶対誤差score平均 |
| 93 | `selector__pred_abs_error__selfgr_hmm_a070__likpf_mean` | `selector_compact` | 0.1246% | 0.3094% | 15/15 |  | `selfgr_hmm_a070__likpf_mean`候補のnested selector予測絶対誤差 |
| 94 | `selector__pred_abs_error__exp226_w500_50_50` | `selector_compact` | 0.1222% | 0.3566% | 15/15 |  | fixed `exp226_w500_50_50`候補のnested selector予測絶対誤差 |
| 95 | `beam_std_d` | `base_replay` | 0.1191% | 0.6195% | 15/15 |  | 7種類のBeam候補deltaの行別標準偏差。 |
| 96 | `selector__primitive_pair_bank__p_within10__top2_score` | `selector_compact` | 0.1168% | 0.2144% | 15/15 |  | primitive+pair domainのwithin10確率2位score |
| 97 | `uproj_absdiff_pf_z_beam_mean` | `u_projection` | 0.1146% | 0.4218% | 15/15 |  | U-spaceの\|pf_z − beam_mean\|。 |
| 98 | `slp_b_d_all` | `base_replay` | 0.1137% | 1.0331% | 15/15 |  | 全prefix slope外挿TVT − anchor。 |
| 99 | `uproj_diff_beam_mean_minus_beam_med` | `u_projection` | 0.1131% | 0.5034% | 15/15 |  | U-spaceのbeam_mean − beam_med。Z/anchor共通項は相殺。 |
| 100 | `selector__p_within10__pf_ancc` | `selector_compact` | 0.1114% | 0.3500% | 15/15 |  | `pf_ancc`候補が誤差10以内となるnested校正確率 |
| 101 | `uproj_diff_pf_ancc_minus_beam_med` | `u_projection` | 0.1108% | 0.4004% | 15/15 |  | U-spaceのpf_ancc − beam_med。Z/anchor共通項は相殺。 |
| 102 | `selector__primitive_fixed_bank__pred_abs_error__top2_score` | `selector_compact` | 0.1095% | 0.3536% | 15/15 |  | primitive+fixed domainの予測誤差2位score |
| 103 | `uproj_absdiff_pf_ancc_pf_z` | `u_projection` | 0.1095% | 0.3882% | 15/15 |  | U-spaceの\|pf_ancc − pf_z\|。 |
| 104 | `tda80` | `base_replay` | 0.1071% | 0.4165% | 15/15 |  | raw GR − typewell GR(anchor TVT +80 ft)。 |
| 105 | `selector__primitive_fixed_bank__pred_abs_error__top1_score` | `selector_compact` | 0.1033% | 0.2972% | 15/15 |  | primitive+fixed domainの予測誤差最小score |
| 106 | `uproj_source_u_range` | `u_projection` | 0.1025% | 0.3367% | 15/15 | u_projection_family_slim_review | 5候補のU値の行別range。 |
| 107 | `selector__primitive_pair_bank__p_within10__top1_score` | `selector_compact` | 0.0968% | 0.2526% | 15/15 |  | primitive+pair domainのwithin10確率最大score |
| 108 | `selector__p_within10_mean` | `selector_compact` | 0.0963% | 0.2222% | 15/15 |  | 12候補のwithin10確率平均 |
| 109 | `uproj_absdiff_pf_z_likpf_mean` | `u_projection` | 0.0962% | 0.4171% | 15/15 |  | U-spaceの\|pf_z − likpf_mean\|。 |
| 110 | `frac` | `base_replay` | 0.0959% | 1.0910% | 15/15 |  | 予測tail内の0〜1正規化行位置。 |
| 111 | `uproj_absdiff_beam_med_likpf_mean` | `u_projection` | 0.0959% | 0.2164% | 15/15 |  | U-spaceの\|beam_med − likpf_mean\|。 |
| 112 | `uproj_source_u_std` | `u_projection` | 0.0885% | 0.3211% | 15/15 | u_projection_family_slim_review | 5候補のU値の行別標準偏差。 |
| 113 | `selector__primitive_fixed_bank__p_within10__top1_score` | `selector_compact` | 0.0800% | 0.2315% | 15/15 |  | primitive+fixed domainのwithin10確率最大score |
| 114 | `uproj_diff_pf_ancc_minus_pf_z` | `u_projection` | 0.0763% | 0.2736% | 15/15 | existing_disagreement_duplicate → keep `pf_vs_z` (r=1.0) | U-spaceのpf_ancc − pf_z。Z/anchor共通項は相殺。 |
| 115 | `md_since` | `base_replay` | 0.0756% | 0.6520% | 15/15 |  | anchor行からのMD距離。 |
| 116 | `selector__primitive_pair_bank__pred_abs_error__top2_value` | `selector_compact` | 0.0733% | 0.2650% | 15/15 |  | primitive+pair domainの予測誤差top2候補TVT値 |
| 117 | `ll_candidate_tvt_pf_ancc_minus_likpf_mean_tvt` | `learned_likelihood` | 0.0730% | 0.2546% | 15/15 | existing_disagreement_duplicate → keep `uproj_diff_pf_ancc_minus_likpf_mean` (r=1.0) | 元のANCC粒子フィルタ TVT − likpf_mean_tvt。learned予測値ではない。 |
| 118 | `selector__primitive_fixed_bank__pred_abs_error__top1_value` | `selector_compact` | 0.0703% | 0.2802% | 15/15 |  | primitive+fixed domainの予測誤差top1候補TVT値 |
| 119 | `selector__primitive_fixed_bank__pred_abs_error__top2_value` | `selector_compact` | 0.0677% | 0.2866% | 15/15 |  | primitive+fixed domainの予測誤差top2候補TVT値 |
| 120 | `selector__pred_abs_error__pf_ancc` | `selector_compact` | 0.0676% | 0.3004% | 15/15 |  | `pf_ancc`候補のnested selector予測絶対誤差 |
| 121 | `dxy` | `base_replay` | 0.0673% | 0.6067% | 15/15 |  | anchorからのXY平面距離。 |
| 122 | `selector__primitive_fixed_bank__p_within10__top2_value` | `selector_compact` | 0.0663% | 0.2694% | 15/15 |  | primitive+fixed domainのwithin10確率top2候補TVT値 |
| 123 | `selector__primitive_pair_bank__p_within10__top1_value` | `selector_compact` | 0.0658% | 0.3087% | 15/15 |  | primitive+pair domainのwithin10確率top1候補TVT値 |
| 124 | `selector__primitive_fixed_bank__p_within10__top1_value` | `selector_compact` | 0.0658% | 0.3119% | 15/15 |  | primitive+fixed domainのwithin10確率top1候補TVT値 |
| 125 | `selector__primitive_pair_bank__pred_abs_error__top1_value` | `selector_compact` | 0.0644% | 0.2924% | 15/15 |  | primitive+pair domainの予測誤差top1候補TVT値 |
| 126 | `selector__primitive_pair_bank__p_within10__top2_value` | `selector_compact` | 0.0638% | 0.2653% | 15/15 |  | primitive+pair domainのwithin10確率top2候補TVT値 |
| 127 | `uproj_likpf_mean_resid` | `u_projection` | 0.0580% | 0.4129% | 15/15 | sign_flip_duplicate → keep `uproj_likpf_mean_corr` (r=-1.0) | likpf_mean U-projection: source U − well内robust polynomial U-trend。 |
| 128 | `uproj_beam_mean_resid` | `u_projection` | 0.0572% | 0.2658% | 15/15 | sign_flip_duplicate → keep `uproj_beam_mean_corr` (r=-1.0) | beam_mean U-projection: source U − well内robust polynomial U-trend。 |
| 129 | `uproj_likpf_mean_corr` | `u_projection` | 0.0562% | 0.4061% | 15/15 |  | likpf_mean U-projection: well内robust polynomial U-trend − source U。 |
| 130 | `uproj_absdiff_pf_ancc_likpf_mean` | `u_projection` | 0.0549% | 0.3123% | 15/15 |  | U-spaceの\|pf_ancc − likpf_mean\|。 |
| 131 | `selector__p_within10_std` | `selector_compact` | 0.0542% | 0.2919% | 15/15 |  | 12候補のwithin10確率標準偏差 |
| 132 | `uproj_beam_mean_corr` | `u_projection` | 0.0534% | 0.2721% | 15/15 |  | beam_mean U-projection: well内robust polynomial U-trend − source U。 |
| 133 | `selector__primitive_fixed_bank__pred_abs_error__margin` | `selector_compact` | 0.0533% | 0.1428% | 15/15 |  | primitive+fixed domainの予測誤差top1/top2 score margin |
| 134 | `selector__primitive_fixed_bank__p_within10__top2_score` | `selector_compact` | 0.0527% | 0.1819% | 15/15 |  | primitive+fixed domainのwithin10確率2位score |
| 135 | `uproj_absdiff_beam_mean_beam_med` | `u_projection` | 0.0509% | 0.3167% | 15/15 |  | U-spaceの\|beam_mean − beam_med\|。 |
| 136 | `uproj_absdiff_pf_ancc_beam_mean` | `u_projection` | 0.0480% | 0.2780% | 15/15 |  | U-spaceの\|pf_ancc − beam_mean\|。 |
| 137 | `selector__p_within10_candidate_entropy` | `selector_compact` | 0.0455% | 0.2632% | 15/15 |  | 12候補のwithin10確率を候補方向へ正規化したentropy |
| 138 | `uproj_beam_med_corr` | `u_projection` | 0.0447% | 0.2432% | 15/15 |  | beam_med U-projection: well内robust polynomial U-trend − source U。 |
| 139 | `uproj_beam_med_resid` | `u_projection` | 0.0427% | 0.2340% | 15/15 | sign_flip_duplicate → keep `uproj_beam_med_corr` (r=-1.0) | beam_med U-projection: source U − well内robust polynomial U-trend。 |
| 140 | `frac2` | `base_replay` | 0.0418% | 0.4552% | 15/15 |  | fracの二乗。 |
| 141 | `uproj_absdiff_pf_ancc_beam_med` | `u_projection` | 0.0383% | 0.2414% | 15/15 |  | U-spaceの\|pf_ancc − beam_med\|。 |
| 142 | `tda-20` | `base_replay` | 0.0350% | 0.0945% | 15/15 |  | raw GR − typewell GR(anchor TVT -20 ft)。 |
| 143 | `dzdmd` | `base_replay` | 0.0316% | 0.3405% | 15/15 |  | 行差分 dZ/dMD。 |
| 144 | `grm101` | `base_replay` | 0.0280% | 0.2007% | 15/15 |  | raw GRのcentered rolling-101平均。 |
| 145 | `sqrt_frac` | `base_replay` | 0.0229% | 0.2631% | 15/15 |  | fracの平方根。 |
| 146 | `tdbc-20` | `base_replay` | 0.0223% | 0.0944% | 15/15 |  | raw GR − typewell GR(Beam reference TVT -20 ft)。 |
| 147 | `tda40` | `base_replay` | 0.0223% | 0.1591% | 15/15 |  | raw GR − typewell GR(anchor TVT +40 ft)。 |
| 148 | `selector__primitive_fixed_bank__p_within10__margin` | `selector_compact` | 0.0189% | 0.1357% | 15/15 |  | primitive+fixed domainのwithin10確率top1/top2 margin |
| 149 | `tda-80` | `base_replay` | 0.0183% | 0.1051% | 15/15 |  | raw GR − typewell GR(anchor TVT -80 ft)。 |
| 150 | `tdbc40` | `base_replay` | 0.0170% | 0.1319% | 15/15 |  | raw GR − typewell GR(Beam reference TVT +40 ft)。 |
| 151 | `tda-40` | `base_replay` | 0.0167% | 0.1461% | 15/15 |  | raw GR − typewell GR(anchor TVT -40 ft)。 |
| 152 | `tda-10` | `base_replay` | 0.0148% | 0.0601% | 15/15 |  | raw GR − typewell GR(anchor TVT -10 ft)。 |
| 153 | `uproj_pf_ancc_resid` | `u_projection` | 0.0148% | 0.2097% | 15/15 | sign_flip_duplicate → keep `uproj_pf_ancc_corr` (r=-1.0) | pf_ancc U-projection: source U − well内robust polynomial U-trend。 |
| 154 | `uproj_pf_ancc_corr` | `u_projection` | 0.0142% | 0.2265% | 15/15 |  | pf_ancc U-projection: well内robust polynomial U-trend − source U。 |
| 155 | `selector__primitive_pair_bank__p_within10__margin` | `selector_compact` | 0.0139% | 0.1080% | 15/15 |  | primitive+pair domainのwithin10確率top1/top2 margin |
| 156 | `tdbc-40` | `base_replay` | 0.0128% | 0.1044% | 15/15 |  | raw GR − typewell GR(Beam reference TVT -40 ft)。 |
| 157 | `tdbc0` | `base_replay` | 0.0112% | 0.0433% | 15/15 |  | raw GR − typewell GR(Beam reference TVT +0 ft)。 |
| 158 | `uproj_pf_z_corr` | `u_projection` | 0.0101% | 0.1790% | 15/15 |  | pf_z U-projection: well内robust polynomial U-trend − source U。 |
| 159 | `selector__primary_error_top1__likpf_mean` | `selector_compact` | 0.0100% | 0.0570% | 15/15 |  | primary予測誤差top1が`likpf_mean`のone-hot |
| 160 | `uproj_pf_z_resid` | `u_projection` | 0.0087% | 0.1593% | 15/15 | sign_flip_duplicate → keep `uproj_pf_z_corr` (r=-1.0) | pf_z U-projection: source U − well内robust polynomial U-trend。 |
| 161 | `tda10` | `base_replay` | 0.0085% | 0.0734% | 15/15 |  | raw GR − typewell GR(anchor TVT +10 ft)。 |
| 162 | `tda5` | `base_replay` | 0.0085% | 0.0397% | 15/15 |  | raw GR − typewell GR(anchor TVT +5 ft)。 |
| 163 | `selector__primitive_pair_bank__pred_abs_error__margin` | `selector_compact` | 0.0084% | 0.0830% | 15/15 |  | primitive+pair domainの予測誤差top1/top2 margin |
| 164 | `selector__primary_error_top1__beam_mean` | `selector_compact` | 0.0081% | 0.0417% | 15/15 |  | primary予測誤差top1が`beam_mean`のone-hot |
| 165 | `selector__primary_error_top1__exp226_k16` | `selector_compact` | 0.0077% | 0.0450% | 15/15 |  | primary予測誤差top1が`exp226_k16`のone-hot |
| 166 | `pf_ancc_std` | `base_replay` | 0.0072% | 0.2064% | 15/15 |  | ANCC particle filter粒子の行別TVT標準偏差。 |
| 167 | `grm51` | `base_replay` | 0.0062% | 0.0581% | 15/15 |  | raw GRのcentered rolling-51平均。 |
| 168 | `grwr_raw_minus_rolling_absmean_w129` | `gr_wavelet_rotation` | 0.0058% | 0.0926% | 15/15 |  | \|raw GR − rolling denoised GR\|のlocal平均（window 129）。 |
| 169 | `uproj_likpf_mean_abs_resid` | `u_projection` | 0.0057% | 0.1426% | 15/15 | u_projection_family_slim_review | likpf_mean U-projection: 上記residualの絶対値。 |
| 170 | `grwr_raw_minus_savgol_absmean_w129` | `gr_wavelet_rotation` | 0.0050% | 0.0954% | 15/15 |  | \|raw GR − savgol denoised GR\|のlocal平均（window 129）。 |
| 171 | `tda20` | `base_replay` | 0.0049% | 0.0481% | 15/15 |  | raw GR − typewell GR(anchor TVT +20 ft)。 |
| 172 | `grwr_dwt_detail_energy_w129` | `gr_wavelet_rotation` | 0.0044% | 0.0973% | 15/15 |  | db4 level-3 DWT detailのdetail二乗平均（window 129）。 |
| 173 | `uproj_beam_mean_abs_resid` | `u_projection` | 0.0040% | 0.1215% | 15/15 | u_projection_family_slim_review | beam_mean U-projection: 上記residualの絶対値。 |
| 174 | `uproj_beam_med_abs_resid` | `u_projection` | 0.0037% | 0.1175% | 15/15 | u_projection_family_slim_review | beam_med U-projection: 上記residualの絶対値。 |
| 175 | `selector__primary_error_top1__likpf_mean__exact_hmm` | `selector_compact` | 0.0034% | 0.0111% | 11/15 |  | primary予測誤差top1が`likpf_mean__exact_hmm`のone-hot |
| 176 | `gr_vs_tw_anc` | `base_replay` | 0.0032% | 0.0289% | 15/15 |  | raw GR − anchor TVTでのtypewell GR。 |
| 177 | `grwr_dwt_detail_absmean_w129` | `gr_wavelet_rotation` | 0.0031% | 0.0679% | 15/15 |  | db4 level-3 DWT detailのdetail絶対値平均（window 129）。 |
| 178 | `selector__primary_error_top1__selfgr_hmm_a070__likpf_mean` | `selector_compact` | 0.0029% | 0.0086% | 14/15 |  | primary予測誤差top1が`selfgr_hmm_a070__likpf_mean`のone-hot |
| 179 | `uproj_pf_ancc_abs_resid` | `u_projection` | 0.0029% | 0.1197% | 15/15 | u_projection_family_slim_review | pf_ancc U-projection: 上記residualの絶対値。 |
| 180 | `uproj_corr_std` | `u_projection` | 0.0028% | 0.1297% | 14/15 | u_projection_family_slim_review | 5候補のpolynomial correction値の行別標準偏差。 |
| 181 | `uproj_corr_range` | `u_projection` | 0.0028% | 0.1145% | 13/15 | u_projection_family_slim_review | 5候補のpolynomial correction値の行別range。 |
| 182 | `grm21` | `base_replay` | 0.0028% | 0.0196% | 15/15 |  | raw GRのcentered rolling-21平均。 |
| 183 | `grwr_dwt_detail_energy_ratio_w129` | `gr_wavelet_rotation` | 0.0027% | 0.1421% | 13/15 |  | db4 level-3 DWT detailのdetail/(raw-local+detail) energy比（window 129）。 |
| 184 | `tdbc-3` | `base_replay` | 0.0027% | 0.0297% | 14/15 |  | raw GR − typewell GR(Beam reference TVT -3 ft)。 |
| 185 | `gr_nrg` | `base_replay` | 0.0026% | 0.0222% | 15/15 |  | raw GR二乗のrolling-21平均平方根。 |
| 186 | `tdbc10` | `base_replay` | 0.0024% | 0.0399% | 15/15 |  | raw GR − typewell GR(Beam reference TVT +10 ft)。 |
| 187 | `uproj_pf_z_abs_resid` | `u_projection` | 0.0024% | 0.1057% | 14/15 | u_projection_family_slim_review | pf_z U-projection: 上記residualの絶対値。 |
| 188 | `gr_vs_slp_all` | `base_replay` | 0.0024% | 0.0261% | 15/15 |  | raw GR − 全prefix slope外挿TVTでのtypewell GR。 |
| 189 | `tdbc3` | `base_replay` | 0.0023% | 0.0286% | 15/15 |  | raw GR − typewell GR(Beam reference TVT +3 ft)。 |
| 190 | `tdbc-10` | `base_replay` | 0.0022% | 0.0245% | 14/15 |  | raw GR − typewell GR(Beam reference TVT -10 ft)。 |
| 191 | `ll_multiobs_mae_pf_ancc` | `learned_likelihood` | 0.0020% | 0.0249% | 15/15 |  | ANCC粒子フィルタのmulti-observation GR MAE。 |
| 192 | `tdbc5` | `base_replay` | 0.0019% | 0.0324% | 15/15 |  | raw GR − typewell GR(Beam reference TVT +5 ft)。 |
| 193 | `tdbc20` | `base_replay` | 0.0018% | 0.0355% | 15/15 |  | raw GR − typewell GR(Beam reference TVT +20 ft)。 |
| 194 | `tda0` | `base_replay` | 0.0017% | 0.0187% | 15/15 | near_exact_public_replay_duplicate → keep `gr_vs_tw_anc` (r=0.9999999707804362) | raw GR − typewell GR(anchor TVT +0 ft)。 |
| 195 | `grwr_raw_std_w129` | `gr_wavelet_rotation` | 0.0017% | 0.1040% | 11/15 |  | raw GRのlocal rolling標準偏差（window 129）。 |
| 196 | `grwr_raw_minus_dwt_absmean_w129` | `gr_wavelet_rotation` | 0.0016% | 0.0262% | 15/15 |  | \|raw GR − dwt denoised GR\|のlocal平均（window 129）。 |
| 197 | `tda-5` | `base_replay` | 0.0016% | 0.0256% | 15/15 |  | raw GR − typewell GR(anchor TVT -5 ft)。 |
| 198 | `gr_env` | `base_replay` | 0.0015% | 0.0316% | 15/15 |  | raw GRのcentered rolling-21最大値。 |
| 199 | `selector__primary_error_top1__pf_ancc` | `selector_compact` | 0.0014% | 0.0267% | 12/15 |  | primary予測誤差top1が`pf_ancc`のone-hot |
| 200 | `tdbc-5` | `base_replay` | 0.0014% | 0.0245% | 14/15 |  | raw GR − typewell GR(Beam reference TVT -5 ft)。 |
| 201 | `selector__primary_error_top1__exp226_k16__likpf_mean` | `selector_compact` | 0.0014% | 0.0183% | 13/15 |  | primary予測誤差top1が`exp226_k16__likpf_mean`のone-hot |
| 202 | `tdpf15` | `base_replay` | 0.0012% | 0.0311% | 12/15 |  | raw GR − typewell GR(ANCC PF TVT +15 ft)。 |
| 203 | `tdpf2` | `base_replay` | 0.0012% | 0.0193% | 12/15 |  | raw GR − typewell GR(ANCC PF TVT +2 ft)。 |
| 204 | `tdpf-30` | `base_replay` | 0.0011% | 0.0301% | 15/15 |  | raw GR − typewell GR(ANCC PF TVT -30 ft)。 |
| 205 | `grm5` | `base_replay` | 0.0010% | 0.0065% | 14/15 |  | raw GRのcentered rolling-5平均。 |
| 206 | `tdpf-2` | `base_replay` | 0.0010% | 0.0257% | 14/15 |  | raw GR − typewell GR(ANCC PF TVT -2 ft)。 |
| 207 | `tdpf30` | `base_replay` | 0.0009% | 0.0337% | 14/15 |  | raw GR − typewell GR(ANCC PF TVT +30 ft)。 |
| 208 | `grwr_raw_rolling_corr_w129` | `gr_wavelet_rotation` | 0.0009% | 0.0629% | 11/15 |  | raw GRとrolling denoised GRのlocal相関（window 129）。 |
| 209 | `ll_multiobs_mae_beam_mean` | `learned_likelihood` | 0.0008% | 0.0267% | 13/15 |  | 複数Beam path平均のmulti-observation GR MAE。 |
| 210 | `glead30` | `base_replay` | 0.0008% | 0.0118% | 10/15 |  | raw GRを30行leadした値。 |
| 211 | `selector__primary_top1_is_pair` | `selector_compact` | 0.0008% | 0.0100% | 14/15 |  | primary予測誤差top1がpairのフラグ |
| 212 | `selector__primary_top1_is_primitive` | `selector_compact` | 0.0008% | 0.0096% | 13/15 |  | primary予測誤差top1がprimitiveのフラグ |
| 213 | `ll_multiobs_mae_likpf_mean` | `learned_likelihood` | 0.0007% | 0.0194% | 14/15 |  | likelihood-weighted PF平均のmulti-observation GR MAE。 |
| 214 | `tdpf-4` | `base_replay` | 0.0007% | 0.0167% | 12/15 |  | raw GR − typewell GR(ANCC PF TVT -4 ft)。 |
| 215 | `grwr_raw_savgol_corr_w129` | `gr_wavelet_rotation` | 0.0007% | 0.0535% | 10/15 |  | raw GRとsavgol denoised GRのlocal相関（window 129）。 |
| 216 | `grs101` | `base_replay` | 0.0006% | 0.0501% | 12/15 |  | raw GRのcentered rolling-101標準偏差。 |
| 217 | `glag30` | `base_replay` | 0.0006% | 0.0107% | 12/15 |  | raw GRを30行lagした値。 |
| 218 | `tdpf-8` | `base_replay` | 0.0006% | 0.0144% | 11/15 |  | raw GR − typewell GR(ANCC PF TVT -8 ft)。 |
| 219 | `glag15` | `base_replay` | 0.0005% | 0.0043% | 8/15 |  | raw GRを15行lagした値。 |
| 220 | `glead15` | `base_replay` | 0.0005% | 0.0065% | 11/15 |  | raw GRを15行leadした値。 |
| 221 | `grwr_raw_dwt_corr_w129` | `gr_wavelet_rotation` | 0.0005% | 0.0448% | 9/15 |  | raw GRとdwt denoised GRのlocal相関（window 129）。 |
| 222 | `tdpf-15` | `base_replay` | 0.0005% | 0.0243% | 11/15 |  | raw GR − typewell GR(ANCC PF TVT -15 ft)。 |
| 223 | `selector__primary_error_top1__selfgr_hmm_a070` | `selector_compact` | 0.0005% | 0.0081% | 11/15 |  | primary予測誤差top1が`selfgr_hmm_a070`のone-hot |
| 224 | `grwr_dwt_detail_energy_ratio_w065` | `gr_wavelet_rotation` | 0.0005% | 0.0536% | 8/15 |  | db4 level-3 DWT detailのdetail/(raw-local+detail) energy比（window 65）。 |
| 225 | `tdpf0` | `base_replay` | 0.0005% | 0.0064% | 7/15 |  | raw GR − typewell GR(ANCC PF TVT +0 ft)。 |
| 226 | `selector__fixed_top1_is_primitive` | `selector_compact` | 0.0005% | 0.0147% | 11/15 |  | fixed domainの予測誤差top1がprimitiveのフラグ |
| 227 | `tdpf8` | `base_replay` | 0.0004% | 0.0163% | 11/15 |  | raw GR − typewell GR(ANCC PF TVT +8 ft)。 |
| 228 | `selector__primary_error_top1__exp226_k16__exact_hmm` | `selector_compact` | 0.0004% | 0.0096% | 9/15 |  | primary予測誤差top1が`exp226_k16__exact_hmm`のone-hot |
| 229 | `grwr_raw_minus_savgol_absmean_w065` | `gr_wavelet_rotation` | 0.0004% | 0.0204% | 11/15 |  | \|raw GR − savgol denoised GR\|のlocal平均（window 65）。 |
| 230 | `selector__primary_error_top1__exp226_k16__selfgr_hmm_a070` | `selector_compact` | 0.0003% | 0.0092% | 10/15 |  | primary予測誤差top1が`exp226_k16__selfgr_hmm_a070`のone-hot |
| 231 | `glead5` | `base_replay` | 0.0003% | 0.0018% | 8/15 |  | raw GRを5行leadした値。 |
| 232 | `grwr_dwt_detail_energy_w065` | `gr_wavelet_rotation` | 0.0003% | 0.0212% | 11/15 |  | db4 level-3 DWT detailのdetail二乗平均（window 65）。 |
| 233 | `grwr_raw_minus_rolling_absmean_w065` | `gr_wavelet_rotation` | 0.0002% | 0.0189% | 11/15 |  | \|raw GR − rolling denoised GR\|のlocal平均（window 65）。 |
| 234 | `tdpf4` | `base_replay` | 0.0002% | 0.0160% | 9/15 |  | raw GR − typewell GR(ANCC PF TVT +4 ft)。 |
| 235 | `grwr_raw_std_w065_x_log1p_md_since` | `gr_wavelet_rotation` | 0.0002% | 0.0243% | 10/15 |  | raw GR local std(w65) × log1p(md_since)。 |
| 236 | `selector__primitive_fixed_bank__top1_objective_agreement` | `selector_compact` | 0.0002% | 0.0129% | 10/15 |  | primitive+fixed domainで2 objectiveのtop1候補が一致したフラグ |
| 237 | `glag5` | `base_replay` | 0.0001% | 0.0021% | 8/15 |  | raw GRを5行lagした値。 |
| 238 | `selector__fixed_top1_is_fixed` | `selector_compact` | 0.0001% | 0.0044% | 7/15 |  | fixed domainの予測誤差top1がfixed候補のフラグ |
| 239 | `glag1` | `base_replay` | 0.0001% | 0.0014% | 9/15 |  | raw GRを1行lagした値。 |
| 240 | `grwr_raw_std_w065` | `gr_wavelet_rotation` | 0.0001% | 0.0190% | 7/15 |  | raw GRのlocal rolling標準偏差（window 65）。 |
| 241 | `grwr_savgol_31_p2_default_candidate_cost` | `gr_wavelet_rotation` | 0.0001% | 0.0094% | 10/15 |  | savgol_31_p2 GR面: default likPFのlocal GR observation cost。 |
| 242 | `sc25_d` | `base_replay` | 0.0001% | 0.0035% | 5/15 |  | half-window 25のmulti-scale NCC候補 − anchor。 |
| 243 | `grwr_dwt_approx_default_candidate_cost` | `gr_wavelet_rotation` | 0.0001% | 0.0081% | 8/15 |  | dwt_approx GR面: default likPFのlocal GR observation cost。 |
| 244 | `glead1` | `base_replay` | 0.0001% | 0.0010% | 5/15 |  | raw GRを1行leadした値。 |
| 245 | `grwr_dwt_detail_absmean_w065` | `gr_wavelet_rotation` | 0.0001% | 0.0121% | 7/15 |  | db4 level-3 DWT detailのdetail絶対値平均（window 65）。 |
| 246 | `selector__primary_error_top1__exact_hmm` | `selector_compact` | 0.0001% | 0.0045% | 8/15 |  | primary予測誤差top1が`exact_hmm`のone-hot |
| 247 | `grwr_raw_default_candidate_cost` | `gr_wavelet_rotation` | 0.0001% | 0.0030% | 8/15 |  | raw GR面: default likPFのlocal GR observation cost。 |
| 248 | `grwr_raw_rolling_corr_w065` | `gr_wavelet_rotation` | 0.0001% | 0.0156% | 6/15 |  | raw GRとrolling denoised GRのlocal相関（window 65）。 |
| 249 | `gr` | `base_replay` | 0.0001% | 0.0012% | 6/15 |  | 水平井の補間済みraw GR。 |
| 250 | `selector__primitive_pair_bank__top1_objective_agreement` | `selector_compact` | 0.0001% | 0.0088% | 7/15 |  | primitive+pair domainで2 objectiveのtop1候補が一致したフラグ |
| 251 | `grs51` | `base_replay` | 0.0001% | 0.0135% | 6/15 |  | raw GRのcentered rolling-51標準偏差。 |
| 252 | `grwr_raw_minus_savgol_absmean_w033` | `gr_wavelet_rotation` | 0.0001% | 0.0074% | 5/15 |  | \|raw GR − savgol denoised GR\|のlocal平均（window 33）。 |
| 253 | `grwr_raw_savgol_corr_w065` | `gr_wavelet_rotation` | 0.0001% | 0.0117% | 5/15 |  | raw GRとsavgol denoised GRのlocal相関（window 65）。 |
| 254 | `grwr_rolling_median_11_default_candidate_cost` | `gr_wavelet_rotation` | 0.0000% | 0.0067% | 4/15 |  | rolling_median_11 GR面: default likPFのlocal GR observation cost。 |
| 255 | `grwr_dwt_approx_best_minus_default_cost` | `gr_wavelet_rotation` | 0.0000% | 0.0017% | 4/15 |  | dwt_approx GR面: 最良候補cost − default候補cost。 |
| 256 | `grwr_raw_dwt_corr_w065` | `gr_wavelet_rotation` | 0.0000% | 0.0094% | 4/15 |  | raw GRとdwt denoised GRのlocal相関（window 65）。 |
| 257 | `grwr_dwt_detail_energy_ratio_w033` | `gr_wavelet_rotation` | 0.0000% | 0.0099% | 4/15 |  | db4 level-3 DWT detailのdetail/(raw-local+detail) energy比（window 33）。 |
| 258 | `grwr_dwt_approx_zero_minus_best_cost` | `gr_wavelet_rotation` | 0.0000% | 0.0046% | 6/15 |  | dwt_approx GR面: anchor固定候補cost − 全候補最良cost。 |
| 259 | `grwr_raw_minus_dwt_absmean_w065` | `gr_wavelet_rotation` | 0.0000% | 0.0033% | 5/15 |  | \|raw GR − dwt denoised GR\|のlocal平均（window 65）。 |
| 260 | `grwr_raw_std_w033` | `gr_wavelet_rotation` | 0.0000% | 0.0065% | 5/15 |  | raw GRのlocal rolling標準偏差（window 33）。 |
| 261 | `grwr_dwt_detail_energy_w033` | `gr_wavelet_rotation` | 0.0000% | 0.0060% | 5/15 |  | db4 level-3 DWT detailのdetail二乗平均（window 33）。 |
| 262 | `ll_multiobs_score_pf_ancc` | `learned_likelihood` | 0.0000% | 0.0029% | 7/15 |  | ANCC粒子フィルタのmulti-observation一致score。 |
| 263 | `ll_multiobs_score_beam_mean` | `learned_likelihood` | 0.0000% | 0.0035% | 6/15 |  | 複数Beam path平均のmulti-observation一致score。 |
| 264 | `ll_multiobs_score_likpf_mean` | `learned_likelihood` | 0.0000% | 0.0023% | 6/15 |  | likelihood-weighted PF平均のmulti-observation一致score。 |
| 265 | `grwr_savgol_31_p2_zero_minus_best_cost` | `gr_wavelet_rotation` | 0.0000% | 0.0039% | 5/15 |  | savgol_31_p2 GR面: anchor固定候補cost − 全候補最良cost。 |
| 266 | `grwr_raw_minus_rolling_absmean_w033` | `gr_wavelet_rotation` | 0.0000% | 0.0057% | 4/15 |  | \|raw GR − rolling denoised GR\|のlocal平均（window 33）。 |
| 267 | `grwr_rolling_median_11_zero_minus_best_cost` | `gr_wavelet_rotation` | 0.0000% | 0.0042% | 6/15 |  | rolling_median_11 GR面: anchor固定候補cost − 全候補最良cost。 |
| 268 | `grwr_raw_rolling_corr_w033` | `gr_wavelet_rotation` | 0.0000% | 0.0049% | 4/15 |  | raw GRとrolling denoised GRのlocal相関（window 33）。 |
| 269 | `grwr_raw_zero_minus_best_cost` | `gr_wavelet_rotation` | 0.0000% | 0.0017% | 5/15 |  | raw GR面: anchor固定候補cost − 全候補最良cost。 |
| 270 | `grwr_dwt_detail_absmean_w033` | `gr_wavelet_rotation` | 0.0000% | 0.0040% | 5/15 |  | db4 level-3 DWT detailのdetail絶対値平均（window 33）。 |
| 271 | `grwr_savgol_31_p2_default_candidate_ncc` | `gr_wavelet_rotation` | 0.0000% | 0.0049% | 4/15 |  | savgol_31_p2 GR面: default likPFのlocal GR NCC。 |
| 272 | `grwr_raw_dwt_corr_w033` | `gr_wavelet_rotation` | 0.0000% | 0.0030% | 5/15 |  | raw GRとdwt denoised GRのlocal相関（window 33）。 |
| 273 | `grwr_dwt_approx_default_candidate_ncc` | `gr_wavelet_rotation` | 0.0000% | 0.0037% | 4/15 |  | dwt_approx GR面: default likPFのlocal GR NCC。 |
| 274 | `grwr_rolling_median_11_default_candidate_ncc` | `gr_wavelet_rotation` | 0.0000% | 0.0029% | 5/15 |  | rolling_median_11 GR面: default likPFのlocal GR NCC。 |
| 275 | `grwr_raw_savgol_corr_w033` | `gr_wavelet_rotation` | 0.0000% | 0.0030% | 2/15 |  | raw GRとsavgol denoised GRのlocal相関（window 33）。 |
| 276 | `grs21` | `base_replay` | 0.0000% | 0.0029% | 3/15 |  | raw GRのcentered rolling-21標準偏差。 |
| 277 | `grwr_rolling_median_11_best_minus_default_cost` | `gr_wavelet_rotation` | 0.0000% | 0.0018% | 4/15 |  | rolling_median_11 GR面: 最良候補cost − default候補cost。 |
| 278 | `grwr_savgol_31_p2_best_minus_default_cost` | `gr_wavelet_rotation` | 0.0000% | 0.0018% | 4/15 |  | savgol_31_p2 GR面: 最良候補cost − default候補cost。 |
| 279 | `grwr_raw_best_minus_default_cost` | `gr_wavelet_rotation` | 0.0000% | 0.0009% | 4/15 |  | raw GR面: 最良候補cost − default候補cost。 |
| 280 | `sc25_sc` | `base_replay` | 0.0000% | 0.0019% | 2/15 |  | half-window 25 NCC matching score。 |
| 281 | `sc15_d` | `base_replay` | 0.0000% | 0.0005% | 4/15 |  | half-window 15のmulti-scale NCC候補 − anchor。 |
| 282 | `grwr_raw_minus_dwt_absmean_w033` | `gr_wavelet_rotation` | 0.0000% | 0.0009% | 4/15 |  | \|raw GR − dwt denoised GR\|のlocal平均（window 33）。 |
| 283 | `ll_multiobs_ncc_pf_ancc` | `learned_likelihood` | 0.0000% | 0.0011% | 2/15 |  | ANCC粒子フィルタのmulti-observation NCC。 |
| 284 | `ll_multiobs_ncc_beam_mean` | `learned_likelihood` | 0.0000% | 0.0013% | 2/15 |  | 複数Beam path平均のmulti-observation NCC。 |
| 285 | `ll_multiobs_ncc_likpf_mean` | `learned_likelihood` | 0.0000% | 0.0011% | 2/15 |  | likelihood-weighted PF平均のmulti-observation NCC。 |
| 286 | `grwr_raw_default_candidate_ncc` | `gr_wavelet_rotation` | 0.0000% | 0.0005% | 3/15 |  | raw GR面: default likPFのlocal GR NCC。 |
| 287 | `sc_vs_beam` | `base_replay` | 0.0000% | 0.0002% | 2/15 |  | NCC ensemble候補 − conservative Beam候補。 |
| 288 | `grwr_rolling_median_11_zero_rank_norm` | `gr_wavelet_rotation` | 0.0000% | 0.0003% | 3/15 |  | rolling_median_11 GR面: anchor固定候補のcost順位（正規化）。 |
| 289 | `grs5` | `base_replay` | 0.0000% | 0.0002% | 2/15 |  | raw GRのcentered rolling-5標準偏差。 |
| 290 | `grwr_dwt_approx_zero_rank_norm` | `gr_wavelet_rotation` | 0.0000% | 0.0002% | 3/15 |  | dwt_approx GR面: anchor固定候補のcost順位（正規化）。 |
| 291 | `grwr_savgol_31_p2_zero_rank_norm` | `gr_wavelet_rotation` | 0.0000% | 0.0002% | 2/15 |  | savgol_31_p2 GR面: anchor固定候補のcost順位（正規化）。 |
| 292 | `sc15_sc` | `base_replay` | 0.0000% | 0.0003% | 2/15 |  | half-window 15 NCC matching score。 |
| 293 | `ll_candidate_tvt_std` | `learned_likelihood` | 0.0000% | 0.0001% | 2/15 | high_corr_or_redundancy_review | exp111の5候補TVTの行別標準偏差。 |
| 294 | `grwr_rolling_median_11_minus_raw_default_candidate_cost` | `gr_wavelet_rotation` | 0.0000% | 0.0001% | 2/15 |  | rolling_median_11面とraw面のdefault likPF cost差。 |
| 295 | `grwr_savgol_31_p2_minus_raw_default_candidate_ncc` | `gr_wavelet_rotation` | 0.0000% | 0.0001% | 2/15 |  | savgol_31_p2面とraw面のdefault likPF ncc差。 |
| 296 | `ll_candidate_tvt_hyb_minus_likpf_mean_tvt` | `learned_likelihood` | 0.0000% | 0.0001% | 3/15 | high_corr_or_redundancy_review | 元のBeam/NCC hybrid TVT − likpf_mean_tvt。learned予測値ではない。 |
| 297 | `sc8_sc` | `base_replay` | 0.0000% | 0.0001% | 2/15 |  | half-window 8 NCC matching score。 |
| 298 | `grwr_raw_zero_rank_norm` | `gr_wavelet_rotation` | 0.0000% | 0.0001% | 2/15 |  | raw GR面: anchor固定候補のcost順位（正規化）。 |
| 299 | `hyb_d` | `base_replay` | 0.0000% | 0.0001% | 2/15 |  | Beam/NCC hybrid候補 − anchor。 |
| 300 | `grwr_rolling_median_11_minus_raw_default_candidate_ncc` | `gr_wavelet_rotation` | 0.0000% | 0.0000% | 2/15 |  | rolling_median_11面とraw面のdefault likPF ncc差。 |
| 301 | `ll_candidate_tvt_sc_ens_minus_likpf_mean_tvt` | `learned_likelihood` | 0.0000% | 0.0001% | 2/15 | high_corr_or_redundancy_review | 元のmulti-scale NCC ensemble TVT − likpf_mean_tvt。learned予測値ではない。 |
| 302 | `grwr_dwt_approx_candidate_cost_std` | `gr_wavelet_rotation` | 0.0000% | 0.0001% | 2/15 |  | dwt_approx GR面: 候補costの標準偏差。 |
| 303 | `sc_cons_d` | `base_replay` | 0.0000% | 0.0001% | 2/15 |  | sc8/sc15/sc25候補の平均 − anchor。 |
| 304 | `grwr_dwt_approx_candidate_cost_entropy` | `gr_wavelet_rotation` | 0.0000% | 0.0001% | 2/15 |  | dwt_approx GR面: 候補cost分布のentropy。 |
| 305 | `grwr_rolling_median_11_candidate_cost_std` | `gr_wavelet_rotation` | 0.0000% | 0.0000% | 1/15 |  | rolling_median_11 GR面: 候補costの標準偏差。 |
| 306 | `ll_candidate_tvt_range` | `learned_likelihood` | 0.0000% | 0.0001% | 2/15 | high_corr_or_redundancy_review | exp111の5候補TVTの行別range。 |
| 307 | `grwr_savgol_31_p2_candidate_cost_entropy` | `gr_wavelet_rotation` | 0.0000% | 0.0001% | 2/15 |  | savgol_31_p2 GR面: 候補cost分布のentropy。 |
| 308 | `grwr_savgol_31_p2_candidate_cost_std` | `gr_wavelet_rotation` | 0.0000% | 0.0000% | 2/15 |  | savgol_31_p2 GR面: 候補costの標準偏差。 |
| 309 | `tdsc-2` | `base_replay` | 0.0000% | 0.0001% | 1/15 |  | raw GR − typewell GR(NCC ensemble TVT -2 ft)。 |
| 310 | `tdsc-30` | `base_replay` | 0.0000% | 0.0001% | 2/15 |  | raw GR − typewell GR(NCC ensemble TVT -30 ft)。 |
| 311 | `sc_ens_d` | `base_replay` | 0.0000% | 0.0000% | 1/15 |  | multi-scale NCC ensemble候補 − anchor。 |
| 312 | `tdsc30` | `base_replay` | 0.0000% | 0.0000% | 1/15 |  | raw GR − typewell GR(NCC ensemble TVT +30 ft)。 |
| 313 | `ll_multiobs_ncc_sc_ens` | `learned_likelihood` | 0.0000% | 0.0000% | 1/15 |  | multi-scale NCC ensembleのmulti-observation NCC。 |
| 314 | `tdsc-4` | `base_replay` | 0.0000% | 0.0000% | 1/15 |  | raw GR − typewell GR(NCC ensemble TVT -4 ft)。 |
| 315 | `ll_multiobs_mae_hyb` | `learned_likelihood` | 0.0000% | 0.0000% | 1/15 |  | Beam/NCC hybridのmulti-observation GR MAE。 |
| 316 | `grwr_raw_candidate_cost_std` | `gr_wavelet_rotation` | 0.0000% | 0.0000% | 2/15 |  | raw GR面: 候補costの標準偏差。 |
| 317 | `sc8_d` | `base_replay` | 0.0000% | 0.0000% | 1/15 |  | half-window 8のmulti-scale NCC候補 − anchor。 |
| 318 | `grwr_raw_candidate_cost_entropy` | `gr_wavelet_rotation` | 0.0000% | 0.0000% | 2/15 |  | raw GR面: 候補cost分布のentropy。 |
| 319 | `tdsc0` | `base_replay` | 0.0000% | 0.0000% | 1/15 |  | raw GR − typewell GR(NCC ensemble TVT +0 ft)。 |
| 320 | `grwr_savgol_31_p2_minus_raw_default_candidate_cost` | `gr_wavelet_rotation` | 0.0000% | 0.0000% | 1/15 |  | savgol_31_p2面とraw面のdefault likPF cost差。 |
| 321 | `tdsc2` | `base_replay` | 0.0000% | 0.0000% | 1/15 |  | raw GR − typewell GR(NCC ensemble TVT +2 ft)。 |
| 322 | `ll_candidate_tvt_sc_ens_minus_last_known_tvt` | `learned_likelihood` | 0.0000% | 0.0000% | 1/15 | existing_delta_duplicate → keep `sc_ens_d` (r=1.0) | 元のmulti-scale NCC ensemble TVT − last_known_tvt。learned予測値ではない。 |
| 323 | `grwr_rolling_median_11_candidate_cost_entropy` | `gr_wavelet_rotation` | 0.0000% | 0.0000% | 1/15 |  | rolling_median_11 GR面: 候補cost分布のentropy。 |
| 324 | `grwr_dwt_approx_minus_raw_default_candidate_cost` | `gr_wavelet_rotation` | 0.0000% | 0.0000% | 1/15 | all models zero split | dwt_approx面とraw面のdefault likPF cost差。 |
| 325 | `grwr_savgol_31_p2_best_is_default_candidate` | `gr_wavelet_rotation` | 0.0000% | 0.0000% | 1/15 | all models zero split | savgol_31_p2 GR面: 最良cost候補がdefault likPFかのflag。 |
| 326 | `tdsc4` | `base_replay` | 0.0000% | 0.0000% | 1/15 |  | raw GR − typewell GR(NCC ensemble TVT +4 ft)。 |
| 327 | `gr_d1` | `base_replay` | 0.0000% | 0.0000% | 1/15 | all models zero split | raw GRの1階行差分。 |
| 328 | `tdsc-8` | `base_replay` | 0.0000% | 0.0000% | 1/15 |  | raw GR − typewell GR(NCC ensemble TVT -8 ft)。 |
| 329 | `ll_multiobs_score_hyb` | `learned_likelihood` | 0.0000% | 0.0000% | 1/15 |  | Beam/NCC hybridのmulti-observation一致score。 |
| 330 | `grwr_dwt_approx_minus_raw_default_candidate_ncc` | `gr_wavelet_rotation` | 0.0000% | 0.0000% | 1/15 | all models zero split | dwt_approx面とraw面のdefault likPF ncc差。 |
| 331 | `gr_d2` | `base_replay` | 0.0000% | 0.0000% | 0/15 | all models zero split | raw GRの2階行差分。 |
| 332 | `grwr_dwt_approx_best_is_default_candidate` | `gr_wavelet_rotation` | 0.0000% | 0.0000% | 0/15 | all models zero split | dwt_approx GR面: 最良cost候補がdefault likPFかのflag。 |
| 333 | `grwr_dwt_minus_raw_ncc_gap_x_dwt_energy_ratio_w065` | `gr_wavelet_rotation` | 0.0000% | 0.0000% | 0/15 | all models zero split | DWT-vs-raw default NCC差 × DWT detail energy比(w65)。 |
| 334 | `grwr_raw_best_is_default_candidate` | `gr_wavelet_rotation` | 0.0000% | 0.0000% | 0/15 | all models zero split | raw GR面: 最良cost候補がdefault likPFかのflag。 |
| 335 | `grwr_rolling_median_11_best_is_default_candidate` | `gr_wavelet_rotation` | 0.0000% | 0.0000% | 0/15 | all models zero split | rolling_median_11 GR面: 最良cost候補がdefault likPFかのflag。 |
| 336 | `grwr_typewell_gr_missing_rate` | `gr_wavelet_rotation` | 0.0000% | 0.0000% | 0/15 | all models zero split | typewell GRのwell内欠損率。 |
| 337 | `ll_candidate_tvt_hyb_minus_last_known_tvt` | `learned_likelihood` | 0.0000% | 0.0000% | 0/15 | existing_delta_duplicate → keep `hyb_d` (r=1.0) | 元のBeam/NCC hybrid TVT − last_known_tvt。learned予測値ではない。 |
| 338 | `ll_candidate_tvt_likpf_mean_minus_likpf_mean_tvt` | `learned_likelihood` | 0.0000% | 0.0000% | 0/15 | constant_zero; all models zero split | 元のlikelihood-weighted PF平均 TVT − likpf_mean_tvt。learned予測値ではない。 |
| 339 | `ll_multiobs_mae_sc_ens` | `learned_likelihood` | 0.0000% | 0.0000% | 0/15 |  | multi-scale NCC ensembleのmulti-observation GR MAE。 |
| 340 | `ll_multiobs_ncc_hyb` | `learned_likelihood` | 0.0000% | 0.0000% | 0/15 | all models zero split | Beam/NCC hybridのmulti-observation NCC。 |
| 341 | `ll_multiobs_score_sc_ens` | `learned_likelihood` | 0.0000% | 0.0000% | 0/15 |  | multi-scale NCC ensembleのmulti-observation一致score。 |
| 342 | `sc_trust` | `base_replay` | 0.0000% | 0.0000% | 0/15 | constant; all models zero split | 既知prefix長から作るNCC trust。exp238 train面では定数。 |
| 343 | `selector__available_count` | `selector_compact` | 0.0000% | 0.0000% | 0/15 |  | 有限値を持つ候補数 |
| 344 | `selector__confidence_valid_count` | `selector_compact` | 0.0000% | 0.0000% | 0/15 |  | source-native confidenceが有効な候補数 |
| 345 | `tdsc-15` | `base_replay` | 0.0000% | 0.0000% | 0/15 |  | raw GR − typewell GR(NCC ensemble TVT -15 ft)。 |
| 346 | `tdsc15` | `base_replay` | 0.0000% | 0.0000% | 0/15 |  | raw GR − typewell GR(NCC ensemble TVT +15 ft)。 |
| 347 | `tdsc8` | `base_replay` | 0.0000% | 0.0000% | 0/15 |  | raw GR − typewell GR(NCC ensemble TVT +8 ft)。 |

### availabilityで除外した107列

| group | 列数 | current-test生成 | 除外理由 |
| --- | ---: | --- | --- |
| formation系 | 74 | 可能 | full-train FormationPlaneKNN / DenseANCCImputer参照をOOFへ使っており非fold-safe |
| exp111 learned score系 | 27 | 可能 | exp111 fold0 target-trained scoreを全773 train wellsへ適用した非nested stacking |
| GRWR推移依存 | 6 | 可能 | 上記formationまたは非nested learned scoreへ依存 |

「testで生成できる」ことと「OOFでfold-safe」なことは別である。旧380列controlやそのOOFは比較に再利用せず、今回の最終347列にも入れていない。

## Sources

- [`candidate_contract.yaml`](../../experiments/exp264_exp263_candidate_confidence_dual_selector/candidate_contract.yaml)
- [`result.md`](../../experiments/exp264_exp263_candidate_confidence_dual_selector/result.md)
- [`selector_feature_readout_corrected_stage_b_v5.md`](../../experiments/exp264_exp263_candidate_confidence_dual_selector/selector_feature_readout_corrected_stage_b_v5.md)
- [`stage_d_feature_importance_readout_corrected_stage_d_v3.md`](../../experiments/exp264_exp263_candidate_confidence_dual_selector/stage_d_feature_importance_readout_corrected_stage_d_v3.md)
- [`formation 74 contract`](../../experiments/exp264_exp263_candidate_confidence_dual_selector/assets/formation_74_contract.csv)
- 380列全体の監査生成手順: [`studies/exp264_feature_availability_audit.py`](../../studies/exp264_feature_availability_audit.py)
- `kaggle/output/stage_b_v5/artifacts/{candidate_score_oof.parquet,selector_metrics.json,selector_selection_rate.csv,feature_catalog.csv,feature_duplicate_correlation_audit.csv}`
- `kaggle/output/stage_c_v6/artifacts/{nested_selector_metrics.json,feature_schema.json,compact_meta_schema.json,nested_feature_importance_by_objective_outer_inner.csv}`
- `kaggle/output/stage_d_v3_corrected/artifacts/{stage_d_metrics.json,stage_d_feature_importance.csv,stage_d_by_well.csv}`
- `kaggle/output/inference_v4_corrected/artifacts/{inference_metrics.json,stage_d_inference_feature_schema.csv}`
