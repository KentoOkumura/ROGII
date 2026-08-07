# exp264 corrected Stage B version 5 selector特徴量・重要度・重複相関 readout

本readoutはtraining-only formation 12特徴を除外した修正版88列schemaのKaggle CPU version 5を正とする。旧version 2/3の100列重要度はfeature availability leakageにより無効であり、本表へ転記していない。

根拠: `kaggle/output/stage_b_v5/artifacts/selector_metrics.json`、`feature_catalog.csv`、`feature_importance_by_objective_fold.csv`、`feature_duplicate_correlation_audit.csv`、`selector_model_manifest.json`、OOF Parquet 2ファイル。

## 結論

- dual score guardはPASS。expected-error MAE、within10 logloss/Brierはpooledかつ5/5 foldsでcandidate別outer-train priorを改善した。
- hard top1はFAIL。fixed `exp226_w500_50_50`よりRMSEが+0.348673悪く、改善foldは0/5。Viterbi、hard推論、submissionには使わない。
- `conf` groupは予測誤差gainの4.267%を占め、`conf__native__sigma_tvt`は予測誤差4位・2.958%。候補confidence追加は有効だが、hard選択を正当化するほどではない。
- score品質はStage Cのnested compact候補へ進める基準を満たす。ただしStage B OOF compactを後段へ直接入れず、Stage Cでinner OOFを作る必要がある。

## scoreとhard readout

| 指標 | selector | prior / fixed | 差 | fold | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| expected-error MAE | 3.795801 | 5.788783 | -1.992982 | 5/5改善 | PASS |
| within10 logloss | 0.359972 | 0.510131 | -0.150160 | 5/5改善 | PASS |
| within10 Brier | 0.112451 | 0.165095 | -0.052644 | 5/5改善 | PASS |
| hard top1 OOF RMSE | 8.587004 | 8.238332 | +0.348673 | 0/5改善 | FAIL |
| near 0–250 RMSE差 | - | - | +0.079326 | - | FAIL |
| 1000+ RMSE差 | - | - | +0.389208 | - | FAIL |
| worst-well RMSE差 | - | - | +14.684481 | - | FAIL |

### 独立hidden-like post-hoc

assignmentは学習・early stopping・閾値選択に使用せず、完了後に`selector_by_well.csv`へjoinした。

| subset | wells | rows | hard RMSE | fixed RMSE | hard-fixed | 改善/悪化well |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| spatial | 200 | 972,463 | 9.516693 | 8.748108 | +0.768585 | 94/106 |
| typewell-purged | 200 | 976,449 | 9.415269 | 8.694132 | +0.721137 | 95/105 |

## OOF・SHA完全性

- candidate-long: 45,407,868行、12候補×3,783,989行。fold count、candidate順、model_fold対応は全件一致。予測誤差の非finite/負値、確率の非finite/範囲外、label異常は0。
- compact: 3,783,989行、key 6列 + compact 74列。欠損・非finiteは0。
- model: 10/10 byte SHAがmanifestと一致。manifest SHA `d5159ed1...a07d`。
- candidate OOF SHA `9a91b625...d48a`、compact OOF SHA `5485ede1...512a`がmetrics/reproducibility manifestと一致。
- feature schema: 88列、logical SHA `aaef4ffd...ddd3a4`。selected側の全欠損・定数・完全重複は0。構造的NaNを持つselected特徴は25列。

## 特徴group別重要度

| group | pred_abs_error gain share | p_within10 gain share |
| --- | ---: | ---: |
| `bank` | 56.789% | 56.765% |
| `ctx` | 24.728% | 33.684% |
| `formula` | 9.137% | 3.567% |
| `cand` | 4.716% | 4.136% |
| `conf` | 4.267% | 1.461% |
| `id` | 0.364% | 0.387% |

候補bank disagreementが両objectiveの約56.8%を占める。confidenceは予測誤差で4.267%、within10で1.461%で、特にHMM由来`sigma_tvt`が予測誤差校正に効いている。candidate one-hotは0.4%未満だが非ゼロで、ordinal IDは使っていない。

## 候補別score品質

| candidate | expected-error MAE | within10 logloss | within10 Brier |
| --- | ---: | ---: | ---: |
| `exp226_k16__exact_hmm` | 3.112552 | 0.325779 | 0.098640 |
| `exp226_w500_50_50` | 3.113805 | 0.325655 | 0.099150 |
| `exp226_k16__selfgr_hmm_a070` | 3.127198 | 0.324873 | 0.098243 |
| `exp226_k16__likpf_mean` | 3.295010 | 0.338377 | 0.104226 |
| `likpf_mean__exact_hmm` | 3.718808 | 0.363235 | 0.113745 |
| `exp226_k16` | 3.720195 | 0.379012 | 0.120733 |
| `selfgr_hmm_a070__likpf_mean` | 3.732081 | 0.364509 | 0.114443 |
| `likpf_mean` | 4.251639 | 0.383646 | 0.122253 |
| `selfgr_hmm_a070` | 4.253823 | 0.361743 | 0.112883 |
| `exact_hmm` | 4.311015 | 0.358570 | 0.111366 |
| `beam_mean` | 4.428738 | 0.413552 | 0.132489 |
| `pf_ancc` | 4.484758 | 0.380708 | 0.121241 |

pair/fixed候補のscore校正が上位で、`beam_mean`と`pf_ancc`は予測誤差MAEが最も弱い。hard top1で直接置換するより、連続scoreを後段へadd-onlyする設計を支持する。

## primary 11候補のtop1選択率

| candidate | pred_abs_error | p_within10 |
| --- | ---: | ---: |
| `selfgr_hmm_a070` | 19.775% | 9.320% |
| `pf_ancc` | 16.507% | 5.163% |
| `exp226_k16__likpf_mean` | 11.523% | 15.115% |
| `exp226_k16__exact_hmm` | 8.933% | 21.682% |
| `exp226_k16` | 7.685% | 12.947% |
| `likpf_mean` | 7.390% | 5.862% |
| `exp226_k16__selfgr_hmm_a070` | 6.735% | 15.797% |
| `likpf_mean__exact_hmm` | 6.649% | 4.582% |
| `selfgr_hmm_a070__likpf_mean` | 6.159% | 4.267% |
| `exact_hmm` | 5.245% | 2.215% |
| `beam_mean` | 3.399% | 3.050% |

`beam_mean`の選択率は低いが0ではない。ただしhard top1全体がfixedより悪いため、この率を候補採用の単独根拠にはしない。
## 採用88特徴の説明と重要度

shareはobjectiveごとの5-fold mean gain合計で正規化した。pred_abs_error順位を主順序にする。

| pred rank | within rank | feature | group | 説明 | pred share | within share | missing |
| ---: | ---: | --- | --- | --- | ---: | ---: | ---: |
| 1 | 1 | `bank__candidate_mean_abs_disagreement` | `bank` | exp263 deployable candidate bank内の位置、spread、disagreement (bank__candidate_mean_abs_disagreement) | 45.401% | 44.100% | 0.00% |
| 2 | 17 | `formula__component_std` | `formula` | 固定formulaの親値・親confidence・weight・lineage (formula__component_std) | 3.993% | 1.349% | 50.00% |
| 3 | 2 | `bank__candidate_abs_minus_median` | `bank` | exp263 deployable candidate bank内の位置、spread、disagreement (bank__candidate_abs_minus_median) | 3.795% | 4.613% | 0.00% |
| 4 | 33 | `conf__native__sigma_tvt` | `conf` | source-native confidenceと有効性。未提供はNaNとvalidityで表現 (conf__native__sigma_tvt) | 2.958% | 0.494% | 75.00% |
| 5 | 3 | `ctx__eval_len` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__eval_len) | 2.592% | 3.613% | 0.00% |
| 6 | 5 | `ctx__raw__y` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__raw__y) | 2.507% | 3.208% | 0.00% |
| 7 | 6 | `cand__minus_last` | `cand` | 現在候補の値、anchor差、局所shape (cand__minus_last) | 2.254% | 2.683% | 0.00% |
| 8 | 4 | `ctx__typewell__rows` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__typewell__rows) | 2.008% | 3.215% | 0.00% |
| 9 | 11 | `ctx__typewell__tvt_min` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__typewell__tvt_min) | 1.983% | 2.204% | 0.00% |
| 10 | 16 | `bank__candidate_minus_median` | `bank` | exp263 deployable candidate bank内の位置、spread、disagreement (bank__candidate_minus_median) | 1.969% | 1.502% | 0.00% |
| 11 | 29 | `formula__parent__exact_hmm__sigma_tvt` | `formula` | 固定formulaの親値・親confidence・weight・lineage (formula__parent__exact_hmm__sigma_tvt) | 1.855% | 0.659% | 75.00% |
| 12 | 8 | `ctx__typewell__gr_mean` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__typewell__gr_mean) | 1.811% | 2.292% | 0.00% |
| 13 | 7 | `ctx__raw__x` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__raw__x) | 1.799% | 2.544% | 0.00% |
| 14 | 26 | `bank__fixed_std` | `bank` | exp263 deployable candidate bank内の位置、spread、disagreement (bank__fixed_std) | 1.539% | 0.745% | 0.00% |
| 15 | 12 | `ctx__typewell__gr_std` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__typewell__gr_std) | 1.521% | 2.176% | 0.00% |
| 16 | 41 | `formula__parent__selfgr_hmm_a070__sigma_tvt` | `formula` | 固定formulaの親値・親confidence・weight・lineage (formula__parent__selfgr_hmm_a070__sigma_tvt) | 1.499% | 0.278% | 83.33% |
| 17 | 13 | `ctx__raw_delta_last__x` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__raw_delta_last__x) | 1.374% | 2.006% | 0.00% |
| 18 | 30 | `formula__component_range` | `formula` | 固定formulaの親値・親confidence・weight・lineage (formula__component_range) | 1.337% | 0.622% | 50.00% |
| 19 | 15 | `ctx__raw_delta_last__z` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__raw_delta_last__z) | 1.245% | 1.616% | 0.00% |
| 20 | 10 | `ctx__raw__z` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__raw__z) | 1.210% | 2.218% | 0.00% |
| 21 | 9 | `ctx__last_known_tvt` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__last_known_tvt) | 1.206% | 2.229% | 0.00% |
| 22 | 23 | `ctx__md_since` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__md_since) | 1.077% | 0.840% | 0.00% |
| 23 | 18 | `bank__primary_std` | `bank` | exp263 deployable candidate bank内の位置、spread、disagreement (bank__primary_std) | 1.040% | 1.254% | 0.00% |
| 24 | 19 | `ctx__typewell__gr_max` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__typewell__gr_max) | 0.950% | 1.094% | 0.00% |
| 25 | 14 | `bank__std` | `bank` | exp263 deployable candidate bank内の位置、spread、disagreement (bank__std) | 0.913% | 1.771% | 0.00% |
| 26 | 20 | `ctx__raw_delta_last__y` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__raw_delta_last__y) | 0.862% | 1.057% | 0.00% |
| 27 | 35 | `cand__slope_512` | `cand` | 現在候補の値、anchor差、局所shape (cand__slope_512) | 0.849% | 0.427% | 0.00% |
| 28 | 25 | `bank__range` | `bank` | exp263 deployable candidate bank内の位置、spread、disagreement (bank__range) | 0.744% | 0.768% | 0.00% |
| 29 | 21 | `cand__tvt` | `cand` | 現在候補の値、anchor差、局所shape (cand__tvt) | 0.736% | 0.898% | 0.00% |
| 30 | 27 | `ctx__typewell__gr_min` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__typewell__gr_min) | 0.648% | 0.707% | 0.00% |
| 31 | 40 | `conf__native__loglik_per_row` | `conf` | source-native confidenceと有効性。未提供はNaNとvalidityで表現 (conf__native__loglik_per_row) | 0.519% | 0.279% | 83.33% |
| 32 | 24 | `ctx__raw__md` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__raw__md) | 0.519% | 0.773% | 0.00% |
| 33 | 22 | `ctx__evaluation_progress` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__evaluation_progress) | 0.476% | 0.897% | 0.00% |
| 34 | 38 | `conf__native__source_loglik` | `conf` | source-native confidenceと有効性。未提供はNaNとvalidityで表現 (conf__native__source_loglik) | 0.459% | 0.300% | 83.33% |
| 35 | 28 | `bank__median` | `bank` | exp263 deployable candidate bank内の位置、spread、disagreement (bank__median) | 0.446% | 0.699% | 0.00% |
| 36 | 32 | `ctx__well_row_idx` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__well_row_idx) | 0.427% | 0.538% | 0.00% |
| 37 | 60 | `cand__slope_128` | `cand` | 現在候補の値、anchor差、局所shape (cand__slope_128) | 0.332% | 0.007% | 0.00% |
| 38 | 34 | `bank__fixed_median` | `bank` | exp263 deployable candidate bank内の位置、spread、disagreement (bank__fixed_median) | 0.322% | 0.467% | 0.00% |
| 39 | 31 | `bank__primary_median` | `bank` | exp263 deployable candidate bank内の位置、spread、disagreement (bank__primary_median) | 0.312% | 0.581% | 0.00% |
| 40 | 55 | `cand__straightness_128` | `cand` | 現在候補の値、anchor差、局所shape (cand__straightness_128) | 0.311% | 0.013% | 0.00% |
| 41 | 37 | `ctx__typewell__tvt_max` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__typewell__tvt_max) | 0.310% | 0.310% | 0.00% |
| 42 | 42 | `bank__candidate_rank_fraction` | `bank` | exp263 deployable candidate bank内の位置、spread、disagreement (bank__candidate_rank_fraction) | 0.293% | 0.257% | 0.00% |
| 43 | 39 | `conf__native__geometry_gr_delta` | `conf` | source-native confidenceと有効性。未提供はNaNとvalidityで表現 (conf__native__geometry_gr_delta) | 0.183% | 0.287% | 91.67% |
| 44 | 36 | `formula__parent__exp226_k16__confidence_valid` | `formula` | 固定formulaの親値・親confidence・weight・lineage (formula__parent__exp226_k16__confidence_valid) | 0.167% | 0.337% | 0.00% |
| 45 | 43 | `id__candidate__exp226_k16` | `id` | candidate/family/kindのone-hot。ordinal indexは不使用 (id__candidate__exp226_k16) | 0.163% | 0.240% | 0.00% |
| 46 | 47 | `conf__native__beam_family_std` | `conf` | source-native confidenceと有効性。未提供はNaNとvalidityで表現 (conf__native__beam_family_std) | 0.145% | 0.100% | 91.67% |
| 47 | 46 | `cand__straightness_512` | `cand` | 現在候補の値、anchor差、局所shape (cand__straightness_512) | 0.139% | 0.106% | 0.00% |
| 48 | 45 | `ctx__raw_delta_last__gr` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__raw_delta_last__gr) | 0.114% | 0.116% | 47.78% |
| 49 | 44 | `formula__parent_valid_count` | `formula` | 固定formulaの親値・親confidence・weight・lineage (formula__parent_valid_count) | 0.100% | 0.125% | 50.00% |
| 50 | 48 | `formula__parent__exact_hmm__loglik_per_row` | `formula` | 固定formulaの親値・親confidence・weight・lineage (formula__parent__exact_hmm__loglik_per_row) | 0.076% | 0.099% | 75.00% |
| 51 | 54 | `ctx__typewell__row_gr_z` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__typewell__row_gr_z) | 0.065% | 0.015% | 31.82% |
| 52 | 50 | `id__candidate__exp226_w500_50_50` | `id` | candidate/family/kindのone-hot。ordinal indexは不使用 (id__candidate__exp226_w500_50_50) | 0.065% | 0.057% | 0.00% |
| 53 | 71 | `cand__slope_32` | `cand` | 現在候補の値、anchor差、局所shape (cand__slope_32) | 0.062% | 0.001% | 0.00% |
| 54 | 49 | `formula__parent__selfgr_hmm_a070__loglik_per_row` | `formula` | 固定formulaの親値・親confidence・weight・lineage (formula__parent__selfgr_hmm_a070__loglik_per_row) | 0.060% | 0.064% | 83.33% |
| 55 | 51 | `id__kind__pair_mean_50` | `id` | candidate/family/kindのone-hot。ordinal indexは不使用 (id__kind__pair_mean_50) | 0.049% | 0.035% | 0.00% |
| 56 | 52 | `id__candidate__beam_mean` | `id` | candidate/family/kindのone-hot。ordinal indexは不使用 (id__candidate__beam_mean) | 0.040% | 0.032% | 0.00% |
| 57 | 57 | `formula__parent_direction_agreement` | `formula` | 固定formulaの親値・親confidence・weight・lineage (formula__parent_direction_agreement) | 0.035% | 0.012% | 50.00% |
| 58 | 53 | `ctx__raw__gr` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__raw__gr) | 0.024% | 0.015% | 31.82% |
| 59 | 73 | `cand__straightness_32` | `cand` | 現在候補の値、anchor差、局所shape (cand__straightness_32) | 0.021% | 0.000% | 0.00% |
| 60 | 59 | `id__candidate__likpf_mean__exact_hmm` | `id` | candidate/family/kindのone-hot。ordinal indexは不使用 (id__candidate__likpf_mean__exact_hmm) | 0.015% | 0.009% | 0.00% |
| 61 | 66 | `id__candidate__likpf_mean` | `id` | candidate/family/kindのone-hot。ordinal indexは不使用 (id__candidate__likpf_mean) | 0.014% | 0.001% | 0.00% |
| 62 | 76 | `cand__step` | `cand` | 現在候補の値、anchor差、局所shape (cand__step) | 0.011% | 0.000% | 0.00% |
| 63 | 56 | `formula__parent__selfgr_hmm_a070__score_margin` | `formula` | 固定formulaの親値・親confidence・weight・lineage (formula__parent__selfgr_hmm_a070__score_margin) | 0.011% | 0.013% | 83.33% |
| 64 | 65 | `bank__candidate_is_max` | `bank` | exp263 deployable candidate bank内の位置、spread、disagreement (bank__candidate_is_max) | 0.010% | 0.003% | 0.00% |
| 65 | 63 | `id__candidate__exp226_k16__likpf_mean` | `id` | candidate/family/kindのone-hot。ordinal indexは不使用 (id__candidate__exp226_k16__likpf_mean) | 0.008% | 0.004% | 0.00% |
| 66 | 61 | `bank__candidate_is_min` | `bank` | exp263 deployable candidate bank内の位置、spread、disagreement (bank__candidate_is_min) | 0.005% | 0.005% | 0.00% |
| 67 | 62 | `id__candidate__pf_ancc` | `id` | candidate/family/kindのone-hot。ordinal indexは不使用 (id__candidate__pf_ancc) | 0.004% | 0.004% | 0.00% |
| 68 | 58 | `formula__weight_entropy` | `formula` | 固定formulaの親値・親confidence・weight・lineage (formula__weight_entropy) | 0.004% | 0.009% | 50.00% |
| 69 | 70 | `conf__native_valid` | `conf` | source-native confidenceと有効性。未提供はNaNとvalidityで表現 (conf__native_valid) | 0.002% | 0.001% | 0.00% |
| 70 | 68 | `cand__curvature_512` | `cand` | 現在候補の値、anchor差、局所shape (cand__curvature_512) | 0.002% | 0.001% | 0.00% |
| 71 | 75 | `id__candidate__exp226_k16__selfgr_hmm_a070` | `id` | candidate/family/kindのone-hot。ordinal indexは不使用 (id__candidate__exp226_k16__selfgr_hmm_a070) | 0.002% | 0.000% | 0.00% |
| 72 | 76 | `id__kind__primitive` | `id` | candidate/family/kindのone-hot。ordinal indexは不使用 (id__kind__primitive) | 0.002% | 0.000% | 0.00% |
| 73 | 64 | `id__candidate__selfgr_hmm_a070__likpf_mean` | `id` | candidate/family/kindのone-hot。ordinal indexは不使用 (id__candidate__selfgr_hmm_a070__likpf_mean) | 0.002% | 0.004% | 0.00% |
| 74 | 67 | `id__candidate__exact_hmm` | `id` | candidate/family/kindのone-hot。ordinal indexは不使用 (id__candidate__exact_hmm) | 0.001% | 0.001% | 0.00% |
| 75 | 76 | `conf__native__score_margin` | `conf` | source-native confidenceと有効性。未提供はNaNとvalidityで表現 (conf__native__score_margin) | 0.001% | 0.000% | 91.67% |
| 76 | 69 | `conf__native__selfgr_quality` | `conf` | source-native confidenceと有効性。未提供はNaNとvalidityで表現 (conf__native__selfgr_quality) | 0.000% | 0.001% | 91.67% |
| 76 | 72 | `id__candidate__exp226_k16__exact_hmm` | `id` | candidate/family/kindのone-hot。ordinal indexは不使用 (id__candidate__exp226_k16__exact_hmm) | 0.000% | 0.001% | 0.00% |
| 76 | 74 | `conf__native__selfgr_peak_tvt` | `conf` | source-native confidenceと有効性。未提供はNaNとvalidityで表現 (conf__native__selfgr_peak_tvt) | 0.000% | 0.000% | 91.67% |
| 76 | 76 | `cand__curvature` | `cand` | 現在候補の値、anchor差、局所shape (cand__curvature) | 0.000% | 0.000% | 0.00% |
| 76 | 76 | `cand__curvature_128` | `cand` | 現在候補の値、anchor差、局所shape (cand__curvature_128) | 0.000% | 0.000% | 0.00% |
| 76 | 76 | `cand__curvature_32` | `cand` | 現在候補の値、anchor差、局所shape (cand__curvature_32) | 0.000% | 0.000% | 0.00% |
| 76 | 76 | `conf__native__candidate_finite_source` | `conf` | source-native confidenceと有効性。未提供はNaNとvalidityで表現 (conf__native__candidate_finite_source) | 0.000% | 0.000% | 91.67% |
| 76 | 76 | `conf__native__selfgr_typewell_agreement` | `conf` | source-native confidenceと有効性。未提供はNaNとvalidityで表現 (conf__native__selfgr_typewell_agreement) | 0.000% | 0.000% | 91.67% |
| 76 | 76 | `conf__native__selfgr_valid` | `conf` | source-native confidenceと有効性。未提供はNaNとvalidityで表現 (conf__native__selfgr_valid) | 0.000% | 0.000% | 91.67% |
| 76 | 76 | `formula__parent__exact_hmm__confidence_valid` | `formula` | 固定formulaの親値・親confidence・weight・lineage (formula__parent__exact_hmm__confidence_valid) | 0.000% | 0.000% | 0.00% |
| 76 | 76 | `formula__parent__selfgr_hmm_a070__confidence_valid` | `formula` | 固定formulaの親値・親confidence・weight・lineage (formula__parent__selfgr_hmm_a070__confidence_valid) | 0.000% | 0.000% | 0.00% |
| 76 | 76 | `formula__weight_max` | `formula` | 固定formulaの親値・親confidence・weight・lineage (formula__weight_max) | 0.000% | 0.000% | 50.00% |
| 76 | 76 | `id__candidate__selfgr_hmm_a070` | `id` | candidate/family/kindのone-hot。ordinal indexは不使用 (id__candidate__selfgr_hmm_a070) | 0.000% | 0.000% | 0.00% |

## 0 gain特徴

| objective | feature |
| --- | --- |
| `pred_abs_error` | `cand__curvature` |
| `pred_abs_error` | `cand__curvature_128` |
| `pred_abs_error` | `cand__curvature_32` |
| `pred_abs_error` | `conf__native__candidate_finite_source` |
| `pred_abs_error` | `conf__native__selfgr_peak_tvt` |
| `pred_abs_error` | `conf__native__selfgr_quality` |
| `pred_abs_error` | `conf__native__selfgr_typewell_agreement` |
| `pred_abs_error` | `conf__native__selfgr_valid` |
| `pred_abs_error` | `formula__parent__exact_hmm__confidence_valid` |
| `pred_abs_error` | `formula__parent__selfgr_hmm_a070__confidence_valid` |
| `pred_abs_error` | `formula__weight_max` |
| `pred_abs_error` | `id__candidate__exp226_k16__exact_hmm` |
| `pred_abs_error` | `id__candidate__selfgr_hmm_a070` |
| `p_within10` | `cand__curvature` |
| `p_within10` | `cand__curvature_128` |
| `p_within10` | `cand__curvature_32` |
| `p_within10` | `cand__step` |
| `p_within10` | `conf__native__candidate_finite_source` |
| `p_within10` | `conf__native__score_margin` |
| `p_within10` | `conf__native__selfgr_typewell_agreement` |
| `p_within10` | `conf__native__selfgr_valid` |
| `p_within10` | `formula__parent__exact_hmm__confidence_valid` |
| `p_within10` | `formula__parent__selfgr_hmm_a070__confidence_valid` |
| `p_within10` | `formula__weight_max` |
| `p_within10` | `id__candidate__selfgr_hmm_a070` |
| `p_within10` | `id__kind__primitive` |

事前固定した88列schemaなので、本runの結果を見た事後dropはしない。次schemaの整理候補として扱う。

## 重複・相関

- Stage Aで全欠損41、定数5、完全重複16を除外。採用88列内の完全重複は0。
- |Pearson|または|Spearman| 0.999以上の14組はreport-only。

### 除外した完全重複16列

| 除外feature | 保持feature |
| --- | --- |
| `ctx__raw_delta_last__md` | `ctx__md_since` |
| `bank__primary_range` | `bank__range` |
| `bank__fixed_range` | `bank__range` |
| `id__kind__named_fixed` | `id__candidate__exp226_w500_50_50` |
| `id__family__beam` | `id__candidate__beam_mean` |
| `id__family__exact_hmm` | `id__candidate__exact_hmm` |
| `id__family__geometry` | `id__candidate__exp226_k16` |
| `id__family__geometry_exact_hmm_pair` | `id__candidate__exp226_k16__exact_hmm` |
| `id__family__geometry_likelihood_pf_exact_hmm_fixed` | `id__candidate__exp226_w500_50_50` |
| `id__family__geometry_likelihood_pf_pair` | `id__candidate__exp226_k16__likpf_mean` |
| `id__family__geometry_self_gr_hmm_pair` | `id__candidate__exp226_k16__selfgr_hmm_a070` |
| `id__family__likelihood_pf` | `id__candidate__likpf_mean` |
| `id__family__likelihood_pf_exact_hmm_pair` | `id__candidate__likpf_mean__exact_hmm` |
| `id__family__particle_filter` | `id__candidate__pf_ancc` |
| `id__family__self_gr_hmm` | `id__candidate__selfgr_hmm_a070` |
| `id__family__self_gr_hmm_likelihood_pf_pair` | `id__candidate__selfgr_hmm_a070__likpf_mean` |

### 採用側の高相関14組

| feature A | feature B | Pearson | Spearman |
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

## 判断

Stage Bはselector scoreの校正能力について信頼でき、score guardはPASS。一方、hard top1はoverall・5 folds・near・1000+・worst-well・hidden-likeの全観点で不採用。次へ進む場合は同じ88列とdual objectiveをStage Cのouter 5 × inner 4へ移し、後段outer-trainにはinner OOF scoreだけを渡す。

## 2026-07-18 修正版Stage C version 6追記

同じ88列と説明を変更せず、outer 5 × inner 4で再学習した40-model平均重要度を監査した。
全88列の定義、欠損率、完全重複16組、高相関14組は上表を正とし、Stage C結果を見た事後dropは行わない。

| objective | bank | ctx | formula | cand | conf | id |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pred-abs-error | 54.909% | 26.848% | 8.756% | 4.902% | 4.247% | 0.337% |
| within10 | 56.684% | 33.899% | 3.294% | 4.328% | 1.446% | 0.349% |

pred-abs-errorの上位は`bank__candidate_mean_abs_disagreement` 42.552%、
`bank__candidate_abs_minus_median` 4.403%、`formula__component_std` 3.755%、
`ctx__eval_len` 2.910%、`conf__native__sigma_tvt` 2.841%。within10の上位は同じbank disagreement
45.072%、`ctx__eval_len` 3.715%、candidate abs-minus-median 3.627%、raw Y 2.972%、
typewell rows 2.843%だった。confidence groupは主に誤差objectiveへ寄与し、`sigma_tvt`は誤差5位、
within10 33位・0.477%。両objectiveでzero-gainは各11特徴だった。

Stage C score guardとnested leakage auditはPASSしたが、hard top1はfixed比+0.414200でFAIL。
したがって、この重要度は候補別scoreから74列compactを作る根拠として使い、hard候補選択の根拠には使わない。
