# exp264 Stage B selector特徴量・重要度・重複相関 readout

> **無効化済み:** training-only formation raw/delta 12特徴をouter-validで使用したため、以下の重要度と
> selector OOF指標はfeature availability leakageを含み、特徴選択や性能解釈に使用しない。

> **修正版:** Stage A version 4で88列schemaを`aaef4ffd...ddd3a4`として凍結し、Kaggle CPU
> Stage B version 5で再学習・監査を完了した。有効な88特徴の全説明・重要度・重複・相関は
> `selector_feature_readout_corrected_stage_b_v5.md`を正とする。以下の本文は旧version 2/3の無効化済み履歴。

Kaggle train version 2のouter-fold OOF結果は失敗履歴としてのみ保持する。重要度は各objectiveで5 foldsのLightGBM gainを平均し、objective内のgain合計で正規化した。異なるobjective間でraw gainを直接比較しない。

根拠ファイル: `kaggle/output/stage_b_v2/artifacts/feature_catalog.csv`、`feature_importance_by_objective_fold.csv`、`feature_duplicate_correlation_audit.csv`、`selector_metrics.json`。

## 結論

- dual score guardはPASS。予測誤差MAEは `5.788783 → 3.742231`、within10 loglossは `0.510131 → 0.355298`、Brierは `0.165095 → 0.110596`で、3指標すべて5/5 folds改善した。
- hard top1はFAIL。固定`exp226_w500_50_50` `8.238332`に対して `8.362844`、差は `+0.124512`。hard selectorの推論・提出には使わない。
- 旧実行ではStage C/Stage Dまで渡したが、feature availability leakage判明後にscore、guard、RMSE差、重要度を全無効化した。

## Stage C nested selector追補

Kaggle train version 3では同じ100特徴をouter 5 × inner 4 × 2 objectivesの40モデルで学習した。
詳細な100特徴表はStage Aで固定した同一schemaなので下表を正とし、nested重要度の実測値は
`kaggle/output/stage_c_v3/artifacts/nested_feature_importance_by_objective_outer_inner.csv`を正とする。

- expected-error MAEは3.762776、within10 logloss/Brierは0.354702/0.110137で、全指標5/5 foldsでprior改善。
- group gain shareは`pred_abs_error`: bank 51.49%、ctx 30.66%、formula 8.33%、cand 5.18%、conf 4.04%、id 0.31%。
- group gain shareは`p_within10`: bank 54.57%、ctx 36.15%、cand 4.25%、formula 3.29%、conf 1.42%、id 0.33%。
- `conf__native__sigma_tvt`はnested `pred_abs_error`で6位、gain share 2.64%。候補固有confidenceの寄与はStage Bと同程度に再現した。
- nested hard top1は8.420613でfixed 8.238332より+0.182281悪く、hard化には使わない。

## scoreとhard readout

| 指標 | selector | prior / fixed | 差 | 判定 |
| --- | ---: | ---: | ---: | --- |
| expected-error MAE | 3.742231 | 5.788783 | -2.046553 | 5/5改善 |
| within10 logloss | 0.355298 | 0.510131 | -0.154834 | 5/5改善 |
| within10 Brier | 0.110596 | 0.165095 | -0.054499 | 5/5改善 |
| hard top1 OOF RMSE | 8.362844 | 8.238332 | +0.124512 | FAIL |
| near 0–250m RMSE差 | - | - | +0.088746 | 上限+0.02を超過 |
| 1000m+ RMSE差 | - | - | +0.135728 | 上限+0.02を超過 |
| worst-well RMSE差 | - | - | +18.258274 | 上限+0.25を超過 |

### 独立hidden-like post-hoc readout

assignmentはselector学習・閾値選択には使用せず、完了後に`selector_by_well.csv`へjoinした。

| subset | wells | rows | hard RMSE | fixed RMSE | hard-fixed | 改善/悪化well |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| spatial valid | 200 | 972,463 | 9.186219 | 8.748108 | +0.438111 | 100/100 |
| typewell-purged valid | 200 | 976,449 | 9.101735 | 8.694132 | +0.407604 | 100/100 |

## 特徴group別重要度

| group | pred_abs_error gain share | p_within10 gain share |
| --- | ---: | ---: |
| `bank` | 53.87% | 54.13% |
| `ctx` | 28.01% | 36.35% |
| `formula` | 8.83% | 3.43% |
| `cand` | 4.91% | 4.25% |
| `conf` | 4.03% | 1.46% |
| `id` | 0.34% | 0.38% |

bank disagreementが両objectiveで約54%を占める。native confidence groupは予測誤差で約4.03%、within10で約1.46%で、特に`sigma_tvt`は予測誤差5位である。candidate ID one-hotは約0.34%で、ordinal IDなしでもcandidate固有差を補助的に表現している。

## primary 11候補のtop1選択率

固定`exp226_w500_50_50`は別の7候補fallback domainに属するため、このprimary top1率には含めない。

| candidate | pred_abs_error | p_within10 |
| --- | ---: | ---: |
| `selfgr_hmm_a070` | 18.52% | 6.47% |
| `pf_ancc` | 16.15% | 4.76% |
| `exp226_k16__likpf_mean` | 11.34% | 15.22% |
| `exp226_k16__exact_hmm` | 8.59% | 25.60% |
| `exp226_k16` | 8.43% | 10.15% |
| `likpf_mean` | 7.33% | 5.09% |
| `exact_hmm` | 6.87% | 2.38% |
| `likpf_mean__exact_hmm` | 6.76% | 4.36% |
| `exp226_k16__selfgr_hmm_a070` | 6.49% | 18.44% |
| `selfgr_hmm_a070__likpf_mean` | 5.62% | 4.55% |
| `beam_mean` | 3.90% | 2.99% |

## 採用100特徴の説明と重要度

`pred rank/share`を主順序にし、`within rank/share`も併記する。shareは各objective内の5-fold mean gain比率。

| pred rank | within rank | feature | group | 説明 | pred share | within share | missing |
| ---: | ---: | --- | --- | --- | ---: | ---: | ---: |
| 1 | 1 | `bank__candidate_mean_abs_disagreement` | `bank` | exp263 deployable candidate bank内の位置、spread、disagreement (bank__candidate_mean_abs_disagreement) | 41.523% | 41.160% | 0.00% |
| 2 | 16 | `formula__component_std` | `formula` | 固定formulaの親値・親confidence・weight・lineage (formula__component_std) | 3.981% | 1.316% | 50.00% |
| 3 | 2 | `bank__candidate_abs_minus_median` | `bank` | exp263 deployable candidate bank内の位置、spread、disagreement (bank__candidate_abs_minus_median) | 3.808% | 4.553% | 0.00% |
| 4 | 3 | `ctx__eval_len` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__eval_len) | 2.904% | 3.797% | 0.00% |
| 5 | 36 | `conf__native__sigma_tvt` | `conf` | source-native confidenceと有効性。未提供はNaNとvalidityで表現 (conf__native__sigma_tvt) | 2.776% | 0.488% | 75.00% |
| 6 | 12 | `bank__candidate_minus_median` | `bank` | exp263 deployable candidate bank内の位置、spread、disagreement (bank__candidate_minus_median) | 2.547% | 2.027% | 0.00% |
| 7 | 5 | `cand__minus_last` | `cand` | 現在候補の値、anchor差、局所shape (cand__minus_last) | 2.533% | 2.733% | 0.00% |
| 8 | 6 | `ctx__raw__y` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__raw__y) | 2.186% | 2.641% | 0.00% |
| 9 | 4 | `ctx__typewell__rows` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__typewell__rows) | 1.939% | 2.803% | 0.00% |
| 10 | 32 | `formula__parent__exact_hmm__sigma_tvt` | `formula` | 固定formulaの親値・親confidence・weight・lineage (formula__parent__exact_hmm__sigma_tvt) | 1.767% | 0.616% | 75.00% |
| 11 | 14 | `ctx__typewell__tvt_min` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__typewell__tvt_min) | 1.754% | 1.921% | 0.00% |
| 12 | 9 | `ctx__typewell__gr_mean` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__typewell__gr_mean) | 1.730% | 2.299% | 0.00% |
| 13 | 7 | `ctx__raw__x` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__raw__x) | 1.713% | 2.391% | 0.00% |
| 14 | 8 | `ctx__raw_delta_last__ancc` | `ctx` | **無効:** training-only `ANCC`のlast-known差分。current-testでは生成不可 | 1.562% | 2.382% | 0.84% |
| 15 | 13 | `ctx__typewell__gr_std` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__typewell__gr_std) | 1.526% | 1.979% | 0.00% |
| 16 | 52 | `formula__parent__selfgr_hmm_a070__sigma_tvt` | `formula` | 固定formulaの親値・親confidence・weight・lineage (formula__parent__selfgr_hmm_a070__sigma_tvt) | 1.436% | 0.262% | 83.33% |
| 17 | 27 | `bank__fixed_std` | `bank` | exp263 deployable candidate bank内の位置、spread、disagreement (bank__fixed_std) | 1.420% | 0.687% | 0.00% |
| 18 | 33 | `formula__component_range` | `formula` | 固定formulaの親値・親confidence・weight・lineage (formula__component_range) | 1.237% | 0.609% | 50.00% |
| 19 | 11 | `bank__std` | `bank` | exp263 deployable candidate bank内の位置、spread、disagreement (bank__std) | 1.234% | 2.046% | 0.00% |
| 20 | 10 | `ctx__last_known_tvt` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__last_known_tvt) | 1.139% | 2.069% | 0.00% |
| 21 | 15 | `ctx__raw_delta_last__z` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__raw_delta_last__z) | 1.118% | 1.371% | 0.00% |
| 22 | 19 | `bank__primary_std` | `bank` | exp263 deployable candidate bank内の位置、spread、disagreement (bank__primary_std) | 1.027% | 1.094% | 0.00% |
| 23 | 25 | `ctx__md_since` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__md_since) | 1.013% | 0.740% | 0.00% |
| 24 | 17 | `ctx__raw_delta_last__x` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__raw_delta_last__x) | 1.001% | 1.254% | 0.00% |
| 25 | 18 | `ctx__typewell__gr_max` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__typewell__gr_max) | 0.926% | 1.097% | 0.00% |
| 26 | 22 | `bank__range` | `bank` | exp263 deployable candidate bank内の位置、spread、disagreement (bank__range) | 0.810% | 0.783% | 0.00% |
| 27 | 29 | `ctx__raw_delta_last__egfdl` | `ctx` | **無効:** training-only `EGFDL`のlast-known差分。current-testでは生成不可 | 0.809% | 0.663% | 0.14% |
| 28 | 41 | `cand__slope_512` | `cand` | 現在候補の値、anchor差、局所shape (cand__slope_512) | 0.803% | 0.414% | 0.00% |
| 29 | 21 | `cand__tvt` | `cand` | 現在候補の値、anchor差、局所shape (cand__tvt) | 0.750% | 0.958% | 0.00% |
| 30 | 24 | `ctx__raw__ancc` | `ctx` | **無効:** training-only `ANCC` raw値。current-testには列自体がない | 0.681% | 0.765% | 0.84% |
| 31 | 20 | `ctx__raw__z` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__raw__z) | 0.631% | 1.070% | 0.00% |
| 32 | 26 | `ctx__typewell__gr_min` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__typewell__gr_min) | 0.613% | 0.703% | 0.00% |
| 33 | 23 | `ctx__raw_delta_last__astnu` | `ctx` | **無効:** training-only `ASTNU`のlast-known差分。current-testでは生成不可 | 0.610% | 0.772% | 0.00% |
| 34 | 50 | `conf__native__loglik_per_row` | `conf` | source-native confidenceと有効性。未提供はNaNとvalidityで表現 (conf__native__loglik_per_row) | 0.537% | 0.280% | 83.33% |
| 35 | 31 | `ctx__raw__md` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__raw__md) | 0.524% | 0.635% | 0.00% |
| 36 | 44 | `bank__candidate_rank_fraction` | `bank` | exp263 deployable candidate bank内の位置、spread、disagreement (bank__candidate_rank_fraction) | 0.454% | 0.360% | 0.00% |
| 37 | 28 | `ctx__evaluation_progress` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__evaluation_progress) | 0.432% | 0.672% | 0.00% |
| 38 | 34 | `bank__median` | `bank` | exp263 deployable candidate bank内の位置、spread、disagreement (bank__median) | 0.409% | 0.541% | 0.00% |
| 39 | 46 | `conf__native__source_loglik` | `conf` | source-native confidenceと有効性。未提供はNaNとvalidityで表現 (conf__native__source_loglik) | 0.404% | 0.289% | 83.33% |
| 40 | 30 | `ctx__raw__astnu` | `ctx` | **無効:** training-only `ASTNU` raw値。current-testには列自体がない | 0.403% | 0.641% | 0.00% |
| 41 | 40 | `ctx__raw_delta_last__y` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__raw_delta_last__y) | 0.391% | 0.438% | 0.00% |
| 42 | 39 | `ctx__well_row_idx` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__well_row_idx) | 0.380% | 0.452% | 0.00% |
| 43 | 35 | `ctx__raw__egfdl` | `ctx` | **無効:** training-only `EGFDL` raw値。current-testには列自体がない | 0.354% | 0.535% | 0.14% |
| 44 | 42 | `bank__fixed_median` | `bank` | exp263 deployable candidate bank内の位置、spread、disagreement (bank__fixed_median) | 0.325% | 0.396% | 0.00% |
| 45 | 71 | `cand__slope_128` | `cand` | 現在候補の値、anchor差、局所shape (cand__slope_128) | 0.300% | 0.007% | 0.00% |
| 46 | 66 | `cand__straightness_128` | `cand` | 現在候補の値、anchor差、局所shape (cand__straightness_128) | 0.295% | 0.025% | 0.00% |
| 47 | 37 | `bank__primary_median` | `bank` | exp263 deployable candidate bank内の位置、spread、disagreement (bank__primary_median) | 0.284% | 0.471% | 0.00% |
| 48 | 48 | `ctx__raw_delta_last__astnl` | `ctx` | **無効:** training-only `ASTNL`のlast-known差分。current-testでは生成不可 | 0.256% | 0.285% | 0.00% |
| 49 | 38 | `ctx__raw__astnl` | `ctx` | **無効:** training-only `ASTNL` raw値。current-testには列自体がない | 0.242% | 0.462% | 0.00% |
| 50 | 43 | `ctx__raw__egfdu` | `ctx` | **無効:** training-only `EGFDU` raw値。current-testには列自体がない | 0.214% | 0.372% | 0.00% |
| 51 | 54 | `ctx__typewell__tvt_max` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__typewell__tvt_max) | 0.194% | 0.202% | 0.00% |
| 52 | 49 | `ctx__raw__buda` | `ctx` | **無効:** training-only `BUDA` raw値。current-testには列自体がない | 0.188% | 0.282% | 0.00% |
| 53 | 47 | `conf__native__geometry_gr_delta` | `conf` | source-native confidenceと有効性。未提供はNaNとvalidityで表現 (conf__native__geometry_gr_delta) | 0.183% | 0.288% | 91.67% |
| 54 | 51 | `ctx__raw_delta_last__egfdu` | `ctx` | **無効:** training-only `EGFDU`のlast-known差分。current-testでは生成不可 | 0.172% | 0.278% | 0.00% |
| 55 | 55 | `ctx__raw_delta_last__buda` | `ctx` | **無効:** training-only `BUDA`のlast-known差分。current-testでは生成不可 | 0.166% | 0.185% | 0.00% |
| 56 | 53 | `id__candidate__exp226_k16` | `id` | candidate/family/kindのone-hot。ordinal indexは不使用 (id__candidate__exp226_k16) | 0.143% | 0.224% | 0.00% |
| 57 | 45 | `formula__parent__exp226_k16__confidence_valid` | `formula` | 固定formulaの親値・親confidence・weight・lineage (formula__parent__exp226_k16__confidence_valid) | 0.140% | 0.290% | 0.00% |
| 58 | 56 | `ctx__raw_delta_last__gr` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__raw_delta_last__gr) | 0.135% | 0.145% | 47.78% |
| 59 | 58 | `cand__straightness_512` | `cand` | 現在候補の値、anchor差、局所shape (cand__straightness_512) | 0.133% | 0.110% | 0.00% |
| 60 | 60 | `conf__native__beam_family_std` | `conf` | source-native confidenceと有効性。未提供はNaNとvalidityで表現 (conf__native__beam_family_std) | 0.127% | 0.109% | 91.67% |
| 61 | 57 | `formula__parent_valid_count` | `formula` | 固定formulaの親値・親confidence・weight・lineage (formula__parent_valid_count) | 0.084% | 0.128% | 50.00% |
| 62 | 65 | `ctx__typewell__row_gr_z` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__typewell__row_gr_z) | 0.084% | 0.029% | 31.82% |
| 63 | 59 | `formula__parent__exact_hmm__loglik_per_row` | `formula` | 固定formulaの親値・親confidence・weight・lineage (formula__parent__exact_hmm__loglik_per_row) | 0.073% | 0.110% | 75.00% |
| 64 | 62 | `id__candidate__exp226_w500_50_50` | `id` | candidate/family/kindのone-hot。ordinal indexは不使用 (id__candidate__exp226_w500_50_50) | 0.070% | 0.057% | 0.00% |
| 65 | 61 | `formula__parent__selfgr_hmm_a070__loglik_per_row` | `formula` | 固定formulaの親値・親confidence・weight・lineage (formula__parent__selfgr_hmm_a070__loglik_per_row) | 0.064% | 0.072% | 83.33% |
| 66 | 79 | `cand__slope_32` | `cand` | 現在候補の値、anchor差、局所shape (cand__slope_32) | 0.056% | 0.001% | 0.00% |
| 67 | 64 | `id__candidate__beam_mean` | `id` | candidate/family/kindのone-hot。ordinal indexは不使用 (id__candidate__beam_mean) | 0.045% | 0.032% | 0.00% |
| 68 | 68 | `formula__parent_direction_agreement` | `formula` | 固定formulaの親値・親confidence・weight・lineage (formula__parent_direction_agreement) | 0.037% | 0.018% | 50.00% |
| 69 | 63 | `id__kind__pair_mean_50` | `id` | candidate/family/kindのone-hot。ordinal indexは不使用 (id__kind__pair_mean_50) | 0.031% | 0.045% | 0.00% |
| 70 | 84 | `cand__straightness_32` | `cand` | 現在候補の値、anchor差、局所shape (cand__straightness_32) | 0.025% | 0.000% | 0.00% |
| 71 | 67 | `ctx__raw__gr` | `ctx` | raw train/current-testの両方で生成する候補非依存の行・well・typewell context (ctx__raw__gr) | 0.021% | 0.025% | 31.82% |
| 72 | 80 | `cand__step` | `cand` | 現在候補の値、anchor差、局所shape (cand__step) | 0.018% | 0.001% | 0.00% |
| 73 | 77 | `id__candidate__likpf_mean` | `id` | candidate/family/kindのone-hot。ordinal indexは不使用 (id__candidate__likpf_mean) | 0.015% | 0.001% | 0.00% |
| 74 | 72 | `bank__candidate_is_max` | `bank` | exp263 deployable candidate bank内の位置、spread、disagreement (bank__candidate_is_max) | 0.015% | 0.005% | 0.00% |
| 75 | 69 | `id__candidate__likpf_mean__exact_hmm` | `id` | candidate/family/kindのone-hot。ordinal indexは不使用 (id__candidate__likpf_mean__exact_hmm) | 0.015% | 0.012% | 0.00% |
| 76 | 73 | `id__candidate__exp226_k16__likpf_mean` | `id` | candidate/family/kindのone-hot。ordinal indexは不使用 (id__candidate__exp226_k16__likpf_mean) | 0.013% | 0.004% | 0.00% |
| 77 | 74 | `bank__candidate_is_min` | `bank` | exp263 deployable candidate bank内の位置、spread、disagreement (bank__candidate_is_min) | 0.013% | 0.003% | 0.00% |
| 78 | 75 | `formula__weight_entropy` | `formula` | 固定formulaの親値・親confidence・weight・lineage (formula__weight_entropy) | 0.008% | 0.003% | 50.00% |
| 79 | 70 | `formula__parent__selfgr_hmm_a070__score_margin` | `formula` | 固定formulaの親値・親confidence・weight・lineage (formula__parent__selfgr_hmm_a070__score_margin) | 0.007% | 0.010% | 83.33% |
| 80 | 76 | `id__candidate__pf_ancc` | `id` | candidate/family/kindのone-hot。ordinal indexは不使用 (id__candidate__pf_ancc) | 0.005% | 0.002% | 0.00% |
| 81 | 91 | `conf__native_valid` | `conf` | source-native confidenceと有効性。未提供はNaNとvalidityで表現 (conf__native_valid) | 0.005% | 0.000% | 0.00% |
| 82 | 78 | `id__candidate__selfgr_hmm_a070__likpf_mean` | `id` | candidate/family/kindのone-hot。ordinal indexは不使用 (id__candidate__selfgr_hmm_a070__likpf_mean) | 0.002% | 0.001% | 0.00% |
| 83 | 91 | `id__kind__primitive` | `id` | candidate/family/kindのone-hot。ordinal indexは不使用 (id__kind__primitive) | 0.001% | 0.000% | 0.00% |
| 84 | 88 | `conf__native__selfgr_valid` | `conf` | source-native confidenceと有効性。未提供はNaNとvalidityで表現 (conf__native__selfgr_valid) | 0.001% | 0.000% | 91.67% |
| 85 | 82 | `id__candidate__exact_hmm` | `id` | candidate/family/kindのone-hot。ordinal indexは不使用 (id__candidate__exact_hmm) | 0.001% | 0.001% | 0.00% |
| 86 | 86 | `conf__native__selfgr_peak_tvt` | `conf` | source-native confidenceと有効性。未提供はNaNとvalidityで表現 (conf__native__selfgr_peak_tvt) | 0.001% | 0.000% | 91.67% |
| 87 | 83 | `conf__native__score_margin` | `conf` | source-native confidenceと有効性。未提供はNaNとvalidityで表現 (conf__native__score_margin) | 0.001% | 0.000% | 91.67% |
| 88 | 91 | `formula__parent__exact_hmm__confidence_valid` | `formula` | 固定formulaの親値・親confidence・weight・lineage (formula__parent__exact_hmm__confidence_valid) | 0.000% | 0.000% | 0.00% |
| 89 | 81 | `conf__native__selfgr_quality` | `conf` | source-native confidenceと有効性。未提供はNaNとvalidityで表現 (conf__native__selfgr_quality) | 0.000% | 0.001% | 91.67% |
| 90 | 89 | `id__candidate__exp226_k16__selfgr_hmm_a070` | `id` | candidate/family/kindのone-hot。ordinal indexは不使用 (id__candidate__exp226_k16__selfgr_hmm_a070) | 0.000% | 0.000% | 0.00% |
| 91 | 85 | `id__candidate__exp226_k16__exact_hmm` | `id` | candidate/family/kindのone-hot。ordinal indexは不使用 (id__candidate__exp226_k16__exact_hmm) | 0.000% | 0.000% | 0.00% |
| 92 | 87 | `cand__curvature_512` | `cand` | 現在候補の値、anchor差、局所shape (cand__curvature_512) | 0.000% | 0.000% | 0.00% |
| 92 | 90 | `cand__curvature_32` | `cand` | 現在候補の値、anchor差、局所shape (cand__curvature_32) | 0.000% | 0.000% | 0.00% |
| 92 | 91 | `cand__curvature` | `cand` | 現在候補の値、anchor差、局所shape (cand__curvature) | 0.000% | 0.000% | 0.00% |
| 92 | 91 | `cand__curvature_128` | `cand` | 現在候補の値、anchor差、局所shape (cand__curvature_128) | 0.000% | 0.000% | 0.00% |
| 92 | 91 | `conf__native__candidate_finite_source` | `conf` | source-native confidenceと有効性。未提供はNaNとvalidityで表現 (conf__native__candidate_finite_source) | 0.000% | 0.000% | 91.67% |
| 92 | 91 | `conf__native__selfgr_typewell_agreement` | `conf` | source-native confidenceと有効性。未提供はNaNとvalidityで表現 (conf__native__selfgr_typewell_agreement) | 0.000% | 0.000% | 91.67% |
| 92 | 91 | `formula__parent__selfgr_hmm_a070__confidence_valid` | `formula` | 固定formulaの親値・親confidence・weight・lineage (formula__parent__selfgr_hmm_a070__confidence_valid) | 0.000% | 0.000% | 0.00% |
| 92 | 91 | `formula__weight_max` | `formula` | 固定formulaの親値・親confidence・weight・lineage (formula__weight_max) | 0.000% | 0.000% | 50.00% |
| 92 | 91 | `id__candidate__selfgr_hmm_a070` | `id` | candidate/family/kindのone-hot。ordinal indexは不使用 (id__candidate__selfgr_hmm_a070) | 0.000% | 0.000% | 0.00% |

## 0 gain特徴

| objective | feature |
| --- | --- |
| `p_within10` | `cand__curvature` |
| `p_within10` | `cand__curvature_128` |
| `p_within10` | `conf__native__candidate_finite_source` |
| `p_within10` | `conf__native__selfgr_typewell_agreement` |
| `p_within10` | `conf__native_valid` |
| `p_within10` | `formula__parent__exact_hmm__confidence_valid` |
| `p_within10` | `formula__parent__selfgr_hmm_a070__confidence_valid` |
| `p_within10` | `formula__weight_max` |
| `p_within10` | `id__candidate__selfgr_hmm_a070` |
| `p_within10` | `id__kind__primitive` |
| `pred_abs_error` | `cand__curvature` |
| `pred_abs_error` | `cand__curvature_128` |
| `pred_abs_error` | `cand__curvature_32` |
| `pred_abs_error` | `cand__curvature_512` |
| `pred_abs_error` | `conf__native__candidate_finite_source` |
| `pred_abs_error` | `conf__native__selfgr_typewell_agreement` |
| `pred_abs_error` | `formula__parent__selfgr_hmm_a070__confidence_valid` |
| `pred_abs_error` | `formula__weight_max` |
| `pred_abs_error` | `id__candidate__selfgr_hmm_a070` |

0 gainでもschemaはStage A後に固定済みなのでStage B結果を見て事後dropしない。次schemaを切る場合の候補として扱う。

## Stage Aで除外した重複・無情報特徴

### 完全重複 16列

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

### 全欠損 41列

- `conf__native__entropy`
- `conf__native__ess_fraction`
- `conf__native__fallback_rate`
- `conf__native__support_count`
- `formula__parent__exp226_k16__sigma_tvt`
- `formula__parent__exp226_k16__loglik_per_row`
- `formula__parent__exp226_k16__entropy`
- `formula__parent__exp226_k16__score_margin`
- `formula__parent__exp226_k16__support_count`
- `formula__parent__exp226_k16__ess_fraction`
- `formula__parent__exp226_k16__fallback_rate`
- `formula__parent__selfgr_hmm_a070__entropy`
- `formula__parent__selfgr_hmm_a070__support_count`
- `formula__parent__selfgr_hmm_a070__ess_fraction`
- `formula__parent__selfgr_hmm_a070__fallback_rate`
- `formula__parent__likpf_mean__sigma_tvt`
- `formula__parent__likpf_mean__loglik_per_row`
- `formula__parent__likpf_mean__entropy`
- `formula__parent__likpf_mean__score_margin`
- `formula__parent__likpf_mean__support_count`
- `formula__parent__likpf_mean__ess_fraction`
- `formula__parent__likpf_mean__fallback_rate`
- `formula__parent__exact_hmm__entropy`
- `formula__parent__exact_hmm__score_margin`
- `formula__parent__exact_hmm__support_count`
- `formula__parent__exact_hmm__ess_fraction`
- `formula__parent__exact_hmm__fallback_rate`
- `formula__parent__pf_ancc__sigma_tvt`
- `formula__parent__pf_ancc__loglik_per_row`
- `formula__parent__pf_ancc__entropy`
- `formula__parent__pf_ancc__score_margin`
- `formula__parent__pf_ancc__support_count`
- `formula__parent__pf_ancc__ess_fraction`
- `formula__parent__pf_ancc__fallback_rate`
- `formula__parent__beam_mean__sigma_tvt`
- `formula__parent__beam_mean__loglik_per_row`
- `formula__parent__beam_mean__entropy`
- `formula__parent__beam_mean__score_margin`
- `formula__parent__beam_mean__support_count`
- `formula__parent__beam_mean__ess_fraction`
- `formula__parent__beam_mean__fallback_rate`

### 定数 5列

- `cand__available`
- `cand__finite`
- `formula__parent__likpf_mean__confidence_valid`
- `formula__parent__pf_ancc__confidence_valid`
- `formula__parent__beam_mean__confidence_valid`

## 高相関report-only 35組

閾値は|Pearson|または|Spearman| >= 0.999。完全重複ではないためStage Aでは削除していない。

| feature A | feature B | Pearson | Spearman |
| --- | --- | ---: | ---: |
| `id__candidate__likpf_mean` | `conf__native_valid` | -1.000000000 | -1.000000000 |
| `id__kind__pair_mean_50` | `formula__weight_entropy` | -1.000000000 | -1.000000000 |
| `id__candidate__exp226_w500_50_50` | `formula__weight_entropy` | 1.000000000 | 1.000000000 |
| `ctx__raw_delta_last__astnu` | `ctx__raw_delta_last__astnl` | 0.999999999 | 0.999999997 |
| `ctx__raw_delta_last__ancc` | `ctx__raw_delta_last__astnu` | 0.999999999 | 0.999999997 |
| `ctx__raw_delta_last__ancc` | `ctx__raw_delta_last__astnl` | 0.999999999 | 0.999999997 |
| `ctx__raw_delta_last__astnl` | `ctx__raw_delta_last__buda` | 0.999999999 | 0.999999997 |
| `ctx__raw_delta_last__ancc` | `ctx__raw_delta_last__buda` | 0.999999999 | 0.999999996 |
| `ctx__raw_delta_last__astnl` | `ctx__raw_delta_last__egfdl` | 0.999999999 | 0.999999996 |
| `ctx__raw_delta_last__astnu` | `ctx__raw_delta_last__egfdl` | 0.999999999 | 0.999999996 |
| `ctx__raw_delta_last__astnu` | `ctx__raw_delta_last__buda` | 0.999999999 | 0.999999996 |
| `ctx__raw_delta_last__astnu` | `ctx__raw_delta_last__egfdu` | 0.999999999 | 0.999999996 |
| `ctx__raw_delta_last__egfdl` | `ctx__raw_delta_last__buda` | 0.999999999 | 0.999999996 |
| `ctx__raw_delta_last__astnl` | `ctx__raw_delta_last__egfdu` | 0.999999999 | 0.999999996 |
| `ctx__raw_delta_last__ancc` | `ctx__raw_delta_last__egfdl` | 0.999999999 | 0.999999996 |
| `ctx__raw_delta_last__ancc` | `ctx__raw_delta_last__egfdu` | 0.999999999 | 0.999999996 |
| `ctx__raw_delta_last__egfdu` | `ctx__raw_delta_last__egfdl` | 0.999999999 | 0.999999996 |
| `ctx__raw_delta_last__egfdu` | `ctx__raw_delta_last__buda` | 0.999999999 | 0.999999996 |
| `bank__median` | `bank__primary_median` | 0.999998044 | 0.999985437 |
| `bank__median` | `bank__fixed_median` | 0.999994597 | 0.999945520 |
| `bank__primary_median` | `bank__fixed_median` | 0.999993612 | 0.999934733 |
| `cand__tvt` | `bank__median` | 0.999943785 | 0.999597517 |
| `cand__tvt` | `bank__primary_median` | 0.999940675 | 0.999576149 |
| `cand__tvt` | `bank__fixed_median` | 0.999938357 | 0.999559470 |
| `bank__std` | `bank__primary_std` | 0.999927764 | 0.999930955 |
| `ctx__raw__egfdu` | `ctx__raw__egfdl` | 0.999907951 | 0.999849485 |
| `ctx__raw__egfdu` | `ctx__raw__buda` | 0.999870408 | 0.999829171 |
| `ctx__raw__egfdl` | `ctx__raw__buda` | 0.999802568 | 0.999749834 |
| `ctx__last_known_tvt` | `bank__fixed_median` | 0.999801966 | 0.998540446 |
| `ctx__last_known_tvt` | `bank__median` | 0.999801211 | 0.998525108 |
| `ctx__last_known_tvt` | `bank__primary_median` | 0.999792151 | 0.998472378 |
| `ctx__last_known_tvt` | `cand__tvt` | 0.999770761 | 0.998395473 |
| `ctx__raw__astnu` | `ctx__raw__astnl` | 0.999608450 | 0.999580382 |
| `ctx__raw__ancc` | `ctx__raw__buda` | 0.999193235 | 0.999072956 |
| `ctx__raw__ancc` | `ctx__raw__egfdu` | 0.999023465 | 0.998919699 |

## 運用判断

- Stage Bの10 modelとmanifest SHAは全件一致した。
- hard top1、Viterbi、softmax TVT平均、submissionは作らない。
- Stage Cへ進む場合は、outer 5 × inner 4 × 2 objectives = 40 CPU selector boostersを別承認し、Stage B OOF compactをそのままdownstream outer-trainへ使わない。
- Stage Dはmatched control/add-only 30 GPU boostersをversion 2で完走した。same-fold比較はpooled、5/5 folds、distance、hidden-likeで改善したがworst-well guardがFAILしたためinferenceへ進まない。
- Stage Dの全74 compact特徴の説明と15-model正規化gain/split重要度は`stage_d_feature_importance_readout.md`を正とする。
