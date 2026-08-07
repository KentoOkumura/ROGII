# exp264 修正版Stage D v3 compact特徴量・重要度 readout

Kaggle T4 version 3の`selector_compact_addonly` 15 modelsについて、各model内でgain/splitを
全347特徴の合計で正規化し、3 configs × 5 outer foldsで平均した。shareはcompact 74列内ではなく、
clean base 273 + compact 74列のadd-onlyモデル全体に占める割合である。

根拠は`kaggle/output/stage_d_v3_corrected/artifacts/stage_d_feature_importance.csv`、
`stage_d_metrics.json`、`stage_d_by_well.csv`。全74列は修正版Stage C v6の
`compact_meta_schema.json`と一致する。

## 結論

- clean 273 control 10.476169に対して347列add-onlyは8.460811、delta -2.015358、5/5 folds改善。
- compact 74列はadd-only全体gainの76.9258%、splitの25.2013%を占める。
- 上位4列は2 legal domainのwithin10/error top1候補値とlast-known anchorとの差で、合計gainは61.0343%。
- 5位は`beam_mean`のnested予測誤差scoreで5.8196%。hard `beam_mean`選択ではなく、候補の危険度を
  後段が連続regime featureとして使った結果と解釈する。
- worst `70925e23`は+14.482873悪化したため、重要度が強くても推論採用条件は満たさない。

## 全74 compact特徴

`gain nonzero`は15 add-only modelsのうちgainが正だった本数。Stage Cでschemaを固定済みなので、
Stage D重要度を見た事後dropは行わない。

| rank | compact feature | 説明 | gain share | split share | gain nonzero |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | `selector__primitive_fixed_bank__p_within10__top1_minus_anchor` | primitive+fixed 7候補domainのwithin10確率最大top1候補TVT−last-known anchor | 17.3289% | 1.3104% | 15/15 |
| 2 | `selector__primitive_pair_bank__p_within10__top1_minus_anchor` | primitive+pair 11候補domainのwithin10確率最大top1候補TVT−last-known anchor | 16.8119% | 1.2408% | 15/15 |
| 3 | `selector__primitive_fixed_bank__pred_abs_error__top1_minus_anchor` | primitive+fixed 7候補domainの予測誤差最小top1候補TVT−last-known anchor | 16.2704% | 1.1686% | 15/15 |
| 4 | `selector__primitive_pair_bank__pred_abs_error__top1_minus_anchor` | primitive+pair 11候補domainの予測誤差最小top1候補TVT−last-known anchor | 10.6231% | 1.2011% | 15/15 |
| 5 | `selector__pred_abs_error__beam_mean` | `beam_mean`候補のnested selector予測絶対誤差。小さいほど有望 | 5.8196% | 0.9611% | 15/15 |
| 6 | `selector__candidate_value_range` | 12候補TVT値の最大−最小 | 1.3595% | 0.6562% | 15/15 |
| 7 | `selector__p_within10__beam_mean` | `beam_mean`候補が誤差10以内となるnested校正確率 | 1.1078% | 0.5154% | 15/15 |
| 8 | `selector__candidate_value_std` | 12候補TVT値の標準偏差 | 0.8626% | 0.5619% | 15/15 |
| 9 | `selector__pred_abs_error__exp226_k16` | `exp226_k16`候補のnested selector予測絶対誤差 | 0.5180% | 1.0627% | 15/15 |
| 10 | `selector__p_within10__exp226_k16` | `exp226_k16`候補が誤差10以内となるnested校正確率 | 0.3758% | 1.0237% | 15/15 |
| 11 | `selector__p_within10__exp226_k16__likpf_mean` | `exp226_k16__likpf_mean`候補が誤差10以内となるnested校正確率 | 0.2914% | 0.5266% | 15/15 |
| 12 | `selector__pred_abs_error__exp226_k16__selfgr_hmm_a070` | `exp226_k16__selfgr_hmm_a070`候補のnested selector予測絶対誤差 | 0.2796% | 0.5685% | 15/15 |
| 13 | `selector__pred_abs_error__exact_hmm` | `exact_hmm`候補のnested selector予測絶対誤差 | 0.2703% | 0.7153% | 15/15 |
| 14 | `selector__pred_abs_error_std` | 12候補の予測絶対誤差score標準偏差 | 0.2611% | 0.4190% | 15/15 |
| 15 | `selector__pred_abs_error__selfgr_hmm_a070` | `selfgr_hmm_a070`候補のnested selector予測絶対誤差 | 0.2409% | 0.7532% | 15/15 |
| 16 | `selector__pred_abs_error__exp226_k16__exact_hmm` | `exp226_k16__exact_hmm`候補のnested selector予測絶対誤差 | 0.2318% | 0.5091% | 15/15 |
| 17 | `selector__p_within10__exp226_k16__selfgr_hmm_a070` | `exp226_k16__selfgr_hmm_a070`候補が誤差10以内となるnested校正確率 | 0.2128% | 0.4415% | 15/15 |
| 18 | `selector__p_within10__likpf_mean__exact_hmm` | `likpf_mean__exact_hmm`候補が誤差10以内となるnested校正確率 | 0.2102% | 0.3660% | 15/15 |
| 19 | `selector__p_within10__likpf_mean` | `likpf_mean`候補が誤差10以内となるnested校正確率 | 0.2045% | 0.3838% | 15/15 |
| 20 | `selector__pred_abs_error__exp226_k16__likpf_mean` | `exp226_k16__likpf_mean`候補のnested selector予測絶対誤差 | 0.1994% | 0.4229% | 15/15 |
| 21 | `selector__p_within10__exact_hmm` | `exact_hmm`候補が誤差10以内となるnested校正確率 | 0.1955% | 0.5686% | 15/15 |
| 22 | `selector__pred_abs_error__likpf_mean` | `likpf_mean`候補のnested selector予測絶対誤差 | 0.1858% | 0.3861% | 15/15 |
| 23 | `selector__pred_abs_error__likpf_mean__exact_hmm` | `likpf_mean__exact_hmm`候補のnested selector予測絶対誤差 | 0.1791% | 0.3143% | 15/15 |
| 24 | `selector__primitive_pair_bank__pred_abs_error__top1_score` | primitive+pair domainの予測誤差最小score | 0.1677% | 0.3205% | 15/15 |
| 25 | `selector__p_within10__selfgr_hmm_a070__likpf_mean` | `selfgr_hmm_a070__likpf_mean`候補が誤差10以内となるnested校正確率 | 0.1552% | 0.2926% | 15/15 |
| 26 | `selector__p_within10__exp226_k16__exact_hmm` | `exp226_k16__exact_hmm`候補が誤差10以内となるnested校正確率 | 0.1520% | 0.3571% | 15/15 |
| 27 | `selector__primitive_pair_bank__pred_abs_error__top2_score` | primitive+pair domainの予測誤差2位score | 0.1493% | 0.3602% | 15/15 |
| 28 | `selector__p_within10__selfgr_hmm_a070` | `selfgr_hmm_a070`候補が誤差10以内となるnested校正確率 | 0.1471% | 0.5014% | 15/15 |
| 29 | `selector__p_within10__exp226_w500_50_50` | fixed `exp226_w500_50_50`候補が誤差10以内となるnested校正確率 | 0.1371% | 0.3703% | 15/15 |
| 30 | `selector__pred_abs_error_mean` | 12候補の予測絶対誤差score平均 | 0.1262% | 0.2482% | 15/15 |
| 31 | `selector__pred_abs_error__selfgr_hmm_a070__likpf_mean` | `selfgr_hmm_a070__likpf_mean`候補のnested selector予測絶対誤差 | 0.1246% | 0.3094% | 15/15 |
| 32 | `selector__pred_abs_error__exp226_w500_50_50` | fixed `exp226_w500_50_50`候補のnested selector予測絶対誤差 | 0.1222% | 0.3566% | 15/15 |
| 33 | `selector__primitive_pair_bank__p_within10__top2_score` | primitive+pair domainのwithin10確率2位score | 0.1168% | 0.2144% | 15/15 |
| 34 | `selector__p_within10__pf_ancc` | `pf_ancc`候補が誤差10以内となるnested校正確率 | 0.1114% | 0.3500% | 15/15 |
| 35 | `selector__primitive_fixed_bank__pred_abs_error__top2_score` | primitive+fixed domainの予測誤差2位score | 0.1095% | 0.3536% | 15/15 |
| 36 | `selector__primitive_fixed_bank__pred_abs_error__top1_score` | primitive+fixed domainの予測誤差最小score | 0.1033% | 0.2972% | 15/15 |
| 37 | `selector__primitive_pair_bank__p_within10__top1_score` | primitive+pair domainのwithin10確率最大score | 0.0968% | 0.2526% | 15/15 |
| 38 | `selector__p_within10_mean` | 12候補のwithin10確率平均 | 0.0963% | 0.2222% | 15/15 |
| 39 | `selector__primitive_fixed_bank__p_within10__top1_score` | primitive+fixed domainのwithin10確率最大score | 0.0800% | 0.2315% | 15/15 |
| 40 | `selector__primitive_pair_bank__pred_abs_error__top2_value` | primitive+pair domainの予測誤差top2候補TVT値 | 0.0733% | 0.2650% | 15/15 |
| 41 | `selector__primitive_fixed_bank__pred_abs_error__top1_value` | primitive+fixed domainの予測誤差top1候補TVT値 | 0.0703% | 0.2802% | 15/15 |
| 42 | `selector__primitive_fixed_bank__pred_abs_error__top2_value` | primitive+fixed domainの予測誤差top2候補TVT値 | 0.0677% | 0.2866% | 15/15 |
| 43 | `selector__pred_abs_error__pf_ancc` | `pf_ancc`候補のnested selector予測絶対誤差 | 0.0676% | 0.3004% | 15/15 |
| 44 | `selector__primitive_fixed_bank__p_within10__top2_value` | primitive+fixed domainのwithin10確率top2候補TVT値 | 0.0663% | 0.2694% | 15/15 |
| 45 | `selector__primitive_pair_bank__p_within10__top1_value` | primitive+pair domainのwithin10確率top1候補TVT値 | 0.0658% | 0.3087% | 15/15 |
| 46 | `selector__primitive_fixed_bank__p_within10__top1_value` | primitive+fixed domainのwithin10確率top1候補TVT値 | 0.0658% | 0.3119% | 15/15 |
| 47 | `selector__primitive_pair_bank__pred_abs_error__top1_value` | primitive+pair domainの予測誤差top1候補TVT値 | 0.0644% | 0.2924% | 15/15 |
| 48 | `selector__primitive_pair_bank__p_within10__top2_value` | primitive+pair domainのwithin10確率top2候補TVT値 | 0.0638% | 0.2653% | 15/15 |
| 49 | `selector__p_within10_std` | 12候補のwithin10確率標準偏差 | 0.0542% | 0.2919% | 15/15 |
| 50 | `selector__primitive_fixed_bank__pred_abs_error__margin` | primitive+fixed domainの予測誤差top1/top2 score margin | 0.0533% | 0.1428% | 15/15 |
| 51 | `selector__primitive_fixed_bank__p_within10__top2_score` | primitive+fixed domainのwithin10確率2位score | 0.0527% | 0.1819% | 15/15 |
| 52 | `selector__p_within10_candidate_entropy` | 12候補のwithin10確率を候補方向へ正規化したentropy | 0.0455% | 0.2632% | 15/15 |
| 53 | `selector__primitive_fixed_bank__p_within10__margin` | primitive+fixed domainのwithin10確率top1/top2 margin | 0.0189% | 0.1357% | 15/15 |
| 54 | `selector__primitive_pair_bank__p_within10__margin` | primitive+pair domainのwithin10確率top1/top2 margin | 0.0139% | 0.1080% | 15/15 |
| 55 | `selector__primary_error_top1__likpf_mean` | primary予測誤差top1が`likpf_mean`のone-hot | 0.0100% | 0.0570% | 15/15 |
| 56 | `selector__primitive_pair_bank__pred_abs_error__margin` | primitive+pair domainの予測誤差top1/top2 margin | 0.0084% | 0.0830% | 15/15 |
| 57 | `selector__primary_error_top1__beam_mean` | primary予測誤差top1が`beam_mean`のone-hot | 0.0081% | 0.0417% | 15/15 |
| 58 | `selector__primary_error_top1__exp226_k16` | primary予測誤差top1が`exp226_k16`のone-hot | 0.0077% | 0.0450% | 15/15 |
| 59 | `selector__primary_error_top1__likpf_mean__exact_hmm` | primary予測誤差top1が`likpf_mean__exact_hmm`のone-hot | 0.0034% | 0.0111% | 11/15 |
| 60 | `selector__primary_error_top1__selfgr_hmm_a070__likpf_mean` | primary予測誤差top1が`selfgr_hmm_a070__likpf_mean`のone-hot | 0.0029% | 0.0086% | 14/15 |
| 61 | `selector__primary_error_top1__pf_ancc` | primary予測誤差top1が`pf_ancc`のone-hot | 0.0014% | 0.0267% | 12/15 |
| 62 | `selector__primary_error_top1__exp226_k16__likpf_mean` | primary予測誤差top1が`exp226_k16__likpf_mean`のone-hot | 0.0014% | 0.0183% | 13/15 |
| 63 | `selector__primary_top1_is_pair` | primary予測誤差top1がpairのフラグ | 0.0008% | 0.0100% | 14/15 |
| 64 | `selector__primary_top1_is_primitive` | primary予測誤差top1がprimitiveのフラグ | 0.0008% | 0.0096% | 13/15 |
| 65 | `selector__primary_error_top1__selfgr_hmm_a070` | primary予測誤差top1が`selfgr_hmm_a070`のone-hot | 0.0005% | 0.0081% | 11/15 |
| 66 | `selector__fixed_top1_is_primitive` | fixed domainの予測誤差top1がprimitiveのフラグ | 0.0005% | 0.0147% | 11/15 |
| 67 | `selector__primary_error_top1__exp226_k16__exact_hmm` | primary予測誤差top1が`exp226_k16__exact_hmm`のone-hot | 0.0004% | 0.0096% | 9/15 |
| 68 | `selector__primary_error_top1__exp226_k16__selfgr_hmm_a070` | primary予測誤差top1が`exp226_k16__selfgr_hmm_a070`のone-hot | 0.0003% | 0.0092% | 10/15 |
| 69 | `selector__primitive_fixed_bank__top1_objective_agreement` | primitive+fixed domainで2 objectiveのtop1候補が一致したフラグ | 0.0002% | 0.0129% | 10/15 |
| 70 | `selector__fixed_top1_is_fixed` | fixed domainの予測誤差top1がfixed候補のフラグ | 0.0001% | 0.0044% | 7/15 |
| 71 | `selector__primary_error_top1__exact_hmm` | primary予測誤差top1が`exact_hmm`のone-hot | 0.0001% | 0.0045% | 8/15 |
| 72 | `selector__primitive_pair_bank__top1_objective_agreement` | primitive+pair domainで2 objectiveのtop1候補が一致したフラグ | 0.0001% | 0.0088% | 7/15 |
| 73 | `selector__confidence_valid_count` | source-native confidenceが有効な候補数 | 0.0000% | 0.0000% | 0/15 |
| 74 | `selector__available_count` | 有限値を持つ候補数 | 0.0000% | 0.0000% | 0/15 |

## 解釈

- raw top1候補値より`top1_minus_anchor`が圧倒的に強い。後段targetがlast-known TVTからのresidualなので、
  selectorが選んだcandidateを同じanchor基準へ変換した表現が直接整合する。
- `p_within10`と`pred_abs_error`の両objectiveが上位4列へ入り、dual-objective scoreを残した設計を支持する。
- `available_count`と`confidence_valid_count`は全15 modelsで0 gain。現行12候補bankでは数がほぼ固定であり、
  confidenceの内容は候補別scoreを介して使われている。
- top1 one-hotとobjective agreementのgainは非常に小さい。candidate IDそのものより、top1候補値のanchor差、
  candidate spread、候補別連続scoreが有効だった。

## tail-risk

by-well delta中央値は-0.783572、90 percentile +1.868145、95 percentile +3.016274、99 percentile
+6.848748。global改善は広いが、255 wellsが悪化し14 wellsは+5 ft超なのでtail問題は残る。

| worst rank | well | rows | control RMSE | add-only RMSE | delta |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `70925e23` | 5,947 | 11.825487 | 26.308360 | +14.482873 |
| 2 | `add9c322` | 3,932 | 3.963402 | 17.404669 | +13.441268 |
| 3 | `14fee784` | 4,910 | 6.411408 | 17.536155 | +11.124747 |
| 4 | `57f05c51` | 4,771 | 11.697719 | 21.425345 | +9.727626 |
| 5 | `c9578d27` | 4,487 | 2.532487 | 11.131354 | +8.598867 |

そのため74列をそのままcurrent-testへportしない。再訪するなら、修正版Stage C/Dを固定入力とし、
outer-valid truthをrisk fitへ使わない0-booster tail-risk readoutを先に行う。
