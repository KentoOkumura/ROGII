# exp264 Stage D compact特徴量・重要度 readout

> **無効化済み:** 入力compactがtraining-only formation 12特徴を使ったStage C score由来であるため、
> 以下のadd-only重要度・RMSE差は有効なOOF根拠ではない。

Kaggle Stage D version 2の`selector_compact_addonly` 15 modelsは失敗履歴としてのみ保持する。各model内でgain/splitを全454特徴の合計で正規化し、3 configs × 5 outer foldsで平均した。したがってshareは「compact 74列内」ではなく「base 380 + compact 74列のadd-onlyモデル全体」に占める割合である。raw gainはmodel間でscaleが異なるため直接加算しない。

根拠ファイルは`kaggle/output/stage_d_v2/artifacts/stage_d_feature_importance.csv`、`stage_d_metrics.json`、`stage_d_by_well.csv`。全74列はStage Cの`compact_meta_schema.json`と一致した。

## 結論

- 旧計算上は`8.545568 → 7.805644`だったが、入力compactのfeature availability leakageにより改善判定を無効化した。
- compact 74列はadd-only全体gainの70.9550%、splitの17.4854%を占め、74列すべてを使う効果は大きい。
- 上位4列は2 legal domainのwithin10/error top1候補とlast-known anchorとの差で、合計gain shareは59.9561%。候補scoreそのものより「scoreで選ばれた候補値のanchor差」が強い。
- 5位は`beam_mean`の予測誤差scoreで5.0341%。`beam_mean`をhard選択する根拠ではなく、直線的なcandidateがどのregimeで信頼できるかを後段がsoftに使った証拠である。
- worst-well `70925e23`は`5.804539 → 23.251280`（`+17.446742`）。773 well中470改善、303悪化、243 wellが+0.25を超えたため、事前guardはFAIL。global gainを理由にcompact inferenceへ進めない。

## 全74 compact特徴

`gain nonzero`は15 add-only modelsのうちgainが正だった本数。0 gain列もStage C schema固定後に生成されたため、Stage D結果を見て事後dropはしない。

| rank | compact feature | 説明 | gain share | split share | gain nonzero |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | `selector__primitive_fixed_bank__p_within10__top1_minus_anchor` | primitive+fixed 7候補domainのwithin10確率最大top1候補TVT−last-known anchor | 18.9033% | 0.9407% | 15/15 |
| 2 | `selector__primitive_pair_bank__p_within10__top1_minus_anchor` | primitive+pair 11候補domainのwithin10確率最大top1候補TVT−last-known anchor | 17.9638% | 0.9608% | 15/15 |
| 3 | `selector__primitive_pair_bank__pred_abs_error__top1_minus_anchor` | primitive+pair 11候補domainの予測誤差最小top1候補TVT−last-known anchor | 12.1969% | 0.9184% | 15/15 |
| 4 | `selector__primitive_fixed_bank__pred_abs_error__top1_minus_anchor` | primitive+fixed 7候補domainの予測誤差最小top1候補TVT−last-known anchor | 10.8921% | 0.8840% | 15/15 |
| 5 | `selector__pred_abs_error__beam_mean` | `beam_mean`候補のnested selector予測絶対誤差。小さいほど有望 | 5.0341% | 0.6726% | 15/15 |
| 6 | `selector__candidate_value_range` | 12候補TVT値の最大−最小 | 1.2395% | 0.4952% | 15/15 |
| 7 | `selector__candidate_value_std` | 12候補TVT値の標準偏差 | 0.8274% | 0.4432% | 15/15 |
| 8 | `selector__pred_abs_error__exp226_k16` | `exp226_k16`候補のnested selector予測絶対誤差 | 0.4009% | 0.7103% | 15/15 |
| 9 | `selector__p_within10__beam_mean` | `beam_mean`候補が誤差10以内となるnested校正確率 | 0.2646% | 0.3928% | 15/15 |
| 10 | `selector__p_within10__exp226_k16` | `exp226_k16`候補が誤差10以内となるnested校正確率 | 0.2330% | 0.6207% | 15/15 |
| 11 | `selector__pred_abs_error__exact_hmm` | `exact_hmm`候補のnested selector予測絶対誤差 | 0.2045% | 0.5297% | 15/15 |
| 12 | `selector__pred_abs_error__selfgr_hmm_a070` | `selfgr_hmm_a070`候補のnested selector予測絶対誤差 | 0.1710% | 0.5505% | 15/15 |
| 13 | `selector__pred_abs_error__likpf_mean` | `likpf_mean`候補のnested selector予測絶対誤差 | 0.1576% | 0.3112% | 15/15 |
| 14 | `selector__pred_abs_error__exp226_k16__exact_hmm` | `exp226_k16__exact_hmm`候補のnested selector予測絶対誤差 | 0.1515% | 0.3489% | 15/15 |
| 15 | `selector__p_within10__exact_hmm` | `exact_hmm`候補が誤差10以内となるnested校正確率 | 0.1460% | 0.4190% | 15/15 |
| 16 | `selector__p_within10__selfgr_hmm_a070` | `selfgr_hmm_a070`候補が誤差10以内となるnested校正確率 | 0.1230% | 0.4435% | 15/15 |
| 17 | `selector__pred_abs_error__exp226_k16__selfgr_hmm_a070` | `exp226_k16__selfgr_hmm_a070`候補のnested selector予測絶対誤差 | 0.1217% | 0.3742% | 15/15 |
| 18 | `selector__pred_abs_error__exp226_k16__likpf_mean` | `exp226_k16__likpf_mean`候補のnested selector予測絶対誤差 | 0.1193% | 0.2889% | 15/15 |
| 19 | `selector__p_within10__exp226_k16__selfgr_hmm_a070` | `exp226_k16__selfgr_hmm_a070`候補が誤差10以内となるnested校正確率 | 0.1095% | 0.2740% | 15/15 |
| 20 | `selector__p_within10__exp226_k16__likpf_mean` | `exp226_k16__likpf_mean`候補が誤差10以内となるnested校正確率 | 0.1094% | 0.2870% | 15/15 |
| 21 | `selector__p_within10__exp226_k16__exact_hmm` | `exp226_k16__exact_hmm`候補が誤差10以内となるnested校正確率 | 0.0957% | 0.2518% | 15/15 |
| 22 | `selector__pred_abs_error_mean` | 12候補の予測絶対誤差score平均 | 0.0914% | 0.2026% | 15/15 |
| 23 | `selector__pred_abs_error__likpf_mean__exact_hmm` | `likpf_mean__exact_hmm`候補のnested selector予測絶対誤差 | 0.0897% | 0.2267% | 15/15 |
| 24 | `selector__primitive_pair_bank__p_within10__top2_score` | primitive+pair domainのwithin10確率2位score | 0.0832% | 0.1687% | 15/15 |
| 25 | `selector__pred_abs_error__exp226_w500_50_50` | fixed `exp226_w500_50_50`候補のnested selector予測絶対誤差 | 0.0817% | 0.2451% | 15/15 |
| 26 | `selector__p_within10__likpf_mean__exact_hmm` | `likpf_mean__exact_hmm`候補が誤差10以内となるnested校正確率 | 0.0767% | 0.2324% | 15/15 |
| 27 | `selector__p_within10__likpf_mean` | `likpf_mean`候補が誤差10以内となるnested校正確率 | 0.0755% | 0.2543% | 15/15 |
| 28 | `selector__pred_abs_error_std` | 12候補の予測絶対誤差score標準偏差 | 0.0727% | 0.3442% | 15/15 |
| 29 | `selector__pred_abs_error__selfgr_hmm_a070__likpf_mean` | `selfgr_hmm_a070__likpf_mean`候補のnested selector予測絶対誤差 | 0.0698% | 0.2158% | 15/15 |
| 30 | `selector__primitive_pair_bank__pred_abs_error__top2_score` | primitive+pair domainの予測誤差2位score | 0.0667% | 0.2523% | 15/15 |
| 31 | `selector__primitive_fixed_bank__pred_abs_error__top2_score` | primitive+fixed domainの予測誤差2位score | 0.0657% | 0.2294% | 15/15 |
| 32 | `selector__p_within10__exp226_w500_50_50` | fixed `exp226_w500_50_50`候補が誤差10以内となるnested校正確率 | 0.0655% | 0.2315% | 15/15 |
| 33 | `selector__primitive_pair_bank__pred_abs_error__top1_score` | primitive+pair domainの予測誤差最小score | 0.0594% | 0.2330% | 15/15 |
| 34 | `selector__primitive_pair_bank__p_within10__top1_score` | primitive+pair domainのwithin10確率最大score | 0.0592% | 0.1660% | 15/15 |
| 35 | `selector__p_within10_mean` | 12候補のwithin10確率平均 | 0.0571% | 0.1830% | 15/15 |
| 36 | `selector__p_within10__selfgr_hmm_a070__likpf_mean` | `selfgr_hmm_a070__likpf_mean`候補が誤差10以内となるnested校正確率 | 0.0492% | 0.2070% | 15/15 |
| 37 | `selector__primitive_fixed_bank__p_within10__top2_score` | primitive+fixed domainのwithin10確率2位score | 0.0489% | 0.1650% | 15/15 |
| 38 | `selector__primitive_fixed_bank__p_within10__top1_score` | primitive+fixed domainのwithin10確率最大score | 0.0405% | 0.1582% | 15/15 |
| 39 | `selector__primitive_fixed_bank__pred_abs_error__top1_score` | primitive+fixed domainの予測誤差最小score | 0.0402% | 0.1938% | 15/15 |
| 40 | `selector__primitive_fixed_bank__pred_abs_error__margin` | primitive+fixed domainの予測誤差top1/top2 score margin | 0.0380% | 0.1175% | 15/15 |
| 41 | `selector__p_within10__pf_ancc` | `pf_ancc`候補が誤差10以内となるnested校正確率 | 0.0329% | 0.2694% | 15/15 |
| 42 | `selector__p_within10_candidate_entropy` | 12候補のwithin10確率を候補方向へ正規化したentropy | 0.0314% | 0.2388% | 15/15 |
| 43 | `selector__p_within10_std` | 12候補のwithin10確率標準偏差 | 0.0281% | 0.2549% | 15/15 |
| 44 | `selector__pred_abs_error__pf_ancc` | `pf_ancc`候補のnested selector予測絶対誤差 | 0.0237% | 0.2539% | 15/15 |
| 45 | `selector__primitive_pair_bank__pred_abs_error__top1_value` | primitive+pair domainの予測誤差top1候補TVT値 | 0.0185% | 0.1053% | 15/15 |
| 46 | `selector__primitive_pair_bank__pred_abs_error__top2_value` | primitive+pair domainの予測誤差top2候補TVT値 | 0.0150% | 0.0904% | 15/15 |
| 47 | `selector__primitive_fixed_bank__pred_abs_error__top1_value` | primitive+fixed domainの予測誤差top1候補TVT値 | 0.0138% | 0.1071% | 15/15 |
| 48 | `selector__primitive_fixed_bank__p_within10__top1_value` | primitive+fixed domainのwithin10確率top1候補TVT値 | 0.0127% | 0.1263% | 15/15 |
| 49 | `selector__primitive_pair_bank__p_within10__top1_value` | primitive+pair domainのwithin10確率top1候補TVT値 | 0.0125% | 0.1144% | 15/15 |
| 50 | `selector__primitive_fixed_bank__pred_abs_error__top2_value` | primitive+fixed domainの予測誤差top2候補TVT値 | 0.0120% | 0.0886% | 15/15 |
| 51 | `selector__primitive_pair_bank__p_within10__top2_value` | primitive+pair domainのwithin10確率top2候補TVT値 | 0.0108% | 0.0838% | 15/15 |
| 52 | `selector__primitive_fixed_bank__p_within10__top2_value` | primitive+fixed domainのwithin10確率top2候補TVT値 | 0.0106% | 0.0928% | 15/15 |
| 53 | `selector__primary_error_top1__beam_mean` | primary予測誤差top1が`beam_mean`のone-hot | 0.0090% | 0.0317% | 15/15 |
| 54 | `selector__primitive_fixed_bank__p_within10__margin` | primitive+fixed domainのwithin10確率top1/top2 margin | 0.0070% | 0.1040% | 15/15 |
| 55 | `selector__primitive_pair_bank__p_within10__margin` | primitive+pair domainのwithin10確率top1/top2 margin | 0.0051% | 0.0764% | 15/15 |
| 56 | `selector__primary_error_top1__exp226_k16` | primary予測誤差top1が`exp226_k16`のone-hot | 0.0032% | 0.0416% | 15/15 |
| 57 | `selector__primary_error_top1__likpf_mean` | primary予測誤差top1が`likpf_mean`のone-hot | 0.0032% | 0.0425% | 15/15 |
| 58 | `selector__primitive_pair_bank__pred_abs_error__margin` | primitive+pair domainの予測誤差top1/top2 margin | 0.0025% | 0.0665% | 15/15 |
| 59 | `selector__primary_error_top1__exp226_k16__likpf_mean` | primary予測誤差top1が`exp226_k16__likpf_mean`のone-hot | 0.0011% | 0.0185% | 15/15 |
| 60 | `selector__primary_top1_is_primitive` | primary予測誤差top1がprimitiveのフラグ | 0.0008% | 0.0094% | 13/15 |
| 61 | `selector__primary_error_top1__likpf_mean__exact_hmm` | primary予測誤差top1が`likpf_mean__exact_hmm`のone-hot | 0.0008% | 0.0114% | 14/15 |
| 62 | `selector__primary_error_top1__pf_ancc` | primary予測誤差top1が`pf_ancc`のone-hot | 0.0008% | 0.0261% | 15/15 |
| 63 | `selector__fixed_top1_is_primitive` | fixed domainの予測誤差top1がprimitiveのフラグ | 0.0007% | 0.0164% | 15/15 |
| 64 | `selector__fixed_top1_is_fixed` | fixed domainの予測誤差top1がfixed候補のフラグ | 0.0006% | 0.0068% | 14/15 |
| 65 | `selector__primary_error_top1__selfgr_hmm_a070__likpf_mean` | primary予測誤差top1が`selfgr_hmm_a070__likpf_mean`のone-hot | 0.0006% | 0.0056% | 7/15 |
| 66 | `selector__primary_top1_is_pair` | primary予測誤差top1がpairのフラグ | 0.0004% | 0.0106% | 14/15 |
| 67 | `selector__primary_error_top1__selfgr_hmm_a070` | primary予測誤差top1が`selfgr_hmm_a070`のone-hot | 0.0004% | 0.0100% | 12/15 |
| 68 | `selector__primary_error_top1__exp226_k16__exact_hmm` | primary予測誤差top1が`exp226_k16__exact_hmm`のone-hot | 0.0003% | 0.0075% | 11/15 |
| 69 | `selector__primary_error_top1__exp226_k16__selfgr_hmm_a070` | primary予測誤差top1が`exp226_k16__selfgr_hmm_a070`のone-hot | 0.0003% | 0.0092% | 13/15 |
| 70 | `selector__primitive_fixed_bank__top1_objective_agreement` | primitive+fixed domainで2 objectiveのtop1候補が一致したフラグ | 0.0001% | 0.0109% | 9/15 |
| 71 | `selector__primitive_pair_bank__top1_objective_agreement` | primitive+pair domainで2 objectiveのtop1候補が一致したフラグ | 0.0001% | 0.0069% | 10/15 |
| 72 | `selector__primary_error_top1__exact_hmm` | primary予測誤差top1が`exact_hmm`のone-hot | 0.0000% | 0.0038% | 9/15 |
| 73 | `selector__confidence_valid_count` | source-native confidenceが有効な候補数 | 0.0000% | 0.0000% | 0/15 |
| 74 | `selector__available_count` | 有限値を持つ候補数 | 0.0000% | 0.0000% | 0/15 |

## 重要度の解釈

- raw `top1_value`より`top1_minus_anchor`が圧倒的に強い。後段targetが`TVT - last_known_tvt` residualなので、selector由来候補を同じanchor基準へ置いた表現がtargetと整合している。
- `p_within10`側anchor差2列がgain 36.87%、予測誤差側2列が23.09%。exp251形式の2 objectiveを両方残した判断は支持された。
- individual scoreでは`beam_mean`が最大だが、hard top1率やcandidate単体RMSEとは同義ではない。直線的なbeam pathの危険度を予測するscoreがregime featureとして働いている。
- `available_count`と`confidence_valid_count`は全15 modelsで0 gain。現行12候補bankではavailabilityがほぼ固定で、valid数だけではconfidence内容を表現できない。source-native confidenceはStage B/Cのcandidate-long scoreへ既に吸収されている。
- top1 one-hotとobjective agreementはgainが非常に小さい。candidate identityを直接使うより、選択された候補値とanchorの差、候補spread、連続scoreを使う方が有効だった。

## tail-risk判断

by-well deltaの中央値は-0.307938、90 percentileは+1.799525、95 percentileは+2.805478、99 percentileは+7.353320。16 wellが+5を超えた。global改善は少数wellだけの効果ではない一方、悪化もworst 1 wellだけに局在していない。

| worst rank | well | rows | control RMSE | add-only RMSE | delta |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `70925e23` | 5,947 | 5.804539 | 23.251280 | +17.446742 |
| 2 | `57f05c51` | 4,771 | 8.650998 | 22.062566 | +13.411568 |
| 3 | `b3388334` | 6,080 | 10.395171 | 21.709893 | +11.314722 |
| 4 | `81bf5923` | 4,543 | 31.182530 | 41.770189 | +10.587659 |
| 5 | `add9c322` | 3,932 | 2.576139 | 11.735094 | +9.158955 |
| 6 | `8c167025` | 5,368 | 7.555368 | 16.646010 | +9.090643 |
| 7 | `4caa7289` | 5,396 | 13.715214 | 22.186060 | +8.470846 |
| 8 | `37344c2a` | 5,197 | 4.580162 | 12.278992 | +7.698830 |
| 9 | `ee0300f7` | 4,544 | 12.202469 | 19.421424 | +7.218956 |
| 10 | `113011eb` | 3,272 | 12.795689 | 19.481102 | +6.685413 |

したがって74列をそのままcurrent-testへportしない。再訪する場合は保存済みOOFで、悪化wellをtarget-freeなcandidate spread、score dispersion、confidence validity、geometry/contextからouter-fold再現可能に識別できるかを先に0 boosterで監査する。guardや特徴schemaをStage D結果に合わせて事後変更しない。
