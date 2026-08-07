# exp239_distribution_matched_multicut_pseudotail

## 状態

- Route: `ml_model`
- 状態: v11 full augmentation不採用、trial submission完了・不採用
- CV / Public LB / Private LB: official-start OOF 8.697380066 / 7.944 / 未確定
- 親実験: `exp218_gr_wavelet_rotation_confidence_features_on_exp148`
- 比較親: `exp023_pseudo_tail_distance_augmentation`
- 作成日: 2026-07-12

## 仮説

exp023で有効だったearly-start pseudo-tailを、全well共通の固定3 quantileではなく、
official-startのprefix長、evaluation長、GR欠損、trajectory phase分布に合わせて選べば、
現行exp218系のofficial-start OOF、long-tail、hidden-like頑健性を改善できる可能性がある。

## 変更点

- quantile、fixed hidden rows、GR change、GR missing boundary、trajectory curvatureからcutoff候補を作る。
- official-start各marginal binのdeficitを埋めるdeterministic quota selectionを行う。
- well/source/total cutoffと推定augmentation rowsを事前capする。
- source-well 5-fold manifestとprefix feature replay requestを保存する。
- tail TVTのfeature参照とfull-prefix生成済みcacheのslice流用をassertで禁止する。

## 検証方針

- primary: official-start OOF RMSE
- group: source well GroupKFold
- stress: exp115 hidden-like、near、1000+、worst-well
- leakage: outer-valid wellの派生sampleはouter-trainへ入れず、synthetic cutoff後のTVTはtarget専用とする。

## 実装範囲

raw trainからcutoff候補とdistribution-matched selectionを作り、v3で800 requestsを
materializeした。最終v11ではofficial/pseudo両方の380-feature cacheをmemmap streamingし、
exp218-family 3 configs x 5 foldsを学習した。v12ではユーザー明示依頼により、保存済み15
boostersをexp218 raw-test replayへ接続し、trial submissionを1件だけ実施した。

late-startを加える`bidirectional_prediction_start_pseudotail_augmentation`は別バックログであり、
exp239はofficial startより前へ動かすearly-startだけを扱う。

## 所見

Kaggle CPU audit v1は773 wells、11,123候補、1,546 selected cutoffsを生成し、fold/leakage/
SHA contractはpassした。一方、official-start targetに対する最大marginal share差はprefix rowsで
0.210220、prefix fraction / trajectory phaseで0.131953と大きい。全wellへ2 cutoffsを割り当てる
制約により短いprefixへ偏ったため、exp218 feature再生成には進まずmatching設計を修正する。

## 実行入口

- train audit: `exp239_distribution_matched_multicut_pseudotail_train.ipynb`
- inference kernel: `kentookumura/exp239-distribution-matched-multicut-inference` v1
- canonical train kernel: `kentookumura/exp239-distribution-matched-multicut-train`
- 初回実行はKaggle CPUとし、pushはユーザー承認後に行う。

## 次

全well一律2-cutoffを外し、0-3 cutoffs/wellを許すglobal quotaへ修正済み。v1生成物preflightは
max marginal差0.030344、well coverage 0.798189、hidden-like 1.0でguard pass。次は同じexp239の
Kaggle CPU audit v2でも同値を再現した。v3は800 requests、799,961行・50列、約96秒、
推定memory 528 MBで全guardをpassし、download後のfeature content SHAも一致した。次はlearned
likelihood / rankerなど小さいdownstreamの設計へ進む。

最初のdownstreamはPF/Beam候補を必要としないresidual probeとした。v3数値特徴だけで
`target_tvt - anchor_tvt_input`を学習し、source-well 5 folds、CPU LightGBM 1 config、合計5
boostersでanchor系baselineに対するlearnabilityを確認する。これはofficial-start OOFではない。

v4はoverall 69.526871から24.349143へ改善したが、210/617 wellsが悪化し、最大well regression
+63.415661でguard failed。direct residual予測や親exp218学習には進まず、後続利用は
cross-fitted confidenceまたはanchor shrinkageに限定する。

v11本評価は15/15 boostersを完走したが、official-start OOF 8.697380066で保存済みexp218
8.475793752から+0.221586314悪化した。cache/schema/SHAとfold-safe除外はpassしているため、
direct pseudo-tail augmentation仮説を不採用とし、weight微調整、inference、submitへ進まない。

ただし2026-07-15、negative CVを承知でLB挙動を確認するユーザー指定trialを実施した。
Inference v1は15 models / 380 features / 14,151 rows / fallback 0で完了し、submit-checkはPASS。
提出ref `54720769` はPublic LB 7.944で完了した。exp218 7.843から+0.101、現ML anchor
exp238 7.775から+0.169悪化し、official-start OOF悪化と方向が一致したため不採用を確定する。
