# exp193_typewell_late_interval_context_features_addonly_on_exp148

## 目的

exp148 の learned-likelihood ML anchor に、target-free な typewell late-interval context feature を add-only で追加し、typewell 後半集中の prior を LightGBM が小さな context signal として吸収できるか確認する。

## 状態

Kaggle train v1 完了。train-side supported。Kaggle inference v2 完了、submit-check PASS。competition submit ref `54347471` は Public LB 7.946 で完了。exp148 GPU inference v7 の 7.960 は上回ったが、exp148 CPU runtime submission の 7.921 には届かないため非採用。

## 背景

- exp174: exp148 ML 予測への late-range hard clip / shrink は no-op または悪化。
- exp176: PF/Beam candidate ranker では `candidate_pct` / `known_last_pct` 系 late-range signal が positive。
- 今回は candidate 別 feature を入れず、row/well context のみで exp148 に効くかを反証する。

## 仮説

typewell 後半区間の min/max/span と observed prefix の `known_last_pct` は、candidate 別 signal なしでも late-range prior の一部を表現し、exp148 の learned likelihood confidence とは別系統の補助特徴として効く可能性がある。

## 変更点

- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- route: `ml_model`
- variant: `typewell_late_interval_context_addonly`
- 追加 feature group: `typewell_late_interval_context`
- control: 再学習しない。保存済み exp148 CV / Public LB を historical baseline として参照する。

追加列は `tlic_typewell_min/max/span`、late50/60/70 の min/max/span、`known_last_pct`、late interval 開始との差分、inside flag に限定する。`candidate_pct_*`、candidate 別 violation、direct clip、blend、postprocess、hard selector は入れない。

## 検証方針

GroupKFold 5 folds を well group で実行し、GPU LightGBM family 3 configs を単一 train notebook 内で学習する。予定は 1 variant、3 configs、5 folds、合計 15 boosters。

比較基準:

- exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960
- exp174 posthoc clip negative
- exp176 selector signal positive
- exp160 / exp162 / exp183 系 exp148 add-only 候補

train CV が positive でも、raw-test/current-test feature parity と submit-check を Kaggle inference output で確認するまで submit しない。

## 所見

Kaggle train v1 は `kentookumura/exp193-typewell-late-context-exp148-train` で完了した。3,783,989 rows / 773 wells / 313 features / 15 boosters。

pooled OOF は `lgb0` 8.553543817、`lgb1` 8.475340902、`lgb2` 8.510015021、`lgb_mean` 8.456665439。exp148 `lgb_mean` 8.501281182 から -0.044615743 改善したため、train-side では supported。

`tlic_known_last_pct` は feature importance rank 46 / 313 で、late interval delta 系も上位寄りに入った。2026-07-05 に same-exp inference port を追加し、current test の horizontal/typewell input から `tlic_` 19 features を再生成して exp193 train v1 の 15 saved boosters を `lgb_mean` 平均する package を作成した。

Inference package は `experiments/exp193_typewell_late_interval_context_features_addonly_on_exp148/kaggle/inference`。kernel id は `kentookumura/exp193-typewell-late-context-exp148-inference`、GPU enabled、internet off。

Kaggle inference v1 は `generator.candidates` が exp193 config に無く失敗した。exp145/exp148 と同じ generator block を追加し、v2 で完了した。v2 は 14,151 rows、313 features、`tlic_` 19 features、fallback 0。train manifest と inference feature schema は exact match、submit-check は PASS。submission SHA256 は `9265e3e19e7eea20c6e0097b3b581b4a15c29353ebb77875d09ac30475502695`。

Code submission ref `54347471` は Public LB 7.946 で、exp148 GPU inference v7 Public LB 7.960 からは -0.014 改善した。一方、ユーザー確認済みの exp148 CPU runtime submission Public LB 7.921 には +0.025 届かないため、exp193 は ML route submitted anchor には採用しない。アンサンブル route anchor の exp082 Public LB 7.601 は引き続き全体最良。
