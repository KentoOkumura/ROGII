# 設計

## 結論

exp410ではacross-seed算術平均が排他的SSE`36.2441%`、重複あり`42.8304%`に
関係し、truth-best seedには大きなheadroomがあった。一方、単純seed medianと
particle modeは悪化した。exp404のx1.0 seed bankには固定temperature 5のtarget-free
likelihood-weighted predictionが保存され、算術平均より約`0.68 ft`よいという先行根拠が
ある。ただしexp404の主仮説はGR sigma x1.3対x1.0であり、scale5対算術平均の
promotion gateではなかった。

そこで新しいPFを回さず、同一x1.0 trajectory bankの2 readoutだけを厳格な
fold / scope / tail gateで評価する。これにより「単純平均をやめる」効果を
PF dynamicsやGR sigma変更から分離する。

## アプローチ

### Stage A: 保存OOF promotion audit

1. exp404 version-1 frozen prediction bundleとscientific contractのSHAを検証する。
2. truthを読む前にID、well、row、`likpf_mean_x1p0`、
   `likpf_scale_5_x1p0`をfreezeし、logical SHAを記録する。
3. 両列が同じx1.0 PF trajectory / seed labelsから作られ、差がaggregationだけである
   ことをcontractで確認する。
4. freeze後にsuffix truth、fold、raw-GR observed/missing、well missing fraction、
   1000+、hidden-like roleを結合する。
5. 事前固定AND gateを判定する。PF再生、学習、selector、temperature探索は行わない。

### 条件付き将来Stage B: raw-test batch inference

Stage A全gate PASSと別のユーザー承認がある場合だけ、同じexp417で設計を追記する。
hidden wellごとにexp072 PF bankを1回再生し、同じbankから算術平均parityと固定scale5を
同時出力する。算術とscale5のためにPFを2回走らせない。今回は実装しない。

## 実験範囲

- 対象実験: `exp417_scale5_seed_aggregation_promotion_audit`
- Route: `pf_beam`
- artifact parent: `exp404_scale5_sigma_gr_likelihood_pf_ablation`
- scientific control: `exp072_exp063_full_replay_feature_cache`
- 原因監査: `exp410_likpf_particle_resampling_basin_audit`
- control:
  `likpf_mean_x1p0 = mean(seed_prediction_s, s=0..127)`
- candidate:
  `likpf_scale_5_x1p0 = sum_s softmax((L_s-max L)/5)_s * seed_prediction_s`
- 変更する変数: seed aggregation weightだけ
- 固定する変数:
  particles / seeds / seed labels / trajectories / GR sigma / transition /
  resampling / roughening / initialization / Type Well / missing補間 / score rows

`L_s`はwellの未知suffix全体のGR likelihood合計である。candidateはsuffix TVTを
使わないtarget-free batch readoutだが、early rowでもfuture suffix GRを使うため
causal online predictionではない。Kaggle hidden inferenceでは全horizontal GRが先に
与えられることを適用条件として明記する。

## 実行量

### Stage A

- saved candidate readouts: 1
- PF well-runs: 0
- control PF reruns: 0
- model configs / trained folds / boosters: `0 / 0 / 0`
- HMM / Beam / GPU: `0 / 0 / 0`
- reporting folds: 5
- expected runtime upper bound: 900秒

### 条件付きStage B

- 現在はdisabled / 未承認
- raw-test PF replay: hidden wellごと1回
- readouts: arithmetic parity + fixed scale5
- model training: 0

## 評価と固定gate

evidence-only期待値はcontrol `11.594894395642696 ft`、candidate
`10.914522073 ft`だが、Stage A実装が全SHAとlate-joinを通すまでは実験結果としない。

全条件をANDで要求する。

1. pooled gain `>=0.05 ft`
2. 改善fold `>=4/5`
3. raw-GR observed gain `>=0.05 ft`
4. raw-GR missing、high-missing、1000+、hidden-like 2面のregression `<=0`
5. by-well delta RMSE p95 `<=0`
6. worst-well regression `<=0.25 ft`
7. fixed HMM/LikPF 50:50 blend regression `<=0`

FAIL時はtemperature、scale、best seed、median、mode、medoid、selector、
well/row gateを同じOOFで試さない。exp413のML branch判断も変更しない。

## 実装時のNotebook契約

2026-07-28にStage A実装が承認されたため、Jupytext percent形式のcompact
self-contained train / inference候補と専用contract testsを実装した。その後の
実行承認で正規train Notebookを採用し、private CPU version 1を実行した。
inference候補はfail-closedのままとする。

Notebook構成:

1. Imports
2. Runtime / binary-gzip / SHA helpers
3. exp404 frozen artifact and contract checks
4. Prediction identity freeze
5. Late truth / fold / hidden-like joins
6. Direct, scope, fold, blend, and well-tail metrics
7. Promotion gate and artifacts

## 再現性設計

- Stage A RNG: なし。保存predictionのdeterministic readout。
- seed policy:
  frozen exp404 / exp072 stable SHA256 seed bankを継承し、Stage Aでは再生成しない。
- SHA:
  exp404 raw、decompressed、logical、schema、scientific contractをすべてmandatoryにする。
- truth分離:
  control/candidate row bundleとlogical SHAをfreezeするまでtruth/error/fold/roleを読まない。
- gzip:
  raw gzip SHAとdecompressed CSV SHAを分け、content比較はdecompressed SHAを使う。
- batch inference:
  将来Stage Bではtrainとhidden testで同じfull-suffix likelihood式、dtype、temperature、
  seed順を使い、算術parityも同時保存する。
- deterministic anchor:
  Stage Aだけではinference anchorと呼ばない。Stage B rerunとsubmission SHAが必要。

## リスク

- 既存結果の再利用:
  exp404の主判定を都合よく読み替えない。新しい問いはaggregation差だけで、
  raw artifactから独立gateを再計算する。
- batch / causal差:
  full-suffix GRを使うためonline用途には移せない。Kaggle batch inference専用。
- long-tail:
  pooled gainが大きくても少数well regressionを隠す可能性がある。
- selection:
  temperature 5は固定し、同じOOFで他temperatureを比較しない。
- route重複:
  exp413はscale5をMLの12候補slotへ入れる別仮説。本実験はPF direct candidateの
  promotion auditで、相互のPASS/FAILを代用しない。

## 対象外

- temperature / likelihood scale探索
- seed best / median / mode / medoid / learned selector
- roughening / process noise / GR sigma変更
- exp413 selector / signed / downstream学習
- inference / submission

## Stage A結果

Kaggle version 1はtechnical gateを全PASSした。scale5は算術平均からpooled RMSEを
`0.680375810 ft`改善し、5/5 folds、全固定scope、固定HMM/LikPF 50:50を通過した。
一方でby-well delta RMSE p95は`+2.941688483 ft`、worst wellは
`+25.311274575 ft`で、事前上限`0.0 / 0.25 ft`を破った。AND gateに従い
`fixed_scale5_seed_aggregation_rejected_close_without_rescue`で閉じ、
Stage B inferenceへ進まない。
