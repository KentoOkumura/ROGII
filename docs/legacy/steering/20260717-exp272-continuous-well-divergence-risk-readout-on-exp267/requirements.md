# 要件

## 依頼

`continuous_well_divergence_risk_readout_on_exp267` を
`exp272_continuous_well_divergence_risk_readout_on_exp267` として実装する。

exp267 が保存した target-free 18 次元 well 署名を K=3 へ離散化せず、exp264 Stage B の
OOF candidate score に含まれる actual absolute error と predicted absolute error の calibration
bias が、連続 divergence 軸に沿って単調に変化するかを 0 booster で監査する。

## 仮説

K=3 の occupancy/profile が不安定でも、target-free divergence の連続順位には candidate-bank
error と calibration drift の fold-stable な単調信号が残る。

## 制約

- Route: `ensemble`。PF/Beam を含む 6 primitive candidate の target-free divergence と、
  exp264 learned selector score の関係を監査するため。
- 主判定軸は early/middle/late × bank range mean/p90・pair absolute gap mean/p90 の
  12 特徴を outer-train だけで median 補完・RobustScaler・clip し、等重み平均する。
- PCA1 は 18 特徴を outer-train だけで前処理・fit し、主判定軸との outer-train 相関で符号を
  固定する report-only sensitivity とする。outer-valid score で軸、符号、特徴 subset を選ばない。
- exp267 の K=3、segment、clip、特徴 subset、candidate winner threshold を再探索しない。
- LightGBM 学習、selector 学習、hard routing、inference、submission は行わない。
- exp267 保存署名と exp264 Stage B v2 candidate score の SHA を fail-closed で照合する。
- well-bootstrap は outer fold 内で復元抽出する deterministic stratified bootstrap とし、
  stable seed を用いる。global RNG や thread scheduling に依存しない。
- Kaggle CPU notebook を最初の full readout 実行先とする。実装ターンでは push しない。

## 受け入れ基準

- 773 wells / 5 outer folds / 18 署名 / 6 primitive candidates の入力契約を検証する。
- 各 outer fold の outer-train だけで fit した主軸と PCA1 を outer-valid wells に OOF 付与する。
- well×candidate と candidate-bank 平均について、actual MAE / calibration bias の fold 別・pooled
  Spearman を保存する。
- candidate-bank 平均の主軸について stratified well-bootstrap 95% 区間を保存する。
- 主判定は actual MAE が 5/5 folds で正、calibration bias が 5/5 folds で負、かつ pooled
  Spearman 95% 区間が actual MAE で `[0.05, +inf)`、calibration bias で
  `(-inf, -0.05]` を満たす場合だけ PASS とする。
- PCA1 と candidate 別結果は report-only で、主判定の救済に使わない。
- config、readable Jupytext notebook、disabled inference、tests、README、SESSION_NOTES、
  result、metrics、Kaggle package を揃える。
- input / schema / logical content / preprocessor / readout artifact SHA を記録する。
- model / prediction / submission は生成しないため、それぞれの SHA は対象外と明記する。
