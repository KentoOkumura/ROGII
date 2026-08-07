# exp272 continuous well divergence risk readout on exp267

> candidate actual MAE readoutだけ保持。exp264 predicted error由来calibrationと総合guardは無効。

## 状態

- Kaggle CPU train v1完了、primary guard FAIL、branch closed
- Route: `ensemble`
- 0 variant / 0 LightGBM config / 0 trained fold / 0 booster
- inference / submission: disabled

## 仮説

exp267 の K=3 cluster は occupancy と profile 再現性を失敗したが、calibration の high-low 方向は
5/5 folds で一致した。18 次元署名を離散化せず、target-free な連続 divergence 軸として使えば、
candidate-bank の actual MAE 増加と calibration bias 低下が outer folds をまたいで単調に残る可能性がある。

## 変更点

exp267 の KMeans assignment / centroid / soft membership は使わず、保存済み18署名を
fold-safeな連続軸へ変換する。exp264 scoreはaxis fit後にだけ接続し、予測やモデルは作らない。

## 検証方針

主軸は early/middle/late の bank range mean/p90 と pair absolute gap mean/p90、計 12 特徴を
outer-train median + RobustScaler + clip 後に等重み平均する。PCA1 は全 18 特徴を outer-train
だけで fit し、主軸との outer-train 相関で符号を固定する report-only sensitivity とする。

exp264 Stage B v2 score を 6 primitive candidates に限定して streaming 集約し、well×candidate と
candidate-bank 等重み平均について actual MAE / calibration bias の fold/pooled Spearman を保存する。
主判定では主軸だけを使い、5/5 folds の方向一致と stratified well-bootstrap 95% 区間の
非自明な効果下限を要求する。

## 所見

primary actual MAE Spearmanは5/5 foldsで正、pooled `0.785818`、bootstrap 95% interval
`[0.749473, 0.817829]`だった。一方、calibration biasは負方向1/5、pooled `0.040968`、interval
`[-0.032918, 0.115824]`で必須guardを失敗した。連続軸はactual-error riskをよく表すが、
事前固定したcalibration drift仮説は再現しないためadd-only候補を支持しない。PCA1やcandidate別の
良い結果で救済せず、K=3、segment、clip、特徴subsetの事後gridも行わない。

## 主要ファイル

- `config.yaml`: 入力 SHA、主軸、bootstrap、guard、0-booster 契約
- `exp272_continuous_well_divergence_risk_readout_on_exp267_train.py/.ipynb`: Kaggle CPU readout
- `exp272_continuous_well_divergence_risk_readout_on_exp267_inference.py/.ipynb`: disabled guard
- `src/continuous_well_divergence_risk.py`: fold-safe axis、stream 集約、Spearman、bootstrap、保存
- `kaggle/output/train_v1/artifacts/`: Kaggle v1 readout、guard、再現性manifest
