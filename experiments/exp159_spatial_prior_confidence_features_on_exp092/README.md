# exp159_spatial_prior_confidence_features_on_exp092

## 状態

実装済み、未実行。Colab 前提の train-side add-only feature audit。

## 仮説

exp114 の spatial neighbor prior は直接補正では強い信号を持つ一方で worst-well regression が大きい。exp118 では exp092 への tiny gated correction が微小改善し、exp129 では spatial candidate に oracle headroom があるが learned selector は崩れた。

したがって spatial prior を直接 TVT candidate / correction / hard selector として使わず、exp092 系 LightGBM に prior value、neighbor quality、PF/Beam/likPF disagreement、high-drift context を confidence feature として渡す。

## 検証方針

GroupKFold by well の full-row exp092 surface 上で `spatial_prior_confidence_addonly` を学習する。既存 exp092 metrics を baseline とし、control 再学習は行わない。

初回 Colab train 対象は 1 variant、LightGBM 3 config、5 folds、合計 15 boosters。

比較基準:

- exp092 `lgb1` CV 9.322479896 / Public LB 8.350
- exp118 best tiny gate RMSE 9.321625436
- exp148 current submitted anchor CV 8.501281182 / Public LB 7.960

## 所見

未実行。OOF、worst-well、near-row、`1000_plus` longtail、feature importance、exp115 hidden-like stress、raw-test/full-train parity を確認するまで inference port / submit はしない。
