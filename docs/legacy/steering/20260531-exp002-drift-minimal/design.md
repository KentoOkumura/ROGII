# 設計

## アプローチ

`TVT` を直接外挿せず、`last_anchor_tvt` を基準値として `TVT - last_anchor_tvt` を row-level regression target にする。モデルは Kaggle offline で標準的に使える `sklearn.ensemble.HistGradientBoostingRegressor` を使い、学習行数を config で cap して runtime と memory を抑える。

特徴量は推論時にも存在する horizontal well の列だけから作る。

- anchor からの距離: `delta_md`, `delta_x`, `delta_y`, `delta_z`, `delta_xy`, `delta_xyz`
- prefix 情報: `last_known_tvt`, `recent_tvt_slope`, prefix GR mean/std/slope
- row trajectory: `MD`, `X`, `Y`, `Z`, `GR`, local `dz/dmd`
- GR 形状: trailing rolling mean/std と anchor/prefix との差分
- evaluation zone 内の相対位置

## 実験範囲

- 対象実験: `exp002_drift_minimal`
- 親実験: `experiments/exp001_baseline`
- 変更する変数: `last_anchor` residual target、推論可能な trajectory / GR / prefix 特徴、HistGradientBoosting residual model
- 固定する変数: GroupKFold、score mask、submission schema、Kaggle offline/runtime path 方針、train-only formation columns 不使用

## リスク

- リークリスク: target 由来特徴は prefix の `TVT_input` だけに限定する。evaluation zone の真値 `TVT` は target/residual 作成以外に使わない。
- CV/LB 不一致リスク: 公開 test は 3 well 例で hidden と異なるため、CV 改善が小さい場合は提出優先度を下げる。
- ランタイム/メモリリスク: row 数が多いため fold ごとに training rows を sample する。full CV が重い場合は debug 結果と validation のみ記録し、Kaggle train notebook で確定する。
