# 要件

## 依頼

`xgboost_pseudo_tail_residual` を実装する。`exp026` の pseudo-tail residual 手順を保ち、残差モデルだけを XGBoost に差し替えた train-side CV 実験として作る。

## 制約

- Route: `ml_model`
- 主評価は既存の well-level GroupKFold を維持する。
- pseudo-tail cutoff、distance-balanced sampling、row cap、no-GR feature set、residual shrink は `exp026` 相当で固定する。
- 固定 bucket shrink は `exp025` で選択済みの `exp014_bucket_shrink_params` をそのまま使い、この実験の OOF に合わせて refit しない。
- direct PF/Beam replacement、Ravaghi feature、推論 port、提出は今回の範囲に含めない。

## 受け入れ基準

- `experiments/exp049_xgboost_pseudo_tail_residual/` に self-contained な実験がある。
- `config.yaml` に `experiment.route: ml_model` と親実験、XGBoost パラメータ、固定比較基準が記録されている。
- train notebook は設定確認、既存 OOF reference、XGBoost pseudo-tail CV、metrics/生成物保存をセル単位で追える。
- `baseline.py` は `XGBRegressor` を設定から作れる。
- raw XGBoost と fixed bucket-shrink 後候補が別 variant として metrics に保存される。
- validation と静的チェックが通る。
