# 設計

## アプローチ

exp002 の residual model 実装を残し、train notebook を variant runner に拡張する。各 variant は base config に nested override を 1 つだけ適用し、同じ fold map、同じ target (`TVT - last_anchor_tvt`)、同じ `HistGradientBoostingRegressor` で OOF RMSE を比較する。

実装する比較:

- `control_exp002`: exp002 設定の再実行。
- `sample_per_well_400`: `max_train_rows_per_well` を 800 から 400 に下げる。
- `sample_total_200k`: CV の total sampling cap を 300k から 200k に下げ、inference 用 `max_train_rows_final` も 450k から 300k に下げる。
- `shrink_100`: `residual_shrink` を 0.85 から 1.00 に上げる。
- `feature_no_gr_signal`: raw / derived GR signal features を落とし、trajectory と prefix TVT 系だけを使う。

## 実験範囲

- 対象実験: `exp003_residual_ablation`
- 親実験: `experiments/exp002_drift_minimal`
- 変更する変数: sampling cap、residual shrink、feature set
- 固定する変数: GroupKFold split、score mask、target、model class、seed、train-only formation columns 不使用、submission schema

## リスク

- リークリスク: exp002 と同じく、evaluation zone の `TVT` は target/residual 作成にだけ使う。feature set ablation でも train-only formation columns は使わない。
- CV/LB 不一致リスク: public test は小さい visible set のため、variant 選定は public LB より well-level CV を優先する。
- ランタイム/メモリリスク: 5 variants x 5 folds の HGB fit になる。Kaggle runtime が重い場合は `EXPERIMENT_VARIANT_LIMIT` で先頭 variant だけに絞れる。
