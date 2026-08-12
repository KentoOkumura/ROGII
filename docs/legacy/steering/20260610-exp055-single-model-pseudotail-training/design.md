# 設計

## アプローチ

`exp048` の `ravaghi_single_lgbm_audit.py` をコピーし、feature variant に `training_policy` を追加する。同じ exp039 feature family を 2 つの variant で使い、学習 row selection だけを変える。

- `exp039_same_surface_control`: exp029 cutoff 0.65 rows を、従来通り距離 bucket で balance して学習する。
- `single_model_pseudotail_training`: exp051 の pseudo-tail 方針に合わせ、cutoff `[0.45, 0.65, 0.82]` filter、`max_rows_per_pseudo_tail=260`、`balanced_rows_per_bucket=60000` を適用する。

現在の local exp029 artifact は cutoff 0.65 しかないため、policy は missing cutoff を `single_lgbm_train_summary.csv` に記録し、存在する cutoff だけで実行できるようにする。multi-cutoff artifact を用意した場合は同じ config のまま full pseudo-tail training distribution として動く。

## 実験範囲

- 対象実験: `exp055_single_model_pseudotail_training`
- Route: `ml_model`
- 親実験: `exp039_ravaghi_single_lgbm_inference_submit`
- 実装親: `exp048_ravaghi_single_model_feature_parity_revisit`
- 変更する変数: training row policy、cutoff filter、per pseudo-tail cap、distance-balanced sampling
- 固定する変数: exp039 feature surface、LightGBM params、residual target、residual shrink/clip、fixed bucket-shrink、well-level split

## リスク

- リークリスク: training policy は train-fold wells だけを選択し、valid-fold wells は scoring にのみ使う。
- CV/LB 不一致リスク: exp039 Public LB 11.740、exp051/052/054 pseudo-tail 基準とは評価 surface が違う。通常 CV / same-surface holdout / Public LB を混ぜない。
- ランタイム/メモリリスク: multi-cutoff exp029 artifact は cutoff 数に応じて行数が増える。初回は cutoff 0.65 artifact で static/smoke し、full artifact がある場合だけ Kaggle full audit に進む。
