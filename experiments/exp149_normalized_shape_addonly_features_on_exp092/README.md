# exp149_normalized_shape_addonly_features_on_exp092

## 概要

exp092 の U-projection correction / disagreement LightGBM surface に、well-local に正規化した shape 特徴を add-only で入れる実験。

PF/Beam 候補を直接置換したり hard switch したりせず、`pf_ancc`、`pf_z`、`beam_mean`、`beam_med`、`likpf_mean` を `U = TVT + Z - (T0 + Z0)` に写し、MD tail scale と target-free U scale で正規化した形状だけを LightGBM に渡す。

## 仮説

exp092 は U-projection correction / disagreement で強く改善したが、raw TVT/MD スケールに依存する情報が残る。候補 path の normalized U shape、slope、curvature、polynomial residual、candidate 間 disagreement を追加すれば、target を変えずに exp092 が外れやすい regime を補助的に表現できる可能性がある。

## 検証方針

GroupKFold by well の full-row exp092 surface 上で `normalized_shape_addonly` を学習する。既存 exp092 metrics を baseline とし、control 再学習は明示承認なしに行わない。

初回 Kaggle train 対象は `normalized_shape_addonly` 1 variant、LightGBM 3 config、5 folds、合計 15 boosters。

## 比較

- baseline: `exp092_u_projection_correction_disagreement_fullrun` `lgb1` CV 9.322479896 / Public LB 8.350
- `normalized_shape_addonly`: exp092 surface に normalized geometry / candidate shape / normalized disagreement features を追加
- reference: `exp098_selector_rank_slot_features_on_exp073`、`exp139_exp092_exp098_small_rank_slot_merge`

## 生成物

- `exp149_normalized_shape_addonly_features_on_exp092_metrics.csv`
- `exp149_normalized_shape_addonly_features_on_exp092_by_well.csv`
- `exp149_normalized_shape_addonly_features_on_exp092_bucket_metrics.csv`
- `exp149_normalized_shape_addonly_features_on_exp092_projection_feature_summary.csv`
- `exp149_normalized_shape_addonly_features_on_exp092_shape_feature_summary.csv`
- `exp149_normalized_shape_addonly_features_on_exp092_feature_importance.csv`
- `exp149_normalized_shape_addonly_features_on_exp092_feature_importance_mean.csv`
- `exp149_normalized_shape_addonly_features_on_exp092_feature_importance_mean_top.png`
- `exp149_normalized_shape_addonly_features_on_exp092_predictions.csv.gz`
- `exp149_normalized_shape_addonly_features_on_exp092_feature_schema.csv`
- `exp149_normalized_shape_addonly_features_on_exp092_lgb_models/manifest.json`
- `exp149_normalized_shape_addonly_features_on_exp092_summary.json`

## 状態

Kaggle train v1 完了。train-side OOF は positive。raw-test parity / hidden-like stress 未確認のため submit は未実施。

## 所見

`normalized_shape_addonly` は exp092 に対して `lgb1` -0.006634、`lgb2` -0.011013、`lgb_mean` -0.001376 改善した。normalized shape features は feature importance 上位に入り、U-state shape 表現は有効信号として支持された。

一方で worst well は残り、raw-test feature parity と exp115 hidden-like stress は未確認。現時点では direct inference port / submit はしない。
