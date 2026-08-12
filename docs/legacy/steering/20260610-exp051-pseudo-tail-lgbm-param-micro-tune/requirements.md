# 要件

## 依頼

`pseudo_tail_lgbm_param_micro_tune` を `exp051_pseudo_tail_lgbm_param_micro_tune` として実装する。

## 制約

- Route: `ml_model`
- 親実験は `exp026_pseudo_tail_bucket_shrink_inference_submit`、実装の土台は `exp049_xgboost_pseudo_tail_residual` とする。
- `exp026` の pseudo-tail 3 cutoffs、distance-balanced sampling、no-GR feature set、residual shrink、固定 `exp014_bucket_shrink_params` を維持する。
- 広い Optuna や多数パラメータ同時探索はしない。
- `num_leaves`、`min_child_samples`、`subsample`、`colsample_bytree`、`reg_lambda`、row cap の狭い変更だけを比較する。
- exp044 補助 fold に合わせたチューニングはしない。主評価 GroupKFold を一次判定にする。
- 推論 port と提出はこの実験範囲に含めない。

## 受け入れ基準

- `config.yaml` に control と micro tune 候補が明示されている。
- variant ごとの LightGBM parameter override と row cap override が training loop で反映される。
- raw CV と固定 bucket-shrink 後 CV が variant 別に保存される。
- fold 別、distance bucket 別、予測範囲確認に必要な CSV/JSON 生成物が保存される。
- Kaggle train notebook を準備できる構造検証が通る。
