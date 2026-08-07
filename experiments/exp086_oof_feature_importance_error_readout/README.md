# exp086_oof_feature_importance_error_readout

## Status

Kaggle train v1 completed. This is a diagnostic readout only; no inference or submission was created.

## Hypothesis

exp073 の raw deterministic OOF 誤差は、LightGBM が高く使っている PF/Beam / dense surface / likelihood-PF 系特徴の特定 value bucket に集中している可能性がある。重要特徴量と誤差 bucket を先に読むことで、次の confidence feature、sample weight、guard 付き後処理候補を絞る。

## Scope

- `exp077_full_replay_postprocess_guard` の policy prediction と fold 平均 feature importance を読む。
- `exp072_exp063_full_replay_feature_cache` の full replay train feature cache から、上位重要特徴量と診断用列だけを読む。
- exp073 baseline policy と exp077 best policy を、全体・feature quantile・well 単位で比較する。
- 新しいモデル学習、推論、提出は行わない。

## Validation Strategy

OOF 診断専用。`baseline_exp073_lgb_mean` と `longtail_likpf_tiny_gate_w006` の同一 OOF rows を比較し、重要特徴量ごとの worst quantile、error lift、誤差相関、well summary を保存する。結果は次実験の仮説選定に使い、anchor 更新根拠にはしない。

## Findings

Kaggle train v1 completed on `kentookumura/exp086-oof-feature-importance-error-readout-train`.

- rows / wells: 3,783,989 / 773
- baseline policy: `baseline_exp073_lgb_mean`
- baseline RMSE: `9.52637482601992`
- compare policy: `longtail_likpf_tiny_gate_w006`
- compare RMSE: `9.47051480056479`
- delta vs baseline: `-0.05586002545513047`
- output path: `/tmp/kaggle-output/exp086_oof_feature_importance_error_readout/train_v1`

Worst feature-value buckets by baseline MAE lift are concentrated in `pf_vs_dense`, dense TVT features (`tvt_densew_d`, `tvt_dense50_d`, `tvt_dense_d`), `pf_vs_z`, `dense_dist`, `dz`, `beam_std_d`, `slp_b_d_50`, and `likpf_mean_d`. The strongest absolute-error correlations among the selected important features are `beam_std_d`, `dense_dist`, `dense_nb_std`, `eval_len`, and `tvt_densew_d`.

The readout supports using PF/Beam/dense-surface disagreement and uncertainty as confidence/sample-weight material, not direct PF/Beam replacement.

## Expected Outputs

- `exp086_oof_feature_importance_error_readout_policy_metrics.csv`
- `exp086_oof_feature_importance_error_readout_feature_summary.csv`
- `exp086_oof_feature_importance_error_readout_feature_quantile_metrics.csv`
- `exp086_oof_feature_importance_error_readout_well_summary.csv`
- `exp086_oof_feature_importance_error_readout_feature_error_lift_top20.png`
- `exp086_oof_feature_importance_error_readout_feature_error_correlation_top20.png`
- `exp086_oof_feature_importance_error_readout_summary.json`
