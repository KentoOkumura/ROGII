# exp224_well_scaled_z_dz_features_on_exp218 結果

## 状態

Kaggle CPU split train 完了。不採用。inference / submit には進めない。

## 仮説

well 内の `z` / `dz` / `dzdmd` / `slp_z` を median / MAD / IQR / p05-p95 range / rank で robust scale し、exp218 の GRWR feature surface に add-only する。

## 設定

- route: `ml_model`
- parent: `exp218_gr_wavelet_rotation_confidence_features_on_exp148`
- active variant: `well_scaled_z_dz_addonly`
- runtime: CPU
- train split: `train_lgb0` / `train_lgb1` / `train_lgb2`
- control retraining: なし
- planned boosters: 各 split 1 LightGBM config x 5 folds = 5 boosters、合計 15 boosters

## 結果

2026-07-09 に以下の CPU split train を push し、3 本とも `KernelWorkerStatus.COMPLETE` を確認した。

| split | Kaggle kernel | RMSE TVT |
| --- | --- | ---: |
| lgb0 | `kentookumura/exp224-wsz-exp218-lgb0` v1 | 8.683606336 |
| lgb1 | `kentookumura/exp224-wsz-exp218-lgb1` v1 | 8.573438105 |
| lgb2 | `kentookumura/exp224-wsz-exp218-lgb2` v1 | 8.534973570 |
| 3-config `lgb_mean` | split OOF aggregate | 8.538687042 |

比較:

- exp218 parent `lgb_mean`: 8.475793752。exp224 は +0.062893290 悪化。
- exp148 feature surface `lgb_mean`: 8.501281182。exp224 は +0.037405860 悪化。
- full train coverage: 3 split すべて pass、3,783,989 rows / 773 wells。

## 解釈

well-scaled z/dz/dzdmd/slp_z は add-only feature としては exp218 anchor を改善しなかった。`wsz_dz_over_likpf_tvt_p05_p95_range` は feature importance 上位に出るが、CV は exp218 / exp148 の両方に届かないため採用しない。

aggregate artifacts:

- `kaggle/output/train_split_aggregate_v1/artifacts/exp224_well_scaled_z_dz_features_on_exp218_split_aggregate_summary.json`
- `kaggle/output/train_split_aggregate_v1/artifacts/exp224_well_scaled_z_dz_features_on_exp218_split_aggregate_metrics.csv`
- `kaggle/output/train_split_aggregate_v1/artifacts/exp224_well_scaled_z_dz_features_on_exp218_split_aggregate_by_well.csv`
- `kaggle/output/train_split_aggregate_v1/artifacts/exp224_well_scaled_z_dz_features_on_exp218_split_aggregate_feature_importance_mean.csv`

worst wells は `86454a6f` RMSE 48.803342、`fb03ae90` RMSE 46.074345、`1b1eba53` RMSE 43.999814。aggregate bucket は local exp072 `md_since` cache がないため tail-rank のみ厳密に再計算し、distance bucket は各 split の既存 CSV を連結して保存した。

## 次

この候補は終了。inference port、saved-booster aggregate manifest 作成、submit は行わない。後続では `likpf_mean_d` 由来 scale を correction に使わず、readout 専用または極小 feature subset に限定する。
